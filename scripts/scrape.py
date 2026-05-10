"""
Scrape member coordinates and activity points from a Skool community.

Uses Playwright with a persistent Chromium profile so the auth state lives
on-disk in ~/.aa-meetup-map-profile/ and the cron doesn't need any token
copy-pasting. One-time setup:

    uv run python scripts/scrape.py --login

opens a non-headless browser; you log into Skool, then close. The session
cookie is now persisted in the profile dir.

Subsequent runs:

    uv run python scripts/scrape.py [--community agent-architects]

reuse the saved auth, fetch member coords from the map page's signed
dataUrl, then per-user GET each profile to read all-time activity points.
Output: members.csv with lat,lng,points (no user_ids on disk).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import html
import json
import re
import sys
from pathlib import Path

from playwright.async_api import async_playwright

PROFILE_DIR = Path.home() / ".aa-meetup-map-profile"
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
    re.DOTALL,
)
CONCURRENCY = 12


async def _login_flow(community: str) -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(PROFILE_DIR), headless=False
        )
        page = await ctx.new_page()
        await page.goto(f"https://www.skool.com/{community}")
        print(
            "log into Skool in the browser window, then close it when done",
            file=sys.stderr,
        )
        # Wait for the user to close the browser
        await ctx.wait_for_event("close", timeout=0)


def _parse_next_data(html_text: str) -> dict:
    m = NEXT_DATA_RE.search(html_text)
    if not m:
        raise SystemExit("could not find __NEXT_DATA__; auth state may have expired")
    return json.loads(html.unescape(m.group(1)))


async def scrape(community: str) -> list[tuple[float, float, int]]:
    """Returns [(lat, lng, points), ...] for every member with a location."""
    if not PROFILE_DIR.exists():
        raise SystemExit(
            f"profile dir {PROFILE_DIR} does not exist; run `uv run python "
            "scripts/scrape.py --login` once first"
        )

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(PROFILE_DIR), headless=True
        )
        page = await ctx.new_page()

        # Map data
        print("fetching member coordinates...", file=sys.stderr)
        await page.goto(f"https://www.skool.com/{community}/-/map")
        map_html = await page.content()
        next_data = _parse_next_data(map_html)
        data_url = next_data.get("props", {}).get("pageProps", {}).get("dataUrl")
        if not data_url:
            raise SystemExit("missing pageProps.dataUrl; auth state likely expired")
        map_payload = await page.evaluate(
            "url => fetch(url).then(r => r.json())", data_url
        )
        entries = [
            (e["u"], e["p"][0], e["p"][1])
            for e in map_payload
            if "u" in e and "p" in e
        ]
        print(f"  {len(entries)} members have a location set", file=sys.stderr)

        # Per-profile points
        print(
            f"fetching activity points (per-profile, {CONCURRENCY} concurrent)...",
            file=sys.stderr,
        )
        sem = asyncio.Semaphore(CONCURRENCY)
        results: dict[str, int] = {}

        async def one(uid: str, idx: int) -> None:
            async with sem:
                pts = await page.evaluate(
                    """async ([uid, slug]) => {
                        const r = await fetch(`/@${uid}?g=${slug}`);
                        const t = await r.text();
                        const m = t.match(/<script id="__NEXT_DATA__"[^>]*>(.+?)<\\/script>/s);
                        if (!m) return 0;
                        try {
                            const cu = JSON.parse(m[1]).props?.pageProps?.currentUser;
                            const sp = cu?.metadata?.spData;
                            if (!sp) return 0;
                            return JSON.parse(sp).pts || 0;
                        } catch { return 0; }
                    }""",
                    [uid, community],
                )
                results[uid] = int(pts or 0)
                if (idx + 1) % 50 == 0 or idx + 1 == len(entries):
                    print(
                        f"  {idx + 1}/{len(entries)} profiles fetched",
                        file=sys.stderr,
                    )

        await asyncio.gather(*[one(uid, i) for i, (uid, _, _) in enumerate(entries)])
        await ctx.close()

    return [(lat, lng, results.get(uid, 0)) for uid, lat, lng in entries]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--community", default="agent-architects", help="Skool community slug"
    )
    parser.add_argument("--out", default="members.csv", help="output CSV path")
    parser.add_argument(
        "--login",
        action="store_true",
        help="open a browser to log into Skool; saves session for future runs",
    )
    args = parser.parse_args()

    if args.login:
        asyncio.run(_login_flow(args.community))
        return 0

    rows = asyncio.run(scrape(args.community))
    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lng", "points"])
        w.writerows(rows)
    active = sum(1 for _, _, p in rows if p > 0)
    print(
        f"wrote {len(rows)} members -> {out} ({active} with activity)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
