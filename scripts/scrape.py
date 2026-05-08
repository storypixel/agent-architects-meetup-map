"""
Scrape member coordinates and activity points from a Skool community.

Reads:
  - SKOOL_AUTH_TOKEN env var (your `auth_token` cookie value, copied from devtools)
  - --community <slug>   e.g. agent-architects

Writes:
  - members.csv with columns lat,lng,points

How it works:
  1. GET /<slug>/-/map -> parse __NEXT_DATA__ for `dataUrl` (signed CDN URL)
     -> GET dataUrl -> array of {u: user_id, p: [lat, lng]} for every member
     who has set a location.
  2. For each user_id, GET /@<user_id>?g=<slug> and parse __NEXT_DATA__ for
     `currentUser.metadata.spData` (a JSON string with `pts` = all-time points).
     This is the only reliable way to get points across all members; Skool's
     /members listing is hard-capped at 30 entries with broken `?p=` pagination.
  3. Drop user_ids on output. The CSV only has lat,lng,points so engagement
     numbers can't be linked back to individuals.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import html
import json
import os
import re
import sys
from pathlib import Path

import httpx

NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>',
    re.DOTALL,
)
CONCURRENCY = 12  # parallel profile fetches; raise if you don't get rate-limited


def fetch_map_points(client: httpx.Client, community: str) -> list[tuple[str, float, float]]:
    """Return [(user_id, lat, lng), ...] for every member with a location set."""
    r = client.get(f"https://www.skool.com/{community}/-/map")
    r.raise_for_status()
    m = NEXT_DATA_RE.search(r.text)
    if not m:
        raise SystemExit("could not find __NEXT_DATA__ on map page; check auth_token")
    payload = json.loads(html.unescape(m.group(1)))
    data_url = payload.get("props", {}).get("pageProps", {}).get("dataUrl")
    if not data_url:
        raise SystemExit("__NEXT_DATA__ missing pageProps.dataUrl; layout changed")
    r2 = httpx.get(data_url, timeout=30)
    r2.raise_for_status()
    return [
        (entry["u"], entry["p"][0], entry["p"][1])
        for entry in r2.json()
        if "u" in entry and "p" in entry
    ]


async def _fetch_pts(client: httpx.AsyncClient, user_id: str, community: str) -> int:
    try:
        r = await client.get(f"https://www.skool.com/@{user_id}?g={community}")
    except httpx.HTTPError:
        return 0
    if r.status_code != 200:
        return 0
    m = NEXT_DATA_RE.search(r.text)
    if not m:
        return 0
    try:
        payload = json.loads(html.unescape(m.group(1)))
        sp = payload["props"]["pageProps"]["currentUser"]["metadata"].get("spData")
        if not sp:
            return 0
        return int(json.loads(sp).get("pts", 0))
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0


async def fetch_all_points(
    user_ids: list[str], community: str, auth_token: str
) -> dict[str, int]:
    sem = asyncio.Semaphore(CONCURRENCY)
    results: dict[str, int] = {}

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (meetup-map scraper)"},
        cookies={"auth_token": auth_token},
        follow_redirects=True,
        timeout=30,
    ) as client:
        async def one(uid: str, idx: int) -> None:
            async with sem:
                pts = await _fetch_pts(client, uid, community)
                results[uid] = pts
                if (idx + 1) % 50 == 0 or idx + 1 == len(user_ids):
                    print(
                        f"  {idx + 1}/{len(user_ids)} profiles fetched",
                        file=sys.stderr,
                    )

        await asyncio.gather(*[one(uid, i) for i, uid in enumerate(user_ids)])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--community", default="agent-architects", help="Skool community slug")
    parser.add_argument("--out", default="members.csv", help="output CSV path")
    args = parser.parse_args()

    auth_token = os.environ.get("SKOOL_AUTH_TOKEN")
    if not auth_token:
        print(
            "error: set SKOOL_AUTH_TOKEN env var to your Skool auth_token cookie\n"
            "  - log into skool.com in a browser\n"
            "  - devtools -> Application -> Cookies -> https://www.skool.com\n"
            "  - copy the value of `auth_token` and export SKOOL_AUTH_TOKEN=<value>",
            file=sys.stderr,
        )
        return 2

    print("fetching member coordinates...", file=sys.stderr)
    with httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (meetup-map scraper)"},
        cookies={"auth_token": auth_token},
        follow_redirects=True,
        timeout=30,
    ) as client:
        map_points = fetch_map_points(client, args.community)
    print(f"  {len(map_points)} members have a location set", file=sys.stderr)

    print(
        f"fetching activity points (per-profile, {CONCURRENCY} concurrent)...",
        file=sys.stderr,
    )
    user_ids = [uid for uid, _, _ in map_points]
    points_by_user = asyncio.run(fetch_all_points(user_ids, args.community, auth_token))

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lng", "points"])
        for uid, lat, lng in map_points:
            w.writerow([lat, lng, points_by_user.get(uid, 0)])
    active = sum(1 for p in points_by_user.values() if p > 0)
    print(
        f"wrote {len(map_points)} members -> {out} ({active} with activity)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
