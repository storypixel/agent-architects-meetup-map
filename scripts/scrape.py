"""
Scrape member coordinates from a Skool community's member map.

Reads:
  - SKOOL_AUTH_TOKEN env var (your `auth_token` cookie value, copied from devtools)
  - --community <slug>            e.g. agent-architects

Writes:
  - members.csv with columns lat,lng

How it works:
  1. GET https://www.skool.com/<slug>/-/map (with auth_token cookie).
  2. Parse the embedded __NEXT_DATA__ JSON; extract `props.pageProps.dataUrl`,
     which is a short-lived signed URL on files.skool.com.
  3. GET dataUrl. The payload is an array of {u: <user_id>, p: [lat, lng]}.
  4. Drop user IDs (we only want coords for clustering) and write CSV.
"""
from __future__ import annotations

import argparse
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


def fetch_data_url(community: str, auth_token: str) -> str:
    map_url = f"https://www.skool.com/{community}/-/map"
    with httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (meetup-map scraper)"},
        cookies={"auth_token": auth_token},
        follow_redirects=True,
        timeout=30,
    ) as client:
        r = client.get(map_url)
    r.raise_for_status()

    m = NEXT_DATA_RE.search(r.text)
    if not m:
        raise SystemExit("could not find __NEXT_DATA__ in map page; check auth_token")
    payload = json.loads(html.unescape(m.group(1)))
    data_url = payload.get("props", {}).get("pageProps", {}).get("dataUrl")
    if not data_url:
        raise SystemExit("__NEXT_DATA__ missing pageProps.dataUrl; layout may have changed")
    return data_url


def fetch_points(data_url: str) -> list[tuple[float, float]]:
    r = httpx.get(data_url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    return [(entry["p"][0], entry["p"][1]) for entry in payload if "p" in entry]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--community", default="agent-architects", help="Skool community slug"
    )
    parser.add_argument(
        "--out", default="members.csv", help="output CSV path"
    )
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

    data_url = fetch_data_url(args.community, auth_token)
    points = fetch_points(data_url)
    if not points:
        raise SystemExit("no member coordinates returned")

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lng"])
        w.writerows(points)
    print(f"wrote {len(points)} members -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
