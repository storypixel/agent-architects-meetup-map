# Agent Architects · Meetup City Analysis

Interactive map of [Agent Architects](https://www.skool.com/agent-architects) Skool community members, k-means clustered into theoretical meetup hubs at k = 10, 15, 20, 25, 30.

**Live:** https://iamnotsam.com/agent-architects-meetup-map/

## What's in here

| File | Purpose |
|---|---|
| `members.csv` | Raw input. One row per member: `lat,lng`. Public dataset; no PII. |
| `scripts/scrape.py` | Fetches member coordinates from Skool, writes `members.csv`. |
| `scripts/cluster.py` | Reads `members.csv`, runs k-means + metro lookup, writes `data.json`. |
| `data.json` | Computed output. Consumed by `index.html`. |
| `index.html` | The visualization. Leaflet + CARTO tiles, fetches `data.json`. |

The methodology is the code. Read `scripts/cluster.py` to see exactly how host cities are picked.

## Run it against your own community

You need [`uv`](https://github.com/astral-sh/uv) and a Skool account that's a member of the community you want to analyze.

```bash
uv sync

# 1. Get your auth_token cookie:
#    - log into skool.com in any browser
#    - devtools -> Application -> Cookies -> https://www.skool.com
#    - copy the value of the `auth_token` cookie
export SKOOL_AUTH_TOKEN=<paste here>

# 2. Scrape coordinates (writes members.csv)
uv run python scripts/scrape.py --community agent-architects

# 3. Cluster + reverse-geocode (writes data.json)
uv run python scripts/cluster.py

# 4. Serve the page locally
python3 -m http.server 8000
# open http://localhost:8000
```

To analyze a different community, change `--community` to that community's slug (the part of the URL after `skool.com/`).

## Algorithm

1. **Reverse-geocode** each member's coordinates to a city, falling back to the offline `reverse_geocoder` dataset for points outside the major-metro list.
2. **Snap to metro.** Members within ~75km of a major world metro (NYC, LA, London, Tokyo, etc — see `METROS` in `cluster.py`) roll up to that metro. This collapses NYC's boroughs into "New York" and LA's suburbs into "Los Angeles" instead of leaving them as 30+ tiny towns.
3. **K-means** on raw lat/lng for k ∈ {10, 15, 20, 25, 30}. Treats coordinates as Euclidean — fine at this scale for picking hubs.
4. **Host city per cluster** = the metro most cluster members already live in. Tiebreak by global metro size (a tie of 9 Austin / 9 Dallas resolves to whichever metro is bigger overall in the dataset, so reruns are stable).
5. **Travel distance** = haversine from each member to the host-metro centroid. Not driving distance.

K-means is initialization-sensitive. Reruns with the same data + same `RANDOM_STATE` (currently 42) produce identical clusters.
