# Agent Architects · Meetup City Analysis

Interactive map of [Agent Architects](https://www.skool.com/agent-architects) Skool community members, k-means clustered into theoretical meetup hubs at k = 10, 15, 20, 25, 30.

**Live:** https://iamnotsam.com/agent-architects-meetup-map/

## What's in here

| File | Purpose |
|---|---|
| `members.csv` | Raw input. One row per member: `lat,lng,points`. No user IDs / PII. |
| `scripts/scrape.py` | Fetches member coordinates + activity points from Skool, writes `members.csv`. |
| `scripts/cluster.py` | Reads `members.csv`, runs k-means + metro lookup, writes `data.json`. |
| `data.json` | Computed output. Consumed by `index.html`. |
| `index.html` | The visualization. Leaflet + CARTO tiles, fetches `data.json`. |

## Two views

The page has a Members / Activity toggle at the bottom.

- **Members** weights every member equally. "Where do MOST people live?"
- **Activity** weights each member by their all-time Skool points. "Where do ENGAGED people live?" Centroids pull toward heavy contributors. James's question.

Both views are URL-shareable:

- `?view=activity` &mdash; opens Activity view by default
- `?k=20` &mdash; opens 20 hubs
- `?view=activity&k=20` &mdash; combined

The methodology is the code. Read `scripts/cluster.py` to see exactly how host cities are picked.

## Run it against your own community

You need [`uv`](https://github.com/astral-sh/uv) and a Skool account that's a member of the community you want to analyze.

```bash
uv sync
uv run playwright install chromium

# One-time login. Opens a browser; log into Skool, then close the window.
# Session cookies persist in ~/.aa-meetup-map-profile/ for all future runs.
uv run python scripts/scrape.py --login

# Scrape coordinates + activity points (writes members.csv)
uv run python scripts/scrape.py --community agent-architects

# Cluster + reverse-geocode (writes data.json)
uv run python scripts/cluster.py

# Serve the page locally
python3 -m http.server 8000
# open http://localhost:8000
```

To analyze a different community, pass `--community <slug>` (the part of the URL after `skool.com/`).

## Weekly auto-refresh (macOS, launchd)

`scripts/refresh.sh` runs scrape + cluster + commit + push. To schedule it Sundays at 7am local time:

```bash
mkdir -p ~/.local/share/aa-meetup-map
cp scripts/com.bonayrindustries.aa-meetup-map.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.bonayrindustries.aa-meetup-map.plist

# Run once now to verify it works:
launchctl start com.bonayrindustries.aa-meetup-map
tail -f ~/.local/share/aa-meetup-map/stderr.log
```

If a Skool session expires (rare; the cookie is good for ~1 year), the cron will fail with "auth state likely expired" in stderr.log. Re-run `uv run python scripts/scrape.py --login` to refresh.

## Algorithm

1. **Reverse-geocode** each member's coordinates to a city, falling back to the offline `reverse_geocoder` dataset for points outside the major-metro list.
2. **Snap to metro.** Members within ~75km of a major world metro (NYC, LA, London, Tokyo, etc — see `METROS` in `cluster.py`) roll up to that metro. This collapses NYC's boroughs into "New York" and LA's suburbs into "Los Angeles" instead of leaving them as 30+ tiny towns.
3. **K-means** on raw lat/lng for k ∈ {10, 15, 20, 25, 30}. Treats coordinates as Euclidean — fine at this scale for picking hubs.
   - In **Activity** view, k-means is fit with `sample_weight=points`. Centroids are pulled toward heavy contributors and per-cluster size becomes the sum of points instead of the member count.
4. **Host city per cluster** = the metro most cluster members already live in (raw count, even in Activity view — a meetup happens where most people live, not where the heaviest one does). Tiebreak by global metro size (a tie of 9 Austin / 9 Dallas resolves to whichever metro is bigger overall, so reruns are stable).
5. **Travel distance** = haversine from each member to the host-metro centroid. Not driving distance.

K-means is initialization-sensitive. Reruns with the same data + same `RANDOM_STATE` (currently 42) produce identical clusters.
