"""
Cluster member coordinates into theoretical meetup hubs.

Reads:
  - members.csv (columns: lat, lng)

Writes:
  - data.json (consumed by index.html)

Algorithm:
  1. Reverse-geocode each member to (city, country) using the offline
     `reverse_geocoder` package — no API key, ships its own dataset.
  2. Apply small ALIASES table to roll boroughs/suburbs up to their parent
     metro (e.g. "The Bronx" -> "New York"). The reverse-geocoder is
     accurate but too granular for "where would we actually meet up".
  3. K-means on raw lat/lng for each k in K_VALUES (10, 15, 20, 25, 30).
     Note: this treats lat/lng as Euclidean. Fine at this scale for picking
     hubs; we could swap to haversine-aware clustering later if needed.
  4. For each cluster, choose host = the metro most cluster members already
     live in (tiebreak by lowest mean haversine distance to the rest of the
     cluster — i.e. the metro that minimizes group travel).
  5. Compute per-cluster stats: n (members), local (members already in host
     metro), mean_km / max_km (haversine distance from each member to host).
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import reverse_geocoder as rg
from sklearn.cluster import KMeans

K_VALUES = [10, 15, 20, 25, 30]
RANDOM_STATE = 42
METRO_RADIUS_KM = 75  # ~1hr drive. Suburbs within this distance roll up to the metro.

# Major world metros where a meetup would plausibly happen. Members within
# METRO_RADIUS_KM of one of these centers are bucketed under the metro
# (so all NYC boroughs become "New York", all LA-area suburbs become "Los
# Angeles", etc). Members outside any metro keep whatever city name
# reverse_geocoder returns. Center coords are rough downtown / centroids.
METROS: list[tuple[str, str, float, float]] = [
    # North America
    ("New York", "United States", 40.7128, -74.0060),
    ("Los Angeles", "United States", 34.0522, -118.2437),
    ("San Francisco", "United States", 37.7749, -122.4194),
    ("Chicago", "United States", 41.8781, -87.6298),
    ("Austin", "United States", 30.2672, -97.7431),
    ("Houston", "United States", 29.7604, -95.3698),
    ("Dallas", "United States", 32.7767, -96.7970),
    ("Atlanta", "United States", 33.7490, -84.3880),
    ("Miami", "United States", 25.7617, -80.1918),
    ("Boston", "United States", 42.3601, -71.0589),
    ("Washington", "United States", 38.9072, -77.0369),
    ("Philadelphia", "United States", 39.9526, -75.1652),
    ("Seattle", "United States", 47.6062, -122.3321),
    ("Denver", "United States", 39.7392, -104.9903),
    ("Phoenix", "United States", 33.4484, -112.0740),
    ("Salt Lake City", "United States", 40.7608, -111.8910),
    ("Las Vegas", "United States", 36.1699, -115.1398),
    ("Nashville", "United States", 36.1627, -86.7816),
    ("Toronto", "Canada", 43.6532, -79.3832),
    ("Vancouver", "Canada", 49.2827, -123.1207),
    ("Montreal", "Canada", 45.5019, -73.5674),
    ("Mexico City", "Mexico", 19.4326, -99.1332),
    # Europe
    ("London", "United Kingdom", 51.5074, -0.1278),
    ("Manchester", "United Kingdom", 53.4808, -2.2426),
    ("Edinburgh", "United Kingdom", 55.9533, -3.1883),
    ("Dublin", "Ireland", 53.3498, -6.2603),
    ("Paris", "France", 48.8566, 2.3522),
    ("Berlin", "Germany", 52.5200, 13.4050),
    ("Frankfurt", "Germany", 50.1109, 8.6821),
    ("Munich", "Germany", 48.1351, 11.5820),
    ("Hamburg", "Germany", 53.5511, 9.9937),
    ("Amsterdam", "Netherlands", 52.3676, 4.9041),
    ("Brussels", "Belgium", 50.8503, 4.3517),
    ("Zurich", "Switzerland", 47.3769, 8.5417),
    ("Vienna", "Austria", 48.2082, 16.3738),
    ("Madrid", "Spain", 40.4168, -3.7038),
    ("Barcelona", "Spain", 41.3851, 2.1734),
    ("Lisbon", "Portugal", 38.7223, -9.1393),
    ("Rome", "Italy", 41.9028, 12.4964),
    ("Milan", "Italy", 45.4642, 9.1900),
    ("Athens", "Greece", 37.9838, 23.7275),
    ("Istanbul", "Turkey", 41.0082, 28.9784),
    ("Stockholm", "Sweden", 59.3293, 18.0686),
    ("Copenhagen", "Denmark", 55.6761, 12.5683),
    ("Oslo", "Norway", 59.9139, 10.7522),
    ("Helsinki", "Finland", 60.1699, 24.9384),
    ("Warsaw", "Poland", 52.2297, 21.0122),
    ("Prague", "Czechia", 50.0755, 14.4378),
    ("Budapest", "Hungary", 47.4979, 19.0402),
    ("Bucharest", "Romania", 44.4268, 26.1025),
    # Middle East / Africa
    ("Dubai", "United Arab Emirates", 25.2048, 55.2708),
    ("Abu Dhabi", "United Arab Emirates", 24.4539, 54.3773),
    ("Riyadh", "Saudi Arabia", 24.7136, 46.6753),
    ("Tel Aviv", "Israel", 32.0853, 34.7818),
    ("Cairo", "Egypt", 30.0444, 31.2357),
    ("Johannesburg", "South Africa", -26.2041, 28.0473),
    ("Cape Town", "South Africa", -33.9249, 18.4241),
    ("Lagos", "Nigeria", 6.5244, 3.3792),
    ("Nairobi", "Kenya", -1.2921, 36.8219),
    # Asia / Oceania
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Delhi", "India", 28.6139, 77.2090),
    ("Bangalore", "India", 12.9716, 77.5946),
    ("Bangkok", "Thailand", 13.7563, 100.5018),
    ("Singapore", "Singapore", 1.3521, 103.8198),
    ("Kuala Lumpur", "Malaysia", 3.1390, 101.6869),
    ("Manila", "Philippines", 14.5995, 120.9842),
    ("Jakarta", "Indonesia", -6.2088, 106.8456),
    ("Ho Chi Minh City", "Vietnam", 10.8231, 106.6297),
    ("Hong Kong", "Hong Kong", 22.3193, 114.1694),
    ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Seoul", "South Korea", 37.5665, 126.9780),
    ("Taipei", "Taiwan", 25.0330, 121.5654),
    ("Shanghai", "China", 31.2304, 121.4737),
    ("Sydney", "Australia", -33.8688, 151.2093),
    ("Melbourne", "Australia", -37.8136, 144.9631),
    ("Brisbane", "Australia", -27.4698, 153.0251),
    ("Perth", "Australia", -31.9505, 115.8605),
    ("Auckland", "New Zealand", -36.8485, 174.7633),
    # South America
    ("Sao Paulo", "Brazil", -23.5505, -46.6333),
    ("Rio de Janeiro", "Brazil", -22.9068, -43.1729),
    ("Buenos Aires", "Argentina", -34.6037, -58.3816),
    ("Santiago", "Chile", -33.4489, -70.6693),
    ("Bogota", "Colombia", 4.7110, -74.0721),
    ("Lima", "Peru", -12.0464, -77.0428),
]


def metro_for(lat: float, lng: float) -> tuple[str, str] | None:
    best = None
    best_d = METRO_RADIUS_KM
    for name, country, mlat, mlng in METROS:
        d = haversine_km(lat, lng, mlat, mlng)
        if d < best_d:
            best_d = d
            best = (name, country)
    return best


def haversine_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    R = 6371.0088
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lng - a_lng)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def load_members(path: Path) -> np.ndarray:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = [(float(r["lat"]), float(r["lng"])) for r in reader]
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return np.array(rows, dtype=float)


def reverse_geocode(points: np.ndarray) -> list[tuple[str, str]]:
    """Return [(city, country), ...] for each point.

    First tries to bucket the point into a major METRO; falls back to
    reverse_geocoder's nearest-city lookup for points outside any metro.
    """
    coords = [(float(p[0]), float(p[1])) for p in points]
    results = rg.search(coords, mode=1, verbose=False)
    out: list[tuple[str, str]] = []
    for (lat, lng), r in zip(coords, results):
        metro = metro_for(lat, lng)
        if metro is not None:
            out.append(metro)
        else:
            out.append((r["name"], country_name(r["cc"])))
    return out


# ISO-3166 alpha-2 -> human name. Extend as needed; reverse_geocoder ships
# alpha-2 codes only. We only translate codes that show up in practice; an
# unknown code falls back to the code itself, which is still readable.
COUNTRY_NAMES = {
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "AU": "Australia",
    "NZ": "New Zealand",
    "DE": "Germany",
    "FR": "France",
    "NL": "Netherlands",
    "BE": "Belgium",
    "IE": "Ireland",
    "ES": "Spain",
    "PT": "Portugal",
    "IT": "Italy",
    "CH": "Switzerland",
    "AT": "Austria",
    "DK": "Denmark",
    "SE": "Sweden",
    "NO": "Norway",
    "FI": "Finland",
    "PL": "Poland",
    "CZ": "Czechia",
    "GR": "Greece",
    "TR": "Turkey",
    "RO": "Romania",
    "HU": "Hungary",
    "BG": "Bulgaria",
    "RU": "Russia",
    "UA": "Ukraine",
    "IL": "Israel",
    "AE": "United Arab Emirates",
    "SA": "Saudi Arabia",
    "EG": "Egypt",
    "ZA": "South Africa",
    "NG": "Nigeria",
    "KE": "Kenya",
    "IN": "India",
    "PK": "Pakistan",
    "BD": "Bangladesh",
    "LK": "Sri Lanka",
    "TH": "Thailand",
    "VN": "Vietnam",
    "PH": "Philippines",
    "MY": "Malaysia",
    "SG": "Singapore",
    "ID": "Indonesia",
    "JP": "Japan",
    "KR": "South Korea",
    "CN": "China",
    "TW": "Taiwan",
    "HK": "Hong Kong",
    "MX": "Mexico",
    "BR": "Brazil",
    "AR": "Argentina",
    "CL": "Chile",
    "CO": "Colombia",
    "PE": "Peru",
    "EC": "Ecuador",
    "VE": "Venezuela",
    "CR": "Costa Rica",
    "DO": "Dominican Republic",
}


def country_name(cc: str) -> str:
    return COUNTRY_NAMES.get(cc, cc)


def cluster_summary(
    points: np.ndarray,
    metros: list[tuple[str, str]],
    k: int,
) -> list[dict]:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(points)
    labels = km.labels_
    centers = km.cluster_centers_
    global_counts = Counter(metros)

    out: list[dict] = []
    for i in range(k):
        mask = labels == i
        cluster_pts = points[mask]
        cluster_metros = [metros[j] for j in range(len(metros)) if mask[j]]
        n = int(mask.sum())
        if n == 0:
            continue

        # Pick host metro: most cluster members already live there.
        # Tiebreak by global metro size (a tie of 9 Austin / 9 Dallas resolves
        # to whichever is bigger overall in the dataset). Stable across reruns.
        counts = Counter(cluster_metros)
        top_count = counts.most_common(1)[0][1]
        candidates = [m for m, c in counts.items() if c == top_count]
        candidates.sort(key=lambda m: (-global_counts[m], m))
        host_city, host_country = candidates[0]

        # Centroid of host-metro members (tighter than KMeans centroid)
        host_pts = [
            cluster_pts[idx]
            for idx, m in enumerate(cluster_metros)
            if m == (host_city, host_country)
        ]
        if host_pts:
            host_lat = float(np.mean([p[0] for p in host_pts]))
            host_lng = float(np.mean([p[1] for p in host_pts]))
        else:
            host_lat, host_lng = float(centers[i][0]), float(centers[i][1])

        dists = [haversine_km(host_lat, host_lng, p[0], p[1]) for p in cluster_pts]

        out.append(
            {
                "i": i,
                "host": host_city,
                "country": host_country,
                "label": host_city,
                "lat": round(host_lat, 5),
                "lng": round(host_lng, 5),
                "n": n,
                "local": int(counts[(host_city, host_country)]),
                "mean_km": float(np.mean(dists)),
                "max_km": float(np.max(dists)),
            }
        )

    out.sort(key=lambda c: -c["n"])
    return out


def build_assignments(points: np.ndarray) -> dict[str, list[int]]:
    """For each k, the cluster index assigned to each member (0..k-1)."""
    out: dict[str, list[int]] = {}
    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(points)
        out[str(k)] = [int(x) for x in km.labels_]
    return out


def build_top(metros: list[tuple[str, str]]):
    metro_counts = Counter(metros)
    country_counts = Counter(c for _, c in metros)
    return (
        [
            {"city": city, "country": country, "n": n}
            for (city, country), n in metro_counts.most_common(20)
        ],
        [
            {"country": c, "n": n}
            for c, n in country_counts.most_common(20)
        ],
    )


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    members_csv = repo / "members.csv"
    out_path = repo / "data.json"

    points = load_members(members_csv)
    metros = reverse_geocode(points)

    clusters = {str(k): cluster_summary(points, metros, k) for k in K_VALUES}
    top_metros, top_countries = build_top(metros)

    data = {
        "members": [{"lat": float(lat), "lng": float(lng)} for lat, lng in points],
        "assignments": build_assignments(points),
        "clusters": clusters,
        "total": len(points),
        "ks": K_VALUES,
        "top_metros": top_metros,
        "top_countries": top_countries,
    }

    with out_path.open("w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out_path} ({len(points)} members, ks={K_VALUES})")
    print("top hubs at k=10:")
    for c in clusters["10"][:5]:
        print(f"  {c['n']:>4}  {c['host']}, {c['country']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
