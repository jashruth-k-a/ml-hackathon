"""Generate a Folium exploration map for training trip 00000001."""
from pathlib import Path

import folium


root = Path(__file__).resolve().parent
trip = root / "Student folder" / "train" / "00000001"


def rows(extension, convert):
    with (trip / f"{trip.name}.{extension}").open() as source:
        return [tuple(map(convert, row.split())) for row in source if row.strip()]


gps = [(lat, lon) for lon, lat, timestamp in rows("track", float)]
nodes = [(lat, lon) for lon, lat in rows("nodes", float)]
arcs = rows("arcs", int)
route = [arcs[index] for (index,) in rows("route", int)]
latitudes, longitudes = zip(*gps)
bounds = [(min(latitudes), min(longitudes)), (max(latitudes), max(longitudes))]
south, west = bounds[0]
north, east = bounds[1]
roads = []
for start, end in arcs:
    a, b = nodes[start], nodes[end]
    if (max(a[0], b[0]) >= south and min(a[0], b[0]) <= north
            and max(a[1], b[1]) >= west and min(a[1], b[1]) <= east):
        roads.append([a, b])

view = folium.Map(prefer_canvas=True)
network = folium.FeatureGroup(name="Road network near GPS track").add_to(view)
folium.PolyLine(roads, color="gray", weight=1).add_to(network)
reference = folium.FeatureGroup(name="Reference route (green)").add_to(view)
folium.PolyLine([[nodes[a], nodes[b]] for a, b in route], color="green",
                weight=4).add_to(reference)
observations = folium.FeatureGroup(name="GPS observations (blue)").add_to(view)
for point in gps:
    folium.CircleMarker(point, radius=2, color="blue", weight=1,
                        fill=True).add_to(observations)
view.fit_bounds(bounds)
folium.LayerControl(collapsed=False).add_to(view)
view.save(str(root / "training_trip_00000001.html"))
print(f"Mapped {len(gps)} observations, {len(route)} route entries and {len(roads)} nearby arcs.")
