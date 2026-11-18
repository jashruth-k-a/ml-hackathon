"""Run pointwise, continuity and compressed-sequence comparisons on development trips."""
import csv
import folium
import joblib
import numpy as np
from decoder import predict, metrics
from prepare import ROOT, load
from threadpoolctl import threadpool_limits


def main():
    out = ROOT / "experiments"
    model = joblib.load(out / (out / "selected_model.txt").read_text().strip())
    rows = []
    for trip in sorted((ROOT / "Student folder/student_test").iterdir()):
        raw, nodes, arcs, _, _, _, base, path, resets = predict(trip, model)
        with np.load(ROOT / "prepared/student_test" / (trip.name + ".labels.npz")) as labels:
            ref = labels["reference_arc"]
        for name, prediction in [("classifier", base), ("viterbi", path)]:
            rows.append(dict(trip=trip.name, method=name, restarts=resets if name=="viterbi" else 0,
                             **metrics(prediction, ref, arcs)))
        if len(rows) == 2:
            view = folium.Map(prefer_canvas=True)
            gps = raw[:, :2][:, ::-1]
            view.fit_bounds([gps.min(axis=0).tolist(), gps.max(axis=0).tolist()])
            xy = nodes[:, ::-1]
            low, high = gps.min(axis=0), gps.max(axis=0)
            visible = ((xy>=low)&(xy<=high)).all(axis=1)
            for title, color, selected in [
                ("Road network", "gray", np.flatnonzero(visible[arcs].any(axis=1))),
                ("Reference", "green", load(trip, "route", int).ravel()),
                ("Classifier", "orange", base), ("Viterbi", "purple", path)]:
                layer = folium.FeatureGroup(name=title).add_to(view)
                segs = xy[arcs[np.unique(selected[selected >= 0])]]
                folium.PolyLine(segs.tolist(), color=color, weight=2).add_to(layer)
            layer = folium.FeatureGroup(name="GPS observations").add_to(view)
            for position in gps:
                folium.CircleMarker(position.tolist(), radius=2, color="blue").add_to(layer)
            folium.LayerControl(collapsed=False).add_to(view)
            view.save(str(out / "decoder_map.html"))
    with (out / "decoder_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()

