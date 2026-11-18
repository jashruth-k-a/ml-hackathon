"""Choose a motion scale on the first five training trips, before student evaluation."""
import csv
import joblib
import numpy as np
from decoder import decode, metrics
from prepare import ROOT, load
from train_models import FEATURES, OUT


def main():
    model = joblib.load(OUT / (OUT / "selected_model.txt").read_text().strip())
    records = []
    for trip in sorted((ROOT/"Student folder/train").iterdir())[:5]:
        folder = ROOT/"prepared/train"
        data = np.load(folder/(trip.name+".features.npz"))["candidates"]
        ref = np.load(folder/(trip.name+".labels.npz"))["reference_arc"]
        arcs = load(trip,"arcs",int)
        times = load(trip,"track",float)[:,2]
        scores = model.predict_proba(np.column_stack([data[k] for k in FEATURES]).astype("float32"))[:,1]
        for scale in [5,15,40]:
            _, path, _ = decode(data,scores,arcs,times,motion_scale=scale)
            records.append((trip.name,scale,len(ref),int((path==ref).sum())))
    with (OUT/"motion_trials.csv").open("w", newline="") as f:
        w=csv.writer(f); w.writerow(["trip","scale","observations","correct"]); w.writerows(records)
    best=max([5,15,40],key=lambda scale:sum(r[3] for r in records if r[1]==scale))
    (OUT/"motion_scale.txt").write_text(str(best))
    print("selected scale",best)


if __name__=="__main__":
    main()
