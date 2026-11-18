# PathSnap

This Machine Learning hackathon project reconstructs vehicle routes from noisy GPS observations. This checkpoint covers data preparation, five-model comparison, Bayesian tuning and HMM/Viterbi development evaluation.

| Work in this checkpoint | Where to find it |
|---|---|
| Development data inventory, GPS quality, graph checks and training-trip map | [Data exploration results](results/data/) |
| Candidate generation and reference-label alignment | [Preparation script](prepare.py) and [verification script](verify_prepared.py) |
| Viterbi recurrence and hand-checked example | [Decoder source](src/hmm/) and [example output](results/hmm/toy_example.txt) |
| Two-laptop connectivity demonstration | [API](api/main.py), [UI](ui/index.html) and [captured evidence](results/api/) |

The networking demonstration returns the observed GPS trace. For the trained pipeline, see the [decoder guide](ROUTE_RECONSTRUCTION.md), [per-trip results](experiments/decoder_results.csv) and [interactive route map](experiments/decoder_map.html).

## Run the existing demonstrations

After installing the dependencies below, run `python -m src.hmm.toy_example` to compare the toy decoder against brute-force enumeration. To rehearse the connection, run `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000` on Laptop 1. The API reads the training trips by default. On Laptop 2 open `ui/index.html`, enter Laptop 1's LAN address and port, and select a development trip.

## Development data handoff

Run from the repository root with Python 3.11 and the supplied `Student folder/train` and `Student folder/student_test` directories in place.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python prepare.py
.\.venv\Scripts\python verify_prepared.py
```

For a single development example use `prepare.py --trip 00000001`. Run the full command to regenerate the committed summary. The raw data and reproducible per-trip arrays stay local. The evaluator directory is outside the loader's two development paths.

## Inputs and inspection

The inventory and track-quality CSVs record counts, malformed rows, invalid coordinates, duplicate or decreasing timestamps, and gaps above five seconds. Run `python inventory.py` or `python audit_tracks.py` to print these tables. Run `python map_training_trip.py` in the environment to regenerate the interactive training-trip map. Blue marks denote observations, green segments the supplied reference route, and grey segments the local road network. The HTML loads external map tiles and browser libraries. The preparation summary adds geographical bounds, isolated node counts, weakly connected component counts and largest component sizes. Weak connectivity ignores direction and does not guarantee legal vehicle movement.

## Candidate features

Observation indices and arc IDs are zero-based and local to a trip. Coordinates use a local equirectangular approximation in metres centred at the trip mean, suitable for local distances and subject to distortion farther from the origin. Whole-trip context assumes offline reconstruction. Each arc is the straight line between its supplied endpoints. For each observation we keep up to 12 directed arcs within 75 m, ordered by exact projected point-to-segment distance and then arc ID. These are initial design settings. The spatial index covers each arc with subsegments at most 20 m long. Querying their centres with 10 m extra margin retains even long arcs before exact filtering.

Distance captures lateral fit, fraction along the arc captures proximity to junctions, length describes geometry, and heading cosine compares observed displacement to directed arc orientation. Elapsed seconds and displacement retain motion context across irregular sampling. Heading cosine is zero for undefined direction. The first observation has zero elapsed time and displacement. A large time gap weakens the meaning of heading. Empty candidate sets remain recorded as misses. Features need no learned scaling. Fit any later transformations within training folds.

## Reference alignment and limitations

For development labels only, dynamic programming minimises summed point-to-segment distance under a nondecreasing index into the supplied route. It permits staying on a route entry or skipping entries across observations. Endpoints are unconstrained. It provides one deterministic assignment, not independently measured pointwise truth. Crossings, reverse arcs, stationary readings and gaps can make several assignments plausible. The ambiguity flag identifies another distinct route arc within 3 m of the assigned distance or closer, ignoring chronology for this diagnostic. Distances above 75 m flag poor geometric fits. Retain these flags for sensitivity analysis. The candidate miss denominator includes every observation, including empty sets. Never append a missing reference arc to the prediction candidates.

## Training data interface

Each prepared partition contains two NPZ files per trip. Load them with NumPy as below. The structured feature array contains one row per candidate. Label arrays have one element per original observation in track order. Trip IDs remain strings with leading zeros. Use trip ID for grouped splits. Arc and observation IDs are join keys. The labels, residuals, ambiguity and hit flags are restricted to supervision and scoring.

```python
import numpy as np
from pathlib import Path

folder = Path("prepared/train")
with np.load(folder / "00000001.features.npz") as data:
    candidates = data["candidates"]
with np.load(folder / "00000001.labels.npz") as data:
    reference = data["reference_arc"]
target = candidates["arc_id"] == reference[candidates["observation_index"]]
```

Observations whose reference is absent have no positive candidate. Preserve them in observation-level evaluation. Candidate coverage measures retrieval against derived reference labels. It is not classifier accuracy. The verifier checks all development files and compares the first, middle and last observations of each trip against exhaustive segment search.

## Measured development results

| Partition | Observations | Candidate rows | Empty sets | Reference misses | Geometrically ambiguous |
|---|---:|---:|---:|---:|---:|
| Training | 208,417 | 1,635,027 | 13,572 | 23,301 | 62,076 |
| Student test | 19,160 | 172,916 | 0 | 35 | 5,852 |

Training trip 00000054 accounts for 12,855 empty candidate sets and 17,401 reference misses. Training trip 00000033 has candidates for every point but 3,888 reference misses. These findings warrant per-trip reporting. Their causes are unverified. No trips were removed. Across both partitions there were no malformed GPS rows, invalid coordinates, repeated timestamps or backwards steps. There were 1,260 intervals over five seconds and a maximum interval of 153,064 seconds. No isolated nodes were found, although 89 of 90 trip networks contained multiple weak components.

## Five-model training experiment

Run the following after preparing the 90 development trips.

```powershell
.\.venv\Scripts\python train_models.py
.\.venv\Scripts\python score_student_trips.py
```

The six numeric features and their column order are defined in train_models.py. The target is whether the candidate arc matches the aligned reference. The split manifest records the validation trips in each of three GroupKFold splits. Models are refitted on the 80 training trips before student testing. Scaling for logistic regression and Gaussian Naive Bayes is fitted on the training portion of each run. Tree ensembles use a fixed seed of 42. There is no resampling. Model constructor arguments in the script and the pinned scikit-learn version specify the initial configurations.

Top-1 selects the candidate with highest positive-class probability. Ties favour distance order, then arc ID. Overall Top-1 includes observations with missing reference candidates. Conditional Top-1 uses only observations whose reference is present. Macro F1 and log loss describe candidate classification. Precision and recall refer to the positive class. Confusion counts are TN, FP, FN, TP. Inference time measures probability prediction on precomputed features. Continuity is the fraction of consecutive observations assigned the same directed arc or two arcs with the first ending where the second begins. Gaps can span intermediate arcs and reduce this diagnostic even for plausible travel.

The five compressed joblib artifacts under experiments contain the refitted classifiers. For inference, use the feature column order in the script and predict_proba(X)[:, 1]. Identifiers and reference diagnostics are excluded from the predictors.

## Decoder, application and report
See [route reconstruction](ROUTE_RECONSTRUCTION.md) for route reconstruction, [deployment](DEPLOYMENT.md) for the two-laptop demonstration and [the technical report](PathSnap-report.pdf) for the measured development results. The final API uses the model named by experiments/selected_model.txt. The five-model comparison is the initial untuned comparison.
