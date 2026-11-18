"""Discrete Gaussian-process Bayesian search with expected improvement."""
import csv
from itertools import product
from time import perf_counter
import joblib
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from train_models import load, evaluate, OUT, GroupKFold, SEED


def forest(params):
    depth, leaf, fraction = params
    return RandomForestClassifier(n_estimators=32, max_depth=int(depth), min_samples_leaf=int(leaf),
                                 max_samples=float(fraction), n_jobs=4, random_state=SEED)


def main():
    x,y,o,g,c = load("train")
    folds = list(GroupKFold(3).split(x,y,g[o]))
    grid = np.array(list(product([10,14,18], [12,30,60], [.25,.5,1.])))
    unit = (grid-grid.min(axis=0))/np.ptp(grid,axis=0)
    visited, objectives, records = [], [], []
    for t in range(6):
        if t < 3:
            idx = [0,14,25][t]
        else:
            gp = GaussianProcessRegressor(kernel=Matern(length_scale=.6,nu=2.5),optimizer=None,
                                         normalize_y=True,alpha=1e-5).fit(unit[visited],objectives)
            mean,std = gp.predict(unit,return_std=True)
            gain = mean-max(objectives)
            z = gain/np.maximum(std,1e-12)
            ei = gain*norm.cdf(z)+std*norm.pdf(z)
            ei[visited] = -np.inf
            idx = int(ei.argmax())
        start = perf_counter()
        values = []
        for fit,val in folds:
            m = forest(grid[idx]).fit(x[fit],y[fit])
            p = m.predict_proba(x[val])[:,1]
            keep = np.isin(g,np.unique(g[o[val]]))
            values.append(evaluate(y[val],o[val],p,keep.sum(),c[keep].sum()))
        objective = np.mean([v["top1"] for v in values])
        visited.append(idx)
        objectives.append(objective)
        records.append([t+1,*grid[idx],objective,np.mean([v["macro_f1"] for v in values]),perf_counter()-start])
        with (OUT/"bayesian_trials.csv").open("w",newline="") as f:
            w=csv.writer(f); w.writerow("trial depth leaf sample_fraction cv_top1 cv_macro_f1 seconds".split()); w.writerows(records)
        print("trial",t+1,objective,flush=True)
    model = forest(grid[visited[np.argmax(objectives)]]).fit(x,y)
    tx,ty,to,tg,tc=load("student_test")
    start=perf_counter()
    prob=model.predict_proba(tx)[:,1]
    elapsed=perf_counter()-start
    scores=evaluate(ty,to,prob,len(tg),tc.sum())
    scores["inference_seconds"]=elapsed
    with (OUT/"tuned_test.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=scores); w.writeheader(); w.writerow(scores)
    joblib.dump(model,OUT/"tuned_forest.joblib",compress=3)


if __name__ == "__main__":
    main()
