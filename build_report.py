"""Build the development report from recorded experiments (requires reportlab)."""
import csv
from pathlib import Path
from statistics import mean
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Image, Spacer, PageBreak

root=Path(__file__).resolve().parent
def read(name):
    with (root/"experiments"/name).open() as f:
        return list(csv.DictReader(f))
styles=getSampleStyleSheet()
story=[]
def para(text,style="BodyText"):
    story.append(Paragraph(text,styles[style]))
def title(text):
    para(text,"Heading1")
def table(rows):
    t=Table(rows,hAlign="LEFT",repeatRows=1)
    t.setStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#174f48")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eef4f0")]),
                ("FONTSIZE",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),5)])
    story.extend([Spacer(1,8),t,Spacer(1,8)])
def page():
    story.append(PageBreak())
def picture(name,w,h):
    story.extend([Spacer(1,8), Image(str(root/name),width=w,height=h,kind="proportional")])
m=read("metrics.csv")
tests=[r for r in m if r["split"]=="student_test"]
dec=read("decoder_results.csv")
trials=read("bayesian_trials.csv")
tuned=read("tuned_test.csv")[0]
title("PathSnap | Development results")
para("Map-matching and vehicle route reconstruction","Heading2")
para("Training used 80 trips. All development comparisons used the ten student-test trips. The ten evaluator trips are reserved for the live assessment. Pointwise scores use labels derived from the supplied route.")
para("Dataset and preprocessing","Heading2")
para("The inventory found 208,417 training observations and 19,160 student-test observations. No invalid coordinates, malformed GPS rows, duplicate times or backwards timestamps were detected in the development tracks. There were 1,260 gaps over five seconds; the longest was 153,064 seconds. All but one of the 90 road graphs contained multiple weak components.")
para("Project coordinates to local metres and find up to 12 directed arc candidates within 75 metres. Features capture distance to the segment, position along it, segment length, heading agreement, displacement and elapsed time. A long gap weakens the heading estimate. Fit scaling inside each training fold. Keep arc IDs and labels outside the predictors.")
para("Align observations to the chronological reference arc list by minimising total distance with nondecreasing route index. Parallel roads, loops and junctions can admit several plausible alignments. The reference was missing from 23,301 training lists and 35 student-test lists. All observations remain in the Top-1 denominator.")
picture("evidence/folium.png",470,265)
para("Student trip 00000000. Folium overlay of GPS observations, road graph, reference and predicted segments. Separate layers can be toggled in experiments/decoder_map.html.")
page()
title("Model experiments")
para("We compared five candidate classifiers using identical three-fold splits grouped by trip. Top-1 selects the highest scoring candidate per observation. Macro F1 and log loss operate on candidate rows. Nearest road alone reached 53.74% student-test Top-1. Times are seconds for probability inference on precomputed student-test features.")
rows=[["Model","CV F1","CV Top-1","Test Top-1","Test F1","Loss","Time","Continuity"]]
trip_rows=read("student_trips.csv")
for r in tests:
    name=r["model"]; folds=[i for i in m if i["model"]==name and i["split"].startswith("cv")]
    ts=[i for i in trip_rows if i["model"]==name]
    vals=[mean(float(i[k]) for i in folds) for k in ("macro_f1","top1")]
    vals.extend(float(r[k]) for k in ("top1","macro_f1","log_loss","inference_seconds"))
    vals.append(sum(int(i["continuous_pairs"]) for i in ts)/sum(int(i["pairs"]) for i in ts))
    rows.append([name,*[f"{v:.3f}" for v in vals]])
table(rows)
notes=[
("Logistic regression","Standardised inputs, L2 regularisation, C=1, maximum 300 iterations. Minimises regularised logistic loss. A simple linear score improved substantially over proximity alone but missed interactions between motion and geometry."),
("Gaussian naive Bayes","Gaussian feature likelihoods and empirical class priors, variance smoothing 1e-9. Correlated displacement and elapsed time violate conditional independence. Its 0.428 test log loss was the weakest of the five."),
("Decision tree","Gini splits, depth 12, minimum 50 rows per leaf. Conditional splits can use heading differently for moving and stationary vehicles. A single tree reached 81.37% Top-1 but had higher log loss than either ensemble."),
("Random forest","32 bootstrap trees, Gini splits, depth 14, minimum 30 rows per leaf, square-root feature sampling. Averaging nonlinear trees produced the best initial mean CV Top-1 and test log loss."),
("Histogram gradient boosting","100 boosting rounds, log loss, learning rate 0.1, up to 31 leaves. Successive trees model residual errors. It achieved 81.88% student-test Top-1, narrowly behind the initial forest.")]
for name,note in notes:
    para(name,"Heading3");para(note)
para("The training and student sets differ in candidate coverage (88.82% and 99.82%). Higher student scores alone cannot establish absence of overfitting. Validation metrics vary substantially by trip fold. Full precision, recall, confusion counts and timing records are in experiments/metrics.csv.")
page()
title("Model comparison across the same partitions")
picture("evidence/model-comparison.png",475,360)
para("Cross-validation macro F1 is averaged over the three held-out training folds. Student-test bars are measured on all ten student-test trips. Lower log loss indicates better candidate probability estimates.")
page()
title("Per-model fit diagnostics")
para("The full-fit training score uses all 80 training trips. The validation score averages three held-out trip folds, each trained on fewer trips. These gaps are descriptive estimates and depend on fold composition.")
comments = {"logistic":"The narrow gap and lower scores on both partitions suggest limited model capacity.", "gaussian_nb":"Both scores lag the tree models. The small gap is consistent with restrictive distributional assumptions.", "decision_tree":"Training exceeds validation by about one F1 point, consistent with mild overfitting.", "random_forest":"The roughly one-point gap indicates modest training optimism even after averaging trees.", "hist_boost":"The gap is under one point and held-out F1 is strongest. There is little evidence of severe overfitting."}
for row in read("fit_diagnostics.csv"):
    para(row["model"],"Heading3")
    para(f"Training macro F1 {float(row['training_macro_f1']):.3f}. Held-out macro F1 {float(row.get('cv_macro_f1')):.3f}. " + comments[row["model"]])
page()
title("Bayesian optimisation")
para("The objective was mean observation-level Top-1 across the same three training folds. The search varied tree depth (10, 14, 18), minimum leaf size (12, 30, 60) and bootstrap sample fraction (0.25, 0.5, 1). These control tree complexity, smoothing and training cost. The forest retained 32 trees. Three initial configurations were followed by three Gaussian-process expected-improvement selections over the remaining grid.")
table([["Trial","Depth","Min. leaf","Fraction","CV Top-1","CV F1","Seconds"]]+[[r["trial"],r["depth"],r["leaf"],r["sample_fraction"],f'{float(r["cv_top1"]):.5f}',f'{float(r["cv_macro_f1"]):.5f}',f'{float(r["seconds"]):.1f}'] for r in trials])
r=next(x for x in tests if x["model"]=="random_forest")
best=max(trials,key=lambda x:float(x["cv_top1"]))
table([["Model","CV F1","Test Top-1","Test F1","Log loss","Inference s"],
["Initial","0.86069",*[f'{float(r[k]):.5f}' for k in ("top1","macro_f1","log_loss","inference_seconds")]],
["Tuned",f'{float(best["cv_macro_f1"]):.5f}',*[f'{float(tuned[k]):.5f}' for k in ("top1","macro_f1","log_loss","inference_seconds")]]])
para("Trial 3 won on the specified CV objective. The refitted model improved student Top-1 by 0.21 percentage points. Candidate Macro F1 and log loss weakened slightly, so the advantage depends on the metric. The small CV difference warrants caution.")
para("Route reconstruction","Heading2")
para("The directed candidate arc is the hidden state. Scores come from the tuned forest. For each current arc, Viterbi retains the predecessor with the largest previous log score plus log transition probability, then adds the current log emission. The decoder backtracks to recover the sequence. Transitions allow staying on an arc or taking a directed outgoing arc. We penalise disagreement between road travel distance and observed displacement.")
para("Training-trip calibration compared motion scales of 5, 15 and 40 metres on the first five training trips; 5 metres maximised correct observations. Missing candidates produce an unmatched index. A gap above 30 seconds or loss of all reachable transitions starts a fresh sequence. This local graph rule cannot bridge every multi-arc movement.")
page()
title("Route-level results")
rows=[["Student trip","Classifier Top-1","Viterbi Top-1","Viterbi continuity","Sequence similarity"]]
for a,b in zip(dec[::2],dec[1::2]):
    rows.append([a["trip"],*[f"{v:.2%}" for v in [int(a["correct"])/int(a["observations"]),int(b["correct"])/int(b["observations"]),int(b["continuous"])/int(b["pairs"]),float(b["sequence_similarity"])]]])
table(rows)
for method in ("classifier","viterbi"):
    r=[x for x in dec if x["method"]==method]
    para(f'{method.capitalize()}: pooled point accuracy {sum(int(x["correct"]) for x in r)/19160:.2%}, pooled adjacent-pair continuity {sum(int(x["continuous"]) for x in r)/19150:.2%}, and mean trip sequence similarity {mean(float(x["sequence_similarity"]) for x in r):.2%}.')
para("Sequence similarity uses ordered matching blocks after consecutive repeats are collapsed. Continuity counts adjacent predictions on the same arc or directly connected directed arcs. Trip 00000077 gained 679 correct points. Trips 00000007 and 00000052 lost 43 and 13 respectively, illustrating that a coherent sequence can still choose the wrong road.")
para("Deployment evidence","Heading2")
para("FastAPI runs prediction and reference-based scoring on Laptop 1. Laptop 2 sends only the selected trip ID and displays the returned geometry and aggregate metrics. The following test ran both services as separate local processes with the student split. The physical LAN demo remains to be performed.")
picture("evidence/student-ui.png",400,277)
page()
title("Application verification and reproducibility")
para("The browser selected each of the ten student trips and submitted a real request through the UI to the inference process. Each response matched the offline Viterbi point count. An unknown trip ID returned HTTP 404. The scorer opens the reference route only after prediction finishes. The response excludes reference labels and reference geometry.")
para("Receiving API process | recorded access log","Heading2")
for line in (root/"evidence/api-access.log").read_text().splitlines():
    if "POST /trips" in line: para(line.split(' - ')[-1],"Code")
para("Repository evidence","Heading2")
para("prepare.py reconstructs the candidate tables. train_models.py records trip-grouped comparisons. tune_forest.py records Bayesian trials. tune_decoder.py and evaluate_decoder.py reproduce sequence comparisons and the Folium map. The deployment guide gives commands for both laptops. The interactive map and recorded browser screenshot were generated from real development trips.")
para("Assessment handoff","Heading2")
para("Supply the public repository URL and capture the UI and receiving terminal on the actual laptops during the demonstration. All reported numbers here come from development runs. AI assistance was used to implement the pipeline and draft this report. The team should verify and explain the code, methods and reported results at review.")
def footer(canvas,doc):
    canvas.setFont("Helvetica",9);canvas.drawString(48,25,"PathSnap  |  Development evaluation");canvas.drawRightString(546,25,str(doc.page))
SimpleDocTemplate(str(root/"PathSnap-report.pdf"),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=38,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)


