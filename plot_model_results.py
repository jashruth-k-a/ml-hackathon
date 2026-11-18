"""Draw the required initial comparison charts from the recorded runs."""
import csv
from statistics import mean
from pathlib import Path

folder = Path(__file__).resolve().parent / "experiments"
with (folder / "metrics.csv").open() as file:
    rows = list(csv.DictReader(file))
models = list(dict.fromkeys(row["model"] for row in rows))
charts = [("Grouped CV macro F1", "cv", "macro_f1"),
          ("Student-test Top-1 accuracy", "student_test", "top1"),
          ("Student-test macro F1", "student_test", "macro_f1"),
          ("Student-test log loss (lower is better)", "student_test", "log_loss")]
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="840" height="600" viewBox="0 0 840 600">',
       '<rect width="100%" height="100%" fill="white"/>', '<g font-family="sans-serif" fill="#183147">']
for i, (title, split, metric) in enumerate(charts):
    x, y = 20 + (i % 2) * 420, 40 + (i // 2) * 290
    svg.append(f'<text x="{x}" y="{y}" font-size="19">{title}</text>')
    for j, model in enumerate(models):
        value = mean(float(row[metric]) for row in rows if row["model"] == model
                     and row["split"].startswith(split))
        h = y + 38 + j * 38
        width = value * 205
        svg.append(f'<text x="{x}" y="{h}">{model}</text>')
        svg.append(f'<rect x="{x+112}" y="{h-14}" width="{width}" height="16" fill="#257786"/>')
        svg.append(f'<text x="{x+118+width}" y="{h}">{value:.3f}</text>')
svg.append("</g></svg>")
(folder / "comparison.svg").write_text("\n".join(svg))
