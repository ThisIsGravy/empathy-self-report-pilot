"""
Agreement between the four blind raters on the 20-reply subset, and how the
model's self-ratings and EIQ compare with the independent raters' consensus.

Run from residency_experiment/:
    pip install pandas scipy krippendorff
    python rater_agreement.py
"""
import json

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = "out"
RATER_FILES = {
    "r2": f"{OUT}/rater2.csv",
    "r3": f"{OUT}/rater3.csv",
    "r4": f"{OUT}/rater4.csv",
}
RATERS = ["rater1", "r2", "r3", "r4"]
rng = np.random.default_rng(0)

df = pd.read_csv(f"{OUT}/rater2.csv")[["id"]]
for name, path in RATER_FILES.items():
    d = pd.read_csv(path)
    d.columns = [str(c).strip().lower() for c in d.columns]
    df = df.merge(d[["id", "score"]].rename(columns={"score": name}), on="id")
eli = pd.read_csv(f"{OUT}/scoring_sheet.csv")[["id", "cue", "human_score"]]
df = df.merge(eli.rename(columns={"human_score": "rater1"}), on="id")
hidden = json.loads(open(f"{OUT}/hidden_scores.json").read())
ids = df["id"].astype(str)
df["self_rating"] = ids.map(lambda i: hidden[i]["self_rating"])
df["eiq_reply_only"] = ids.map(lambda i: hidden[i]["eiq_auto"]["reply_only"])
df["eiq_final"] = ids.map(lambda i: hidden[i]["eiq_auto"]["eiq_final"])

print(f"Items: {len(df)} | missing scores per rater: {df[RATERS].isna().sum().to_dict()}")
print("Rater means (scales differ, so everything below uses ranks or z-scores):",
      df[RATERS].mean().round(2).to_dict())

print("\nPairwise Spearman between raters:")
for i, a in enumerate(RATERS):
    for b in RATERS[i + 1:]:
        d = df.dropna(subset=[a, b])
        r, p = spearmanr(d[a], d[b])
        print(f"  {a}-{b}: rho = {r:.2f} (p = {p:.3f}, n = {len(d)})")

def alpha(cols):
    data = df[cols].T.values.astype(float)
    return krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")

# 1 = perfect agreement, 0 = no better than chance, below 0 = systematic disagreement
print(f"\nKrippendorff's alpha (ordinal): all four = {alpha(RATERS):.2f}, "
      f"three independent = {alpha(['r2', 'r3', 'r4']):.2f}")

for c in RATERS:  # z-score each rater so harsh and lenient raters count equally
    df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std()

print("\nEach rater vs the mean of the other three:")
for c in RATERS:
    others = df[[f"z_{x}" for x in RATERS if x != c]].mean(axis=1)
    d = pd.DataFrame({"a": df[c], "m": others}).dropna()
    r, p = spearmanr(d["a"], d["m"])
    print(f"  {c}: rho = {r:.2f} (p = {p:.3f}, n = {len(d)})")

df["independent"] = df[["z_r2", "z_r3", "z_r4"]].mean(axis=1)

def boot_ci(a, b, n=5000):
    vals = df[[a, b]].dropna().values
    rs = []
    for _ in range(n):
        x = vals[rng.integers(0, len(vals), len(vals))]
        if len(set(x[:, 0])) > 1 and len(set(x[:, 1])) > 1:
            rs.append(spearmanr(x[:, 0], x[:, 1])[0])
    return np.percentile(rs, [2.5, 97.5])

print("\nAgainst the three independent raters' consensus:")
for c in ["self_rating", "eiq_reply_only", "eiq_final", "rater1"]:
    r, p = spearmanr(df[c], df["independent"])
    lo, hi = boot_ci(c, "independent")
    print(f"  {c:15s} rho = {r:.2f} (p = {p:.3f}, 95% CI {lo:.2f} to {hi:.2f})")

df["self_z"] = (df["self_rating"] - df["self_rating"].mean()) / df["self_rating"].std()
df["gap_z"] = df["self_z"] - df["independent"]
print("\nBy cue (z units; positive gap = model rates itself above the consensus):")
print(df.groupby("cue").agg(consensus=("independent", "mean"), gap=("gap_z", "mean"),
                            n=("id", "count")).round(2))
df.to_csv(f"{OUT}/rater_agreement_results.csv", index=False)
