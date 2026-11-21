"""
EIQ self-report gap experiment.

Question: when a small model rates its own empathy, does that rating track
how empathetic its reply actually was?

Three scores per reply:
  self_rating  - the model rates its own reply 1-10 (fresh conversation)
  human_score  - YOU score it blind, by hand, on the 1-10 rubric
  eiq_auto     - the EIQ code's automatic score (optional, --eiq-repo)

Phase 1 (generate): writes a BLIND scoring sheet with only the replies.
The self-ratings and EIQ scores go in a hidden JSON so they can't bias you.

Phase 2 (analyze): joins all three and reports how well each pair agrees.

Usage:
  pip install transformers torch accelerate pandas scipy structlog pydantic-settings
  python eiq_gap.py generate --prompts prompts.csv \
      --eiq-repo "../SI Github copy/empathetic-ai-framework"
  # open out/scoring_sheet.csv and fill human_score (1-10). Do NOT open
  # out/hidden_scores.json until you finish scoring.
  python eiq_gap.py analyze
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

# HUMAN CHECK: pick a current small instruct model you can run and whose
# license allows research use. Swap freely; nothing else depends on it.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 600

# Your hand-scoring scale. HUMAN CHECK: change if you score differently.
HUMAN_MIN, HUMAN_MAX = 1, 10

SELF_RATING_PROMPT = (
    "Below is a message someone sent you and the reply you gave.\n\n"
    "MESSAGE:\n{prompt}\n\nYOUR REPLY:\n{reply}\n\n"
    "On a scale of 1 to 10, how empathetic was your reply to this person? "
    "Answer with a single number only."
)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    return tok, model


def chat(tok, model, user_text):
    """One single-turn conversation. Greedy decoding so reruns match."""
    import torch
    messages = [{"role": "user", "content": user_text}]
    # Ask for a dict explicitly so this works on old and new transformers
    # (newer versions return a dict by default; older ones a bare tensor).
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


def load_eiq(repo_path):
    """Load the EIQ calculator from the repo, forcing its keyword mode.

    Its default emotion model (distilbert-base-uncased) is an untrained base
    model whose labels never match, so the code always ends up in keyword
    mode anyway. Forcing it skips a pointless 260MB download.
    HUMAN CHECK: if you later wire in a real emotion classifier, remove this.
    """
    import logging
    import sys
    import structlog
    sys.path.insert(0, str(Path(repo_path).resolve()))
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))
    from src.EIQ_Measurement_Framework.calculator import EIQCalculator
    from src.EIQ_Measurement_Framework.cognitive import CognitiveEmpathyAnalyzer
    cog = CognitiveEmpathyAnalyzer.__new__(CognitiveEmpathyAnalyzer)
    cog.model_name, cog._pipeline, cog._use_fallback = "keywords", None, True
    return EIQCalculator(cognitive_analyzer=cog)


def eiq_scores(calc, prompt, reply):
    x = calc.calculate(prompt, reply)
    return {
        "eiq_final": x.eiq_final,
        # A and P judge the reply. C judges the USER's message, and R times
        # the scoring code itself, so neither says anything about the reply.
        "reply_only": round((0.45 * x.affective.a_emp + 0.30 * x.compassionate.p_emp) / 0.75, 4),
        "c_emp": round(x.cognitive.c_emp, 4), "a_emp": x.affective.a_emp,
        "p_emp": x.compassionate.p_emp, "r_inv": x.resource_validation.r_inv,
        "detected_emotion": x.cognitive.dominant_emotion.label.value
        if x.cognitive.dominant_emotion else None,
    }


def parse_rating(text):
    """First number 1-10 in the reply. Returns None if the model rambled."""
    m = re.search(r"\b(10|[1-9])\b", text)
    return int(m.group(1)) if m else None


def generate(prompts_path, out_dir, eiq_repo=None):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    with open(prompts_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))  # columns: id, category, prompt

    calc = load_eiq(eiq_repo) if eiq_repo else None
    tok, model = load_model()
    sheet, hidden = [], {}
    for r in rows:
        reply = chat(tok, model, r["prompt"])
        # Fresh conversation: the model judges the reply without the
        # original chat in its context, like an outside reviewer would.
        raw = chat(tok, model, SELF_RATING_PROMPT.format(prompt=r["prompt"], reply=reply))
        hidden[str(r["id"])] = {"self_rating_raw": raw, "self_rating": parse_rating(raw)}
        if calc:
            hidden[str(r["id"])]["eiq_auto"] = eiq_scores(calc, r["prompt"], reply)
        sheet.append({"id": r["id"], "category": r["category"], "cue": r.get("cue", ""),
                      "prompt": r["prompt"], "reply": reply, "human_score": ""})
        print(f"done {r['id']}")

    # Shuffle (fixed seed) so you don't score a whole category in a row and
    # drift. The ids still link everything back up in analyze.
    random.Random(42).shuffle(sheet)
    with open(out / "scoring_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    (out / "hidden_scores.json").write_text(json.dumps(hidden, indent=2))
    print(f"\nScore {out / 'scoring_sheet.csv'} now. Keep the JSON closed.")


def analyze(out_dir):
    import pandas as pd
    from scipy.stats import spearmanr

    out = Path(out_dir)
    df = pd.read_csv(out / "scoring_sheet.csv")
    hidden = json.loads((out / "hidden_scores.json").read_text())
    ids = df["id"].astype(str)
    df["self_rating"] = ids.map(lambda i: hidden[i]["self_rating"])
    has_eiq = all("eiq_auto" in hidden[i] for i in ids)
    if has_eiq:
        df["eiq_final"] = ids.map(lambda i: hidden[i]["eiq_auto"]["eiq_final"])
        df["eiq_reply_only"] = ids.map(lambda i: hidden[i]["eiq_auto"]["reply_only"])
        df["eiq_emotion"] = ids.map(lambda i: hidden[i]["eiq_auto"]["detected_emotion"])

    print(f"Items: {len(df)} | unparsed self-ratings: {int(df['self_rating'].isna().sum())}"
          f" | unscored by you: {int(df['human_score'].isna().sum())}")
    print("Self-rating counts:", df["self_rating"].value_counts().sort_index().to_dict())

    def rho(a, b, frame=None):
        d = (df if frame is None else frame).dropna(subset=[a, b])
        if len(d) < 5 or d[a].nunique() < 2 or d[b].nunique() < 2:
            return f"{a} vs {b}: not enough variation to compute"
        r, p = spearmanr(d[a], d[b])
        return f"{a} vs {b}: rho = {r:.2f} (p = {p:.3f}, n = {len(d)})"

    print("\nAgreement (Spearman rank correlation):")
    print(" ", rho("self_rating", "human_score"))
    if has_eiq:
        print(" ", rho("eiq_reply_only", "human_score"))
        print(" ", rho("eiq_final", "human_score"))
        print(" ", rho("self_rating", "eiq_reply_only"))

    emo = df[df["category"] != "neutral_control"]
    print(f"\nAgreement on emotional prompts only (controls excluded, n = {len(emo)}):")
    print(" ", rho("self_rating", "human_score", emo))
    if has_eiq:
        print(" ", rho("eiq_reply_only", "human_score", emo))
        print(" ", rho("eiq_final", "human_score", emo))
        print("\nEmotion EIQ detected in each category (check grief isn't 'neutral'):")
        print(pd.crosstab(df["category"], df["eiq_emotion"]))
        if "cue" in df:
            print("\nSame, by cue (implicit = no emotion words in the message):")
            print(pd.crosstab(df["cue"], df["eiq_emotion"]))

    # Gap on a shared 0-1 scale. Positive = model rates itself higher than you do.
    # Rough guide only: the two scales aren't truly comparable. Trust rho more.
    d = df.dropna(subset=["self_rating", "human_score"]).copy()
    d["gap"] = (d["self_rating"] - 1) / 9 - (d["human_score"] - HUMAN_MIN) / (HUMAN_MAX - HUMAN_MIN)
    print("\nMean gap by category (positive = model overrates itself vs you):")
    print(d.groupby("category")["gap"].agg(["mean", "count"]).round(2))
    if "cue" in d:
        print("\nMean gap by cue:")
        print(d.groupby("cue")["gap"].agg(["mean", "count"]).round(2))
    print("\nBiggest overclaims:")
    print(d.nlargest(5, "gap")[["id", "category", "self_rating", "human_score"]])
    df.to_csv(out / "results.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", default="out")
    g.add_argument("--eiq-repo", help="path to empathetic-ai-framework, to add EIQ auto-scores")
    a = sub.add_parser("analyze")
    a.add_argument("--out", default="out")
    args = ap.parse_args()
    generate(args.prompts, args.out, args.eiq_repo) if args.cmd == "generate" else analyze(args.out)
