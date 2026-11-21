# Empathy self-report pilot

<!-- WRITE THIS SECTION YOURSELF: 3-6 plain sentences on what you tested, what you found,
and why it matters. Anthropic asks that application materials start as your own draft. -->

## What's here

| Path | What it is |
|---|---|
| `prompts.csv` | 50 invented prompts: 40 emotional (6 categories, `cue` = explicit or implicit), 10 neutral controls |
| `eiq_gap.py` | Generates replies and self-ratings with Qwen2.5-1.5B-Instruct, then analyses them |
| `rater_agreement.py` | Agreement between the four blind raters, and each score against their consensus |
| `out/scoring_sheet.csv` | Every reply, with rater 1's blind score in `human_score` |
| `out/hidden_scores.json` | The model's self-rating per reply (raw text + parsed 1-10) and the EIQ auto-scores |
| `out/rater2.csv` … `rater4.csv` | Independent blind scores on a 20-reply subset of the emotional prompts |
| `out/results.csv`, `out/rater_agreement_results.csv` | Joined outputs of the two analyses |

## Data notes

- All prompts are invented. No real customer data.
- Replies come from Qwen2.5-1.5B-Instruct with greedy decoding, capped at 600 new tokens (no reply was cut off).
- The self-rating is asked in a fresh conversation that shows the model its own reply.
- All four raters scored blind, 1-10, from the same one-line instruction and no rubric. Rater 1 is the author.
- Rater 1 scored the neutral controls 0, meaning "no empathy needed". Analyses report results with the controls excluded.
- Rater 2 left 3 of 20 replies blank. Rank-based statistics are used throughout because raters used the scale differently.
- EIQ auto-scores come from the separate EIQ framework, which is not included here.

## Reproduce

    pip install -r requirements.txt
    python eiq_gap.py analyze          # needs out/scoring_sheet.csv and out/hidden_scores.json
    python rater_agreement.py

Regenerating replies (`python eiq_gap.py generate --prompts prompts.csv`) downloads the model
and needs the EIQ repo only for the optional `--eiq-repo` auto-scores.
