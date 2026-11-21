# Empathy self-report pilot

Does a model's rating of its own empathy match how empathetic its replies actually are?
This pilot tested that on a small open model, Qwen2.5-1.5B-Instruct.

The model answered 50 invented messages (40 emotional, 10 neutral controls). In a fresh
conversation, it then rated its own empathy on each reply from 1 to 10. I scored every reply
blind. I also ran my own automated scorer, EIQ, on the same replies.

## Findings

- On the 40 emotional prompts, the model's self-ratings tracked my blind scores weakly
  (Spearman rho = 0.29, p = 0.07).
- EIQ did worse (rho = 0.08). It rewarded sympathy keywords over useful help, and it read grief
  stated without emotion words as neutral.
- Three more people scored 20 of the replies blind, with the same one-line instruction. The four
  of us barely agreed: Krippendorff's alpha was 0.03 across all four raters and -0.20 across the
  three independent ones. Against their consensus, neither the self-ratings (rho = 0.13) nor EIQ
  (rho = -0.19) tracked.
- The model overrated itself most on messages that described a situation without naming the
  feeling. With these sample sizes, that's a hint, not a result.

## What it means

Before you can check whether a model's account of its own empathy is accurate, you need a
reference that people agree on. Overall "how empathetic is this?" judgments didn't give one.
A next step is to score separate parts (reading the person's state, acknowledging it, helping)
and test whether raters agree on those.

## Limits

One small model, 50 prompts, one rater (the author) on all 50 replies and three independent
raters on 20. Treat it as a pilot.

## How this was made

I chose the research question, scored all 50 replies, designed the study recruited the other raters, and reviewed
every result. Claude (Anthropic) helped write the prompts, the generation
and analysis code, and the first draft of this README.

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
