# Per-query ranks (M2-6)

For every query image: where the correct class appeared in the ranked prototype list
(PROTOCOL.md §13.2). Internal validation only; not an article result. Encoder
`dinov3_vitb16`, M0 global-mean prototypes, exact cosine search (`avr.retrieval.exact`, same
rounding and class-id tie rule as the M1 gate).

Query sets: Track A is the official query half (14,922 images, 431 classes, label
"official CVPR 2015 half/half classification split ... hashes recorded"); Track B is the seed 73
`test` partition restricted to known-viewpoint rows, with provisional model-year classes
(`mapping_status = provisional`). Row counts are in the table below.

| track | variant | rows |
| ----- | ------- | ---- |
| track_a | p1 | 14,922 |
| track_a | p3 | 14,922 |
| track_b | p1 | 23,160 |
| track_b | p3 | 23,143 |

## Files

* `data/results/query_ranks/<track>/<encoder>/<variant>.parquet`: one row per query. The
  Parquet bytes are DVC tracked (`<variant>.parquet.dvc` is committed); restore with
  `uv run dvc checkout` or rebuild with
  `uv run python scripts/build_query_ranks.py <variant>`.
* `<track>_<variant>_per_class.csv`: one row per true class.
* `cmc.csv`: cumulative hit rate for k = 1..50 per track, encoder and variant.

## Per-query columns

| column | meaning |
| ------ | ------- |
| `image_id`, `track`, `encoder`, `variant`, `partition` | identity; partition is `query` (Track A) or `test` (Track B) |
| `split_label`, `mapping_status` | Track A: official-split label, `not_applicable`; Track B: `provisional` |
| `true_class_id`, `make_id`, `model_id`, `year`, `viewpoint` | ground truth (`year` is null on Track A, whose class is make-model) |
| `rank_true` | 1-based position of the true class in the full ranking (1 = best) |
| `score_true` | cosine score of the true class prototype (rounded to 6 decimals) |
| `top1_class_id`, `top1_score`, `margin_top1_top2` | the prediction and the gap between the first two scores |
| `top10_class_ids`, `top10_scores` | the ten best classes and their scores, best first |
| `hit_at_1`, `hit_at_5`, `hit_at_10`, `hit_at_20` | `rank_true <= k` |
| `tied` | an exact score tie among positions 1..11 (rule of `Ranking.tied_queries_at(10)`): the order inside the top 10 was then decided by class id |
| `rank_make` | first rank at which a prototype of the right make appears |
| `rank_model` | first rank at which a prototype of the right make and model appears (Track A: equals `rank_true`) |

`rank_make <= rank_model <= rank_true` always. `rank_make == 1` means the make was right at
the top; `rank_model == 1 < rank_true` means the model was right but the year was wrong
(Track B).

## Per-class columns

`class_id`, `make_name`, `model_name`, `year`, `queries`, `hits_at_{1,5,10,20}`,
`hit_rate_at_{1,5,10,20}`, `mean_rank`, `median_rank`, `mrr`, and `top_wrong_class_id` /
`top_wrong_count` (the most frequent wrong top-1 class and how often it was predicted; ties go
to the smaller class id, null and 0 if the class was never mistaken). Only classes with at
least one query appear.

## Two-line answers

```python
import pandas as pd

q = pd.read_parquet("data/results/query_ranks/track_a/dinov3_vitb16/p1.parquet")
q.loc[q.image_id == "00602699b3eb7cad", ["true_class_id", "rank_true", "score_true"]]  # car X
q[q.true_class_id == "081_0105"].hit_at_5.mean()  # how often class Y reaches the top 5 (as truth)
q.top10_class_ids.map(lambda ids: "081_0105" in ids[:5]).mean()  # ... or appears in any top-5
```

## Worked example (Track A, p1)

Image `00602699b3eb7cad`, a BMW X5 (CompCars spells the make "BWM"), viewpoint 5, true class
`081_0105`. Its ranking starts:

| rank | class | name | score |
| ---- | ----- | ---- | ----- |
| 1 | 081_0069 | BWM X1 | 0.6705 |
| 2 | 081_0104 | BWM X4 | 0.6565 |
| 3 | 081_0105 | BWM X5 (true) | 0.6294 |
| 4 | 081_0106 | BWM X6 | 0.6290 |
| 5 | 081_0103 | BWM X3 | 0.6181 |

So `rank_true = 3`, `hit_at_1` is false and `hit_at_5` is true; the margin between the first
two scores is 0.0140; `rank_make = 1` (the right make is first) and `rank_model = 3`; `tied`
is false. Over all 74 queries of the same class the top-5 hit rate is 0.973, and the class
appears in the top-5 list of 1.5 percent of all Track A queries.
