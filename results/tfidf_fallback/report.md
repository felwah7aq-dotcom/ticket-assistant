# Evaluation report

**Embedding backend used for these numbers: `tfidf-svd (offline fallback)`**

Data: train 1129 / val 634 / test 510 tickets (test = 48 distinct core phrasings never seen in train; synthetic data, see README).

## 1. Intent classification (macro-F1, test)

| model | all | en | ar | msa | gulf | egy | lev |
|---|---|---|---|---|---|---|---|
| majority class | 0.043 | 0.041 | 0.043 | 0.045 | 0.043 | 0.042 | 0.043 |
| word TF-IDF + LR (baseline) | 0.741 | 0.553 | 0.787 | 0.805 | 0.667 | 0.983 | 0.821 |
| tfidf-svd (offline fallback) + LR | 0.802 | 0.597 | 0.848 | 0.805 | 0.729 | 1.000 | 1.000 |

## 2. Abstain (thresholds chosen on validation by error cost)

- tau (min confidence) = 0.68, sigma (min similarity to a past ticket) = 0.48; cost: wrong auto-reply = 5.0, send to human = 1.0
- Test coverage 68.0%; accuracy on answered 89.0% (vs 78.8% without abstain)
- Wrong automatic replies: 108 -> 38; expected cost per ticket 1.059 -> 0.692

## 3. Hand-checked set (44 tickets: 32 in-scope, 12 out-of-scope; never used for tuning)

- In-scope accuracy without abstain: 78.1% (EN 60.0%, AR 94.1%)
- With abstain: coverage 75.0%, accuracy on answered 91.7%
- Out-of-scope tickets correctly sent to a human: 66.7%
- Wrong automatic replies (in-scope errors + answered out-of-scope): 6 of 44
- Output schema valid: 100%

## 4. Similar-ticket retrieval (query = test ticket, corpus = train tickets)

| relevance definition | scope | Hit@5 | MRR | queries |
|---|---|---|---|---|
| same intent | all | 0.875 | 0.795 | 510 |
| same intent | en | 0.872 | 0.752 | 86 |
| same intent | ar | 0.875 | 0.804 | 424 |
| same underlying issue | all | 0.598 | 0.573 | 510 |
| same underlying issue | en | 0.000 | 0.003 | 86 |
| same underlying issue | ar | 0.719 | 0.688 | 424 |
| same issue, other language only | all | 0.308 | 0.250 | 344 |
| same issue, other language only | en | 0.279 | 0.239 | 86 |
| same issue, other language only | ar | 0.318 | 0.254 | 258 |

## 5. Latency

p50 2.7 ms, p95 3.0 ms per ticket (CPU).

## 6. Failures logged: 215 (see errors.csv)

- confusion pair: 119
- OVER_ABSTAIN (correct but sent to human): 96
- ERROR_CAUGHT_BY_ABSTAIN: 75
- CONFIDENT_ERROR (auto-reply would be wrong): 44
- low-resource dialect (fewer training templates): 20
- code-switching AR+EN: 4
- OUT_OF_SCOPE answered (should abstain): 4
- very short / little context: 1
