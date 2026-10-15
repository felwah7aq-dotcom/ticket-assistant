# Evaluation report

**Embedding backend used for these numbers: `multilingual-e5-small`**

Data: train 1129 / val 634 / test 510 tickets (test = 48 distinct core phrasings never seen in train; synthetic data, see README).

## 1. Intent classification (macro-F1, test)

| model | all | en | ar | msa | gulf | egy | lev |
|---|---|---|---|---|---|---|---|
| majority class | 0.043 | 0.041 | 0.043 | 0.045 | 0.043 | 0.042 | 0.043 |
| word TF-IDF + LR (baseline) | 0.741 | 0.553 | 0.787 | 0.805 | 0.667 | 0.983 | 0.821 |
| multilingual-e5-small + LR | 0.924 | 1.000 | 0.908 | 0.780 | 0.903 | 0.967 | 0.959 |

## 2. Abstain (thresholds chosen on validation by error cost)

- tau (min confidence) = 0.54, sigma (min similarity to a past ticket) = 0.86; cost: wrong auto-reply = 5.0, send to human = 1.0
- Test coverage 88.0%; accuracy on answered 98.7% (vs 92.2% without abstain)
- Wrong automatic replies: 40 -> 6; expected cost per ticket 0.392 -> 0.178

## 3. Hand-checked set (44 tickets: 32 in-scope, 12 out-of-scope; never used for tuning)

- In-scope accuracy without abstain: 96.9% (EN 93.3%, AR 100.0%)
- With abstain: coverage 93.8%, accuracy on answered 96.7%
- Out-of-scope tickets correctly sent to a human: 83.3%
- Wrong automatic replies (in-scope errors + answered out-of-scope): 3 of 44
- Output schema valid: 100%

## 4. Similar-ticket retrieval (query = test ticket, corpus = train tickets)

| relevance definition | scope | Hit@5 | MRR | queries |
|---|---|---|---|---|
| same intent | all | 0.959 | 0.884 | 510 |
| same intent | en | 0.965 | 0.907 | 86 |
| same intent | ar | 0.958 | 0.879 | 424 |
| same underlying issue | all | 0.731 | 0.688 | 510 |
| same underlying issue | en | 0.512 | 0.438 | 86 |
| same underlying issue | ar | 0.776 | 0.739 | 424 |
| same issue, other language only | all | 0.895 | 0.784 | 344 |
| same issue, other language only | en | 1.000 | 0.994 | 86 |
| same issue, other language only | ar | 0.860 | 0.715 | 258 |

## 5. Latency

p50 51.6 ms, p95 73.4 ms per ticket (Colab CPU, uncached; measured in a separate cell because the first run timed cache hits).

## 6. Failures logged: 72 (see errors.csv)

- confusion pair: 43
- ERROR_CAUGHT_BY_ABSTAIN: 34
- OVER_ABSTAIN (correct but sent to human): 29
- low-resource dialect (fewer training templates): 22
- CONFIDENT_ERROR (auto-reply would be wrong): 9
- very short / little context: 3
- OUT_OF_SCOPE answered (should abstain): 2
- code-switching AR+EN: 1
