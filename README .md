# Ticket Assistant — bilingual (Arabic / English) support-ticket triage

*Capstone project built as part of the Applied NLP course at [SDAIA Academy](https://github.com/SDAIAAcademy).*

> **Our system takes** a customer support ticket written in English or Arabic (MSA, Gulf, Egyptian, Levantine, mixed with English)
> **and produces** a validated JSON decision (intent, confidence, route, priority, order id), the most similar past tickets and a
> knowledge-base-grounded reply with citations — or hands the ticket to a human when it is unsure,
> **measured by** macro-F1 (by language/dialect), Hit@5 / MRR for similar-case retrieval, and the cost of wrong automatic replies.

## Pipeline

```
Ticket (EN/AR) → Embed → Classify intent + confidence ──► abstain? ──► human (with similar cases)
                    │                                        │ no
                    ├──► Retrieve top-k similar past tickets │
                    └──► Retrieve KB articles of the predicted intent → Draft reply with [n] citations → Validate
```

| Stage | What it does | File |
|---|---|---|
| Embed | `intfloat/multilingual-e5-small` (384-d, normalised). Offline fallback: char n-gram TF-IDF → SVD | `core.py` |
| Classify | Standardise + Logistic Regression on the embeddings (8 intents), `C` tuned on validation macro-F1 | `core.py` |
| Abstain | Answer only if `confidence ≥ τ` **and** `similarity to nearest past ticket ≥ σ`; τ, σ chosen on **validation** by minimising error cost (wrong auto-reply = 5, send to human = 1) | `core.py` |
| Retrieve | Cosine top-5 over historical (train) tickets; KB articles filtered by predicted intent | `core.py` |
| Draft | Reply assembled from KB text in the ticket's language, every sentence cited `[kb id]` (no free-form generation, so no hallucination) | `core.py` |
| Validate | Schema check, intent in label set, every `[n]` matches a KB article, human-routed tickets carry no reply | `core.py` |

## Run it

```bash
pip install -r requirements.txt
python make_data.py                      # regenerates data/tickets.csv (deterministic, seed 7)
python evaluate.py --backend e5          # real run: downloads multilingual-e5-small (~0.5 GB) -> results/e5/
python evaluate.py --backend tfidf       # offline fallback -> results/tfidf_fallback/
python make_examples.py --backend e5     # sample inputs and real outputs -> examples/
BACKEND=e5 python app.py                 # Gradio demo on http://localhost:7861 (PORT env to change)
```

`--backend auto` (default) uses e5 if it can be loaded and otherwise falls back with a loud warning.
Each backend writes to its own folder, so results never overwrite each other.

## Data (read this first)

* **Synthetic**, template-based: 8 intents × 5 dialect groups (EN, MSA, Gulf, Egyptian, Levantine), with optional greeting/closing,
  random order ids and items, Arabic-Indic digits, hamza/ta-marbuta spelling noise, EN typos and ~10 % Arabic–English code-switching.
  Class imbalance is intentional.
* **Split by template, not by row**: a test ticket never shares its core phrasing with a train ticket (test = 48 distinct phrasings),
  and exact duplicates across splits are removed (0 found). This is much stricter than a random split but numbers are
  still **optimistic compared with real tickets**.
* `data/handchecked_eval.json`: 44 tickets written separately by hand (32 in-scope, 12 out-of-scope), never used for tuning.
* 24 out-of-scope tickets exist **only in validation**, to choose the abstain thresholds.
* `data/kb.json`: 16 knowledge-base articles (EN + AR) with **fictional store policies** (demo only).

## Results

Full tables: `results/<backend>/report.md`, failures: `results/<backend>/error_analysis.md` and `errors.csv`, charts in the same folder.
The numbers below are from **`multilingual-e5-small`** (`python evaluate.py --backend e5`, run on Colab CPU).
`results/tfidf_fallback/` holds an earlier run with the offline lexical encoder, kept for comparison.

| Test set (510 tickets) macro-F1 | all | English | Arabic | MSA | Gulf | Egyptian | Levantine |
|---|---|---|---|---|---|---|---|
| Majority class | 0.043 | 0.041 | 0.043 | 0.045 | 0.043 | 0.042 | 0.043 |
| Word TF-IDF + LR (baseline) | 0.741 | 0.553 | 0.787 | 0.805 | 0.667 | 0.983 | 0.821 |
| multilingual-e5-small + LR | 0.924 | 1.000 | 0.908 | 0.780 | 0.903 | 0.967 | 0.959 |

e5 improves overall (0.741 -> 0.924), most in English, Gulf and Levantine. It is **not** better on every slice: MSA (0.780 vs 0.805) and
Egyptian (0.967 vs 0.983) are slightly below the baseline. Each dialect slice has only about a dozen test phrasings, so these gaps may be
noise; this was not tested statistically.

| Abstain on test (tau 0.54, sigma 0.86, chosen on validation) | value |
|---|---|
| Coverage (answered automatically) | 88.0 % |
| Accuracy on answered / without abstain | 98.7 % / 92.2 % |
| Wrong automatic replies | 40 -> 6 (of 510) |
| Expected cost per ticket (wrong reply 5, human 1) | 0.392 -> 0.178 |

| Hand-checked set (44 tickets) | value |
|---|---|
| In-scope accuracy without abstain (EN / AR) | 96.9 % (93.3 % / 100 %) |
| With abstain: coverage / accuracy on answered | 93.8 % / 96.7 % |
| Out-of-scope tickets sent to a human | 83.3 % (10 of 12) |
| Wrong automatic replies | 3 of 44 |
| Output schema valid | 100 % |

Retrieval (Hit@5 / MRR): same intent 0.959 / 0.884; same underlying issue 0.731 / 0.688 (English queries 0.512 / 0.438, Arabic 0.776 / 0.739);
same issue, other language only 0.895 / 0.784 (the lexical fallback scored 0.308 / 0.250 here, so the multilingual encoder is what makes
cross-language retrieval work).

Latency (single ticket, Colab CPU, uncached e5 inference + classify + retrieve + draft, 100 random test tickets): p50 51.6 ms, p95 73.4 ms.

## Error analysis
`results/<backend>/error_analysis.md` lists 12 categorised failures (confident errors, over-abstentions, errors caught by abstain,
out-of-scope tickets that were answered) with the most frequent confusion pairs; `errors.csv` has every failure (72 for e5) with its category.

What the e5 run shows:
* **Failures are concentrated.** 67 test failures come from only 11 of the 48 test phrasings (each phrasing appears about 10 times with noise),
  and the five worst phrasings account for 55 of them. The failure count therefore overstates the number of distinct problems.
* Biggest confusion: formal Arabic delivery-status tickets (`track_order`) read as `change_address` (12 errors) or `cancel_order` (7).
  Hypothesis: "delivery" also appears in address-change tickets.
* One Gulf phrasing ("... ما شفت فلوسي في حسابي") is read as `login_problem` (7 errors), probably because of the word "account";
  all of them were caught by the abstain gate.
* On the hand-checked set, 2 of 12 out-of-scope tickets were answered (a used-car question as `product_question`, a courier complaint as
  `damaged_item`) and one in-scope shipping-coverage question was answered as `track_order`.
* 29 correct predictions were sent to a human (over-abstain): lost automation, not wrong replies.

The "likely cause" entries are hypotheses read from the examples; they were not tested.

## Fine-tuning decision
The encoder is **not fine-tuned**; the adaptation is a trained classification head plus validation-tuned thresholds. Reasoning: (1) the data are template-generated, so fine-tuning would mostly memorise templates; (2) logged failures are about
unseen phrasing and vocabulary overlap between intents — a data-coverage problem that more diverse labelled tickets fix better than a larger
trainable model; (3) a frozen encoder keeps latency and cost minimal and the model reproducible. Since MSA is the weakest slice (macro-F1 0.780, below the word baseline's 0.805; small slice, treat with caution), the next experiment is fine-tuning e5-small with a classification head and comparing it on this same test split.

## Limitations
* Synthetic data → optimistic numbers; real tickets are longer, noisier and multi-intent.
* Replies are KB extracts, not generated text; policies are fictional.
* Abstain relies on a similarity gate tuned on only 24 out-of-scope examples; some out-of-scope tickets are still answered (see error analysis).
* No fine-tuning experiment was run.

## Repo layout
```
make_data.py   data generator            core.py      pipeline (embed, classify, abstain, retrieve, draft, validate)
evaluate.py    metrics, charts, errors   app.py       Gradio demo
data/          tickets.csv, kb.json, handchecked_eval.json
results/       one folder per backend (e5/, tfidf_fallback/)
examples/      sample_inputs.json, sample_outputs.json (real pipeline outputs, incl. known failures)
```
