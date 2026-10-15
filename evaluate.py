"""Evaluate the Ticket Assistant: baseline vs embeddings, abstention, retrieval, error analysis.

Run:  python evaluate.py                 # auto: e5 if available, else offline fallback
      python evaluate.py --backend e5    # force multilingual-e5-small (needs internet + torch)
      python evaluate.py --backend tfidf # force the offline fallback

Outputs in results/: metrics.json, report.md, errors.csv, three PNG charts.
All numbers use the SAME held-out test split; thresholds/hyper-parameters are chosen on validation only.
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from core import INTENTS, OOS, ROOT, TicketAssistant, detect_lang, make_backend, normalize_text

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="auto", choices=["auto", "e5", "tfidf"])
args = ap.parse_args()

# ---------------------------------------------------------------- data + model
df = pd.read_csv(os.path.join(ROOT, "data", "tickets.csv"), encoding="utf-8-sig")
train, val, test = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
hand = pd.DataFrame(json.load(open(os.path.join(ROOT, "data", "handchecked_eval.json"), encoding="utf-8")))

asst = TicketAssistant(make_backend(args.backend)).fit(train, val)
BACKEND = asst.backend.name
RES = os.path.join(ROOT, "results", "e5" if "e5" in BACKEND else "tfidf_fallback")  # never overwrite the other backend's results
os.makedirs(RES, exist_ok=True)
print(f"backend: {BACKEND} | C={asst.C} | tau={asst.tau} | sigma={asst.sigma}")

Zt = asst.backend.encode(test.text)
P = asst.clf.predict_proba(Zt)
pred = asst.clf.classes_[P.argmax(1)]
conf = P.max(1)
Sim = Zt @ asst.Xh.T
sim1 = Sim.max(1)
answered = asst.gate(conf, sim1)
ok = pred == test.intent.values


def mf1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def groups(frame, y_pred):
    out = {"all": mf1(frame.intent, y_pred)}
    for lang in ("en", "ar"):
        m = (frame.lang == lang).values
        out[lang] = mf1(frame.intent[m], y_pred[m])
    for d in ("msa", "gulf", "egy", "lev"):
        m = (frame.dialect == d).values
        out[d] = mf1(frame.intent[m], y_pred[m])
    return out


M = {"backend": BACKEND, "n_train": len(train), "n_val": len(val), "n_test": len(test),
     "test_distinct_cores": int(test.groupby(["dialect", "issue"]).ngroups),
     "chosen": {"C": asst.C, "tau": asst.tau, "sigma": asst.sigma, "val_macro_f1": asst.val_macro_f1}}

# ------------------------------------------------------- 1. classification
maj = np.array([train.intent.value_counts().idxmax()] * len(test))
tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, preprocessor=normalize_text)
Xtr_w = tf.fit_transform(train.text)
val_in = val[val.intent != OOS]
best = (-1, None)
for C in (0.3, 1, 3, 10, 30, 100):
    f = f1_score(val_in.intent, LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
                 .fit(Xtr_w, train.intent).predict(tf.transform(val_in.text)), average="macro")
    best = max(best, (f, C))
base = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(Xtr_w, train.intent)
pred_base = base.predict(tf.transform(test.text))

M["classification"] = {
    "majority": groups(test, maj),
    "baseline_word_tfidf_LR": groups(test, pred_base),
    "embeddings_LR": groups(test, pred),
}
M["accuracy"] = {"baseline": float(accuracy_score(test.intent, pred_base)), "embeddings_LR": float(accuracy_score(test.intent, pred))}

# ---------------------------------------------------------- 2. abstention
cost_no = float(np.where(ok, 0, asst.wrong_auto_cost).mean())
cost_ab = float(np.where(answered, np.where(ok, 0, asst.wrong_auto_cost), asst.abstain_cost).mean())
M["abstention_test"] = {
    "coverage": float(answered.mean()),
    "accuracy_on_answered": float(ok[answered].mean()),
    "accuracy_no_abstain": float(ok.mean()),
    "macro_f1_on_answered": mf1(test.intent[answered], pred[answered]),
    "cost_no_abstain": cost_no, "cost_with_abstain": cost_ab,
    "wrong_auto_replies_no_abstain": int((~ok).sum()), "wrong_auto_replies_with_abstain": int((~ok & answered).sum()),
}

# ----------------------- 3. hand-checked set (never used for tuning) + schema check
outs = [asst.run(t) for t in hand.text]
hand["pred"] = [o["top_guess"] for o in outs]
hand["conf"] = [o["confidence"] for o in outs]
hand["route"] = [o["route"] for o in outs]
hand["valid"] = [not o.get("validation_errors") for o in outs]
ins, oos_m = hand.expected != OOS, hand.expected == OOS
auto = hand.route == "auto"
M["handchecked"] = {
    "n": len(hand), "n_in_scope": int(ins.sum()), "n_out_of_scope": int(oos_m.sum()),
    "in_scope_accuracy_no_abstain": float((hand.pred == hand.expected)[ins].mean()),
    "in_scope_coverage": float(auto[ins].mean()),
    "in_scope_accuracy_on_answered": float((hand.pred == hand.expected)[ins & auto].mean()),
    "out_of_scope_rejected": float((~auto)[oos_m].mean()),
    "wrong_auto_replies": int((auto & ((hand.pred != hand.expected) | oos_m)).sum()),
    "schema_valid_rate": float(hand.valid.mean()),
}
for lang in ("en", "ar"):
    m = ins & (hand.lang == lang)
    M["handchecked"][f"in_scope_accuracy_no_abstain_{lang}"] = float((hand.pred == hand.expected)[m].mean())

# ------------------------------------------------------------ 4. retrieval
def retrieval(S, rel, k=5):
    valid = rel.any(1)
    order = np.argsort(-S, axis=1)
    hits = np.take_along_axis(rel, order, axis=1)
    first = hits.argmax(1)
    return hits[:, :k].any(1)[valid], (1.0 / (first + 1))[valid], valid


h_int, h_iss, h_lang = (np.asarray(train[c], dtype=object) for c in ("intent", "issue", "lang"))
t_int, t_iss, t_lang = (np.asarray(test[c], dtype=object) for c in ("intent", "issue", "lang"))
rel_loose = t_int[:, None] == h_int[None, :]
rel_strict = t_iss[:, None] == h_iss[None, :]
other_lang = t_lang[:, None] != h_lang[None, :]
S_x = np.where(other_lang, Sim, -np.inf)
R = {}
for name, S, rel in (("loose_same_intent", Sim, rel_loose), ("strict_same_issue", Sim, rel_strict),
                     ("strict_cross_language", S_x, rel_strict & other_lang)):
    R[name] = {}
    for lang in ("all", "en", "ar"):
        m = np.ones(len(test), bool) if lang == "all" else (test.lang == lang).values
        hit, rr, valid = retrieval(S[m], rel[m])
        R[name][lang] = {"Hit@5": float(hit.mean()), "MRR": float(rr.mean()), "n_queries": int(valid.sum())}
M["retrieval"] = R

# ------------------------------------------------------------- 5. latency
import time
if hasattr(asst.backend, "cache"):  # e5 caches embeddings by text; clear it so we time REAL inference, not cache hits
    asst.backend.cache, asst.backend.cache_path = {}, None
lat = []
for t in test.text.sample(150, random_state=0):
    t0 = time.perf_counter(); asst.run(t); lat.append((time.perf_counter() - t0) * 1000)
M["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                   "note": "single ticket, CPU, includes embedding + classify + retrieve + draft"}

# ------------------------------------------------------ 6. error analysis
def cats(text, true, p, lang, dialect, was_answered, was_correct):
    c = []
    if was_answered and not was_correct: c.append("CONFIDENT_ERROR (auto-reply would be wrong)")
    if (not was_answered) and was_correct: c.append("OVER_ABSTAIN (correct but sent to human)")
    if (not was_answered) and not was_correct: c.append("ERROR_CAUGHT_BY_ABSTAIN")
    if dialect in ("egy", "lev"): c.append("low-resource dialect (fewer training templates)")
    if len(text.split()) <= 5: c.append("very short / little context")
    if re.search(r"[A-Za-z]{3,}", text) and re.search(r"[؀-ۿ]", text): c.append("code-switching AR+EN")
    if not was_correct: c.append(f"confusion {true} -> {p}")
    return " | ".join(c)


err = []
for i in range(len(test)):
    if (not ok[i]) or (not answered[i]):
        err.append({"source": "test", "id": int(test.id[i]), "text": test.text[i], "true": test.intent[i],
                    "pred": pred[i], "conf": round(float(conf[i]), 3), "sim": round(float(sim1[i]), 3),
                    "answered": bool(answered[i]), "lang": test.lang[i], "dialect": test.dialect[i],
                    "category": cats(test.text[i], test.intent[i], pred[i], test.lang[i], test.dialect[i],
                                     bool(answered[i]), bool(ok[i]))})
for _, r in hand.iterrows():
    correct = r.pred == r.expected
    a = r.route == "auto"
    if (r.expected == OOS and a) or (r.expected != OOS and (not correct or not a)):
        c = cats(r.text, r.expected, r.pred, r.lang, "misc", a, correct and r.expected != OOS)
        if r.expected == OOS and a: c = "OUT_OF_SCOPE answered (should abstain) | " + c
        err.append({"source": "handchecked", "id": int(r.id), "text": r.text, "true": r.expected, "pred": r.pred,
                    "conf": r.conf, "sim": None, "answered": a, "lang": r.lang, "dialect": "misc", "category": c})
E = pd.DataFrame(err)
E.to_csv(os.path.join(RES, "errors.csv"), index=False, encoding="utf-8-sig")
M["error_analysis"] = {"n_failures_logged": len(E),
                       "by_category": E.category.str.split(r" \| ").explode().str.replace(r"confusion .*", "confusion pair", regex=True)
                                       .value_counts().to_dict()}

# --------------------------------------------------------------- 7. charts
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
keys = ["all", "en", "ar", "msa", "gulf", "egy", "lev"]
fig, ax = plt.subplots(figsize=(8, 3.8))
w = 0.27
for j, (lab, col, key) in enumerate((("Majority class", "#b8b8b8", "majority"),
                                     ("Word TF-IDF + LR (baseline)", "#6f93b8", "baseline_word_tfidf_LR"),
                                     (f"{BACKEND} + LR", "#d9772b", "embeddings_LR"))):
    ax.bar(np.arange(len(keys)) + (j - 1) * w, [M["classification"][key][k] for k in keys], w, label=lab, color=col)
ax.set_xticks(range(len(keys)))
ax.set_xticklabels(["All", "English", "Arabic (all)", "MSA", "Gulf", "Egyptian", "Levantine"])
ax.set_ylabel("macro-F1 (test)"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
ax.set_title("Same test set, by language / dialect", loc="left")
fig.tight_layout(); fig.savefig(os.path.join(RES, "macro_f1_by_language.png"), dpi=160); plt.close(fig)

# coverage vs accuracy as the confidence threshold moves (sigma fixed at its chosen value)
cov, acc = [], []
ts = np.linspace(0.2, 0.99, 60)
for t in ts:
    m = (conf >= t) & (sim1 >= asst.sigma)
    cov.append(m.mean()); acc.append(ok[m].mean() if m.any() else np.nan)
fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.plot(cov, acc, color="#d9772b", lw=2)
ax.scatter([M["abstention_test"]["coverage"]], [M["abstention_test"]["accuracy_on_answered"]], color="black", zorder=3)
ax.annotate(f"chosen on validation\n(tau={asst.tau}, sigma={asst.sigma})",
            (M["abstention_test"]["coverage"], M["abstention_test"]["accuracy_on_answered"]),
            textcoords="offset points", xytext=(-95, -34), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
ax.set_xlabel("coverage (share answered automatically)"); ax.set_ylabel("accuracy on answered tickets")
ax.set_title("Abstain trade-off (test)", loc="left"); ax.set_ylim(0.5, 1.02)
fig.tight_layout(); fig.savefig(os.path.join(RES, "coverage_vs_accuracy.png"), dpi=160); plt.close(fig)

cm = confusion_matrix(test.intent, pred, labels=INTENTS)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.imshow(cm, cmap="Oranges")
ax.set_xticks(range(len(INTENTS))); ax.set_yticks(range(len(INTENTS)))
ax.set_xticklabels(INTENTS, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(INTENTS, fontsize=8)
for i in range(len(INTENTS)):
    for j in range(len(INTENTS)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="black")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (test, no abstain)", loc="left")
fig.tight_layout(); fig.savefig(os.path.join(RES, "confusion_matrix.png"), dpi=160); plt.close(fig)

# ------------------------------------------------------------- 8. report
json.dump(M, open(os.path.join(RES, "metrics.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def row(label, d):
    return f"| {label} | " + " | ".join(f"{d[k]:.3f}" for k in keys) + " |"


A, H = M["abstention_test"], M["handchecked"]
rep = [f"# Evaluation report\n\n**Embedding backend used for these numbers: `{BACKEND}`**\n",
       f"Data: train {len(train)} / val {len(val)} / test {len(test)} tickets "
       f"(test = {M['test_distinct_cores']} distinct core phrasings never seen in train; synthetic data, see README).\n",
       "## 1. Intent classification (macro-F1, test)\n",
       "| model | " + " | ".join(["all", "en", "ar", "msa", "gulf", "egy", "lev"]) + " |", "|---|" + "---|" * len(keys),
       row("majority class", M["classification"]["majority"]),
       row("word TF-IDF + LR (baseline)", M["classification"]["baseline_word_tfidf_LR"]),
       row(f"{BACKEND} + LR", M["classification"]["embeddings_LR"]), "",
       "## 2. Abstain (thresholds chosen on validation by error cost)\n",
       f"- tau (min confidence) = {asst.tau}, sigma (min similarity to a past ticket) = {asst.sigma}; "
       f"cost: wrong auto-reply = {asst.wrong_auto_cost}, send to human = {asst.abstain_cost}",
       f"- Test coverage {A['coverage']:.1%}; accuracy on answered {A['accuracy_on_answered']:.1%} "
       f"(vs {A['accuracy_no_abstain']:.1%} without abstain)",
       f"- Wrong automatic replies: {A['wrong_auto_replies_no_abstain']} -> {A['wrong_auto_replies_with_abstain']}; "
       f"expected cost per ticket {A['cost_no_abstain']:.3f} -> {A['cost_with_abstain']:.3f}", "",
       "## 3. Hand-checked set (44 tickets: 32 in-scope, 12 out-of-scope; never used for tuning)\n",
       f"- In-scope accuracy without abstain: {H['in_scope_accuracy_no_abstain']:.1%} "
       f"(EN {H['in_scope_accuracy_no_abstain_en']:.1%}, AR {H['in_scope_accuracy_no_abstain_ar']:.1%})",
       f"- With abstain: coverage {H['in_scope_coverage']:.1%}, accuracy on answered {H['in_scope_accuracy_on_answered']:.1%}",
       f"- Out-of-scope tickets correctly sent to a human: {H['out_of_scope_rejected']:.1%}",
       f"- Wrong automatic replies (in-scope errors + answered out-of-scope): {H['wrong_auto_replies']} of {H['n']}",
       f"- Output schema valid: {H['schema_valid_rate']:.0%}", "",
       "## 4. Similar-ticket retrieval (query = test ticket, corpus = train tickets)\n",
       "| relevance definition | scope | Hit@5 | MRR | queries |", "|---|---|---|---|---|"]
for name, label in (("loose_same_intent", "same intent"), ("strict_same_issue", "same underlying issue"),
                    ("strict_cross_language", "same issue, other language only")):
    for lang in ("all", "en", "ar"):
        r = R[name][lang]
        rep.append(f"| {label} | {lang} | {r['Hit@5']:.3f} | {r['MRR']:.3f} | {r['n_queries']} |")
rep += ["", f"## 5. Latency\n\np50 {M['latency_ms']['p50']:.1f} ms, p95 {M['latency_ms']['p95']:.1f} ms per ticket (CPU).", "",
        f"## 6. Failures logged: {len(E)} (see errors.csv)\n"]
rep += [f"- {k}: {v}" for k, v in M["error_analysis"]["by_category"].items()]
open(os.path.join(RES, "report.md"), "w", encoding="utf-8").write("\n".join(rep) + "\n")

# ------------------------------------------- 9. categorised failure write-up (auto-regenerated per backend)
HYP = {
    "CONFIDENT_ERROR": "Wrong and still above both thresholds: phrasing unseen in train that shares vocabulary with the predicted intent. Would send a wrong auto-reply -> most expensive failure. Fix: more diverse data / fine-tuning.",
    "OVER_ABSTAIN": "Prediction was right but confidence/similarity fell under the threshold, so a human handles it. Cost = lost automation, not a wrong reply. Fix: more data near this phrasing, or re-tune thresholds.",
    "ERROR_CAUGHT_BY_ABSTAIN": "Wrong prediction that the abstain rule routed to a human. The safety net worked as designed.",
    "OUT_OF_SCOPE": "Out-of-scope text that resembles a known intent (e.g. a product or delivery question) and passed the gate. Fix: add out-of-scope examples to validation/training or a stricter similarity gate.",
}
E["primary"] = E.category.str.split(r" \| ").str[0].str.split(" ").str[0]
pick = []
for prim, n in (("CONFIDENT_ERROR", 4), ("OVER_ABSTAIN", 3), ("ERROR_CAUGHT_BY_ABSTAIN", 3), ("OUT_OF_SCOPE", 3)):
    sub = E[E.primary == prim]
    pick.append(sub.sample(min(n, len(sub)), random_state=0))
pick = pd.concat(pick)
conf_pairs = (pd.Series([(t, p) for t, p in zip(test.intent, pred) if t != p]).value_counts().head(5))
ea = ["# Error analysis (auto-generated; regenerate with `python evaluate.py`)\n",
      f"Backend: `{BACKEND}`. Failures logged in errors.csv: {len(E)}. Categories are assigned by rules in evaluate.py; "
      "the 'likely cause' column is a hypothesis to verify by reading the example, not a proven fact.\n",
      "## Top confusion pairs on the test set (no abstain)\n", "| true -> predicted | count |", "|---|---|"]
ea += [f"| {t} -> {p} | {c} |" for (t, p), c in conf_pairs.items()]
ea += ["", f"## {len(pick)} categorised failures\n", "| # | source | ticket | true -> pred | conf | category | likely cause |", "|---|---|---|---|---|---|---|"]
for i, (_, r) in enumerate(pick.iterrows(), 1):
    txt = r.text.replace("|", "/")
    cause = HYP.get(r.primary, "")
    cat = r.category.replace("|", ";")
    ea.append(f"| {i} | {r.source} | {txt} | {r.true} -> {r.pred} | {r.conf} | {cat} | {cause} |")
open(os.path.join(RES, "error_analysis.md"), "w", encoding="utf-8").write("\n".join(ea) + "\n")
print("\n".join(rep))
