"""Ticket Assistant core: embeddings -> classify (+abstain) -> retrieve -> grounded reply -> validate.

Pipeline (same shape as the workshop diagram):

    Ticket (EN/AR) -> Embed -> Classify intent + confidence -> abstain if unsure
                            -> Retrieve similar past tickets (top-k)
                            -> Retrieve KB articles for the predicted intent
                            -> Draft a reply from the KB with citations
                            -> Validate (schema, citations, needs_human)
"""
import hashlib
import json
import os
import pickle
import re
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
INTENTS = ["track_order", "track_refund", "cancel_order", "change_address",
           "payment_failed", "damaged_item", "login_problem", "product_question"]
OOS = "out_of_scope"

# ----------------------------------------------------------------- text utils
_AR = re.compile(r"[؀-ۿ]")
_DIACRITICS = re.compile(r"[ً-ْـ]")
_AR_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}


def detect_lang(text):
    ar = len(_AR.findall(text))
    letters = len(re.findall(r"[A-Za-z؀-ۿ]", text)) or 1
    return "ar" if ar / letters > 0.3 else "en"


def normalize_text(text):
    """Light Arabic normalisation (used by the lexical models, NOT by e5)."""
    t = text.translate(_AR_DIGITS).lower()
    t = _DIACRITICS.sub("", t)
    t = re.sub("[إأآ]", "ا", t)
    t = t.replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"\s+", " ", t).strip()


def extract_order_id(text):
    m = re.search(r"(?<!\d)(\d{5,8})(?!\d)", text.translate(_AR_DIGITS))
    return m.group(1) if m else None


# ------------------------------------------------------------------ backends
class TfidfSvdBackend:
    """Offline fallback 'embedding': char n-gram TF-IDF -> SVD -> L2 norm.
    Lexical only (cannot match across languages) -- use e5 for the real thing."""
    name = "tfidf-svd (offline fallback)"

    def __init__(self, dim=200):
        self.dim = dim

    def fit(self, texts):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                   min_df=2, preprocessor=normalize_text)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(min(self.dim, X.shape[1] - 1), random_state=0).fit(X)
        return self

    def encode(self, texts):
        return normalize(self.svd.transform(self.vec.transform(list(texts)))).astype(np.float32)


class E5Backend:
    """intfloat/multilingual-e5-small (384-d). e5 expects a 'query: ' prefix for symmetric tasks."""
    name = "multilingual-e5-small"

    def __init__(self, model="intfloat/multilingual-e5-small", cache=True):
        from sentence_transformers import SentenceTransformer  # lazy: torch only needed here
        self.model = SentenceTransformer(model)
        self.cache_path = os.path.join(ROOT, ".cache", "e5_embeddings.pkl") if cache else None
        self.cache = {}
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.cache = pickle.load(f)

    def fit(self, texts):  # nothing to fit: the encoder is pretrained
        return self

    def encode(self, texts):
        texts = list(texts)
        keys = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]
        todo = [(k, t) for k, t in zip(keys, texts) if k not in self.cache]
        if todo:
            vecs = self.model.encode(["query: " + t for _, t in todo], normalize_embeddings=True,
                                     batch_size=64, show_progress_bar=False)
            for (k, _), v in zip(todo, vecs):
                self.cache[k] = v.astype(np.float32)
            if self.cache_path:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.cache, f)
        return np.stack([self.cache[k] for k in keys])


def make_backend(name="auto"):
    """auto = e5 if it can be loaded, otherwise the offline fallback (with a loud warning)."""
    if name in ("auto", "e5"):
        try:
            return E5Backend()
        except Exception as e:  # no torch / no internet / model blocked
            if name == "e5":
                raise
            print(f"[WARN] multilingual-e5-small unavailable ({type(e).__name__}); "
                  f"using the offline TF-IDF+SVD fallback. Numbers are NOT e5 numbers.")
    return TfidfSvdBackend()


# ------------------------------------------------------------------- assistant
PRIORITY = {"payment_failed": "high", "cancel_order": "high", "damaged_item": "medium",
            "login_problem": "medium", "track_order": "medium", "track_refund": "medium",
            "change_address": "medium", "product_question": "low"}
URGENT = re.compile(r"urgent|asap|immediately|عاجل|فورا|الحين", re.I)

REPLY = {
    "en": {"hello": "Hello,\nThanks for contacting us.", "order": " Regarding order {oid}:",
           "bye": "If this doesn't solve it, just reply to this message and our team will follow up."},
    "ar": {"hello": "مرحبا،\nشكرا لتواصلك معنا.", "order": " بخصوص الطلب {oid}:",
           "bye": "إذا لم يحل هذا مشكلتك، رد على هذه الرسالة وسيتابع فريقنا معك."},
}


class TicketAssistant:
    def __init__(self, backend, kb_path=None, wrong_auto_cost=5.0, abstain_cost=1.0, k_similar=5):
        self.backend = backend
        with open(kb_path or os.path.join(ROOT, "data", "kb.json"), encoding="utf-8") as f:
            self.kb = json.load(f)
        self.wrong_auto_cost, self.abstain_cost, self.k = wrong_auto_cost, abstain_cost, k_similar

    # ---- training ------------------------------------------------------
    def fit(self, train, val):
        kb_texts = [f'{a["title_en"]}. {a["en"]} {a["title_ar"]}. {a["ar"]}' for a in self.kb]
        self.backend.fit(list(train.text) + kb_texts)
        self.hist = train.reset_index(drop=True)                  # "historical tickets"
        self.Xh = self.backend.encode(self.hist.text)
        self.Xkb = self.backend.encode(kb_texts)
        val_in = val[val.intent != OOS]
        Xv_in = self.backend.encode(val_in.text)
        best = (-1, None)
        for C in (0.01, 0.03, 0.1, 0.3, 1, 3):                    # tune on validation, never on test
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
            m.fit(self.Xh, self.hist.intent)
            f1 = f1_score(val_in.intent, m.predict(Xv_in), average="macro")
            if f1 > best[0]:
                best = (f1, C)
        self.C, self.val_macro_f1 = best[1], best[0]
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced"))
        self.clf.fit(self.Xh, self.hist.intent)
        self.tau, self.sigma, self.cost_grid = self._choose_thresholds(val)
        return self

    def _choose_thresholds(self, val):
        """Pick two abstain thresholds that minimise expected cost ON VALIDATION data:
          tau   : minimum classifier confidence (max predicted probability)
          sigma : minimum cosine similarity to the closest historical ticket ("is this familiar?")
        Cost: wrong auto-reply = wrong_auto_cost, sending to a human = abstain_cost, correct auto = 0.
        Out-of-scope tickets can never be answered correctly, so answering one counts as wrong.
        Why two signals: a linear head is over-confident on text unlike anything it saw in training,
        but such text is far from every historical ticket."""
        Z = self.backend.encode(val.text)
        P = self.clf.predict_proba(Z)
        conf, pred = P.max(1), self.clf.classes_[P.argmax(1)]
        sim = (Z @ self.Xh.T).max(1)
        ok = pred == val.intent.values
        taus = np.round(np.arange(0.20, 1.0, 0.02), 2)
        sigmas = np.round(np.arange(0.0, 1.0, 0.02), 2)
        grid = np.zeros((len(taus), len(sigmas)))
        for i, t in enumerate(taus):
            for j, s in enumerate(sigmas):
                answered = (conf >= t) & (sim >= s)
                grid[i, j] = np.where(answered, np.where(ok, 0.0, self.wrong_auto_cost), self.abstain_cost).mean()
        best = grid.min()
        ii, jj = np.where(grid <= best + 1e-9)           # take the middle of the optimal plateau (robust)
        return float(np.median(taus[ii])), float(np.median(sigmas[jj])), (taus, sigmas, grid)

    def gate(self, conf, sim):
        """True -> answer automatically, False -> abstain (route to a human)."""
        return (np.asarray(conf) >= self.tau) & (np.asarray(sim) >= self.sigma)

    # ---- inference -----------------------------------------------------
    def predict_proba(self, texts):
        return self.clf.predict_proba(self.backend.encode(texts))

    def similar(self, Z, k=None):
        """Top-k most similar historical tickets for each embedded query (cosine = dot product)."""
        S = Z @ self.Xh.T
        k = k or self.k
        idx = np.argsort(-S, axis=1)[:, :k]
        return idx, np.take_along_axis(S, idx, axis=1)

    def run(self, text):
        t0 = time.perf_counter()
        z = self.backend.encode([text])
        p = self.clf.predict_proba(z)[0]
        top = self.clf.classes_[p.argmax()]
        conf = float(p.max())
        lang = detect_lang(text)
        oid = extract_order_id(text)
        idx, sims = self.similar(z, self.k)
        auto = bool(self.gate(conf, sims[0][0]))
        similar = [{"id": int(self.hist.id[i]), "intent": self.hist.intent[i], "text": self.hist.text[i],
                    "similarity": round(float(s), 3)} for i, s in zip(idx[0], sims[0])]
        priority = "high" if URGENT.search(text) else PRIORITY[top]
        out = {"intent": top if auto else "unknown", "top_guess": top, "confidence": round(conf, 3),
               "route": "auto" if auto else "human", "priority": priority, "language": lang,
               "order_id": oid, "similar_tickets": [s["id"] for s in similar],
               "reply": None, "citations": [], "needs_human": not auto}
        if auto:
            arts = self._kb_for(top, z[0])
            out["reply"] = self._draft(lang, arts, oid)
            out["citations"] = [a["id"] for a in arts]
        problems = self.validate(out)
        if problems:
            out["needs_human"], out["route"], out["validation_errors"] = True, "human", problems
        out["_similar_detail"] = similar
        out["_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    def _kb_for(self, intent, z, n=2):
        cand = [i for i, a in enumerate(self.kb) if a["intent"] == intent]
        sims = [float(self.Xkb[i] @ z) for i in cand]
        order = [cand[j] for j in np.argsort(sims)[::-1][:n]]
        return [self.kb[i] for i in order]

    def _draft(self, lang, arts, oid):
        t = REPLY[lang]
        head = t["hello"] + (t["order"].format(oid=oid) if oid else "")
        body = "\n".join(f'{a[lang]} [{a["id"]}]' for a in arts)
        return f"{head}\n{body}\n{t['bye']}"

    # ---- validation ("schemas are contracts") ---------------------------
    def validate(self, out):
        errs = []
        need = {"intent", "confidence", "route", "priority", "language", "order_id",
                "similar_tickets", "reply", "citations", "needs_human"}
        if not need <= set(out):
            errs.append(f"missing keys: {sorted(need - set(out))}")
        if out["intent"] not in INTENTS + ["unknown"]:
            errs.append("intent not in label set")
        if not 0 <= out["confidence"] <= 1:
            errs.append("confidence out of range")
        if out["route"] == "human" and out["reply"]:
            errs.append("human-routed ticket must not carry an auto reply")
        if out["reply"]:
            kb_ids = {a["id"] for a in self.kb}
            cited = {int(x) for x in re.findall(r"\[(\d+)\]", out["reply"])}
            if not cited:
                errs.append("reply has no citation")
            if not cited <= set(out["citations"]) or not set(out["citations"]) <= kb_ids:
                errs.append("citation does not match a KB article")
        return errs
