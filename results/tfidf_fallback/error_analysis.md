# Error analysis (auto-generated; regenerate with `python evaluate.py`)

Backend: `tfidf-svd (offline fallback)`. Failures logged in errors.csv: 215. Categories are assigned by rules in evaluate.py; the 'likely cause' column is a hypothesis to verify by reading the example, not a proven fact.

## Top confusion pairs on the test set (no abstain)

| true -> predicted | count |
|---|---|
| track_order -> change_address | 18 |
| track_order -> damaged_item | 16 |
| track_order -> track_refund | 12 |
| damaged_item -> track_order | 12 |
| track_refund -> product_question | 11 |

## 13 categorised failures

| # | source | ticket | true -> pred | conf | category | likely cause |
|---|---|---|---|---|---|---|
| 1 | test | Hi, phone case from order ORD-37534 is defective and stopped working after one day. Thank you! | damaged_item -> track_order | 0.703 | CONFIDENT_ERROR (auto-reply would be wrong) ; confusion damaged_item -> track_order | Wrong and still above both thresholds: phrasing unseen in train that shares vocabulary with the predicted intent. Would send a wrong auto-reply -> most expensive failure. Fix: more diverse data / fine-tuning. |
| 2 | test | my package hasn't arrived yet tracking says it's stuck thanks | track_order -> damaged_item | 0.739 | CONFIDENT_ERROR (auto-reply would be wrong) ; confusion track_order -> damaged_item | Wrong and still above both thresholds: phrasing unseen in train that shares vocabulary with the predicted intent. Would send a wrong auto-reply -> most expensive failure. Fix: more diverse data / fine-tuning. |
| 3 | test | السلام عليكم أرجو إفادتي بحالة التوصيل للطلب 93431 | track_order -> change_address | 0.839 | CONFIDENT_ERROR (auto-reply would be wrong) ; confusion track_order -> change_address | Wrong and still above both thresholds: phrasing unseen in train that shares vocabulary with the predicted intent. Would send a wrong auto-reply -> most expensive failure. Fix: more diverse data / fine-tuning. |
| 4 | test | أرجو إفادتي بحالة التوصيل للطلب #48825. ارجو المساعدة. | track_order -> change_address | 0.784 | CONFIDENT_ERROR (auto-reply would be wrong) ; confusion track_order -> change_address | Wrong and still above both thresholds: phrasing unseen in train that shares vocabulary with the predicted intent. Would send a wrong auto-reply -> most expensive failure. Fix: more diverse data / fine-tuning. |
| 5 | test | الطلب تاخر كتير والتتبع ما عم يتغير جزاكم الله خير | track_order -> track_order | 0.619 | OVER_ABSTAIN (correct but sent to human) ; low-resource dialect (fewer training templates) | Prediction was right but confidence/similarity fell under the threshold, so a human handles it. Cost = lost automation, not a wrong reply. Fix: more data near this phrasing, or re-tune thresholds. |
| 6 | test | the payment page shows an error when i try to pay. thanks. | payment_failed -> payment_failed | 0.55 | OVER_ABSTAIN (correct but sent to human) | Prediction was right but confidence/similarity fell under the threshold, so a human handles it. Cost = lost automation, not a wrong reply. Fix: more data near this phrasing, or re-tune thresholds. |
| 7 | test | مرحبا، أرجو إرسال الساعه الذكيه إلى عنوان منزلي الجديد. جزاكم الله خير. | change_address -> change_address | 0.632 | OVER_ABSTAIN (correct but sent to human) | Prediction was right but confidence/similarity fell under the threshold, so a human handles it. Cost = lost automation, not a wrong reply. Fix: more data near this phrasing, or re-tune thresholds. |
| 8 | test | عاجل ارجو افادتي بحاله التوصيل للطلب 57632 ارجو المساعده | track_order -> change_address | 0.669 | ERROR_CAUGHT_BY_ABSTAIN ; confusion track_order -> change_address | Wrong prediction that the abstain rule routed to a human. The safety net worked as designed. |
| 9 | test | Hi team, How long does a refund take? I was told 5 to 7 days. Thanks. | track_refund -> product_question | 0.495 | ERROR_CAUGHT_BY_ABSTAIN ; confusion track_refund -> product_question | Wrong prediction that the abstain rule routed to a human. The safety net worked as designed. |
| 10 | test | عاجل: طلبت مقاس غلط من الخلاط ابغي الغيه الحين جزاكم الله خير. | cancel_order -> track_order | 0.412 | ERROR_CAUGHT_BY_ABSTAIN ; confusion cancel_order -> track_order | Wrong prediction that the abstain rule routed to a human. The safety net worked as designed. |
| 11 | handchecked | المندوب كان وقح معي وابغى اقدم شكوى عليه | out_of_scope -> track_order | 0.879 | OUT_OF_SCOPE answered (should abstain) ; CONFIDENT_ERROR (auto-reply would be wrong) ; confusion out_of_scope -> track_order | Out-of-scope text that resembles a known intent (e.g. a product or delivery question) and passed the gate. Fix: add out-of-scope examples to validation/training or a stricter similarity gate. |
| 12 | handchecked | ابغى اشتري سيارة مستعملة، عندكم؟ | out_of_scope -> track_order | 0.844 | OUT_OF_SCOPE answered (should abstain) ; CONFIDENT_ERROR (auto-reply would be wrong) ; very short / little context ; confusion out_of_scope -> track_order | Out-of-scope text that resembles a known intent (e.g. a product or delivery question) and passed the gate. Fix: add out-of-scope examples to validation/training or a stricter similarity gate. |
| 13 | handchecked | ما هي ساعات العمل يوم الجمعة؟ | out_of_scope -> damaged_item | 0.95 | OUT_OF_SCOPE answered (should abstain) ; CONFIDENT_ERROR (auto-reply would be wrong) ; confusion out_of_scope -> damaged_item | Out-of-scope text that resembles a known intent (e.g. a product or delivery question) and passed the gate. Fix: add out-of-scope examples to validation/training or a stricter similarity gate. |
