# Error analysis (auto-generated; regenerate with `python evaluate.py`)

Backend: `multilingual-e5-small`. Failures logged in errors.csv: 72. Categories are assigned by rules in evaluate.py; the 'likely cause' column is a hypothesis to verify by reading the example, not a proven fact.

## Top confusion pairs on the test set (no abstain)

| true -> predicted | count |
|---|---|
| track_order -> change_address | 12 |
| track_refund -> login_problem | 7 |
| track_order -> cancel_order | 7 |
| cancel_order -> damaged_item | 4 |
| payment_failed -> track_refund | 3 |

## 12 categorised failures

| # | source | ticket | true -> pred | conf | category | likely cause |
|---|---|---|---|---|---|---|
| 1 | handchecked | do you deliver the blender to Jeddah and how long does it take? | product_question -> track_order | 0.684 | CONFIDENT_ERROR (auto-reply would be wrong) ; confusion product_question -> track_order | Wrong and still above both thresholds: phrasing unseen in train that shares vocabulary with the predicted intent. Would send a wrong auto-reply -> most expensive failure. Fix: more diverse data / fine-tuning. |
| 2 | test | ارجو افادتي بحاله التوصيل للطلب 47632. | track_order -> change_address | 0.557 | CONFIDENT_ERROR (auto-reply would be wrong) ; confusion track_order -> change_address | Wrong and still above both thresholds: phrasing unseen in train that shares vocabulary with the predicted intent. Would send a wrong auto-reply -> most expensive failure. Fix: more diverse data / fine-tuning. |
| 3 | test | طلبت مقاس غلط من الخلاط ابغى الغيه الحين | cancel_order -> damaged_item | 0.68 | CONFIDENT_ERROR (auto-reply would be wrong) ; confusion cancel_order -> damaged_item | Wrong and still above both thresholds: phrasing unseen in train that shares vocabulary with the predicted intent. Would send a wrong auto-reply -> most expensive failure. Fix: more diverse data / fine-tuning. |
| 4 | test | مساء الخير، ارجو افادتي بحالة التوصيل للطلب 94654. جزاكم الله خير. | track_order -> cancel_order | 0.571 | CONFIDENT_ERROR (auto-reply would be wrong) ; confusion track_order -> cancel_order | Wrong and still above both thresholds: phrasing unseen in train that shares vocabulary with the predicted intent. Would send a wrong auto-reply -> most expensive failure. Fix: more diverse data / fine-tuning. |
| 5 | test | الكارت اترفض بس الفلوس اتخصمت من حسابي للأوردر ٦٨٤٥٤ | payment_failed -> payment_failed | 0.405 | OVER_ABSTAIN (correct but sent to human) ; low-resource dialect (fewer training templates) | Prediction was right but confidence/similarity fell under the threshold, so a human handles it. Cost = lost automation, not a wrong reply. Fix: more data near this phrasing, or re-tune thresholds. |
| 6 | test | طلبت مقاس غلط من الشنطة ابغى الغيه الحين | cancel_order -> cancel_order | 0.497 | OVER_ABSTAIN (correct but sent to human) | Prediction was right but confidence/similarity fell under the threshold, so a human handles it. Cost = lost automation, not a wrong reply. Fix: more data near this phrasing, or re-tune thresholds. |
| 7 | test | الكارت اترفض بس الفلوس اتخصمت من حسابي للاوردر ٧٥١٦٤ ارجو المساعدة. | payment_failed -> payment_failed | 0.437 | OVER_ABSTAIN (correct but sent to human) ; low-resource dialect (fewer training templates) | Prediction was right but confidence/similarity fell under the threshold, so a human handles it. Cost = lost automation, not a wrong reply. Fix: more data near this phrasing, or re-tune thresholds. |
| 8 | test | ارجو افادتي بحالة التوصيل للطلب 93122. | track_order -> change_address | 0.518 | ERROR_CAUGHT_BY_ABSTAIN ; confusion track_order -> change_address | Wrong prediction that the abstain rule routed to a human. The safety net worked as designed. |
| 9 | test | السلام عليكم، قلتوا استلمتوا المرتجع بس ما شفت فلوسي في حسابي | track_refund -> login_problem | 0.38 | ERROR_CAUGHT_BY_ABSTAIN ; confusion track_refund -> login_problem | Wrong prediction that the abstain rule routed to a human. The safety net worked as designed. |
| 10 | test | السلام عليكم، الطلب تأخر كتير والتتبع ما عم يتغير جزاكم الله خير. | track_order -> cancel_order | 0.449 | ERROR_CAUGHT_BY_ABSTAIN ; low-resource dialect (fewer training templates) ; confusion track_order -> cancel_order | Wrong prediction that the abstain rule routed to a human. The safety net worked as designed. |
| 11 | handchecked | ابغى اشتري سيارة مستعملة، عندكم؟ | out_of_scope -> product_question | 0.914 | OUT_OF_SCOPE answered (should abstain) ; CONFIDENT_ERROR (auto-reply would be wrong) ; very short / little context ; confusion out_of_scope -> product_question | Out-of-scope text that resembles a known intent (e.g. a product or delivery question) and passed the gate. Fix: add out-of-scope examples to validation/training or a stricter similarity gate. |
| 12 | handchecked | المندوب كان وقح معي وابغى اقدم شكوى عليه | out_of_scope -> damaged_item | 0.668 | OUT_OF_SCOPE answered (should abstain) ; CONFIDENT_ERROR (auto-reply would be wrong) ; confusion out_of_scope -> damaged_item | Out-of-scope text that resembles a known intent (e.g. a product or delivery question) and passed the gate. Fix: add out-of-scope examples to validation/training or a stricter similarity gate. |
