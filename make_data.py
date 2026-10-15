"""Generate the synthetic bilingual ticket dataset (EN + Arabic dialects).

Why synthetic?  No labelled Arabic support-ticket data was available offline.
Everything here is template-based, so numbers on it are OPTIMISTIC compared
with real tickets.  To limit that, the split is done BY TEMPLATE (a test ticket
never shares its core phrasing with a train ticket) and exact duplicates across
splits are removed.

Run:  python make_data.py
Out:  data/tickets.csv   (train / val / test)
"""
import csv
import json
import os
import random
import re

random.seed(7)
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- templates
# T[intent][dialect] -> list of core phrasings.  Index i means the SAME
# sub-issue in every language (used as a cross-lingual "issue id" for retrieval).
T = {
    "track_order": {
        "en": [
            "Where is my order {order}? It was supposed to arrive {days} days ago.",
            "I ordered {item} last week and I still have no tracking update.",
            "Can you tell me the delivery status of {order}?",
            "My package hasn't arrived yet, tracking says it's stuck.",
            "when will my {item} be delivered? ordered it a while back",
            "Order {order} shows shipped but nothing for {days} days. What's going on?",
        ],
        "msa": [
            "أين طلبي رقم {order}؟ كان من المفترض أن يصل قبل {days} أيام.",
            "طلبت {item} الأسبوع الماضي ولم يصلني أي تحديث عن الشحنة.",
            "أرجو إفادتي بحالة التوصيل للطلب {order}.",
            "لم تصلني الشحنة حتى الآن، وحالة التتبع لم تتغير.",
            "متى سيتم تسليم {item}؟ لقد طلبته منذ فترة.",
            "الطلب {order} مسجل كمشحون منذ {days} أيام دون أي تحديث.",
        ],
        "gulf": [
            "وين طلبي {order}؟ المفروض يوصل قبل {days} ايام",
            "طلبت {item} من اسبوع وللحين ما وصلني شي",
            "ابغى اعرف وين وصلت شحنتي {order}",
            "الشحنة متأخرة كثير والتتبع واقف مكانه",
            "متى بيوصلني {item}؟ طلبته من زمان",
            "الطلب {order} مكتوب انه انشحن بس ما تحرك من {days} ايام",
        ],
        "egy": [
            "فين طلبي {order}؟ المفروض يوصل من {days} ايام",
            "طلبت {item} من اسبوع ولسه مجاش",
            "عايز اعرف الشحنة بتاعتي وصلت فين",
            "الأوردر متأخر اوي والتراكينج مش بيتحدث",
        ],
        "lev": [
            "وين طلبي {order}؟ كان لازم يوصل من {days} ايام",
            "طلبت {item} من اسبوع ولهلق ما وصل",
            "بدي اعرف وين صارت شحنتي",
            "الطلب تأخر كتير والتتبع ما عم يتغير",
        ],
    },
    "track_refund": {
        "en": [
            "I returned {item} {days} days ago and haven't received my refund yet.",
            "Where is my refund for order {order}?",
            "It's been over a week since you approved my refund but the money hasn't hit my card.",
            "How long does a refund take? I was told 5 to 7 days.",
            "refund still pending for {order}, please check",
            "You confirmed the return of {item} but I see no money back in my account.",
        ],
        "msa": [
            "أعدت {item} قبل {days} أيام ولم أستلم المبلغ المسترد بعد.",
            "أين المبلغ المسترد للطلب رقم {order}؟",
            "مضى أكثر من أسبوع على الموافقة على الاسترجاع ولم يصل المبلغ إلى بطاقتي.",
            "كم تستغرق عملية استرداد المبلغ؟ أخبروني أنها من 5 إلى 7 أيام.",
            "ما زال الاسترداد معلقا للطلب {order}، أرجو المتابعة.",
            "أكدتم استلام المرتجع {item} لكن لا أرى المبلغ في حسابي.",
        ],
        "gulf": [
            "رجعت {item} من {days} ايام وللحين ما جاني المبلغ",
            "وين فلوس الاسترجاع حق طلبي {order}؟",
            "صار لي اكثر من اسبوع من وافقتوا على الاسترجاع والمبلغ ما نزل في بطاقتي",
            "كم ياخذ الاسترداد؟ قالوا لي من ٥ الى ٧ ايام عمل",
            "الاسترجاع للطلب {order} معلق للحين، ممكن تتأكدون؟",
            "قلتوا استلمتوا المرتجع بس ما شفت فلوسي في حسابي",
        ],
        "egy": [
            "رجعت {item} من {days} ايام ولسه الفلوس مجاتش",
            "فين فلوس الاسترجاع بتاعة الأوردر {order}؟",
            "بقالي اكتر من اسبوع وافقتوا على الاسترجاع والمبلغ مرجعش على الكارت",
            "الريفند بياخد قد ايه؟ قالولي ٥ لـ ٧ ايام",
        ],
        "lev": [
            "رجّعت {item} من {days} ايام ولهلق ما إجا المبلغ",
            "وين مصاري الاسترجاع تبع الطلب {order}؟",
            "صار إلي اكتر من اسبوع وافقتو على الاسترجاع والمبلغ ما نزل ع كرتي",
            "قديش بياخد الاسترداد؟ قالولي من ٥ لـ ٧ ايام",
        ],
    },
    "cancel_order": {
        "en": [
            "I want to cancel order {order} please.",
            "Please cancel my {item} order, I ordered it by mistake.",
            "How can I cancel my purchase before it ships?",
            "I changed my mind. Cancel {order} and stop the shipment.",
            "can i still cancel my order? it was placed an hour ago",
            "Ordered the wrong size of {item}, need to cancel right away.",
        ],
        "msa": [
            "أرغب في إلغاء الطلب رقم {order} من فضلكم.",
            "أرجو إلغاء طلب {item}، فقد طلبته بالخطأ.",
            "كيف يمكنني إلغاء الطلب قبل شحنه؟",
            "غيرت رأيي، ألغوا الطلب {order} وأوقفوا الشحن.",
            "هل ما زال بإمكاني إلغاء طلبي؟ تم الطلب قبل ساعة.",
            "طلبت المقاس الخاطئ من {item} وأحتاج إلى الإلغاء فورا.",
        ],
        "gulf": [
            "ابغى الغي الطلب {order} لو سمحتوا",
            "الغوا طلب {item} طلبته بالغلط",
            "كيف اقدر الغي طلبي قبل لا ينشحن؟",
            "غيرت رايي، الغوا {order} ووقفوا الشحن",
            "للحين اقدر الغي الطلب؟ توني طالبه من ساعة",
            "طلبت مقاس غلط من {item} ابغى الغيه الحين",
        ],
        "egy": [
            "عايز الغي الأوردر {order} لو سمحتوا",
            "الغوا طلب {item} انا طلبته بالغلط",
            "ازاي اكنسل الأوردر قبل ما يتشحن؟",
            "غيرت رأيي كنسلوا {order} وبلاش تشحنوه",
        ],
        "lev": [
            "بدي ألغي الطلب {order} لو سمحتو",
            "الغو طلب {item} طلبتو بالغلط",
            "كيف فيني الغي طلبي قبل ما ينشحن؟",
            "غيرت رأيي الغو {order} ولا تشحنوه",
        ],
    },
    "change_address": {
        "en": [
            "I need to change the delivery address for order {order}.",
            "I entered the wrong address, can you update it before shipping?",
            "Please send my {item} to my new home address instead.",
            "How do I edit the shipping address on an existing order?",
            "moved to a new apartment, need the parcel redirected for {order}",
            "The address on {order} has a typo in the street name, please correct it.",
        ],
        "msa": [
            "أحتاج إلى تغيير عنوان التوصيل للطلب رقم {order}.",
            "أدخلت عنوانا خاطئا، هل يمكنكم تعديله قبل الشحن؟",
            "أرجو إرسال {item} إلى عنوان منزلي الجديد.",
            "كيف أعدّل عنوان الشحن لطلب قائم؟",
            "انتقلت إلى شقة جديدة وأريد تحويل الشحنة للطلب {order}.",
            "يوجد خطأ إملائي في اسم الشارع في عنوان الطلب {order}، أرجو التصحيح.",
        ],
        "gulf": [
            "ابغى اغير عنوان التوصيل للطلب {order}",
            "كتبت العنوان غلط، ممكن تعدلونه قبل الشحن؟",
            "ارسلوا {item} على عنواني الجديد بدل القديم",
            "كيف اعدل عنوان الشحن لطلب موجود؟",
            "انتقلت لشقة جديدة وابغى تحولون الشحنة {order}",
            "فيه غلط في اسم الشارع بعنوان {order}، صححوه لو سمحتوا",
        ],
        "egy": [
            "عايز اغير عنوان التوصيل للأوردر {order}",
            "كتبت العنوان غلط ينفع تعدلوه قبل الشحن؟",
            "ابعتوا {item} على عنواني الجديد بدل القديم",
            "ازاي اعدل عنوان الشحن لأوردر موجود؟",
        ],
        "lev": [
            "بدي غيّر عنوان التوصيل للطلب {order}",
            "كتبت العنوان غلط فيكن تعدلوه قبل الشحن؟",
            "ابعتو {item} ع عنواني الجديد بدل القديم",
            "كيف بعدل عنوان الشحن لطلب موجود؟",
        ],
    },
    "payment_failed": {
        "en": [
            "My card was declined but the money was deducted for order {order}.",
            "Payment keeps failing at checkout even though my card works fine.",
            "I was charged twice for {item}. Please fix this.",
            "The payment page shows an error when I try to pay.",
            "my bank sent an sms that the money was taken but the order is not confirmed",
            "Can't complete payment with Apple Pay, it just spins.",
        ],
        "msa": [
            "تم رفض بطاقتي لكن خُصم المبلغ من حسابي للطلب {order}.",
            "عملية الدفع تفشل باستمرار رغم أن بطاقتي تعمل بشكل سليم.",
            "تم الخصم مرتين مقابل {item}، أرجو معالجة ذلك.",
            "تظهر صفحة الدفع رسالة خطأ عند محاولة السداد.",
            "وصلتني رسالة من البنك بخصم المبلغ لكن الطلب لم يتأكد.",
            "لا أستطيع إتمام الدفع عبر Apple Pay، الصفحة تبقى تحمّل.",
        ],
        "gulf": [
            "البطاقة انرفضت بس الفلوس انسحبت من حسابي للطلب {order}",
            "الدفع يفشل كل مرة مع ان بطاقتي شغالة عادي",
            "انسحب المبلغ مرتين حق {item} ابغاكم تصلحونها",
            "تطلع لي صفحة الدفع برسالة خطأ لما احاول ادفع",
            "جاني رسالة من البنك انه انخصم المبلغ بس الطلب ما تأكد",
            "ما اقدر اكمل الدفع بـ Apple Pay الصفحة تدور بس",
        ],
        "egy": [
            "الكارت اترفض بس الفلوس اتخصمت من حسابي للأوردر {order}",
            "الدفع بيفشل كل مرة والكارت شغال عادي",
            "اتخصم المبلغ مرتين على {item} حد يصلحها",
            "صفحة الدفع بتطلعلي رسالة خطأ",
        ],
        "lev": [
            "الكرت انرفض بس المصاري انسحبت من حسابي للطلب {order}",
            "الدفع بيفشل كل مرة والكرت شغال عادي",
            "انسحب المبلغ مرتين ع {item} صلحوها",
            "صفحة الدفع بتطلعلي رسالة خطأ ما بتخلصلي الدفع",
        ],
    },
    "damaged_item": {
        "en": [
            "My {item} arrived broken. I want a replacement.",
            "The package was crushed and the product inside is damaged.",
            "I received the wrong item, I ordered {item} but got something else.",
            "{item} from order {order} is defective and stopped working after one day.",
            "there's a crack on the screen of my {item}, it came like this",
            "Parts are missing from the box, the {item} is incomplete.",
        ],
        "msa": [
            "وصلني {item} مكسورا وأريد استبداله.",
            "كانت العبوة محطمة والمنتج بداخلها تالف.",
            "استلمت منتجا خاطئا، طلبت {item} لكن وصلني شيء آخر.",
            "{item} من الطلب {order} معيب وتوقف عن العمل بعد يوم واحد.",
            "توجد شروخ في شاشة {item} وقد وصل بهذه الحالة.",
            "هناك قطع ناقصة في الصندوق و{item} غير مكتمل.",
        ],
        "gulf": [
            "وصلني {item} مكسور وابغى بدله",
            "العلبة وصلت مكسرة والمنتج اللي داخلها خربان",
            "جاني منتج غلط طلبت {item} وجاني شي ثاني",
            "{item} من الطلب {order} خربان وتوقف عن الشغل بعد يوم",
            "فيه كسر في شاشة {item} ووصل كذا",
            "فيه قطع ناقصة بالصندوق و{item} مو كامل",
        ],
        "egy": [
            "وصلني {item} مكسور وعايز استبدله",
            "العلبة جت مفتوحة والمنتج جواها باظ",
            "جالي منتج غلط طلبت {item} وجالي حاجة تانية",
            "{item} من الأوردر {order} بايظ ووقف بعد يوم",
        ],
        "lev": [
            "وصلني {item} مكسور وبدي بدله",
            "العلبة وصلت مدعوسة والمنتج جواتها خربان",
            "إجاني منتج غلط طلبت {item} وإجاني شي تاني",
            "{item} من الطلب {order} خربان ووقف بعد يوم",
        ],
    },
    "login_problem": {
        "en": [
            "I can't log in to my account, it says wrong password.",
            "I never received the verification code on my phone.",
            "My account is locked after too many attempts.",
            "How do I reset my password? The reset email isn't arriving.",
            "can't sign in with my email, the app keeps showing an error",
            "I want to change my registered phone number but can't access my account.",
        ],
        "msa": [
            "لا أستطيع تسجيل الدخول إلى حسابي، تظهر رسالة كلمة المرور خاطئة.",
            "لم يصلني رمز التحقق على هاتفي.",
            "تم إغلاق حسابي بعد محاولات كثيرة.",
            "كيف أعيد تعيين كلمة المرور؟ رسالة الاستعادة لا تصل.",
            "لا أتمكن من الدخول ببريدي الإلكتروني ويظهر التطبيق خطأ.",
            "أريد تغيير رقم هاتفي المسجل لكن لا أستطيع الدخول إلى حسابي.",
        ],
        "gulf": [
            "ما اقدر ادخل حسابي يقول الباسوورد غلط",
            "ما وصلني رمز التحقق على جوالي",
            "حسابي انقفل بعد محاولات كثيرة",
            "كيف اغير الباسوورد؟ رسالة الاستعادة ما تجي",
            "ما اقدر ادخل بايميلي والتطبيق يطلع خطأ",
            "ابغى اغير رقم جوالي المسجل بس ما اقدر ادخل حسابي",
        ],
        "egy": [
            "مش قادر ادخل حسابي بيقولي الباسورد غلط",
            "كود التفعيل مجاليش على الموبايل",
            "حسابي اتقفل بعد محاولات كتير",
            "ازاي اغير الباسورد؟ رسالة الاستعادة مبتوصلش",
        ],
        "lev": [
            "ما فيني فوت ع حسابي بيقول الباسوورد غلط",
            "ما وصلني كود التحقق ع موبايلي",
            "حسابي انقفل بعد محاولات كتير",
            "كيف بغير الباسوورد؟ رسالة الاستعادة ما بتوصل",
        ],
    },
    "product_question": {
        "en": [
            "Does the {item} come with a warranty?",
            "Is the {item} available in black, and when will it be back in stock?",
            "What are the dimensions and weight of the {item}?",
            "Do you ship {item} to other cities?",
            "can you tell me if {item} is compatible with my old model?",
            "Is {item} original and does it have a one year guarantee?",
        ],
        "msa": [
            "هل يأتي {item} مع ضمان؟",
            "هل {item} متوفر باللون الأسود ومتى سيعود إلى المخزون؟",
            "ما هي أبعاد ووزن {item}؟",
            "هل تشحنون {item} إلى مدن أخرى؟",
            "هل {item} متوافق مع الطراز القديم لدي؟",
            "هل {item} أصلي وهل يشمله ضمان لمدة سنة؟",
        ],
        "gulf": [
            "{item} فيه ضمان؟",
            "{item} متوفر باللون الاسود ومتى يرجع في المخزون؟",
            "كم مقاس ووزن {item}؟",
            "تشحنون {item} لمدن ثانية؟",
            "{item} يشتغل مع جهازي القديم؟",
            "{item} اصلي وفيه ضمان سنة؟",
        ],
        "egy": [
            "{item} جاي بضمان؟",
            "{item} متاح باللون الاسود وهيرجع امتى؟",
            "ابعاد ووزن {item} كام؟",
            "بتشحنوا {item} لمحافظات تانية؟",
        ],
        "lev": [
            "{item} إلو كفالة؟",
            "{item} متوفر باللون الأسود ومتى بيرجع؟",
            "كم حجم ووزن {item}؟",
            "بتشحنو {item} لمدن تانية؟",
        ],
    },
}

# class imbalance on purpose (real support queues are imbalanced)
MULT = {"track_order": 1.5, "track_refund": 1.2, "cancel_order": 0.9, "change_address": 0.6,
        "payment_failed": 0.9, "damaged_item": 1.0, "login_problem": 0.9, "product_question": 0.6}
BASE_K = 12

# split by template group; the offset differs per dialect so that a sub-issue that is
# "test" in one dialect is usually "train" in another (needed for cross-lingual retrieval)
OFFSET = {"en": 0, "msa": 1, "gulf": 2, "egy": 3, "lev": 0}


def split_of(dialect, idx):
    g = (idx + OFFSET[dialect]) % 4
    return "train" if g in (0, 1) else ("val" if g == 2 else "test")


ITEM_EN = ["headphones", "phone case", "laptop", "blender", "sneakers", "smart watch", "air fryer", "backpack"]
ITEM_AR = {
    "msa": ["السماعة", "الهاتف", "الحاسوب المحمول", "الخلاط", "الحذاء الرياضي", "الساعة الذكية", "القلاية الهوائية", "الحقيبة"],
    "gulf": ["السماعة", "الجوال", "اللابتوب", "الخلاط", "الجزمة", "الساعة الذكية", "القلاية", "الشنطة"],
    "egy": ["السماعة", "الموبايل", "اللاب توب", "الخلاط", "الكوتشي", "الساعة الذكية", "القلاية الهوائية", "الشنطة"],
    "lev": ["السماعة", "الموبايل", "اللابتوب", "الخلاطة", "الكندرة", "الساعة الذكية", "القلاية", "الشنطة"],
}
EN_PRE = ["", "", "Hi, ", "Hello, ", "Hi team, ", "Urgent: "]
EN_SUF = ["", "", " Thanks.", " Please help.", " Thank you!"]
AR_PRE = ["", "", "السلام عليكم، ", "مرحبا، ", "مساء الخير، ", "عاجل: "]
AR_SUF = ["", "", " شكرا.", " ارجو المساعدة.", " جزاكم الله خير."]
AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def ar_noise(t):
    if random.random() < 0.35:
        t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    if random.random() < 0.25:
        t = t.replace("ة", "ه")
    if random.random() < 0.15:
        t = t.replace("ى", "ي")
    if random.random() < 0.20:
        t = re.sub(r"[،؟!.:]", "", t)
    return t


def en_noise(t):
    if random.random() < 0.25:
        t = t.lower()
    if random.random() < 0.20:
        words = t.split()
        i = random.randrange(len(words))
        w = words[i]
        if len(w) > 4 and not any(c.isdigit() for c in w):
            j = random.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            t = " ".join(words)
    if random.random() < 0.15:
        t = re.sub(r"[,?!.]", "", t)
    return t


def make_one(tpl, dialect):
    is_en = dialect == "en"
    num = random.randint(10000, 99999)
    if is_en:
        order = random.choice([f"#{num}", f"ORD-{num}", str(num)])
        days = str(random.choice([2, 3, 4, 5, 7, 10]))
        item = random.choice(ITEM_EN)
    else:
        order = random.choice([str(num), f"#{num}", str(num).translate(AR_DIGITS)])
        days = random.choice([2, 3, 4, 5, 7, 10])
        days = str(days).translate(AR_DIGITS) if random.random() < 0.3 else str(days)
        k = random.randrange(len(ITEM_EN))
        item = ITEM_EN[k] if random.random() < 0.10 else ITEM_AR[dialect][k]  # 10% code-switching
    core = tpl.format(order=order, days=days, item=item)
    pre, suf = (EN_PRE, EN_SUF) if is_en else (AR_PRE, AR_SUF)
    text = random.choice(pre) + core + random.choice(suf)
    return en_noise(text) if is_en else ar_noise(text)


rows = []
for intent, per_dialect in T.items():
    for dialect, tpls in per_dialect.items():
        for idx, tpl in enumerate(tpls):
            n = max(4, round(BASE_K * MULT[intent]))
            for _ in range(n):
                rows.append({
                    "text": make_one(tpl, dialect).strip(),
                    "intent": intent,
                    "lang": "en" if dialect == "en" else "ar",
                    "dialect": dialect,
                    "issue": f"{intent}:{idx}",
                    "split": split_of(dialect, idx),
                })

# --- out-of-scope tickets: used ONLY in validation, to choose the abstain threshold
OOS_VAL = [
    ("en", "Do you have any job openings at your company?"), ("en", "What's the weather going to be like tomorrow?"),
    ("en", "Can you recommend a good restaurant nearby?"), ("en", "How do I invest in the stock market?"),
    ("en", "I'd like to become a supplier for your store."), ("en", "Tell me a joke."),
    ("en", "What is the capital of France?"), ("en", "Can you help me write my CV?"),
    ("en", "Who won the match yesterday?"), ("en", "I want to advertise my business on your website."),
    ("en", "How do I learn Python?"), ("en", "Is it a good time to buy gold?"),
    ("ar", "هل عندكم وظائف شاغرة في الشركة؟"), ("ar", "كيف حالة الطقس غدا؟"),
    ("ar", "ممكن ترشحون لي مطعم كويس؟"), ("ar", "كيف استثمر في سوق الأسهم؟"),
    ("ar", "ابغى اصير مورد لمتجركم"), ("ar", "قل لي نكتة"),
    ("ar", "ما هي عاصمة فرنسا؟"), ("ar", "ساعدوني اكتب سيرتي الذاتية"),
    ("ar", "مين فاز بالمباراة امس؟"), ("ar", "ابغى اعلن عن مشروعي في موقعكم"),
    ("ar", "كيف اتعلم البرمجة؟"), ("ar", "هل الوقت مناسب لشراء الذهب؟"),
]
for lang, text in OOS_VAL:
    rows.append({"text": text, "intent": "out_of_scope", "lang": lang, "dialect": "misc",
                 "issue": "out_of_scope", "split": "val"})

# --- de-duplicate and remove cross-split leakage (identical text in two splits)
seen = {}
clean = []
leaks = 0
for split in ("train", "val", "test"):  # train wins
    for r in [r for r in rows if r["split"] == split]:
        key = r["text"]
        if key in seen:
            if seen[key] != split:
                leaks += 1
            continue
        seen[key] = split
        clean.append(r)
random.shuffle(clean)
for i, r in enumerate(clean):
    r["id"] = 10000 + i

path = os.path.join(OUT, "tickets.csv")
with open(path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["id", "text", "intent", "lang", "dialect", "issue", "split"])
    w.writeheader()
    w.writerows(clean)

# summary
from collections import Counter
print("rows:", len(clean), "| cross-split duplicates removed:", leaks)
for s in ("train", "val", "test"):
    sub = [r for r in clean if r["split"] == s]
    cores = len({(r["dialect"], r["issue"]) for r in sub})
    print(f"{s:5s} n={len(sub):4d}  distinct (dialect, template) cores={cores:3d}  "
          f"lang={dict(Counter(r['lang'] for r in sub))}")
print("intent counts (train):", dict(Counter(r["intent"] for r in clean if r["split"] == "train")))
print("saved ->", path)
