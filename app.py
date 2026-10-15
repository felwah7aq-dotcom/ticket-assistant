"""Gradio demo for the Ticket Assistant.

Run:  python app.py                 # auto: e5 if available, else offline fallback
      BACKEND=e5 python app.py      # force e5
"""
import html
import os

import gradio as gr
import pandas as pd

from core import ROOT, TicketAssistant, make_backend

EXAMPLES = [
    "Hello, my order 48213 hasn't arrived for 8 days and tracking has not moved.",
    "ابغى اعرف متى ينزل مبلغ الاسترجاع في حسابي، رجعت المنتج من اسبوعين",
    "انخصم مني المبلغ مرتين لنفس الطلب، ابغى استرجع الزيادة",
    "فين طلبي 52719؟ المفروض يوصل من 5 ايام",
    "The mixer arrived with a broken lid and the box was soaked. I need a replacement.",
    "ما يوصلني كود التحقق ابدا وحسابي مقفل",
    "Write me a poem about the sea.",        # out of scope -> should be sent to a human
    "المندوب كان وقح معي وابغى اقدم شكوى عليه",  # out of scope: a KNOWN failure case (see error analysis)
]


def build(backend="auto"):
    df = pd.read_csv(os.path.join(ROOT, "data", "tickets.csv"), encoding="utf-8-sig")
    asst = TicketAssistant(make_backend(backend)).fit(df[df.split == "train"], df[df.split == "val"])

    def triage(text):
        if not text or not text.strip():
            return "Type a ticket first.", {}, "", []
        o = asst.run(text.strip())
        detail, lat = o.pop("_similar_detail"), o.pop("_latency_ms")
        badge = ("AUTO-REPLY" if o["route"] == "auto" else "SENT TO A HUMAN (low confidence / unfamiliar)")
        status = (f"**{badge}** &nbsp;|&nbsp; top guess `{o['top_guess']}` &nbsp;|&nbsp; confidence {o['confidence']} "
                  f"&nbsp;|&nbsp; priority {o['priority']} &nbsp;|&nbsp; {lat} ms")
        reply = o["reply"] or "No automatic reply: the ticket is routed to a human agent together with the similar cases below."
        reply_html = (f'<div dir="auto" style="white-space:pre-wrap;padding:14px;border:1px solid #ccc;'
                      f'border-radius:8px;font-size:16px;line-height:1.7">{html.escape(reply)}</div>')
        rows = [[d["id"], d["intent"], d["similarity"], d["text"]] for d in detail]
        return status, o, reply_html, rows

    with gr.Blocks(title="Ticket Assistant") as demo:
        gr.Markdown(
            "# Ticket Assistant  ·  مساعد التذاكر\n"
            "Classify an Arabic/English support ticket, find similar past cases, and draft a reply grounded in the "
            "knowledge base, or abstain and hand it to a human.\n\n"
            f"`backend: {asst.backend.name}` · `confidence threshold {asst.tau}` · `similarity threshold {asst.sigma}`")
        inp = gr.Textbox(label="Ticket / التذكرة", lines=3, placeholder="Paste a ticket in English or Arabic…")
        btn = gr.Button("Triage", variant="primary")
        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=inp, label="Try these")
        status = gr.Markdown()
        with gr.Row():
            js = gr.JSON(label="Structured output (validated)")
            with gr.Column():
                gr.Markdown("**Draft reply (grounded in the KB, with citations)**")
                reply = gr.HTML()
        gr.Markdown("**Similar historical tickets**")
        sim = gr.Dataframe(headers=["ticket id", "intent", "similarity", "text"], wrap=True)
        for trigger in (btn.click, inp.submit):
            trigger(triage, inp, [status, js, reply, sim])
    return demo, triage


if __name__ == "__main__":
    demo, _ = build(os.environ.get("BACKEND", "auto"))
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7861)),
                share=bool(os.environ.get("SHARE")))
