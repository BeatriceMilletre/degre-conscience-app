# app.py
import io, base64, json, re, secrets
from datetime import datetime

import streamlit as st
import yaml, pandas as pd, numpy as np, matplotlib.pyplot as plt
from fpdf import FPDF  # PDF simple (sans accents pour éviter bug Unicode)

# ====== CONFIG APP ======
st.set_page_config(page_title="Questionnaire Degré de Conscience – HPI", page_icon="🧭", layout="wide")
APP_TITLE = "🧭 Questionnaire de Degré de Conscience – Profils HPI"

# ====== CHARGER QUESTIONS ======
@st.cache_data
def load_questions():
    with open("questions.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

data = load_questions()
LIKERT_MIN = data["likert"]["min"]
LIKERT_MAX = data["likert"]["max"]
dimensions = data["dimensions"]

# Index question -> meta
ITEMS = {it["id"]: {"dim_code": d["code"], "dim_label": d["label"], "text": it["text"], "sub": it.get("sub","")}
         for d in dimensions for it in d["items"]}
ALL_ITEM_IDS = [it["id"] for d in dimensions for it in d["items"]]
DIM_LABELS = {d["code"]: d["label"] for d in dimensions}
DIM_ITEMS = {d["code"]: [it["id"] for it in d["items"]] for d in dimensions}

# ====== ENREGISTREMENT SIMPLIFIÉ (mémoire seulement) ======
if "memory_db" not in st.session_state:
    st.session_state["memory_db"] = {}  # {code: payload}

def save_result(code: str, payload: dict):
    st.session_state["memory_db"][code] = payload

def fetch_result(code: str):
    return st.session_state["memory_db"].get(code)

# ====== SCORING ======
def compute_scores(responses: dict):
    scores_dim, max_dim = {}, {}
    for d in dimensions:
        code = d["code"]
        ids = [it["id"] for it in d["items"]]
        s = sum(int(responses.get(i, 0)) for i in ids)
        scores_dim[code] = s
        max_dim[code] = len(ids) * LIKERT_MAX
    total = sum(scores_dim.values())
    max_total = sum(max_dim.values())
    return scores_dim, max_dim, total, max_total

def interpret_overall(total):
    if total <= 140: return "Conscience émergente"
    elif total <= 224: return "Conscience développée"
    elif total <= 308: return "Conscience intégrée"
    else: return "Conscience transcendante"

def map_spiral_hawkins_dabrowski(total):
    if total < 150: spiral = "Beige / Violet"
    elif 150 <= total <= 200: spiral = "Rouge / Bleu"
    elif 200 < total <= 280: spiral = "Orange / Vert"
    else: spiral = "Jaune / Turquoise"

    if total < 150: hawkins = "<150 : en-dessous de Courage"
    elif 150 <= total <= 250: hawkins = "150–250 : Courage / Neutralité"
    elif 250 < total <= 350: hawkins = "250–350 : Volonté / Acceptation"
    else: hawkins = ">350 : Raison / Amour"

    if total < 160: dab = "Niveau I – Intégration primaire"
    elif 160 <= total <= 200: dab = "Niveau II – Désintégration unilatérale"
    elif 200 < total <= 260: dab = "Niveau III – Désintégration multilatérale"
    elif 260 < total <= 320: dab = "Niveau IV – Désintégration organisée"
    else: dab = "Niveau V – Intégration secondaire"
    return spiral, hawkins, dab

def interpret_dimension(code, score, max_score):
    r = score / max_score
    band = "faible" if r < 0.45 else ("modérée" if r < 0.65 else ("élevée" if r < 0.85 else "très élevée"))
    return f"{DIM_LABELS.get(code, code)} {band}"

# ====== GRAPHIQUES ======
def plot_radar(scores_dim, max_dim):
    labels = [d["label"] for d in dimensions]
    codes = [d["code"] for d in dimensions]
    values = [scores_dim[c] / max_dim[c] for c in codes]
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
    fig = plt.figure(figsize=(5.8, 5.8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,1.0)
    ax.plot(angles, values, linewidth=2); ax.fill(angles, values, alpha=0.25)
    st.pyplot(fig)

def plot_bars(scores_dim, max_dim):
    labels = [d["label"] for d in dimensions]
    codes = [d["code"] for d in dimensions]
    values = [scores_dim[c] for c in codes]
    maxv = [max_dim[c] for c in codes]
    fig, ax = plt.subplots(figsize=(6.5,3.8))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylim(0,max(maxv)); ax.set_ylabel("Score"); ax.set_title("Scores par dimension")
    st.pyplot(fig)

# ====== PDF (simple sans accents) ======
def build_pdf(responses, scores_dim, max_dim, total, max_total, level, spiral, hawkins, dab):
    def sanitize(txt): return txt.encode("latin-1", "ignore").decode("latin-1")
    pdf = FPDF(format="A4", unit="mm"); pdf.set_auto_page_break(auto=True, margin=12)
    def h1(t): pdf.set_font("Helvetica","B",16); pdf.cell(0,10,sanitize(t),ln=1)
    def h2(t): pdf.set_font("Helvetica","B",12); pdf.cell(0,8,sanitize(t),ln=1)
    def p(t): pdf.set_font("Helvetica","",10); pdf.multi_cell(0,6,sanitize(t))
    pdf.add_page(); h1("Questionnaire de Degre de Conscience - Profils HPI")
    p(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}"); p(f"Score global : {total}/{max_total} | {level}")
    p(f"Spirale : {spiral}"); p(f"Hawkins : {hawkins}"); p(f"Dabrowski : {dab}")
    for d in dimensions: p(f"- {d['label']} : {scores_dim[d['code']]}/{max_dim[d['code']]}")
    buf = io.BytesIO(pdf.output(dest="S").encode("latin-1")); buf.seek(0); return buf

def download_button_pdf(buf, filename="rapport.pdf"):
    b64 = base64.b64encode(buf.read()).decode()
    st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📄 Télécharger le PDF</a>', unsafe_allow_html=True)

# ====== PARAMÈTRES URL ======
params = st.experimental_get_query_params()
mode = (params.get("mode") or [""])[0].lower()
qp_code = (params.get("code") or [""])[0]

# ====== UI ======
st.title(APP_TITLE)

tabs = st.tabs(["📝 Passer le test", "📤 Accès praticien"])

# ---- Onglet 1 : Passer le test ----
with tabs[0]:
    st.subheader("Questionnaire complet")
    with st.form("form"):
        responses = {}
        for d in dimensions:
            st.markdown(f"### {d['label']}")
            for it in d["items"]:
                responses[it["id"]] = st.slider(
                    f"Q{it['id']}. {it['text']}",
                    min_value=LIKERT_MIN, max_value=LIKERT_MAX,
                    value=int((LIKERT_MIN+LIKERT_MAX)//2),
                    help=it.get("sub","")
                )
        submitted = st.form_submit_button("Envoyer")
    if submitted:
        scores_dim, max_dim, total, max_total = compute_scores(responses)
        level = interpret_overall(total)
        spiral, hawkins, dab = map_spiral_hawkins_dabrowski(total)
        code = secrets.token_hex(3).upper()
        save_result(code, {"responses": responses, "scores_dim": scores_dim,
                           "total": total, "max_total": max_total,
                           "level": level, "spiral": spiral, "hawkins": hawkins, "dab": dab})
        st.success(f"Résultats enregistrés avec le code **{code}**")
        st.info("Communiquez ce code au praticien pour qu’il consulte vos résultats.")
        pdf_buf = build_pdf(responses, scores_dim, max_dim, total, max_total, level, spiral, hawkins, dab)
        download_button_pdf(pdf_buf, filename=f"rapport_{code}.pdf")

# ---- Onglet 2 : Accès praticien ----
with tabs[1]:
    st.subheader("Consulter les résultats d’un patient")
    code_lookup = st.text_input("Code patient")
    if st.button("Afficher"):
        rec = fetch_result(code_lookup.strip().upper())
        if not rec:
            st.error("Aucun résultat trouvé.")
        else:
            st.success(f"Résultats pour code {code_lookup.strip().upper()}")
            st.write(f"Score global : {rec['total']}/{rec['max_total']} – {rec['level']}")
            st.write(f"Spirale : {rec['spiral']} | Hawkins : {rec['hawkins']} | Dabrowski : {rec['dab']}")
            plot_radar(rec["scores_dim"], {c: len(DIM_ITEMS[c])*LIKERT_MAX for c in rec["scores_dim"]})
            pdf_buf = build_pdf(rec["responses"], rec["scores_dim"],
                                {c: len(DIM_ITEMS[c])*LIKERT_MAX for c in rec["scores_dim"]},
                                rec["total"], rec["max_total"], rec["level"],
                                rec["spiral"], rec["hawkins"], rec["dab"])
            download_button_pdf(pdf_buf, filename=f"rapport_{code_lookup}.pdf")

# ---- Mode patient direct via URL ----
if mode == "patient":
    st.subheader("Mode patient (via lien)")
    code_from_link = qp_code or ""
    if not code_from_link:
        st.error("Code manquant dans l’URL.")
    else:
        with st.form("form_patient"):
            responses = {}
            for d in dimensions:
                st.markdown(f"### {d['label']}")
                for it in d["items"]:
                    responses[it["id"]] = st.slider(
                        f"Q{it['id']}. {it['text']}",
                        min_value=LIKERT_MIN, max_value=LIKERT_MAX,
                        value=int((LIKERT_MIN+LIKERT_MAX)//2),
                        help=it.get("sub","")
                    )
            submitted = st.form_submit_button("Envoyer au praticien")
        if submitted:
            scores_dim, max_dim, total, max_total = compute_scores(responses)
            level = interpret_overall(total)
            spiral, hawkins, dab = map_spiral_hawkins_dabrowski(total)
            save_result(code_from_link.upper(), {"responses": responses, "scores_dim": scores_dim,
                                                 "total": total, "max_total": max_total,
                                                 "level": level, "spiral": spiral, "hawkins": hawkins, "dab": dab})
            st.success("Vos réponses ont été enregistrées. Merci !")
