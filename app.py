# app.py
import io
import base64
from datetime import datetime

import streamlit as st
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF  # PDF sans dépendances système

# -----------------------------
# Config de l’app
# -----------------------------
st.set_page_config(
    page_title="Questionnaire de Degré de Conscience – Profils HPI",
    page_icon="🧭",
    layout="wide"
)

# -----------------------------
# Chargement des questions
# -----------------------------
@st.cache_data
def load_questions():
    with open("questions.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

data = load_questions()
LIKERT_MIN = data["likert"]["min"]
LIKERT_MAX = data["likert"]["max"]
LIKERT_LABELS = data["likert"]["labels"]
dimensions = data["dimensions"]

# Index id -> meta
ITEMS = {}
for dim in dimensions:
    for it in dim["items"]:
        ITEMS[it["id"]] = {
            "dim_code": dim["code"],
            "dim_label": dim["label"],
            "text": it["text"],
            "sub": it.get("sub", "")
        }

ALL_ITEM_IDS = [it["id"] for d in dimensions for it in d["items"]]
TOTAL_ITEMS = len(ALL_ITEM_IDS)

# -----------------------------
# Scoring & interprétation
# -----------------------------
def compute_scores(responses: dict):
    scores_dim, max_dim = {}, {}
    for dim in dimensions:
        code = dim["code"]
        ids = [it["id"] for it in dim["items"]]
        s = sum(responses.get(i, 0) for i in ids)
        scores_dim[code] = s
        max_dim[code] = len(ids) * LIKERT_MAX
    total = sum(scores_dim.values())
    max_total = sum(max_dim.values())
    return scores_dim, max_dim, total, max_total

def interpret_overall(total):
    # Version longue : 56 items, échelle 1–7 -> min 56, max 392
    if total <= 140:
        return "Conscience émergente"
    elif total <= 224:
        return "Conscience développée"
    elif total <= 308:
        return "Conscience intégrée"
    else:
        return "Conscience transcendante"

def map_spiral_hawkins_dabrowski(total):
    # Spirale Dynamique
    if total < 150:
        spiral = "Beige / Violet"
    elif 150 <= total <= 200:
        spiral = "Rouge / Bleu"
    elif 200 < total <= 280:
        spiral = "Orange / Vert"
    else:
        spiral = "Jaune / Turquoise"
    # Hawkins
    if total < 150:
        hawkins = "< 150 : en-dessous de Courage (Honte–Peur–Colère variables)"
    elif 150 <= total <= 250:
        hawkins = "150–250 : Courage / Neutralité"
    elif 250 < total <= 350:
        hawkins = "250–350 : Volonté / Acceptation"
    else:
        hawkins = "> 350 : Raison / Amour (et au-delà)"
    # Dabrowski
    if total < 160:
        dab = "Niveau I – Intégration primaire (conformité, peu de conflits internes)"
    elif 160 <= total <= 200:
        dab = "Niveau II – Désintégration unilatérale (ambition/compétition, tensions)"
    elif 200 < total <= 260:
        dab = "Niveau III – Désintégration spontanée multilatérale (conflits, quête de sens)"
    elif 260 < total <= 320:
        dab = "Niveau IV – Désintégration organisée (valeurs personnelles, alignement)"
    else:
        dab = "Niveau V – Intégration secondaire (autonomie, altruisme authentique)"
    return spiral, hawkins, dab

def interpret_dimension(code, score, max_score):
    ratio = score / max_score
    if ratio < 0.45:
        band = "faible"
    elif ratio < 0.65:
        band = "modérée"
    elif ratio < 0.85:
        band = "élevée"
    else:
        band = "très élevée"
    if code == "SOI":
        msg = f"Conscience de soi {band} : introspection, conscience émotionnelle et alignement intérieur."
    elif code == "SOC":
        msg = f"Conscience sociale {band} : empathie, lecture des dynamiques et sens de la justice."
    elif code == "SYS":
        msg = f"Conscience élargie {band} : pensée systémique, vision globale, sens du lien et du sens."
    elif code == "HPI":
        msg = f"Rapport au monde HPI {band} : intensité, hypersensibilité, créativité, décalage et mission."
    else:
        msg = f"Dimension {code} {band}."
    return msg

# -----------------------------
# Graphiques
# -----------------------------
def plot_radar(scores_dim, max_dim):
    labels = [d["label"] for d in dimensions]
    codes = [d["code"] for d in dimensions]
    values = [scores_dim[c] / max_dim[c] for c in codes]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(5.8, 5.8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1.0)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_title("Profil radar (normalisé)")
    st.pyplot(fig)

def plot_bars(scores_dim, max_dim):
    labels = [d["label"] for d in dimensions]
    codes = [d["code"] for d in dimensions]
    values = [scores_dim[c] for c in codes]
    maxv = [max_dim[c] for c in codes]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylim(0, max(maxv))
    ax.set_ylabel("Score")
    ax.set_title("Scores par dimension")
    st.pyplot(fig)

# -----------------------------
# PDF (fpdf2)
# -----------------------------
def build_pdf(responses, scores_dim, max_dim, total, max_total, level, spiral, hawkins, dab):
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=12)

    def h1(txt):
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, txt, ln=1)

    def h2(txt):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, txt, ln=1)

    def p(txt):
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, txt)

    # Page 1 — Synthèse
    pdf.add_page()
    h1("Questionnaire de Degré de Conscience – Profils HPI")
    p(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    p("Version : Longue (56 items, échelle 1–7)")
    pdf.ln(1)
    p(f"Score global : {total}/{max_total}  |  Niveau global : {level}")
    p(f"Spirale Dynamique : {spiral}")
    p(f"Hawkins : {hawkins}")
    p(f"Dabrowski : {dab}")
    pdf.ln(2)
    h2("Scores par dimension")
    for d in dimensions:
        code = d["code"]; lbl = d["label"]
        p(f"- {lbl} : {scores_dim[code]}/{max_dim[code]}")

    # Page 2 — Interprétation par dimension
    pdf.add_page()
    h1("Interprétation par dimension")
    for d in dimensions:
        code = d["code"]; lbl = d["label"]
        h2(lbl)
        p(interpret_dimension(code, scores_dim[code], max_dim[code]))
        pdf.ln(1)

    # Page 3+ — Réponses détaillées
    pdf.add_page()
    h1("Réponses détaillées")
    for dim in dimensions:
        h2(dim["label"])
        for it in dim["items"]:
            rid = it["id"]; val = responses.get(rid, "")
            q = f"Q{rid}. {it['text']}  →  {val}"
            p(q)

    buf = io.BytesIO(pdf.output(dest="S").encode("latin-1"))
    buf.seek(0)
    return buf

def download_button_pdf(buf, filename="rapport_conscience.pdf"):
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📄 Télécharger le rapport PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

# -----------------------------
# UI
# -----------------------------
st.title("🧭 Questionnaire de Degré de Conscience – Profils HPI (version longue)")

with st.expander("ℹ️ À propos", expanded=True):
    st.markdown("""
Ce questionnaire explore **4 dimensions** :  
- **Conscience de soi**, **Conscience sociale**, **Conscience élargie**, **Rapport au monde HPI**.  
Interprétation croisée : **Spirale Dynamique**, **Hawkins**, **Dabrowski**.  
**Échelle** : 1 (Totalement en désaccord) → 7 (Totalement d'accord).  
> Outil d’exploration, non diagnostique.
""")

with st.form("form"):
    st.subheader("Vos réponses")
    responses = {}
    for dim in dimensions:
        st.markdown(f"### {dim['label']}")
        for it in dim["items"]:
            key = f"q_{it['id']}"
            val = st.slider(
                label=f"Q{it['id']}. {it['text']}",
                min_value=LIKERT_MIN, max_value=LIKERT_MAX,
                value=int((LIKERT_MIN+LIKERT_MAX)//2),
                help=it.get("sub",""),
                key=key
            )
            responses[it["id"]] = val
    submitted = st.form_submit_button("Calculer mes scores")

if submitted:
    scores_dim, max_dim, total, max_total = compute_scores(responses)
    level = interpret_overall(total)
    spiral, hawkins, dab = map_spiral_hawkins_dabrowski(total)

    col1, col2 = st.columns([1,1])
    with col1:
        st.success(f"**Score global : {total}/{max_total}**")
        st.write(f"**Niveau global** : {level}")
        st.write(f"**Spirale Dynamique** : {spiral}")
        st.write(f"**Hawkins** : {hawkins}")
        st.write(f"**Dabrowski** : {dab}")

        df_scores = pd.DataFrame({
            "Dimension": [d["label"] for d in dimensions],
            "Score": [scores_dim[d["code"]] for d in dimensions],
            "Max": [max_dim[d["code"]] for d in dimensions],
            "Ratio": [round(scores_dim[d["code"]]/max_dim[d["code"]], 3) for d in dimensions]
        })
        st.dataframe(df_scores, use_container_width=True)

        st.markdown("#### Interprétations par dimension")
        for d in dimensions:
            code = d["code"]
            st.write(f"- **{d['label']}** : {interpret_dimension(code, scores_dim[code], max_dim[code])}")

    with col2:
        plot_radar(scores_dim, max_dim)
        plot_bars(scores_dim, max_dim)

    # Exports
    st.markdown("### Export des résultats")
    df_resp = pd.DataFrame(
        [{"id": i, "dimension": ITEMS[i]["dim_label"], "question": ITEMS[i]["text"], "reponse": responses[i]} for i in ALL_ITEM_IDS]
    )
    csv_bytes = df_resp.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger les réponses (CSV)", data=csv_bytes, file_name="reponses_conscience.csv", mime="text/csv")

    pdf_buf = build_pdf(responses, scores_dim, max_dim, total, max_total, level, spiral, hawkins, dab)
    download_button_pdf(pdf_buf, filename="rapport_conscience.pdf")

else:
    st.info("Répondez aux items puis cliquez sur **Calculer mes scores**.")
