# app.py
import io
import base64
from datetime import datetime

import streamlit as st
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import simpleSplit

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

# Construit un index rapide id->(dim_code, label, text, sub)
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
# Fonctions de scoring
# -----------------------------
def compute_scores(responses: dict):
    """
    responses: {id:int -> score:int}
    Retourne:
      - scores_dim: dict dim_code -> sum
      - max_dim: dict dim_code -> max théorique
      - total, max_total
    """
    scores_dim = {}
    max_dim = {}
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
    """
    Seuils globaux (version longue 56 items, échelle 1–7)
    Min = 56, Max = 392
    """
    if total <= 140:
        level = "Conscience émergente"
    elif total <= 224:
        level = "Conscience développée"
    elif total <= 308:
        level = "Conscience intégrée"
    else:
        level = "Conscience transcendante"
    return level

def map_spiral_hawkins_dabrowski(total):
    # Vos correspondances proposées
    # Spirale
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
        dab = "Niveau II – Désintégration unilatérale (ambition, compétition, premières tensions)"
    elif 200 < total <= 260:
        dab = "Niveau III – Désintégration spontanée multilatérale (conflits, quête de sens)"
    elif 260 < total <= 320:
        dab = "Niveau IV – Désintégration organisée (hiérarchie de valeurs personnelle, alignement)"
    else:
        dab = "Niveau V – Intégration secondaire (personnalité autonome, altruisme authentique)"
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
        msg = f"Conscience sociale {band} : empathie, lecture fine des dynamiques et sens de la justice."
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
    values = [scores_dim[c] / max_dim[c] for c in codes]  # normalisé 0–1
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
# PDF report
# -----------------------------
def build_pdf(responses, scores_dim, max_dim, total, max_total, level, spiral, hawkins, dab):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    W, H = A4

    def draw_title(txt, y):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2*cm, y, txt)

    def draw_text(txt, x, y, size=10, max_width=17*cm, leading=14):
        c.setFont("Helvetica", size)
        wrapped = simpleSplit(txt, "Helvetica", size, max_width)
        for line in wrapped:
            c.drawString(x, y, line)
            y -= leading
        return y

    # Page 1: en-tête
    draw_title("Questionnaire de Degré de Conscience – Profils HPI", H-2.5*cm)
    y = H-3.5*cm
    y = draw_text(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}", 2*cm, y)
    y = draw_text("Version : Longue (56 items, échelle 1–7)", 2*cm, y)
    y -= 0.5*cm
    y = draw_text(f"Score global : {total}/{max_total}  |  Niveau global : {level}", 2*cm, y)
    y = draw_text(f"Spirale Dynamique : {spiral}", 2*cm, y)
    y = draw_text(f"Hawkins : {hawkins}", 2*cm, y)
    y = draw_text(f"Dabrowski : {dab}", 2*cm, y)

    y -= 0.8*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Scores par dimension")
    y -= 0.6*cm
    c.setFont("Helvetica", 10)
    for d in dimensions:
        code = d["code"]
        lbl = d["label"]
        c.drawString(2*cm, y, f"- {lbl} : {scores_dim[code]}/{max_dim[code]}")
        y -= 0.45*cm

    # Page 2: interprétation dimensionnelle
    c.showPage()
    draw_title("Interprétation par dimension", H-2.5*cm)
    y = H-3.5*cm
    for d in dimensions:
        code = d["code"]
        lbl = d["label"]
        msg = interpret_dimension(code, scores_dim[code], max_dim[code])
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, f"{lbl}")
        y -= 0.5*cm
        y = draw_text(msg, 2*cm, y, size=10)
        y -= 0.5*cm

    # Page 3: récapitulatif des réponses (optionnel)
    c.showPage()
    draw_title("Réponses détaillées (réduction pour synthèse)", H-2.5*cm)
    y = H-3.5*cm
    c.setFont("Helvetica", 9)
    for dim in dimensions:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2*cm, y, dim["label"])
        y -= 0.45*cm
        c.setFont("Helvetica", 9)
        for it in dim["items"]:
            rid = it["id"]
            txt = it["text"]
            val = responses.get(rid, "")
            line = f"Q{rid}. {txt}  -> {val}"
            y = draw_text(line, 2*cm, y, size=9, leading=12)
            y -= 0.2*cm
            if y < 3*cm:
                c.showPage()
                y = H-2.5*cm

    c.save()
    buffer.seek(0)
    return buffer

def download_button_pdf(buf, filename="rapport_conscience.pdf"):
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📄 Télécharger le rapport PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

# -----------------------------
# UI
# -----------------------------
st.title("🧭 Questionnaire de Degré de Conscience – Profils HPI (version longue)")

with st.expander("ℹ️ À propos de ce questionnaire", expanded=True):
    st.markdown("""
Ce questionnaire explore **4 dimensions** :  
- **Conscience de soi**, **Conscience sociale**, **Conscience élargie**, et **Rapport au monde HPI**.  
Il intègre les grilles **Spirale Dynamique**, **Hawkins** et **Dabrowski** pour une interprétation **multidimensionnelle**.  
**Échelle de réponse** : 1 (Totalement en désaccord) → 7 (Totalement d'accord).  
> Outil d’exploration, non diagnostique.
""")

# Formulaire
with st.form("form"):
    st.subheader("Vos réponses")
    responses = {}
    for dim in dimensions:
        st.markdown(f"### {dim['label']}")
        cols = st.columns(1)
        for it in dim["items"]:
            key = f"q_{it['id']}"
            val = st.slider(
                label=f"Q{it['id']}. {it['text']}",
                min_value=LIKERT_MIN, max_value=LIKERT_MAX, value=int((LIKERT_MIN+LIKERT_MAX)//2),
                help=it.get("sub",""),
                key=key
            )
            responses[it["id"]] = val

    submitted = st.form_submit_button("Calculer mes scores")

if submitted:
    # Calculs
    scores_dim, max_dim, total, max_total = compute_scores(responses)
    level = interpret_overall(total)
    spiral, hawkins, dab = map_spiral_hawkins_dabrowski(total)

    # Résultats (gauche) + Graphiques (droite)
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

    # CSV (réponses brutes)
    df_resp = pd.DataFrame(
        [{"id": i, "dimension": ITEMS[i]["dim_label"], "question": ITEMS[i]["text"], "reponse": responses[i]} for i in ALL_ITEM_IDS]
    )
    csv_bytes = df_resp.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger les réponses (CSV)", data=csv_bytes, file_name="reponses_conscience.csv", mime="text/csv")

    # PDF (rapport)
    pdf_buf = build_pdf(responses, scores_dim, max_dim, total, max_total, level, spiral, hawkins, dab)
    download_button_pdf(pdf_buf, filename="rapport_conscience.pdf")

    st.markdown("> 🔒 **Confidentialité** : vos réponses ne sont ni stockées ni partagées par l’application, sauf si vous exportez les fichiers vous-même.")
else:
    st.info("Répondez aux items puis cliquez sur **Calculer mes scores**.")

