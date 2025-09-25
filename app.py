# app.py
import io, base64, json, re, secrets
from datetime import datetime

import streamlit as st
import yaml, pandas as pd, numpy as np, matplotlib.pyplot as plt
from fpdf import FPDF  # PDF simple (sans accents pour éviter les soucis Unicode)

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

# ====== PERSISTENCE (Google Sheets si dispo, sinon mémoire) ======
USE_SHEETS = False
try:
    from google.oauth2.service_account import Credentials
    import gspread
    # Secrets attendus :
    # st.secrets["gcp_service_account"] = {... service account json ...}
    # st.secrets["sheets"]["spreadsheet_id"] = "..."
    # st.secrets["sheets"]["worksheet"] = "reponses"
    svc_info = st.secrets.get("gcp_service_account", None)
    sheet_cfg = st.secrets.get("sheets", {})
    if svc_info and sheet_cfg.get("spreadsheet_id"):
        creds = Credentials.from_service_account_info(dict(svc_info), scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_cfg["spreadsheet_id"])
        ws_name = sheet_cfg.get("worksheet", "reponses")
        try:
            ws = sh.worksheet(ws_name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=ws_name, rows="1000", cols="12")
            ws.append_row(["code", "ts", "scores_dim_json", "total", "max_total", "level", "spiral", "hawkins", "dab", "responses_json"])
        USE_SHEETS = True
except Exception as _e:
    USE_SHEETS = False

if "memory_db" not in st.session_state:
    st.session_state["memory_db"] = {}  # {code: payload_json}

def save_result(code: str, payload: dict):
    """Payload keys: scores_dim, max_dim, total, max_total, level, spiral, hawkins, dab, responses"""
    record = {
        "code": code,
        "ts": datetime.utcnow().isoformat(),
        "scores_dim_json": json.dumps(payload["scores_dim"], ensure_ascii=False),
        "total": int(payload["total"]),
        "max_total": int(payload["max_total"]),
        "level": payload["level"],
        "spiral": payload["spiral"],
        "hawkins": payload["hawkins"],
        "dab": payload["dab"],
        "responses_json": json.dumps(payload["responses"], ensure_ascii=False),
    }
    if USE_SHEETS:
        ws.append_row([
            record["code"], record["ts"], record["scores_dim_json"], record["total"], record["max_total"],
            record["level"], record["spiral"], record["hawkins"], record["dab"], record["responses_json"]
        ])
    else:
        st.session_state["memory_db"][code] = record

def fetch_result(code: str):
    if USE_SHEETS:
        try:
            cells = ws.get_all_records()
            for row in cells:
                if str(row.get("code","")).strip().upper() == code.upper():
                    # Normalise pour retour homogène
                    return {
                        "code": row["code"],
                        "ts": row["ts"],
                        "scores_dim": json.loads(row["scores_dim_json"]),
                        "total": int(row["total"]),
                        "max_total": int(row["max_total"]),
                        "level": row["level"],
                        "spiral": row["spiral"],
                        "hawkins": row["hawkins"],
                        "dab": row["dab"],
                        "responses": json.loads(row["responses_json"])
                    }
        except Exception as e:
            st.error(f"Erreur lecture Google Sheets : {e}")
            return None
        return None
    else:
        rec = st.session_state["memory_db"].get(code)
        if not rec: return None
        return {
            "code": rec["code"],
            "ts": rec["ts"],
            "scores_dim": json.loads(rec["scores_dim_json"]),
            "total": rec["total"],
            "max_total": rec["max_total"],
            "level": rec["level"],
            "spiral": rec["spiral"],
            "hawkins": rec["hawkins"],
            "dab": rec["dab"],
            "responses": json.loads(rec["responses_json"])
        }

# ====== SMTP (optionnel) ======
def send_email_notification(code: str, total: int, level: str):
    """Envoi d'un mail si st.secrets['smtp'] est configuré."""
    cfg = st.secrets.get("smtp", None)
    if not cfg: return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        host, port = cfg.get("host"), int(cfg.get("port", 587))
        user, pwd = cfg.get("user"), cfg.get("password")
        to_addr = cfg.get("to", user)

        body = f"Un patient a terminé le questionnaire.\nCode: {code}\nScore global: {total}\nNiveau: {level}\nAccédez aux résultats dans l'onglet 'Accéder aux résultats' de l'app et saisissez le code."
        msg = MIMEText(body, _charset="utf-8")
        msg["Subject"] = f"[Conscience HPI] Nouveau résultat – Code {code}"
        msg["From"] = user
        msg["To"] = to_addr

        server = smtplib.SMTP(host, port)
        server.starttls()
        server.login(user, pwd)
        server.sendmail(user, [to_addr], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.warning(f"Notification email non envoyée ({e})")
        return False

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
    # Spirale
    if total < 150: spiral = "Beige / Violet"
    elif 150 <= total <= 200: spiral = "Rouge / Bleu"
    elif 200 < total <= 280: spiral = "Orange / Vert"
    else: spiral = "Jaune / Turquoise"
    # Hawkins
    if total < 150: hawkins = "<150 : en-dessous de Courage"
    elif 150 <= total <= 250: hawkins = "150–250 : Courage / Neutralité"
    elif 250 < total <= 350: hawkins = "250–350 : Volonté / Acceptation"
    else: hawkins = ">350 : Raison / Amour (et +)"
    # Dabrowski
    if total < 160: dab = "Niveau I – Intégration primaire"
    elif 160 <= total <= 200: dab = "Niveau II – Désintégration unilatérale"
    elif 200 < total <= 260: dab = "Niveau III – Désintégration spontanée multilatérale"
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
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(5.8, 5.8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1.0)
    ax.plot(angles, values, linewidth=2); ax.fill(angles, values, alpha=0.25)
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
    ax.set_ylim(0, max(maxv)); ax.set_ylabel("Score"); ax.set_title("Scores par dimension")
    st.pyplot(fig)

# ====== PDF (simple sans accents) ======
def build_pdf(responses, scores_dim, max_dim, total, max_total, level, spiral, hawkins, dab):
    def sanitize(txt: str) -> str:
        return txt.encode("latin-1", "ignore").decode("latin-1")
    pdf = FPDF(format="A4", unit="mm"); pdf.set_auto_page_break(auto=True, margin=12)
    def h1(t): pdf.set_font("Helvetica","B",16); pdf.cell(0,10,sanitize(t),ln=1)
    def h2(t): pdf.set_font("Helvetica","B",12); pdf.cell(0,8,sanitize(t),ln=1)
    def p(t):  pdf.set_font("Helvetica","",10);  pdf.multi_cell(0,6,sanitize(t))
    # p1
    pdf.add_page()
    h1("Questionnaire de Degre de Conscience - Profils HPI")
    p(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    p("Version : Longue (56 items, echelle 1-7)")
    p(f"Score global : {total}/{max_total}  |  Niveau global : {level}")
    p(f"Spirale Dynamique : {spiral}")
    p(f"Hawkins : {hawkins}")
    p(f"Dabrowski : {dab}")
    h2("Scores par dimension")
    for d in dimensions:
        code = d["code"]; lbl = d["label"]
        p(f"- {lbl} : {scores_dim[code]}/{max_dim[code]}")
    # p2
    pdf.add_page(); h1("Interpretation par dimension")
    for d in dimensions:
        code = d["code"]; lbl = d["label"]
        h2(lbl); p(interpret_dimension(code, scores_dim[code], max_dim[code]))
    # p3
    pdf.add_page(); h1("Reponses detaillees")
    for dim in dimensions:
        h2(dim["label"])
        for it in dim["items"]:
            rid = it["id"]; val = responses.get(rid, "")
            p(f"Q{rid}. {it['text']} -> {val}")
    buf = io.BytesIO(pdf.output(dest="S").encode("latin-1")); buf.seek(0); return buf

def download_button_pdf(buf, filename="rapport_conscience.pdf"):
    b64 = base64.b64encode(buf.read()).decode()
    st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📄 Télécharger le rapport PDF</a>', unsafe_allow_html=True)

# ====== UPLOAD CSV AIDE ======
@st.cache_data
def csv_template_bytes():
    rows = [{"id": i, "dimension": ITEMS[i]["dim_label"], "question": ITEMS[i]["text"], "reponse": ""} for i in ALL_ITEM_IDS]
    return pd.DataFrame(rows, columns=["id","dimension","question","reponse"]).to_csv(index=False).encode("utf-8")

def parse_uploaded_csv(file) -> dict:
    try:
        df = pd.read_csv(file)
    except Exception:
        file.seek(0); df = pd.read_csv(file, sep=";")
    df.columns = [c.lower() for c in df.columns]
    if "id" in df.columns and "reponse" in df.columns:
        pass
    elif {"id","réponse"}.issubset(df.columns):
        df.rename(columns={"réponse":"reponse"}, inplace=True)
    else:
        if not {"id","dimension","question","reponse"}.issubset(df.columns):
            raise ValueError("Colonnes attendues: au moins 'id' et 'reponse'.")
    df = df.dropna(subset=["id"]).copy(); df["id"] = df["id"].astype(int)
    def coerce(x):
        try:
            v = int(x); return v if LIKERT_MIN <= v <= LIKERT_MAX else None
        except: return None
    df["reponse"] = df["reponse"].apply(coerce)
    resp = {}
    for _, r in df.iterrows():
        i = int(r["id"]); v = r["reponse"]
        if i in ALL_ITEM_IDS and v is not None: resp[i] = v
    if len(resp)==0: raise ValueError("Aucune réponse valide (1..7) trouvée.")
    return resp

# ====== UTILES CODE / LIEN ======
def gen_code(n=6) -> str:
    raw = secrets.token_urlsafe(8).upper()
    raw = re.sub(r"[^A-Z0-9]", "", raw)
    return raw[:n]

def current_base_url_fallback():
    # Streamlit ne donne pas l'URL publique; on propose un champ pour la saisir si besoin
    return "https://VOTRE-URL-APP.streamlit.app"

def build_patient_url(base_url: str, code: str) -> str:
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}mode=patient&code={code}"

# ====== UI ======
st.title(APP_TITLE)

# Lire query params (mode/ code)
try:
    q = st.query_params  # Streamlit >=1.37
except Exception:
    q = st.experimental_get_query_params()
mode = (q.get("mode",[None])[0] if isinstance(q.get("mode"), list) else q.get("mode")) or ""
qp_code = (q.get("code",[None])[0] if isinstance(q.get("code"), list) else q.get("code")) or ""

tabs = st.tabs(["🔗 Lien patient & accès praticien", "📝 Passer le test", "📤 Téléverser des résultats"])

# ====== Onglet 1 : Lien & Accès ======
with tabs[0]:
    st.subheader("Créer un lien patient avec code")
    colA, colB = st.columns([1,1])
    with colA:
        base_url = st.text_input("URL publique de l’app (colle-la ici une fois déployée)", value=current_base_url_fallback())
        code = st.text_input("Code unique", value=gen_code(), help="Tu peux en régénérer un autre si besoin.")
        if st.button("Générer le lien patient"):
            url = build_patient_url(base_url.strip(), code.strip())
            st.success("Lien patient généré :")
            st.code(url, language="text")
            st.info("Envoie ce lien au patient. Il remplira le questionnaire ; tu pourras ensuite accéder aux résultats avec le même code (voir ci-dessous).")

    with colB:
        st.subheader("Accéder aux résultats (praticien)")
        code_lookup = st.text_input("Saisir le code patient pour retrouver les résultats")
        if st.button("Afficher les résultats"):
            if not code_lookup.strip():
                st.warning("Renseigne un code.")
            else:
                rec = fetch_result(code_lookup.strip())
                if not rec:
                    st.error("Aucun résultat trouvé pour ce code.")
                else:
                    st.success(f"Résultats trouvés (code {rec['code']}) – {rec['ts']}")
                    scores_dim = rec["scores_dim"]; total = rec["total"]; max_total = rec["max_total"]
                    level, spiral, hawkins, dab = rec["level"], rec["spiral"], rec["hawkins"], rec["dab"]
                    # Affichage
                    col1, col2 = st.columns([1,1])
                    with col1:
                        st.write(f"**Score global : {total}/{max_total}**")
                        st.write(f"Niveau global : **{level}**")
                        st.write(f"Spirale : {spiral}"); st.write(f"Hawkins : {hawkins}"); st.write(f"Dabrowski : {dab}")
                        df_scores = pd.DataFrame({
                            "Dimension": [DIM_LABELS[c] for c in scores_dim.keys()],
                            "Score": [scores_dim[c] for c in scores_dim.keys()],
                            "Max": [len(DIM_ITEMS[c])*LIKERT_MAX for c in scores_dim.keys()],
                        })
                        st.dataframe(df_scores, use_container_width=True)
                    with col2:
                        # Recalcule max_dim pour graphiques
                        max_dim = {c: len(DIM_ITEMS[c])*LIKERT_MAX for c in scores_dim.keys()}
                        # Radar & bars
                        def plot_radar_view():
                            labels = [DIM_LABELS[c] for c in scores_dim.keys()]
                            values = [scores_dim[c]/max_dim[c] for c in scores_dim.keys()]
                            values2 = values + values[:1]
                            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
                            fig = plt.figure(figsize=(5.8,5.8))
                            ax = plt.subplot(111, polar=True)
                            ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
                            ax.set_thetagrids(np.degrees(angles[:-1]), labels); ax.set_ylim(0,1.0)
                            ax.plot(angles, values2, linewidth=2); ax.fill(angles, values2, alpha=0.25)
                            ax.set_title("Profil radar (normalisé)"); st.pyplot(fig)
                        plot_radar_view()
                    # Exports
                    st.markdown("### Export")
                    responses = {int(k): int(v) for k,v in rec["responses"].items()}
                    df_resp = pd.DataFrame(
                        [{"id": i, "dimension": ITEMS[i]["dim_label"], "question": ITEMS[i]["text"], "reponse": responses.get(i,"")} for i in ALL_ITEM_IDS]
                    )
                    csv_bytes = df_resp.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Télécharger les réponses (CSV)", data=csv_bytes, file_name=f"reponses_{rec['code']}.csv", mime="text/csv")
                    # PDF
                    # Reconstruire max_dim pour PDF
                    max_dim = {c: len(DIM_ITEMS[c])*LIKERT_MAX for c in scores_dim.keys()}
                    pdf_buf = build_pdf(responses, scores_dim, max_dim, total, max_total, level, spiral, hawkins, dab)
                    download_button_pdf(pdf_buf, filename=f"rapport_{rec['code']}.pdf")

    st.divider()
    st.markdown("#### ⚙️ Comment ça marche ?")
    st.markdown("""
1) **Créez un code** et un **lien patient**.  
2) Le patient ouvre le lien, remplit le test.  
3) Les résultats sont **enregistrés** (Google Sheets si configuré, sinon mémoire).  
4) Dans *Accéder aux résultats*, saisissez le **code** pour voir le rapport et télécharger PDF/CSV.  
🔔 Optionnel : configurez un **SMTP** dans `st.secrets` pour recevoir un email automatique.
    """)

# ====== Onglet 2 : Passer le test (usage direct) ======
with tabs[1]:
    st.subheader("Passer le test (usage direct)")
    with st.form("form_direct"):
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
        submitted = st.form_submit_button("Calculer mes scores")
    if submitted:
        scores_dim, max_dim, total, max_total = compute_scores(responses)
        level = interpret_overall(total)
        spiral, hawkins, dab = map_spiral_hawkins_dabrowski(total)
        col1, col2 = st.columns([1,1])
        with col1:
            st.success(f"**Score global : {total}/{max_total}**")
            st.write(f"Niveau global : {level}")
            st.write(f"Spirale : {spiral}"); st.write(f"Hawkins : {hawkins}"); st.write(f"Dabrowski : {dab}")
            df_scores = pd.DataFrame({
                "Dimension": [d["label"] for d in dimensions],
                "Score": [scores_dim[d["code"]] for d in dimensions],
                "Max": [max_dim[d["code"]] for d in dimensions],
                "Ratio": [round(scores_dim[d["code"]]/max_dim[d["code"]], 3) for d in dimensions]
            }); st.dataframe(df_scores, use_container_width=True)
        with col2:
            plot_radar(scores_dim, max_dim); plot_bars(scores_dim, max_dim)
        # Export CSV
        df_resp = pd.DataFrame(
            [{"id": i, "dimension": ITEMS[i]["dim_label"], "question": ITEMS[i]["text"], "reponse": responses[i]} for i in ALL_ITEM_IDS]
        )
        st.download_button("⬇️ Télécharger les réponses (CSV)", data=df_resp.to_csv(index=False).encode("utf-8"), file_name="reponses_conscience.csv", mime="text/csv")
        # PDF
        pdf_buf = build_pdf(responses, scores_dim, max_dim, total, max_total, level, spiral, hawkins, dab)
        download_button_pdf(pdf_buf, filename="rapport_conscience.pdf")

# ====== Onglet 3 : Téléverser des résultats (CSV) ======
with tabs[2]:
    st.subheader("Téléverser un fichier de réponses (CSV)")
    st.download_button("📄 Télécharger le modèle CSV", data=csv_template_bytes(), file_name="modele_reponses_conscience.csv", mime="text/csv")
    uploaded = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
    if uploaded is not None:
        try:
            responses_up = parse_uploaded_csv(uploaded)
            # Compléter si partiel
            for i in ALL_ITEM_IDS: responses_up.setdefault(i, LIKERT_MIN)
            scores_dim, max_dim, total, max_total = compute_scores(responses_up)
            level = interpret_overall(total)
            spiral, hawkins, dab = map_spiral_hawkins_dabrowski(total)
            col1, col2 = st.columns([1,1])
            with col1:
                st.success(f"**Score global : {total}/{max_total}**"); st.write(f"Niveau global : {level}")
                st.write(f"Spirale : {spiral}"); st.write(f"Hawkins : {hawkins}"); st.write(f"Dabrowski : {dab}")
                df_scores = pd.DataFrame({
                    "Dimension": [d["label"] for d in dimensions],
                    "Score": [scores_dim[d["code"]] for d in dimensions],
                    "Max": [max_dim[d["code"]] for d in dimensions],
                    "Ratio": [round(scores_dim[d["code"]]/max_dim[d["code"]], 3) for d in dimensions]
                }); st.dataframe(df_scores, use_container_width=True)
            with col2:
                plot_radar(scores_dim, max_dim); plot_bars(scores_dim, max_dim)
            # Export CSV reformaté
            df_resp2 = pd.DataFrame(
                [{"id": i, "dimension": ITEMS[i]["dim_label"], "question": ITEMS[i]["text"], "reponse": responses_up[i]} for i in ALL_ITEM_IDS]
            )
            st.download_button("⬇️ Télécharger ces réponses (CSV)", data=df_resp2.to_csv(index=False).encode("utf-8"), file_name="reponses_conscience_import.csv", mime="text/csv")
            # PDF
            pdf_buf2 = build_pdf(responses_up, scores_dim, max_dim, total, max_total, level, spiral, hawkins, dab)
            download_button_pdf(pdf_buf2, filename="rapport_conscience_import.pdf")
        except Exception as e:
            st.error(f"Fichier non reconnu : {e}")

# ====== MODE PATIENT VIA LIEN ======
if mode.lower() == "patient":
    st.sidebar.markdown("### Mode patient")
    code_from_link = qp_code or st.sidebar.text_input("Code (fourni par le praticien)", value="")
    st.sidebar.info("Remplissez le questionnaire puis validez pour transmettre vos résultats.")
    if code_from_link:
        st.header("Questionnaire (mode patient)")
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
            # Calculs et sauvegarde
            scores_dim, max_dim, total, max_total = compute_scores(responses)
            level = interpret_overall(total)
            spiral, hawkins, dab = map_spiral_hawkins_dabrowski(total)
            payload = {
                "scores_dim": scores_dim, "max_dim": max_dim,
                "total": total, "max_total": max_total,
                "level": level, "spiral": spiral, "hawkins": hawkins, "dab": dab,
                "responses": responses
            }
            save_result(code_from_link.strip(), payload)
            # Email notif (optionnel)
            sent = send_email_notification(code_from_link.strip(), total, level)
            st.success("Merci ! Vos réponses ont été enregistrées et transmises.")
            if sent:
                st.info("✅ Une notification a été envoyée au praticien.")
            else:
                st.info("ℹ️ Le praticien peut consulter vos résultats avec le **code** fourni.")
