from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DEFAULT_DATA = BASE / "data" / "bank_dataset.csv"
EXPORTS = BASE / "exports"
MODELS_DIR = BASE / "models"
EXPORTS.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
HISTORY_FILE = EXPORTS / "historique_analyses.csv"

FEATURES = [
    "credit_score", "country", "gender", "age", "tenure", "balance",
    "products_number", "credit_card", "active_member", "estimated_salary",
]
NUM = [
    "credit_score", "age", "tenure", "balance", "products_number",
    "credit_card", "active_member", "estimated_salary",
]
CAT = ["country", "gender"]
TARGET = "churn"
MODEL_NAMES = ["Régression logistique", "Random Forest", "Gradient Boosting", "Extra Trees"]

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
st.markdown(
    """
<style>
:root{--navy:#071b33;--blue:#0f4c75;--cyan:#2e9cca;--soft:#f4f8fc;--line:#dce6f0}
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1500px}
.hero{padding:1.5rem 1.7rem;border-radius:22px;background:linear-gradient(120deg,#071b33,#0f4c75 65%,#2e9cca);color:white;margin-bottom:1rem;box-shadow:0 12px 30px rgba(7,27,51,.18)}
.hero-grid{display:flex;align-items:center;gap:1rem}.logo-shield{font-size:3rem}.hero h1{margin:0}.hero p{margin:.25rem 0 0;opacity:.9}
.kpi-card{padding:1rem;border:1px solid var(--line);border-radius:16px;background:white;min-height:112px}
.kpi-label{font-size:.85rem;color:#64748b}.kpi-value{font-size:1.7rem;font-weight:750;color:#071b33;margin:.25rem 0}.kpi-note{font-size:.78rem;color:#64748b}
.section-note{padding:.85rem 1rem;border-radius:12px;background:#eef6fb;border-left:4px solid #0f4c75}
div[data-testid="stSidebar"]{background:#071b33}div[data-testid="stSidebar"] *{color:white}.stButton>button{border-radius:11px;font-weight:650}.stDownloadButton>button{border-radius:11px}
[data-testid="stMetric"]{background:white;border:1px solid var(--line);padding:.8rem;border-radius:14px}
</style>
""",
    unsafe_allow_html=True,
)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DATA)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = normalise_columns(df)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(sorted(missing)))
    df = df.drop_duplicates().copy()
    for c in NUM + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=FEATURES + [TARGET])


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUM),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT),
    ])


def model_catalog() -> dict:
    return {
        "Régression logistique": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400, max_depth=14, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }


def train_all(df: pd.DataFrame, progress=None):
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    results = {}
    catalog = model_catalog()
    for idx, (name, algo) in enumerate(catalog.items(), start=1):
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", algo)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.50).astype(int)
        results[name] = {
            "pipe": pipe,
            "proba": proba,
            "pred": pred,
            "metrics": {
                "Accuracy": accuracy_score(y_test, pred),
                "Précision": precision_score(y_test, pred, zero_division=0),
                "Rappel": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, proba),
            },
        }
        if progress is not None:
            progress.progress(idx / len(catalog), text=f"Entraînement : {name}")
    best = max(results, key=lambda n: results[n]["metrics"]["ROC-AUC"])
    return results, best, X_test, y_test


def metrics_frame(models: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Modèle": name, **obj["metrics"]} for name, obj in models.items()])


def fit_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]
    scaler = StandardScaler()
    xs = scaler.fit_transform(df[cols])
    iso = IsolationForest(contamination=0.05, n_estimators=300, random_state=42).fit(xs)
    raw = -iso.score_samples(xs)
    out = df.copy()
    span = raw.max() - raw.min()
    out["score_anomalie"] = 100 * (raw - raw.min()) / (span if span else 1)
    out["alerte_anomalie"] = (iso.predict(xs) == -1).astype(int)
    return out


def load_history() -> pd.DataFrame:
    if HISTORY_FILE.exists():
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            pass
    return pd.DataFrame()


def save_history(row: dict) -> None:
    hist = load_history()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()


def pdf_summary(title: str, lines: list[str]) -> bytes:
    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(48, height - 58, title)
    c.setFont("Helvetica", 10)
    c.drawString(48, height - 78, datetime.now().strftime("Généré le %d/%m/%Y à %H:%M"))
    y = height - 115
    for line in lines:
        if y < 70:
            c.showPage(); y = height - 60
        c.drawString(55, y, str(line)[:110])
        y -= 18
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(48, 45, "FraudShield AI — outil pédagogique d'aide à la décision. Validation humaine requise.")
    c.save()
    return buff.getvalue()


def render_kpi(label: str, value: str, note: str):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


if "training_version" not in st.session_state:
    st.session_state.training_version = 0
if "models_bundle" not in st.session_state:
    st.session_state.models_bundle = None

st.sidebar.markdown("## 🛡️ FraudShield AI")
st.sidebar.caption("Analyse intelligente du risque bancaire")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Tableau de bord", "🧠 Entraîner les modèles", "🧪 Comparateur interactif",
        "⚡ Simulateur IA", "🕘 Historique", "📂 Données & corrélation",
        "📉 Évaluation", "🚨 Anomalies", "📦 Scoring par lot", "📑 Centre d’exports",
    ],
)
upload = st.sidebar.file_uploader("Importer un CSV compatible", type=["csv"])

try:
    raw_df = pd.read_csv(upload) if upload else load_default_data()
    df = clean(raw_df)
except Exception as exc:
    st.error(str(exc)); st.stop()

if st.session_state.models_bundle is None:
    with st.spinner("Initialisation des quatre modèles..."):
        st.session_state.models_bundle = train_all(df)
models, best_name, X_test, y_test = st.session_state.models_bundle
best = models[best_name]
scores = metrics_frame(models)
df_an = fit_anomaly(df)

st.markdown(
    '<div class="hero"><div class="hero-grid"><div class="logo-shield">🛡️</div><div><h1>FraudShield AI</h1><p>Plateforme professionnelle de comparaison, simulation et surveillance du risque bancaire par intelligence artificielle</p></div></div></div>',
    unsafe_allow_html=True,
)

if page == "🏠 Tableau de bord":
    cols = st.columns(5)
    with cols[0]: render_kpi("Clients analysés", f"{len(df):,}".replace(",", " "), "Portefeuille courant")
    with cols[1]: render_kpi("Classe à risque", f"{df[TARGET].mean()*100:.1f} %", "Cible pédagogique")
    with cols[2]: render_kpi("Anomalies", str(int(df_an["alerte_anomalie"].sum())), "Isolation Forest")
    with cols[3]: render_kpi("Meilleur modèle", best_name, "Sélection ROC-AUC")
    with cols[4]: render_kpi("Meilleure ROC-AUC", f"{best['metrics']['ROC-AUC']:.3f}", "Jeu de test 25 %")
    left, right = st.columns(2)
    left.plotly_chart(px.pie(df, names=TARGET, title="Répartition des classes", hole=0.55), use_container_width=True)
    right.plotly_chart(px.bar(scores.sort_values("ROC-AUC", ascending=False), x="Modèle", y="ROC-AUC", text_auto=".3f", range_y=[0,1], title="Classement ROC-AUC"), use_container_width=True)
    st.subheader("Profils nécessitant une revue prioritaire")
    st.dataframe(df_an.sort_values("score_anomalie", ascending=False).head(12), use_container_width=True, hide_index=True)

elif page == "🧠 Entraîner les modèles":
    st.subheader("Entraînement contrôlé des quatre modèles")
    st.markdown('<div class="section-note">Ce module réentraîne les quatre algorithmes sur le même découpage des données et met à jour toutes les métriques de l’application.</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", len(df)); c2.metric("Variables utilisées", len(FEATURES)); c3.metric("Jeu de test", "25 %")
    if st.button("🚀 Entraîner les 4 modèles", type="primary", use_container_width=True):
        bar = st.progress(0, text="Préparation du pipeline")
        st.session_state.models_bundle = train_all(df, progress=bar)
        models, best_name, X_test, y_test = st.session_state.models_bundle
        scores = metrics_frame(models)
        joblib.dump(models[best_name]["pipe"], MODELS_DIR / "meilleur_modele.joblib")
        scores.to_csv(EXPORTS / "comparaison_modeles.csv", index=False, encoding="utf-8-sig")
        with open(MODELS_DIR / "metrics_4_modeles.json", "w", encoding="utf-8") as fh:
            json.dump(scores.set_index("Modèle").to_dict(orient="index"), fh, ensure_ascii=False, indent=2)
        bar.progress(1.0, text="Entraînement terminé")
        st.success(f"Entraînement terminé. Modèle recommandé : {best_name}.")
    st.dataframe(scores.style.format({c:"{:.3f}" for c in ["Accuracy","Précision","Rappel","F1","ROC-AUC"]}).highlight_max(axis=0), use_container_width=True, hide_index=True)

elif page == "🧪 Comparateur interactif":
    st.subheader("Comparateur interactif des quatre modèles")
    metric = st.selectbox("Critère de classement", ["ROC-AUC", "F1", "Rappel", "Précision", "Accuracy"])
    ranked = scores.sort_values(metric, ascending=False).reset_index(drop=True)
    winner = ranked.iloc[0]["Modèle"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Modèles comparés", 4); c2.metric("Modèle gagnant", winner); c3.metric(f"Score {metric}", f"{ranked.iloc[0][metric]:.3f}")
    st.plotly_chart(px.bar(ranked, x="Modèle", y=metric, text_auto=".3f", range_y=[0,1], title=f"Classement selon {metric}"), use_container_width=True)
    st.dataframe(ranked.style.format({c:"{:.3f}" for c in ["Accuracy","Précision","Rappel","F1","ROC-AUC"]}).highlight_max(axis=0), use_container_width=True, hide_index=True)
    selected = st.selectbox("Inspecter la matrice de confusion", MODEL_NAMES, index=MODEL_NAMES.index(winner))
    cm = confusion_matrix(y_test, models[selected]["pred"])
    a, b = st.columns([1.15, .85])
    a.plotly_chart(px.imshow(cm, text_auto=True, x=["Normal", "Alerte"], y=["Normal", "Alerte"], labels={"x":"Classe prédite","y":"Classe réelle"}, title=f"Matrice de confusion — {selected}"), use_container_width=True)
    b.markdown("### Lecture métier")
    b.write(f"Le meilleur modèle selon **{metric}** est **{winner}**.")
    b.write("Le choix de la métrique dépend du coût métier : le rappel limite les risques manqués, tandis que la précision limite les fausses alertes.")

elif page == "⚡ Simulateur IA":
    st.subheader("Formulaire de simulation — réponse simultanée des quatre modèles")
    threshold = st.slider("Seuil de décision (%)", 10, 90, 50, 5)
    decision_mode = st.radio("Modèle de décision", ["Automatique — meilleure ROC-AUC"] + MODEL_NAMES, horizontal=True)
    with st.form("simulation_form"):
        a, b, c, d = st.columns(4)
        credit = a.slider("Score de crédit", 300, 850, 650)
        age = a.slider("Âge", 18, 95, 40)
        tenure = a.slider("Ancienneté (années)", 0, 10, 5)
        country = b.selectbox("Pays", sorted(df["country"].dropna().unique()))
        gender = b.selectbox("Genre", sorted(df["gender"].dropna().unique()))
        products = b.slider("Nombre de produits", 1, 4, 2)
        balance = c.number_input("Solde (€)", 0.0, 500000.0, 75000.0, step=1000.0)
        salary = c.number_input("Salaire estimé (€)", 0.0, 500000.0, 60000.0, step=1000.0)
        card = c.selectbox("Carte de crédit", [1,0], format_func=lambda x: "Oui" if x else "Non")
        active = d.selectbox("Membre actif", [1,0], format_func=lambda x: "Oui" if x else "Non")
        tx_date = d.date_input("Date de l’analyse")
        tx_time = d.time_input("Heure de l’analyse")
        submitted = st.form_submit_button("🔍 Analyser le profil avec les 4 modèles", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{
            "credit_score":credit, "country":country, "gender":gender, "age":age,
            "tenure":tenure, "balance":balance, "products_number":products,
            "credit_card":card, "active_member":active, "estimated_salary":salary,
        }])
        predictions = []
        for name, obj in models.items():
            p = float(obj["pipe"].predict_proba(row)[0,1] * 100)
            predictions.append({"Modèle":name, "Score de risque (%)":round(p,2), "Décision":"Alerte" if p >= threshold else "Normal"})
        pred_df = pd.DataFrame(predictions)
        chosen = best_name if decision_mode.startswith("Automatique") else decision_mode
        risk = float(pred_df.loc[pred_df["Modèle"] == chosen, "Score de risque (%)"].iloc[0])
        final_decision = "Alerte" if risk >= threshold else "Normal"
        level = "Élevé" if risk >= 70 else "Moyen" if risk >= 35 else "Faible"
        left, right = st.columns([.9, 1.4])
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=risk, number={"suffix":" %"}, title={"text":f"Risque {level} — {chosen}"}, gauge={"axis":{"range":[0,100]}, "threshold":{"line":{"width":4},"value":threshold}, "steps":[{"range":[0,35],"color":"#d9f7e8"},{"range":[35,70],"color":"#fff2cc"},{"range":[70,100],"color":"#fde2e2"}]}))
        left.plotly_chart(gauge, use_container_width=True)
        right.plotly_chart(px.bar(pred_df, x="Modèle", y="Score de risque (%)", text_auto=".2f", range_y=[0,100], title="Réponse comparée des quatre modèles"), use_container_width=True)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        if final_decision == "Alerte":
            st.error(f"⚠️ Alerte détectée par {chosen} : score {risk:.2f} %. Une validation humaine est recommandée.")
        else:
            st.success(f"✅ Profil classé normal par {chosen} : score {risk:.2f} %.")
        history_row = {
            "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_saisie": str(tx_date), "heure_saisie": str(tx_time),
            "jour": tx_date.day, "mois": tx_date.month, "annee": tx_date.year,
            "modele_decision": chosen, "seuil": threshold, "score_final": risk,
            "niveau_risque": level, "decision": final_decision,
            "credit_score":credit, "age":age, "tenure":tenure, "balance":balance,
            "estimated_salary":salary, "products_number":products,
            "country":country, "gender":gender, "credit_card":card, "active_member":active,
            **{f"score_{r['Modèle'].lower().replace(' ','_')}":r["Score de risque (%)"] for r in predictions},
        }
        save_history(history_row)
        simulation_pdf = pdf_summary("FraudShield AI — Résultat de simulation", [f"Modèle : {chosen}", f"Score : {risk:.2f} %", f"Niveau : {level}", f"Décision : {final_decision}", f"Seuil : {threshold} %", f"Client : âge {age}, score de crédit {credit}, solde {balance:.2f} €"])
        st.download_button("Télécharger cette analyse en PDF", simulation_pdf, "analyse_fraudshield.pdf", "application/pdf")

elif page == "🕘 Historique":
    st.subheader("Historique horodaté des analyses")
    hist = load_history()
    if hist.empty:
        st.info("Aucune simulation enregistrée pour le moment.")
    else:
        st.dataframe(hist.sort_values("horodatage", ascending=False), use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.download_button("Exporter l’historique CSV", hist.to_csv(index=False).encode("utf-8-sig"), "historique_fraudshield.csv", "text/csv", use_container_width=True)
        c2.download_button("Exporter l’historique Excel", excel_bytes({"Historique":hist}), "historique_fraudshield.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        if c3.button("Réinitialiser l’historique", use_container_width=True):
            HISTORY_FILE.unlink(missing_ok=True); st.success("Historique supprimé."); st.rerun()

elif page == "📂 Données & corrélation":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", len(df)); c2.metric("Colonnes", len(df.columns)); c3.metric("Valeurs manquantes", int(df.isna().sum().sum())); c4.metric("Doublons", int(df.duplicated().sum()))
    tab1, tab2, tab3 = st.tabs(["Aperçu", "Statistiques", "Corrélation"])
    with tab1: st.dataframe(df.head(300), use_container_width=True, hide_index=True)
    with tab2: st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    with tab3:
        corr = df.select_dtypes("number").drop(columns=["customer_id"], errors="ignore").corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Matrice de corrélation"), use_container_width=True)
        st.info("Une corrélation décrit une association linéaire, sans prouver une causalité.")

elif page == "📉 Évaluation":
    fig = go.Figure()
    for name, obj in models.items():
        fpr, tpr, _ = roc_curve(y_test, obj["proba"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} — AUC {obj['metrics']['ROC-AUC']:.3f}"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Aléatoire", line=dict(dash="dash")))
    fig.update_layout(title="Courbes ROC comparées", xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(scores.style.format({c:"{:.3f}" for c in ["Accuracy","Précision","Rappel","F1","ROC-AUC"]}).highlight_max(axis=0), use_container_width=True, hide_index=True)

elif page == "🚨 Anomalies":
    threshold = st.slider("Seuil minimal du score d’anomalie", 0, 100, 70)
    subset = df_an[df_an["score_anomalie"] >= threshold].sort_values("score_anomalie", ascending=False)
    c1, c2 = st.columns(2)
    c1.metric("Profils au-dessus du seuil", len(subset)); c2.metric("Anomalies Isolation Forest", int(df_an["alerte_anomalie"].sum()))
    st.plotly_chart(px.scatter(df_an, x="balance", y="estimated_salary", color="score_anomalie", size="age", hover_data=["customer_id"], title="Cartographie des profils atypiques"), use_container_width=True)
    st.dataframe(subset, use_container_width=True, hide_index=True)

elif page == "📦 Scoring par lot":
    st.subheader("Scoring automatique d’un portefeuille CSV")
    file = st.file_uploader("Charger un fichier contenant les variables attendues", type=["csv"], key="batch")
    if file:
        batch = normalise_columns(pd.read_csv(file))
        missing = set(FEATURES) - set(batch.columns)
        if missing:
            st.error("Colonnes manquantes : " + ", ".join(sorted(missing)))
        else:
            proba = best["pipe"].predict_proba(batch[FEATURES])[:,1]
            batch["score_risque"] = (proba * 100).round(2)
            batch["prediction_alerte"] = (proba >= .50).astype(int)
            st.dataframe(batch.head(300), use_container_width=True, hide_index=True)
            st.download_button("Télécharger le portefeuille scoré", batch.to_csv(index=False).encode("utf-8-sig"), "portefeuille_score.csv", "text/csv")

else:
    st.subheader("Centre d’exports")
    hist = load_history()
    alerts = df_an[df_an["alerte_anomalie"] == 1].copy()
    report_lines = [
        f"Clients analysés : {len(df)}",
        f"Modèle recommandé : {best_name}",
        *[f"{k} : {v:.3f}" for k, v in best["metrics"].items()],
        f"Anomalies détectées : {len(alerts)}",
    ]
    report_pdf = pdf_summary("FraudShield AI — Rapport de synthèse", report_lines)
    workbook = excel_bytes({"Comparaison modèles":scores, "Alertes":alerts, "Historique":hist if not hist.empty else pd.DataFrame({"Information":["Aucun historique"]})})
    c1, c2, c3 = st.columns(3)
    c1.download_button("📄 Rapport PDF", report_pdf, "rapport_fraudshield.pdf", "application/pdf", use_container_width=True)
    c2.download_button("📊 Classeur Excel", workbook, "exports_fraudshield.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    c3.download_button("📁 Alertes CSV", alerts.to_csv(index=False).encode("utf-8-sig"), "alertes_fraudshield.csv", "text/csv", use_container_width=True)
    st.caption("Le dataset fourni concerne l’attrition client. Dans ce projet pédagogique, la cible est exploitée comme indicateur binaire de risque et complétée par une détection d’anomalies.")
