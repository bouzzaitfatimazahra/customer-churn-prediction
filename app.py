import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load("churn_model.joblib")
    columns = joblib.load("model_columns.joblib")
    return model, columns


@st.cache_data
def load_data():
    return pd.read_csv("telco_clean.csv")


model, model_columns = load_model()
df = load_data()

st.title("📉 Customer Churn Prediction & Analytics")
st.caption("Dataset IBM Telco : 7 043 clients, 21 colonnes. Modèle : régression logistique.")

tab_analyse, tab_predict = st.tabs(["📊 Analyse des données", "🔮 Prédiction client"])

# --------------------------------------------------------------------------
# Onglet 1 : analyse
# --------------------------------------------------------------------------
with tab_analyse:
    churn_rate = (df["Churn"] == "Yes").mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Nombre de clients", f"{len(df):,}".replace(",", " "))
    c2.metric("Taux de churn global", f"{churn_rate:.1f} %")
    c3.metric("Charges mensuelles moyennes", f"{df['MonthlyCharges'].mean():.2f} $")

    def churn_by(column):
        return (
            df.groupby(column)["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .round(1)
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Churn (%) par type de contrat")
        st.bar_chart(churn_by("Contract"))
    with right:
        st.subheader("Churn (%) par service Internet")
        st.bar_chart(churn_by("InternetService"))

    left, right = st.columns(2)
    with left:
        st.subheader("Churn (%) par mode de paiement")
        st.bar_chart(churn_by("PaymentMethod"))
    with right:
        st.subheader("Churn (%) par ancienneté")
        tenure_groups = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12 mois", "13-24 mois", "25-48 mois", "49-72 mois"],
        )
        by_tenure = (
            df.groupby(tenure_groups, observed=False)["Churn"]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .round(1)
        )
        st.bar_chart(by_tenure)

# --------------------------------------------------------------------------
# Onglet 2 : prédiction
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Estimer le risque de départ d'un client")

    def options(column):
        return sorted(df[column].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Profil**")
        gender = st.selectbox("Genre", options("gender"))
        senior = st.selectbox("Senior (65+)", [0, 1], format_func=lambda x: "Oui" if x else "Non")
        partner = st.selectbox("En couple", options("Partner"))
        dependents = st.selectbox("Personnes à charge", options("Dependents"))
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)

    with col2:
        st.markdown("**Services**")
        phone = st.selectbox("Service téléphone", options("PhoneService"))
        multiple = st.selectbox("Lignes multiples", options("MultipleLines"))
        internet = st.selectbox("Service Internet", options("InternetService"))
        security = st.selectbox("Sécurité en ligne", options("OnlineSecurity"))
        backup = st.selectbox("Sauvegarde en ligne", options("OnlineBackup"))

    with col3:
        st.markdown("**Options et facturation**")
        protection = st.selectbox("Protection d'appareil", options("DeviceProtection"))
        support = st.selectbox("Support technique", options("TechSupport"))
        streaming_tv = st.selectbox("Streaming TV", options("StreamingTV"))
        streaming_movies = st.selectbox("Streaming films", options("StreamingMovies"))
        contract = st.selectbox("Contrat", options("Contract"))

    col4, col5, col6 = st.columns(3)
    with col4:
        paperless = st.selectbox("Facture électronique", options("PaperlessBilling"))
    with col5:
        payment = st.selectbox("Mode de paiement", options("PaymentMethod"))
    with col6:
        monthly = st.number_input("Charges mensuelles ($)", 0.0, 200.0, 70.0, step=1.0)

    total = st.number_input(
        "Charges totales ($)",
        min_value=0.0,
        value=float(round(tenure * monthly, 2)),
        step=10.0,
        help="Par défaut : ancienneté × charges mensuelles.",
    )

    if st.button("Prédire le risque de churn", type="primary"):
        client = pd.DataFrame(
            [{
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }]
        )[model_columns]  # même ordre de colonnes qu'à l'entraînement

        proba = float(model.predict_proba(client)[0, 1])

        st.metric("Probabilité de départ", f"{proba * 100:.1f} %")
        st.progress(min(max(proba, 0.0), 1.0))

        if proba >= 0.60:
            st.error("🔴 Risque élevé : ce client a une forte probabilité de partir.")
        elif proba >= 0.30:
            st.warning("🟠 Risque moyen : à surveiller.")
        else:
            st.success("🟢 Risque faible : ce client devrait rester.")

        st.caption(
            "Les seuils 30 % / 60 % sont des repères indicatifs, "
            "pas une vérité statistique du modèle."
        )
