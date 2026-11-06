# 📉 Customer Churn Prediction & Analytics

Projet personnel d'analyse et de prédiction du désabonnement des clients d'un opérateur télécom, avec un dashboard interactif.

Réalisé par **Fatima Zahra Bouzzait**, étudiante en 3ᵉ année de licence Génie Digital et Intelligence Artificielle (FST Settat).

## 🎯 Objectif

Une entreprise perd chaque mois une partie de ses clients. L'objectif est d'analyser leurs données, de prédire quels clients risquent de partir, et de présenter les résultats dans un dashboard.

## 📦 Données

- Dataset public **IBM Telco Customer Churn** : 7 043 clients, 21 colonnes (contrat, services, facturation, ancienneté, statut de désabonnement).
- Taux de churn global : **26,5 %** (1 869 clients partis).
- Nettoyage : les 11 valeurs vides de `TotalCharges` correspondent à des clients avec `tenure = 0` (pas encore facturés), remplacées par 0.

## 🛠️ Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · SQLite (SQL) · Streamlit · Jupyter · Git/GitHub

## 🔬 Méthode

1. **Analyse exploratoire (EDA)** : churn selon le contrat, l'ancienneté, les charges mensuelles, le mode de paiement et le service Internet.
2. **Prétraitement** : `StandardScaler` sur les variables numériques, `OneHotEncoder` sur les variables catégorielles, regroupés dans un `Pipeline` scikit-learn.
3. **Modélisation** : séparation 80/20 stratifiée, puis comparaison de trois modèles par validation croisée à 5 plis.
4. **Analyse SQL** : base SQLite et cinq requêtes d'analyse.
5. **Dashboard** : application Streamlit avec analyse des données et prédiction pour un client.

## 📊 Résultats

### Comparaison des modèles (validation croisée, jeu d'entraînement)

| Modèle | Recall | ROC-AUC |
|---|---|---|
| **Régression logistique** | **0,804** | 0,845 |
| Random Forest | 0,645 | 0,825 |
| Gradient Boosting | 0,528 | 0,847 |

Les ROC-AUC sont très proches. La régression logistique a été retenue pour son meilleur recall et son interprétabilité.

### Performance finale sur le jeu de test (1 409 clients)

| Métrique | Valeur |
|---|---|
| Recall | 0,78 |
| Precision | 0,50 |
| F1-score | 0,61 |
| ROC-AUC | 0,84 |
| Accuracy | 0,74 |

Le **recall** a été privilégié : rater un client qui part coûte plus cher que contacter par erreur un client qui serait resté. En contrepartie, la precision est de 0,50 (une alerte sur deux est une fausse alerte). L'accuracy est peu informative ici : un modèle qui prédirait « personne ne part » obtiendrait environ 73 %.

### Facteurs associés au churn

- Facteurs qui augmentent le risque : fibre optique, contrat mensuel, paiement par chèque électronique, absence de sécurité en ligne et de support technique.
- Facteurs qui le diminuent : ancienneté élevée, contrat de 2 ans, Internet DSL.

### Analyse SQL

| Segment | Clients | Taux de churn |
|---|---|---|
| Contrat mensuel | 3 875 | 42,7 % |
| Contrat 1 an | 1 473 | 11,3 % |
| Contrat 2 ans | 1 695 | 2,8 % |
| 0-12 mois d'ancienneté | 2 186 | 47,4 % |
| 49-72 mois d'ancienneté | 2 239 | 9,5 % |
| Mensuel + fibre + 12 mois max | 916 | 70,2 % |

Les clients partis paient en moyenne 74,44 $ par mois, contre 61,27 $ pour les clients restés.

## 🖥️ Dashboard

![Analyse des données](images/dashboard_analyse.png)
![Prédiction client à risque](images/prediction_risque.png)
![Prédiction client fidèle](images/prediction_fidele.png)

## ▶️ Lancer le projet

```bash
pip install -r requirements.txt
streamlit run app.py
```

Les fichiers `churn_model.joblib`, `model_columns.joblib` et `telco_clean.csv` doivent être dans le même dossier que `app.py`. Ils sont générés par le notebook `Customer_Churn_Analysis.ipynb`.

## 📁 Structure du projet

```
├── Customer_Churn_Analysis.ipynb   # EDA, modèles, évaluation, SQL
├── app.py                          # Dashboard Streamlit
├── churn_model.joblib              # Modèle entraîné
├── model_columns.joblib            # Colonnes attendues par le modèle
├── telco_clean.csv                 # Données nettoyées
├── requirements.txt
└── images/                         # Captures d'écran du dashboard
```

## ⚠️ Limites

- Les coefficients indiquent des **associations**, pas des causes.
- `MonthlyCharges`, `TotalCharges` et `tenure` sont fortement liées entre elles : leurs coefficients individuels sont à interpréter avec prudence.
- Le modèle est entraîné sur un dataset public d'illustration, pas sur des données d'entreprise réelles.
- Le dashboard tourne en local ; il n'est pas déployé en ligne.

## 🚀 Pistes d'amélioration

- Ajuster le seuil de décision pour équilibrer recall et precision.
- Tester d'autres modèles avec réglage des hyperparamètres.
- Déployer le dashboard en ligne.
