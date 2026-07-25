# FraudShield AI — Version Excellence

Application Streamlit professionnelle d’analyse du risque bancaire par intelligence artificielle.

## Fonctionnalités

- Bouton **Entraîner les 4 modèles** avec barre de progression
- Régression logistique, Random Forest, Gradient Boosting et Extra Trees
- Comparateur interactif par ROC-AUC, F1, rappel, précision ou accuracy
- Sélection automatique du meilleur modèle
- Matrices de confusion et courbes ROC
- Formulaire de simulation donnant la réponse des quatre modèles
- Jauge de risque et seuil réglable
- Historique persistant des simulations avec jour, mois, date et heure
- Détection d’anomalies par Isolation Forest
- Scoring d’un portefeuille CSV
- Exports PDF, Excel et CSV directement depuis l’application
- Interface moderne avec KPI, cartes, logo et tableaux de bord

## Installation Windows / VS Code

Ouvrir le terminal dans le dossier contenant `app.py`, puis exécuter :

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Structure utile

- `app.py` : application principale
- `data/bank_dataset.csv` : dataset intégré
- `exports/` : historique et fichiers générés
- `models/` : modèle recommandé et métriques après réentraînement
- `reports/` : rapport du projet

## Note méthodologique

Le dataset fourni est un dataset d’attrition client. Sa cible binaire est utilisée comme indicateur pédagogique de risque, complété par une détection non supervisée d’anomalies.
