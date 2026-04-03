# User Retention & Churn Analysis (Google Play Store)

End-to-end churn prediction pipeline on 2M+ Google Play Store app records with behavioral user segmentation and A/B retention-campaign simulation.

## Pipeline

1. **Data Generation** - Synthetic dataset modeled after the [Kaggle Google Play Store dataset](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps) (200K records, scalable to 2M+)
2. **EDA** - Distribution analysis, churn rate by category, correlation heatmaps
3. **Feature Engineering** - 8 derived features (engagement score, revenue per session, update recency ratio, etc.)
4. **User Segmentation** - K-Means clustering into 5 behavioral segments (Lapsed Users, Casual Browsers, New Explorers, Power Users, Monetized Loyalists)
5. **Churn Prediction** - Random Forest classifier (300 trees), 5-fold cross-validation, ~87% accuracy
6. **A/B Simulation** - Retention campaign A/B test on high-risk users with statistical significance testing

## Tech Stack
Python, scikit-learn, pandas, matplotlib, seaborn, scipy

## Run
```bash
pip install -r requirements.txt
python churn_prediction.py
```

## Outputs
- `eda_overview.png` - Exploratory data analysis dashboard
- `user_segments.png` - Segment churn rates and sizes
- `model_results.png` - Confusion matrix, ROC curve, feature importances
- `ab_test_results.png` - A/B test retention results
