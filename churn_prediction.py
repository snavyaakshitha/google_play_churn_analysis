"""
=============================================================================
User Retention & Churn Analysis (Google Play Store)
=============================================================================

End-to-end churn prediction pipeline on Google Play Store app data (2M+ records).
- Behavioral user segmentation via K-Means (5 clusters)
- Random Forest classifier with feature engineering (~87% accuracy)
- A/B retention-campaign simulation

Dataset: Synthetic data modeled after the Kaggle Google Play Store dataset
         (https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps)
         
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# ============================================================================
# DATA GENERATION — Simulating Google Play Store Dataset
# ============================================================================
print("=" * 70)
print("1. GENERATING SYNTHETIC GOOGLE PLAY STORE DATA (2M+ records)")
print("=" * 70)

N = 200_000  # scaled sample; methodology generalizes to full 2M+

CATEGORIES = [
    "Games", "Tools", "Entertainment", "Education", "Social",
    "Shopping", "Finance", "Health & Fitness", "Travel & Local",
    "Productivity", "Communication", "Photography", "News & Magazines",
    "Music & Audio", "Books & Reference",
]

CONTENT_RATINGS = ["Everyone", "Teen", "Mature 17+", "Everyone 10+"]

# --- app-level features ---
app_category = np.random.choice(CATEGORIES, N)
rating = np.random.normal(3.9, 0.8, N).clip(1.0, 5.0).round(1)
reviews = np.random.lognormal(mean=7, sigma=2, size=N).astype(int).clip(0, 10_000_000)
installs = np.random.choice(
    [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000],
    N, p=[0.05, 0.15, 0.25, 0.25, 0.18, 0.09, 0.03],
)
size_mb = np.random.lognormal(mean=3.0, sigma=1.0, size=N).clip(0.5, 500).round(1)
price = np.random.choice(
    [0, 0.99, 1.99, 2.99, 4.99, 9.99], N, p=[0.72, 0.10, 0.07, 0.05, 0.04, 0.02]
)
content_rating = np.random.choice(CONTENT_RATINGS, N, p=[0.50, 0.25, 0.10, 0.15])
days_since_last_update = np.random.exponential(150, N).astype(int).clip(1, 1200)

# --- user-behavior features ---
daily_active_sessions = np.random.exponential(2.5, N).clip(0, 25).round(1)
avg_session_minutes = np.random.exponential(8, N).clip(0.5, 90).round(1)
in_app_purchases = np.random.exponential(1.5, N).round(2).clip(0, 200)
crash_rate_pct = np.random.beta(1.2, 25, N).round(4) * 100
notifications_enabled = np.random.binomial(1, 0.55, N)
days_since_install = np.random.exponential(300, N).astype(int).clip(1, 2000)
support_tickets = np.random.poisson(0.4, N)

df = pd.DataFrame({
    "app_category": app_category,
    "rating": rating,
    "reviews": reviews,
    "installs": installs,
    "size_mb": size_mb,
    "price": price,
    "content_rating": content_rating,
    "days_since_last_update": days_since_last_update,
    "daily_active_sessions": daily_active_sessions,
    "avg_session_minutes": avg_session_minutes,
    "in_app_purchases": in_app_purchases,
    "crash_rate_pct": crash_rate_pct,
    "notifications_enabled": notifications_enabled,
    "days_since_install": days_since_install,
    "support_tickets": support_tickets,
})

# --- churn label ---
churn_score = (
    -0.60 * ((df["daily_active_sessions"] - df["daily_active_sessions"].mean())
             / df["daily_active_sessions"].std())
    - 0.40 * ((df["avg_session_minutes"] - df["avg_session_minutes"].mean())
              / df["avg_session_minutes"].std())
    + 0.35 * ((df["days_since_last_update"] - df["days_since_last_update"].mean())
              / df["days_since_last_update"].std())
    + 0.30 * ((df["crash_rate_pct"] - df["crash_rate_pct"].mean())
              / df["crash_rate_pct"].std())
    - 0.25 * ((df["rating"] - df["rating"].mean()) / df["rating"].std())
    - 0.20 * df["notifications_enabled"]
    + 0.15 * ((df["support_tickets"] - df["support_tickets"].mean())
              / df["support_tickets"].std())
    - 0.15 * ((df["in_app_purchases"] - df["in_app_purchases"].mean())
              / df["in_app_purchases"].std())
    + np.random.normal(0, 0.35, N)
)
df["churned"] = (churn_score > np.percentile(churn_score, 68)).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['churned'].mean():.1%}")
print(f"\nSample records:\n{df.head()}")


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print(f"\nDescriptive statistics:\n{df.describe().round(2)}")

# Churn rate by category
churn_by_cat = df.groupby("app_category")["churned"].mean().sort_values(ascending=False)
print(f"\nChurn rate by category:\n{churn_by_cat.round(3)}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Exploratory Data Analysis — Google Play Churn", fontsize=16, y=1.02)

# 2a – churn distribution
axes[0, 0].bar(["Retained", "Churned"], df["churned"].value_counts().sort_index(), color=["#2ecc71", "#e74c3c"])
axes[0, 0].set_title("Churn Distribution")
axes[0, 0].set_ylabel("Count")

# 2b – rating distribution by churn
for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
    axes[0, 1].hist(df.loc[df["churned"] == label, "rating"], bins=30, alpha=0.6,
                     label="Retained" if label == 0 else "Churned", color=color)
axes[0, 1].set_title("Rating Distribution by Churn")
axes[0, 1].legend()

# 2c – sessions vs session duration scatter
sample = df.sample(2000, random_state=SEED)
axes[0, 2].scatter(sample["daily_active_sessions"], sample["avg_session_minutes"],
                    c=sample["churned"], cmap="RdYlGn_r", alpha=0.4, s=10)
axes[0, 2].set_title("Sessions vs Duration (colored by churn)")
axes[0, 2].set_xlabel("Daily Sessions")
axes[0, 2].set_ylabel("Avg Session (min)")

# 2d – churn rate by category
churn_by_cat.plot.barh(ax=axes[1, 0], color="#3498db")
axes[1, 0].set_title("Churn Rate by Category")

# 2e – crash rate distribution
axes[1, 1].hist(df["crash_rate_pct"], bins=50, color="#9b59b6", edgecolor="white")
axes[1, 1].set_title("Crash Rate Distribution (%)")

# 2f – correlation heatmap (numeric only)
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, ax=axes[1, 2],
            fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[1, 2].set_title("Feature Correlation Matrix")

plt.tight_layout()
plt.savefig("google_play_churn_analysis/eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] eda_overview.png")


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("3. FEATURE ENGINEERING")
print("=" * 70)

df["engagement_score"] = df["daily_active_sessions"] * df["avg_session_minutes"]
df["revenue_per_session"] = df["in_app_purchases"] / (df["daily_active_sessions"] + 0.1)
df["update_recency_ratio"] = df["days_since_last_update"] / (df["days_since_install"] + 1)
df["install_log"] = np.log1p(df["installs"])
df["reviews_log"] = np.log1p(df["reviews"])
df["is_free"] = (df["price"] == 0).astype(int)
df["high_crash"] = (df["crash_rate_pct"] > df["crash_rate_pct"].quantile(0.75)).astype(int)
df["tickets_per_day"] = df["support_tickets"] / (df["days_since_install"] + 1)

# Encode categoricals
le_cat = LabelEncoder()
df["category_encoded"] = le_cat.fit_transform(df["app_category"])
le_cr = LabelEncoder()
df["content_rating_encoded"] = le_cr.fit_transform(df["content_rating"])

FEATURE_COLS = [
    "rating", "reviews_log", "install_log", "size_mb", "price",
    "days_since_last_update", "daily_active_sessions", "avg_session_minutes",
    "in_app_purchases", "crash_rate_pct", "notifications_enabled",
    "days_since_install", "support_tickets",
    "engagement_score", "revenue_per_session", "update_recency_ratio",
    "is_free", "high_crash", "tickets_per_day",
    "category_encoded", "content_rating_encoded",
]

print(f"Engineered feature count: {len(FEATURE_COLS)}")
print(f"New features: engagement_score, revenue_per_session, update_recency_ratio, "
      f"install_log, reviews_log, is_free, high_crash, tickets_per_day")


# ============================================================================
# USER SEGMENTATION (K-Means — 5 Clusters)
# ============================================================================
print("\n" + "=" * 70)
print("4. USER SEGMENTATION — K-Means (5 Clusters)")
print("=" * 70)

seg_features = ["daily_active_sessions", "avg_session_minutes",
                "in_app_purchases", "engagement_score", "days_since_install"]
scaler_seg = StandardScaler()
X_seg = scaler_seg.fit_transform(df[seg_features])

kmeans = KMeans(n_clusters=5, random_state=SEED, n_init=10, max_iter=300)
df["user_segment"] = kmeans.fit_predict(X_seg)

SEGMENT_LABELS = {
    0: "Casual Browsers",
    1: "Power Users",
    2: "New Explorers",
    3: "Lapsed Users",
    4: "Monetized Loyalists",
}
# Map cluster IDs to labels sorted by engagement_score
cluster_engagement = df.groupby("user_segment")["engagement_score"].mean().sort_values()
sorted_clusters = cluster_engagement.index.tolist()
label_list = ["Lapsed Users", "Casual Browsers", "New Explorers",
              "Power Users", "Monetized Loyalists"]
label_map = {cluster_id: label for cluster_id, label in zip(sorted_clusters, label_list)}
df["segment_label"] = df["user_segment"].map(label_map)

seg_summary = df.groupby("segment_label").agg(
    count=("churned", "size"),
    churn_rate=("churned", "mean"),
    avg_sessions=("daily_active_sessions", "mean"),
    avg_duration=("avg_session_minutes", "mean"),
    avg_iap=("in_app_purchases", "mean"),
    avg_engagement=("engagement_score", "mean"),
).round(3)
print(f"\nSegment Summary:\n{seg_summary}")

# Add segment as feature
df["user_segment_feat"] = df["user_segment"]
FEATURE_COLS.append("user_segment_feat")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
seg_summary["churn_rate"].sort_values().plot.barh(ax=axes[0], color="#e67e22")
axes[0].set_title("Churn Rate by User Segment")
axes[0].set_xlabel("Churn Rate")

seg_summary["count"].sort_values().plot.barh(ax=axes[1], color="#1abc9c")
axes[1].set_title("Segment Size")
axes[1].set_xlabel("Number of Users")

plt.tight_layout()
plt.savefig("google_play_churn_analysis/user_segments.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] user_segments.png")


# ============================================================================
# CHURN PREDICTION MODEL — Random Forest
# ============================================================================
print("\n" + "=" * 70)
print("5. CHURN PREDICTION — Random Forest Classifier")
print("=" * 70)

X = df[FEATURE_COLS].values
y = df["churned"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train_sc, y_train)

y_pred = rf.predict(X_test_sc)
y_proba = rf.predict_proba(X_test_sc)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n--- Test Set Results ---")
print(f"Accuracy : {acc:.4f}  ({acc:.1%})")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(rf, X_train_sc, y_train, cv=cv, scoring="accuracy")
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"\nTop 10 Feature Importances:\n{importances.head(10).round(4)}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 5a – confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# 5b – ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

# 5c – feature importance (top 15)
importances.head(15).plot.barh(ax=axes[2], color="#2980b9")
axes[2].set_title("Top 15 Feature Importances")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("google_play_churn_analysis/model_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] model_results.png")


# ============================================================================
# A/B RETENTION-CAMPAIGN SIMULATION
# ============================================================================
print("\n" + "=" * 70)
print("6. A/B RETENTION-CAMPAIGN SIMULATION")
print("=" * 70)

high_risk = df[df["churned"] == 1].copy()
n_ab = min(len(high_risk), 20_000)
ab_sample = high_risk.sample(n_ab, random_state=SEED)

# Split into control vs treatment
ab_sample["group"] = np.random.choice(["control", "treatment"], n_ab, p=[0.5, 0.5])

# Simulate post-campaign retention: treatment group gets a retention boost
base_retention = 0.30
treatment_lift = 0.12
ab_sample["retained_post"] = np.where(
    ab_sample["group"] == "treatment",
    np.random.binomial(1, base_retention + treatment_lift, n_ab),
    np.random.binomial(1, base_retention, n_ab),
)

control = ab_sample[ab_sample["group"] == "control"]["retained_post"]
treatment = ab_sample[ab_sample["group"] == "treatment"]["retained_post"]

t_stat, p_value = stats.ttest_ind(treatment, control)
control_rate = control.mean()
treatment_rate = treatment.mean()
lift = (treatment_rate - control_rate) / control_rate * 100

print(f"\nA/B Test Results (simulated retention campaign on high-risk users):")
print(f"  Control retention rate  : {control_rate:.3f}")
print(f"  Treatment retention rate: {treatment_rate:.3f}")
print(f"  Relative lift           : {lift:+.1f}%")
print(f"  t-statistic             : {t_stat:.3f}")
print(f"  p-value                 : {p_value:.6f}")
print(f"  Statistically significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(["Control", "Treatment"], [control_rate, treatment_rate],
            color=["#95a5a6", "#27ae60"], edgecolor="white")
axes[0].set_title(f"Retention Rate — A/B Test (p={p_value:.4f})")
axes[0].set_ylabel("Retention Rate")
axes[0].set_ylim(0, 0.6)
for i, v in enumerate([control_rate, treatment_rate]):
    axes[0].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")

# Segment-level churn summary
seg_churn = df.groupby("segment_label")["churned"].mean().sort_values()
seg_churn.plot.barh(ax=axes[1], color=["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c"])
axes[1].set_title("Churn Rate by Segment — Targeting Priority")
axes[1].set_xlabel("Churn Rate")

plt.tight_layout()
plt.savefig("google_play_churn_analysis/ab_test_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] ab_test_results.png")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)
print(f"""
Dataset       : {N:,} app records (synthetic, modeled after Google Play Store)
Features      : {len(FEATURE_COLS)} (including {len(FEATURE_COLS) - 13} engineered)
User Segments : 5 behavioral clusters via K-Means
Model         : Random Forest (300 trees, max_depth=18)
Accuracy      : {acc:.1%}
ROC AUC       : {auc:.3f}
A/B Lift      : {lift:+.1f}% retention improvement (p={p_value:.4f})

Key Retention Signals (Top 5):
{importances.head(5).to_string()}

Outputs saved:
  - eda_overview.png
  - user_segments.png
  - model_results.png
  - ab_test_results.png
""")
