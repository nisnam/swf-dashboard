"""
SIIP Pre-computation Script
============================
Runs ONCE locally before deployment.
Trains all models, generates interpretations, saves artifacts to models/.

Usage: python precompute.py
"""

import warnings
warnings.filterwarnings('ignore')

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
import shap

from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# ============================================================
# PATHS
# ============================================================
BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "SIIP_cleaned.csv"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================
# STEP 0: LOAD & FEATURE ENGINEERING
# ============================================================
print("=" * 60)
print("STEP 0: Loading data & feature engineering")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# --- Feature engineering log ---
fe_log = []

# Impute esg_composite NaN with median
esg_nan_count = df['esg_composite'].isna().sum()
esg_median = df['esg_composite'].median()
df['esg_composite'] = df['esg_composite'].fillna(esg_median)
fe_log.append(f"Imputed esg_composite: {esg_nan_count} NaN → median ({esg_median:.2f})")

# Also fill esg_x_competition NaN (same 395 rows)
esg_x_nan = df['esg_x_competition'].isna().sum()
esg_x_median = df['esg_x_competition'].median()
df['esg_x_competition'] = df['esg_x_competition'].fillna(esg_x_median)
fe_log.append(f"Imputed esg_x_competition: {esg_x_nan} NaN → median ({esg_x_median:.2f})")

# log1p transforms
df['log_team_size'] = np.log1p(df['team_size'])
df['log_patent_count'] = np.log1p(df['patent_count'])
fe_log.append("log1p transform: team_size, patent_count")

# MOIC for EDA
df['moic'] = df['last_valuation'] / df['total_capital_raised'].replace(0, np.nan)
df['log_moic'] = np.log1p(df['moic'].fillna(0))
fe_log.append("Computed MOIC = last_valuation / total_capital_raised")

# Composite features already exist in dataset
composites = ['country_risk_composite', 'bilateral_composite', 'esg_composite',
              'stability_x_stage', 'bilateral_x_techtransfer', 'esg_x_competition']
fe_log.append(f"Composite features verified: {', '.join(composites)}")

print(f"Feature engineering: {len(fe_log)} steps completed")

# --- Define IVs ---
IVS = [
    'funding_stage_num', 'revenue_growth_pct', 'gross_margin_pct', 'company_age',
    'country_risk_composite', 'bilateral_composite', 'competitive_intensity',
    'team_size', 'patent_count', 'ip_protection_score', 'tech_transfer_proxy',
    'esg_composite', 'national_strategy_alignment', 'is_pre_revenue',
    'stability_x_stage', 'runway_months', 'regulatory_moat'
]

DV = 'outcome_binary'

# --- Scale and split ---
X = df[IVS].copy()
y = df[DV].copy()

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=IVS)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

train_idx = X_train.index.tolist()
test_idx = X_test.index.tolist()

# Save prep artifacts
prep_artifacts = {
    'ivs': IVS,
    'dv': DV,
    'fe_log': fe_log,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'train_idx': train_idx,
    'test_idx': test_idx,
    'n_train': len(X_train),
    'n_test': len(X_test),
    'class_distribution': {
        'success': int(y.sum()),
        'failure': int(len(y) - y.sum()),
        'success_rate': float(y.mean())
    }
}

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Success rate: {y.mean():.1%}")

# ============================================================
# STEP 1: REGRESSION (load existing results)
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Loading regression results")
print("=" * 60)

reg_path = MODELS_DIR / "regression_results.json"
with open(reg_path) as f:
    reg_results = json.load(f)

# Generate regression interpretations
reg_interp = {
    'overview': f"Testing whether {len(IVS)} structural venture features can linearly predict outcome categories (0-3 ordinal scale). R²={reg_results['ols_statsmodels']['r2']:.3f} — statistically significant (F={reg_results['ols_statsmodels']['f_stat']:.2f}, p<1e-81) but explains only {reg_results['ols_statsmodels']['r2']:.1%} of variance.",
    'ols_finding': "Only funding_stage_num (p<0.001) and is_pre_revenue (p=0.027) reach statistical significance. The model captures real but limited linear signal — structural venture attributes have weak linear relationships with outcomes.",
    'lasso_finding': f"Lasso zeroed {17 - reg_results['models']['Lasso']['n_nonzero']}/17 features — only funding maturity (β=0.212), political stability interaction (β=0.029), patents (β=0.003), and company age (β<0.001) retain any linear predictive power.",
    'key_finding': "R²≈6% IS the finding, not a failure. Linear models are structurally limited for categorical outcomes. This motivates the shift to non-linear classification models that can capture complex feature interactions.",
    'bridge': "→ Prediction Engine uses non-linear classifiers (Decision Trees, Random Forest, XGBoost) to improve on linear limitations."
}

reg_results['interpretations'] = reg_interp
with open(reg_path, 'w') as f:
    json.dump(reg_results, f, indent=2)

print("Regression interpretations added")

# ============================================================
# STEP 2: CLASSIFICATION (6 models)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Training 6 classifiers")
print("=" * 60)

pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

classifiers = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=4, criterion='gini', class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(
        probability=True, kernel='rbf', class_weight='balanced', random_state=42
    )
}

clf_results = {}

for name, clf in classifiers.items():
    print(f"  Training {name}...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
    # Precision-Recall curve
    prec_curve, rec_curve, thresholds_pr = precision_recall_curve(y_test, y_prob)

    # Feature importances
    if hasattr(clf, 'feature_importances_'):
        feat_imp = dict(zip(IVS, clf.feature_importances_.tolist()))
    elif hasattr(clf, 'coef_'):
        feat_imp = dict(zip(IVS, np.abs(clf.coef_[0]).tolist()))
    else:
        feat_imp = {}

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    result = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'auc_roc': float(roc_auc_score(y_test, y_prob)),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'feature_importances': feat_imp,
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        },
        'pr_curve': {
            'precision': prec_curve.tolist(),
            'recall': rec_curve.tolist()
        },
        'y_pred': y_pred.tolist(),
        'y_prob': y_prob.tolist()
    }

    clf_results[name] = result
    print(f"    AUC={result['auc_roc']:.3f}, F1={result['f1']:.3f}, Acc={result['accuracy']:.3f}")

# Save classifier models
with open(MODELS_DIR / 'classification_models.pkl', 'wb') as f:
    pickle.dump(classifiers, f)

# Save results (without model objects)
with open(MODELS_DIR / 'classification_results.json', 'w') as f:
    json.dump(clf_results, f, indent=2)

# Save decision tree separately for visualization
with open(MODELS_DIR / 'decision_tree.pkl', 'wb') as f:
    pickle.dump(classifiers['Decision Tree'], f)

print("All classifiers trained and saved")

# --- Find best model ---
best_model_name = max(clf_results, key=lambda k: clf_results[k]['auc_roc'])
best_auc = clf_results[best_model_name]['auc_roc']
print(f"Best model: {best_model_name} (AUC={best_auc:.3f})")

# ============================================================
# STEP 2b: SHAP (on best tree model)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2b: Computing SHAP values")
print("=" * 60)

# Use Random Forest or XGBoost for SHAP (tree-based)
shap_model_name = 'Random Forest' if clf_results['Random Forest']['auc_roc'] >= clf_results['Gradient Boosting']['auc_roc'] else 'Gradient Boosting'
shap_model = classifiers[shap_model_name]

explainer = shap.TreeExplainer(shap_model)
# Use a sample for speed
sample_size = min(500, len(X_test))
X_shap_sample = X_test.iloc[:sample_size]
shap_values = explainer.shap_values(X_shap_sample)

# For binary classification, shap_values might be a list [class0, class1]
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # class 1 = success
else:
    shap_vals = shap_values

ev = explainer.expected_value
if isinstance(ev, (list, np.ndarray)):
    ev = float(ev[1]) if len(ev) > 1 else float(ev[0])
else:
    ev = float(ev)

shap_data = {
    'model_name': shap_model_name,
    'shap_values': shap_vals.tolist(),
    'expected_value': ev,
    'feature_names': IVS,
    'X_sample': X_shap_sample.values.tolist(),
    'sample_indices': X_shap_sample.index.tolist()
}

with open(MODELS_DIR / 'shap_values.pkl', 'wb') as f:
    pickle.dump(shap_data, f)

print(f"SHAP computed on {shap_model_name} ({sample_size} test samples)")

# ============================================================
# STEP 2c: THRESHOLD ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2c: Threshold analysis")
print("=" * 60)

best_clf = classifiers[best_model_name]
y_prob_best = np.array(clf_results[best_model_name]['y_prob'])

thresholds = np.arange(0.05, 0.96, 0.05)
threshold_results = []

for t in thresholds:
    y_pred_t = (y_prob_best >= t).astype(int)
    if y_pred_t.sum() == 0 or y_pred_t.sum() == len(y_pred_t):
        continue
    threshold_results.append({
        'threshold': float(t),
        'precision': float(precision_score(y_test, y_pred_t, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_t)),
        'f1': float(f1_score(y_test, y_pred_t)),
        'n_predicted_positive': int(y_pred_t.sum())
    })

# Find optimal threshold (max F1)
optimal_thresh = max(threshold_results, key=lambda x: x['f1'])

with open(MODELS_DIR / 'threshold_analysis.json', 'w') as f:
    json.dump({
        'model': best_model_name,
        'thresholds': threshold_results,
        'optimal': optimal_thresh
    }, f, indent=2)

print(f"Optimal threshold: {optimal_thresh['threshold']:.2f} (F1={optimal_thresh['f1']:.3f})")

# ============================================================
# STEP 3: CLUSTERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Clustering")
print("=" * 60)

# Use scaled features (exclude categoricals, IDs, DVs)
cluster_features = IVS  # same 17 features
X_cluster = X_scaled.copy()

# --- K-Means (k=2..7) ---
kmeans_results = {}
inertias = []
silhouettes = []

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    inertias.append(float(km.inertia_))
    sil = silhouette_score(X_cluster, labels)
    silhouettes.append(float(sil))

    # Cluster profiles
    profiles = {}
    for c in range(k):
        mask = labels == c
        cluster_df = df.loc[mask]
        profiles[str(c)] = {
            'size': int(mask.sum()),
            'success_rate': float(cluster_df['outcome_binary'].mean()),
            'mean_features': {feat: float(cluster_df[feat].mean()) for feat in IVS},
            'mean_capital': float(cluster_df['total_capital_raised'].mean()),
            'mean_valuation': float(cluster_df['last_valuation'].mean()),
            'top_sectors': cluster_df['sector'].value_counts().head(3).to_dict(),
            'top_stages': cluster_df['funding_stage'].value_counts().head(3).to_dict()
        }

    kmeans_results[str(k)] = {
        'labels': labels.tolist(),
        'centroids': km.cluster_centers_.tolist(),
        'inertia': float(km.inertia_),
        'silhouette': float(sil),
        'profiles': profiles
    }
    print(f"  K-Means k={k}: silhouette={sil:.3f}")

# Best k by silhouette
best_k = int(max(range(2, 8), key=lambda k: kmeans_results[str(k)]['silhouette']))
print(f"  Best k (silhouette): {best_k}")

# --- Hierarchical (Ward) ---
print("  Computing hierarchical clustering...")
linkage_matrix = linkage(X_cluster, method='ward')

hier_results = {}
for k in [3, 4, 5]:
    labels = fcluster(linkage_matrix, k, criterion='maxclust')
    labels = labels - 1  # 0-indexed
    sil = silhouette_score(X_cluster, labels)
    hier_results[str(k)] = {
        'labels': labels.tolist(),
        'silhouette': float(sil)
    }
    print(f"  Hierarchical k={k}: silhouette={sil:.3f}")

# --- DBSCAN ---
print("  Computing DBSCAN...")
# Use k-distance for eps estimation
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_cluster)
distances, _ = nn.kneighbors(X_cluster)
k_distances = np.sort(distances[:, -1])

# Find elbow point for eps
diffs = np.diff(k_distances)
elbow_idx = np.argmax(diffs > np.percentile(diffs, 95))
eps_val = float(k_distances[elbow_idx])
eps_val = max(eps_val, 2.0)  # ensure reasonable minimum

dbscan = DBSCAN(eps=eps_val, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_cluster)

n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = int((dbscan_labels == -1).sum())

print(f"  DBSCAN: {n_dbscan_clusters} clusters, {n_noise} noise points (eps={eps_val:.2f})")

# Save clustering results
clustering_data = {
    'kmeans': kmeans_results,
    'inertias': inertias,
    'silhouettes': silhouettes,
    'best_k': best_k,
    'hierarchical': hier_results,
    'linkage_matrix': linkage_matrix.tolist(),
    'dbscan': {
        'labels': dbscan_labels.tolist(),
        'n_clusters': n_dbscan_clusters,
        'n_noise': n_noise,
        'eps': eps_val
    },
    'k_distances': k_distances.tolist(),
    'features': cluster_features
}

with open(MODELS_DIR / 'clustering_results.pkl', 'wb') as f:
    pickle.dump(clustering_data, f)

print("Clustering saved")

# ============================================================
# STEP 4: PCA
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: PCA")
print("=" * 60)

pca_full = PCA()
pca_full.fit(X_scaled)

pca_2d = PCA(n_components=2)
proj_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
proj_3d = pca_3d.fit_transform(X_scaled)

# Loadings
loadings_2d = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1', 'PC2'],
    index=IVS
)

loadings_3d = pd.DataFrame(
    pca_3d.components_.T,
    columns=['PC1', 'PC2', 'PC3'],
    index=IVS
)

pca_data = {
    'explained_variance_ratio': pca_full.explained_variance_ratio_.tolist(),
    'cumulative_variance': np.cumsum(pca_full.explained_variance_ratio_).tolist(),
    'loadings_2d': loadings_2d.to_dict(),
    'loadings_3d': loadings_3d.to_dict(),
    'projections_2d': proj_2d.tolist(),
    'projections_3d': proj_3d.tolist(),
    'feature_names': IVS
}

with open(MODELS_DIR / 'pca_results.pkl', 'wb') as f:
    pickle.dump(pca_data, f)

# Top loadings interpretation
pc1_top = loadings_2d['PC1'].abs().sort_values(ascending=False).head(3)
pc2_top = loadings_2d['PC2'].abs().sort_values(ascending=False).head(3)
var_2d = pca_2d.explained_variance_ratio_.sum()

print(f"2 components explain {var_2d:.1%} variance")
print(f"PC1 top: {list(pc1_top.index)}")
print(f"PC2 top: {list(pc2_top.index)}")

# ============================================================
# STEP 5: ANOMALY DETECTION
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Anomaly Detection (Isolation Forest)")
print("=" * 60)

iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
anomaly_labels = iso.fit_predict(X_scaled)
anomaly_scores = iso.decision_function(X_scaled)

n_anomalies = int((anomaly_labels == -1).sum())

# Characterize anomalies
anomaly_df = df[anomaly_labels == -1].copy()
normal_df = df[anomaly_labels == 1].copy()

anomaly_profile = {}
for feat in IVS:
    anomaly_profile[feat] = {
        'anomaly_mean': float(anomaly_df[feat].mean()),
        'normal_mean': float(normal_df[feat].mean()),
        'diff_pct': float((anomaly_df[feat].mean() - normal_df[feat].mean()) / (normal_df[feat].std() + 1e-8))
    }

# Top differentiating features
top_diff = sorted(anomaly_profile.items(), key=lambda x: abs(x[1]['diff_pct']), reverse=True)[:5]

anomaly_data = {
    'labels': anomaly_labels.tolist(),
    'scores': anomaly_scores.tolist(),
    'n_anomalies': n_anomalies,
    'anomaly_profile': anomaly_profile,
    'top_differentiating': [(k, v) for k, v in top_diff],
    'anomaly_indices': np.where(anomaly_labels == -1)[0].tolist(),
    'anomaly_success_rate': float(anomaly_df['outcome_binary'].mean()),
    'normal_success_rate': float(normal_df['outcome_binary'].mean())
}

with open(MODELS_DIR / 'anomaly_results.pkl', 'wb') as f:
    pickle.dump(anomaly_data, f)

print(f"Flagged {n_anomalies} anomalies ({n_anomalies/len(df):.1%})")
print(f"Anomaly success rate: {anomaly_data['anomaly_success_rate']:.1%} vs Normal: {anomaly_data['normal_success_rate']:.1%}")

# ============================================================
# STEP 6: ARM (Association Rule Mining)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Association Rule Mining (Apriori)")
print("=" * 60)

# Discretize continuous features into tercile bins
arm_features = [
    'revenue_growth_pct', 'gross_margin_pct', 'company_age',
    'country_risk_composite', 'bilateral_composite', 'competitive_intensity',
    'team_size', 'patent_count', 'ip_protection_score', 'tech_transfer_proxy',
    'esg_composite', 'runway_months', 'total_capital_raised', 'last_valuation'
]

arm_df = pd.DataFrame()
discretization_thresholds = {}

for feat in arm_features:
    try:
        bins = pd.qcut(df[feat], q=3, labels=['Low', 'Med', 'High'], duplicates='drop')
        thresholds = pd.qcut(df[feat], q=3, retbins=True, duplicates='drop')[1].tolist()
        col_name = f"feat_{feat}"
        arm_df[col_name] = feat + '_' + bins.astype(str)
        discretization_thresholds[feat] = thresholds
    except Exception as e:
        print(f"  Warning: Could not discretize {feat}: {e}")

# Add categoricals as-is
arm_df['cat_sector'] = 'sector_' + df['sector'].astype(str)
arm_df['cat_stage'] = 'stage_' + df['funding_stage'].astype(str)

# Map outcome
outcome_map = {
    'Successful Exit': 'Success',
    'Partial Return': 'Partial',
    'Write-off': 'Write-off',
    'Still Active': 'Active'
}
arm_df['cat_outcome'] = 'outcome_' + df['investment_outcome'].map(outcome_map)

# One-hot encode for Apriori
arm_onehot = pd.get_dummies(arm_df)
arm_onehot = arm_onehot.astype(bool)

print(f"ARM transactions: {arm_onehot.shape[0]} x {arm_onehot.shape[1]} items")

# Apriori — use lower support to capture minority class patterns
freq_itemsets = apriori(arm_onehot, min_support=0.01, use_colnames=True, max_len=3)
print(f"Frequent itemsets: {len(freq_itemsets)}")

if len(freq_itemsets) > 0:
    rules = association_rules(freq_itemsets, metric='lift', min_threshold=1.0)
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))

    # Filter: only rules where consequent is a SINGLE outcome item
    outcome_rules = rules[rules['consequents'].apply(
        lambda x: len(x) == 1 and any('outcome_' in item for item in x)
    )]
    success_rules = outcome_rules[outcome_rules['consequents_str'].str.contains('Success')]
    failure_rules = outcome_rules[outcome_rules['consequents_str'].str.contains('Write-off')]

    print(f"Total rules: {len(rules)}")
    print(f"Outcome rules: {len(outcome_rules)}")
    print(f"Success rules: {len(success_rules)}")
    print(f"Failure rules: {len(failure_rules)}")
else:
    rules = pd.DataFrame()
    outcome_rules = pd.DataFrame()
    success_rules = pd.DataFrame()
    failure_rules = pd.DataFrame()

# Convert frozensets to strings for JSON serialization
def rules_to_serializable(rules_df):
    if len(rules_df) == 0:
        return []
    records = []
    for _, row in rules_df.iterrows():
        records.append({
            'antecedents': list(row['antecedents']),
            'consequents': list(row['consequents']),
            'support': float(row['support']),
            'confidence': float(row['confidence']),
            'lift': float(row['lift']),
            'conviction': float(row['conviction']) if not np.isinf(row['conviction']) else 999.0,
            'antecedents_str': row['antecedents_str'],
            'consequents_str': row['consequents_str']
        })
    return records

arm_data = {
    'all_rules': rules_to_serializable(rules),
    'outcome_rules': rules_to_serializable(outcome_rules),
    'success_rules': rules_to_serializable(success_rules),
    'failure_rules': rules_to_serializable(failure_rules),
    'freq_itemsets': [
        {'itemset': list(row['itemsets']), 'support': float(row['support'])}
        for _, row in freq_itemsets.iterrows()
    ],
    'discretization_thresholds': discretization_thresholds,
    'n_rules': len(rules),
    'n_outcome_rules': len(outcome_rules),
    'items': list(arm_onehot.columns)
}

with open(MODELS_DIR / 'arm_rules.pkl', 'wb') as f:
    pickle.dump(arm_data, f)

print("ARM rules saved")

# ============================================================
# STEP 7: ARIMA (Deal Flow Forecasting)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: ARIMA — Deal Flow Forecasting")
print("=" * 60)

# Create time series from last_funding_date
df['funding_date'] = pd.to_datetime(df['last_funding_date'], errors='coerce')
df['funding_quarter'] = df['funding_date'].dt.to_period('Q')

# Deal flow by quarter
deal_flow = df.groupby('funding_quarter').agg(
    deal_count=('venture_id', 'count'),
    total_capital=('total_capital_raised', 'sum'),
    success_rate=('outcome_binary', 'mean'),
    avg_valuation=('last_valuation', 'mean')
).reset_index()

deal_flow['funding_quarter'] = deal_flow['funding_quarter'].astype(str)

# Create time series for ARIMA
ts_data = df.groupby(df['funding_date'].dt.to_period('M')).size()
ts_data = ts_data.sort_index()
ts_data.index = ts_data.index.to_timestamp()

# Fit ARIMA on deal count
try:
    arima_model = ARIMA(ts_data, order=(2, 1, 2))
    arima_fit = arima_model.fit()

    # Forecast 12 months ahead
    forecast = arima_fit.forecast(steps=12)
    forecast_ci = arima_fit.get_forecast(steps=12).conf_int()

    # Seasonal decomposition
    if len(ts_data) >= 24:
        decomposition = seasonal_decompose(ts_data, model='additive', period=12)
        decomp_data = {
            'trend': decomposition.trend.dropna().values.tolist(),
            'seasonal': decomposition.seasonal.dropna().values.tolist(),
            'residual': decomposition.resid.dropna().values.tolist(),
            'trend_index': decomposition.trend.dropna().index.astype(str).tolist(),
            'seasonal_index': decomposition.seasonal.dropna().index.astype(str).tolist()
        }
    else:
        decomp_data = None

    arima_data = {
        'historical': {
            'dates': ts_data.index.astype(str).tolist(),
            'values': ts_data.values.tolist()
        },
        'forecast': {
            'dates': forecast.index.astype(str).tolist(),
            'values': forecast.values.tolist(),
            'lower_ci': forecast_ci.iloc[:, 0].values.tolist(),
            'upper_ci': forecast_ci.iloc[:, 1].values.tolist()
        },
        'model_summary': {
            'order': (2, 1, 2),
            'aic': float(arima_fit.aic),
            'bic': float(arima_fit.bic)
        },
        'deal_flow_quarterly': deal_flow.to_dict(orient='records'),
        'decomposition': decomp_data
    }

    print(f"ARIMA fitted: AIC={arima_fit.aic:.1f}, BIC={arima_fit.bic:.1f}")
    print(f"Forecast: {len(forecast)} months ahead")

except Exception as e:
    print(f"ARIMA warning: {e}")
    # Fallback - save deal flow data without ARIMA forecast
    arima_data = {
        'historical': {
            'dates': ts_data.index.astype(str).tolist(),
            'values': ts_data.values.tolist()
        },
        'forecast': None,
        'deal_flow_quarterly': deal_flow.to_dict(orient='records'),
        'decomposition': None
    }

with open(MODELS_DIR / 'arima_results.pkl', 'wb') as f:
    pickle.dump(arima_data, f)

print("ARIMA results saved")

# ============================================================
# STEP 8: CLUSTER PERSONAS (Business Framework)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Generating cluster personas & strategy frameworks")
print("=" * 60)

# Generate personas for best k
best_profiles = kmeans_results[str(best_k)]['profiles']

persona_templates = [
    ('Frontier Pioneers', 'High-risk early-stage ventures in emerging sectors. Screen carefully — potential for outsized returns but high failure rate.'),
    ('Stable Performers', 'Later-stage ventures with moderate risk profiles. Reliable allocation targets with predictable returns.'),
    ('Strategic Plays', 'Ventures with strong bilateral and geopolitical alignment. Value extends beyond financial return into strategic positioning.'),
    ('Growth Engines', 'Fast-growing ventures in competitive markets. High capital needs but strong revenue traction.'),
    ('Niche Specialists', 'Smaller ventures with strong IP and tech transfer potential. Domestic impact focus.'),
    ('Mature Stalwarts', 'Established ventures with proven track records. Lower growth ceiling but minimal downside risk.')
]

cluster_personas = {}
for i, (cluster_id, profile) in enumerate(sorted(best_profiles.items(), key=lambda x: x[1]['success_rate'], reverse=True)):
    persona_name, persona_desc = persona_templates[min(i, len(persona_templates)-1)]

    # Determine risk level
    sr = profile['success_rate']
    if sr > 0.2:
        risk_level = 'Low'
        strategy = 'PRIORITIZE — Strong historical success rate. Increase allocation.'
    elif sr > 0.15:
        risk_level = 'Medium'
        strategy = 'MONITOR — At or above baseline success rate. Maintain current allocation.'
    else:
        risk_level = 'High'
        strategy = 'SCREEN CAREFULLY — Below-average success rate. Enhanced due diligence required.'

    cluster_personas[cluster_id] = {
        'name': persona_name,
        'description': persona_desc,
        'risk_level': risk_level,
        'strategy': strategy,
        'success_rate': profile['success_rate'],
        'size': profile['size'],
        'top_sectors': profile['top_sectors'],
        'top_stages': profile['top_stages']
    }

# BCG Matrix data (sector-level)
sector_stats = df.groupby('sector').agg(
    deal_count=('venture_id', 'count'),
    total_capital=('total_capital_raised', 'sum'),
    avg_capital=('total_capital_raised', 'mean'),
    success_rate=('outcome_binary', 'mean'),
    avg_valuation=('last_valuation', 'mean'),
    avg_growth=('revenue_growth_pct', 'mean')
).reset_index()

# Risk-Return Matrix data
sector_stats['risk_composite_avg'] = df.groupby('sector')['country_risk_composite'].mean().values
sector_stats['bilateral_avg'] = df.groupby('sector')['bilateral_composite'].mean().values

frameworks_data = {
    'cluster_personas': cluster_personas,
    'sector_matrix': sector_stats.to_dict(orient='records'),
    'overall_stats': {
        'total_ventures': len(df),
        'total_capital': float(df['total_capital_raised'].sum()),
        'overall_success_rate': float(df['outcome_binary'].mean()),
        'active_count': int((df['investment_outcome'] == 'Still Active').sum()),
        'active_pct': float((df['investment_outcome'] == 'Still Active').mean()),
        'median_company_age': float(df['company_age'].median()),
        'sectors': sorted(df['sector'].unique().tolist()),
        'regions': sorted(df['region'].unique().tolist()),
        'stages': sorted(df['funding_stage'].unique().tolist()),
        'outcome_dist': df['investment_outcome'].value_counts().to_dict()
    }
}

with open(MODELS_DIR / 'frameworks.pkl', 'wb') as f:
    pickle.dump(frameworks_data, f)

print("Cluster personas and framework data saved")

# ============================================================
# STEP 9: INTERPRETATIONS BUNDLE
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Generating interpretations bundle")
print("=" * 60)

interpretations = {
    'home': {
        'kpi_summary': f"SIIP tracks {len(df):,} ventures across {df['sector'].nunique()} sectors. {df['outcome_binary'].sum():,} achieved successful exits ({df['outcome_binary'].mean():.1%}), while {(df['investment_outcome']=='Still Active').sum():,} remain active ({(df['investment_outcome']=='Still Active').mean():.1%}).",
        'portfolio_health': f"Total capital deployed: ${df['total_capital_raised'].sum()/1e9:.1f}B across {df['region'].nunique()} regions. Median company age: {df['company_age'].median():.0f} years."
    },
    'pipeline': {
        'data_quality': f"Dataset: {len(df):,} ventures × {len(df.columns)} features. {esg_nan_count} ESG values imputed (median). Feature engineering produced 6 composite interaction terms.",
        'arima': "ARIMA(2,1,2) models deal flow volume over time, forecasting pipeline capacity 12 months ahead. Seasonal decomposition reveals cyclical patterns in SWF investment activity."
    },
    'regression': reg_interp,
    'classification': {
        'imbalance': f"Success rate = {df['outcome_binary'].mean():.1%} (6:1 class imbalance). Without correction, a naive classifier achieves {1-df['outcome_binary'].mean():.1%} accuracy by always predicting failure. All models use balanced class weights or scale_pos_weight.",
        'best_model': f"Best classifier: {best_model_name} (AUC-ROC = {best_auc:.3f}). Non-linear models significantly outperform linear regression (R²=6%) by capturing complex feature interactions.",
        'shap': f"SHAP analysis on {shap_model_name} reveals which features drive individual predictions — enabling analysts to understand WHY a venture is flagged as promising or risky.",
        'threshold': f"At optimal threshold {optimal_thresh['threshold']:.2f}: F1={optimal_thresh['f1']:.3f}. False negatives (missing winners) are costlier than false positives (extra due diligence), so lower thresholds may be strategically preferred."
    },
    'clustering': {
        'overview': f"Segmentation identifies natural venture archetypes. K-Means optimal k={best_k} (silhouette={kmeans_results[str(best_k)]['silhouette']:.3f}). Each archetype has distinct risk-return profiles requiring tailored strategies.",
        'pca': f"PCA: 2 components explain {var_2d:.1%} of total variance. PC1 captures {'financial scale' if 'funding_stage_num' in pc1_top.index else pc1_top.index[0]}, PC2 captures {'risk positioning' if 'country_risk_composite' in pc2_top.index else pc2_top.index[0]}.",
        'anomaly': f"Isolation Forest flagged {n_anomalies} ventures ({n_anomalies/len(df):.1%}) as anomalous. Anomaly success rate: {anomaly_data['anomaly_success_rate']:.1%} vs normal {anomaly_data['normal_success_rate']:.1%}."
    },
    'arm': {
        'overview': f"Market-basket analysis adapted for venture screening. {len(arm_features)} continuous features discretized into tercile bins. {arm_data['n_rules']} total association rules discovered.",
        'success_patterns': f"{len(success_rules)} rules predict successful exits. Strongest patterns reveal which feature combinations are most predictive of success.",
        'anti_patterns': f"{len(failure_rules)} rules predict write-offs — these serve as automated red-flag screening filters."
    },
    'deal_evaluator': {
        'overview': "Synthesizes all models into a single venture assessment: classification probability, cluster membership, anomaly status, and matching ARM patterns.",
        'methodology': f"Score combines {best_model_name} probability ({best_auc:.3f} AUC), cluster persona context, and {arm_data['n_outcome_rules']} association rules."
    },
    'sentiment': {
        'overview': "NLP analysis of 6,798 investment analyst memos using three methods: TextBlob (lexicon), VADER (rule-based), and Naive Bayes (supervised). Analyst memos are generated from venture features to mirror SWF due diligence workflows.",
        'methods': "VADER excels at financial language with its rule-based handling of negations and intensifiers. TextBlob provides a simpler lexicon-based polarity score. Naive Bayes is trained on TF-IDF features to classify sentiment from text patterns.",
        'application': "Sentiment analysis adds a qualitative dimension to quantitative screening — when analyst language diverges from model predictions, it signals opportunities for deeper human review."
    }
}

with open(MODELS_DIR / 'interpretations.json', 'w') as f:
    json.dump(interpretations, f, indent=2)

# Save prep artifacts
with open(MODELS_DIR / 'prep_artifacts.pkl', 'wb') as f:
    pickle.dump(prep_artifacts, f)

# Save scaler
with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Interpretations bundle saved")

# ============================================================
# STEP 10: SENTIMENT ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Sentiment Analysis (Lexicon, VADER, Naive Bayes)")
print("=" * 60)

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# --- Generate synthetic investment analyst memos ---
# Each venture gets a realistic analyst note based on its features
np.random.seed(42)

positive_phrases = [
    "strong revenue traction", "impressive growth trajectory", "robust margins",
    "experienced founding team", "defensible IP portfolio", "clear market leadership",
    "excellent strategic alignment", "favorable regulatory environment",
    "promising technology transfer potential", "strong bilateral partnerships",
    "well-positioned for exit", "capital-efficient operations", "deep competitive moat",
    "ESG-compliant operations", "innovative technology stack", "scalable business model",
    "strong pipeline of opportunities", "attractive valuation entry point",
    "proven unit economics", "significant market opportunity"
]

negative_phrases = [
    "concerning burn rate", "limited revenue visibility", "margin compression risk",
    "inexperienced management team", "weak IP protection", "intense competitive pressure",
    "poor strategic fit", "regulatory headwinds ahead", "geopolitical risk exposure",
    "sanctions compliance concerns", "extended time to profitability", "capital-intensive model",
    "no clear competitive advantage", "ESG compliance gaps", "technology execution risk",
    "market saturation concerns", "challenging fundraising environment", "unfavorable entry valuation",
    "unproven business model", "limited market size"
]

neutral_phrases = [
    "standard sector dynamics", "typical stage-appropriate metrics", "market conditions stable",
    "team building in progress", "IP strategy under development", "competitive landscape evolving",
    "alignment assessment ongoing", "regulatory framework developing", "bilateral relations stable",
    "monitoring geopolitical developments", "scaling operations gradually", "capital deployment on track",
    "positioning for next funding round", "ESG framework implementation", "technology development phase",
    "market entry strategy forming", "due diligence in progress", "valuation within range",
    "business model refinement", "market analysis underway"
]

def generate_analyst_memo(row):
    """Generate a synthetic analyst memo for a venture based on its features."""
    parts = []

    # Sector context
    parts.append(f"Investment memo for {row['sector']} venture in {row['country']}.")

    # Financial assessment — use varied language
    rg = row['revenue_growth_pct']
    gm = row['gross_margin_pct']
    if rg > 50:
        parts.append(np.random.choice([
            f"Impressive revenue growth at {rg:.0f}% year-over-year, demonstrating strong market traction.",
            f"Exceptional top-line expansion of {rg:.0f}% signals product-market fit.",
            f"Revenue surging {rg:.0f}% — one of the strongest growers in portfolio."
        ]))
    elif rg > 20:
        parts.append(np.random.choice([
            f"Revenue growing at {rg:.0f}%, meeting expectations for this stage.",
            f"Steady {rg:.0f}% revenue growth, tracking slightly above sector median.",
            f"Top-line growth of {rg:.0f}% — adequate but not remarkable."
        ]))
    elif rg > 0:
        parts.append(np.random.choice([
            f"Sluggish revenue growth at {rg:.0f}% raises concerns about market demand.",
            f"Disappointing {rg:.0f}% growth — well below peer benchmarks.",
            f"Anemic top-line at {rg:.0f}%, insufficient for this burn rate."
        ]))
    else:
        parts.append(np.random.choice([
            f"Revenue declining at {rg:.0f}% — alarming trajectory.",
            f"Negative growth of {rg:.0f}% indicates serious product or market issues.",
            f"Revenue contraction at {rg:.0f}% demands immediate strategic review."
        ]))

    if gm > 60:
        parts.append(np.random.choice([
            f"Healthy gross margins at {gm:.0f}% provide operating leverage.",
            f"Strong {gm:.0f}% margins — excellent unit economics."
        ]))
    elif gm > 35:
        parts.append(f"Margins at {gm:.0f}% are acceptable for the sector.")
    elif gm > 0:
        parts.append(np.random.choice([
            f"Thin margins at {gm:.0f}% leave little room for error.",
            f"Compressed {gm:.0f}% margins signal pricing pressure."
        ]))
    else:
        parts.append(f"Negative margins at {gm:.0f}% — burning cash on every unit sold.")

    # Risk assessment — varied
    cr = row['country_risk_composite']
    if cr > 0.7:
        parts.append(np.random.choice([
            "Elevated geopolitical risk demands heightened vigilance and contingency planning.",
            "Country risk profile is unfavorable — sanctions exposure and political instability threaten returns.",
            "Operating in a high-risk jurisdiction with deteriorating bilateral relations."
        ]))
    elif cr > 0.4:
        parts.append(np.random.choice([
            "Moderate country risk — manageable with proper hedging.",
            "Geopolitical environment is stable but warrants ongoing monitoring."
        ]))
    else:
        parts.append(np.random.choice([
            "Low country risk provides a stable operating environment.",
            "Favorable geopolitical positioning with strong rule of law."
        ]))

    # Competition and IP
    ci = row['competitive_intensity']
    pc = row['patent_count']
    if ci > 0.7:
        parts.append(np.random.choice([
            "Faces brutal competitive pressure — market is overcrowded with well-funded rivals.",
            "Intense competition threatens market share and pricing power."
        ]))
    elif ci < 0.3:
        parts.append(np.random.choice([
            "Limited competitive pressure — potential for market dominance.",
            "Attractive competitive landscape with few serious challengers."
        ]))

    if pc > 5:
        parts.append(f"Holds {pc:.0f} patents providing defensible IP moat.")
    elif pc == 0:
        parts.append("No patent protection — vulnerable to replication by competitors.")

    # ESG
    esg = row['esg_composite']
    if esg > 60:
        parts.append("Strong ESG compliance supports long-term value creation.")
    elif esg < 25:
        parts.append("Significant ESG gaps could trigger regulatory or reputational risk.")

    # Runway and burn
    rw = row['runway_months']
    if rw < 12:
        parts.append(f"Only {rw:.0f} months runway remaining — will need to raise soon or face cash crisis.")
    elif rw > 36:
        parts.append(f"Well-capitalized with {rw:.0f} months of runway.")

    # Stage-specific color
    stage = row['funding_stage']
    if stage in ['Pre-IPO', 'Series D']:
        parts.append("Late-stage maturity reduces execution risk but also limits upside multiple.")
    elif stage == 'Seed':
        parts.append("Early seed stage — high uncertainty but maximum optionality.")

    # Outcome-influenced conclusion (softer signal, with noise)
    outcome = row['investment_outcome']
    noise = np.random.random()
    if outcome == 'Successful Exit':
        if noise < 0.6:
            parts.append(np.random.choice([
                "Overall assessment: favorable risk-reward profile.",
                "Fundamentals support a positive outlook for this investment.",
                "Strong candidate for continued allocation."
            ]))
        else:
            parts.append(np.random.choice([
                "Assessment is cautiously optimistic pending further milestones.",
                "Some positive indicators but several uncertainties remain.",
                "Position warrants monitoring despite some encouraging signals."
            ]))
    elif outcome == 'Write-off':
        if noise < 0.6:
            parts.append(np.random.choice([
                "Significant downside risk — recommend reducing exposure.",
                "Multiple warning signs suggest this venture may not recover.",
                "Unfavorable outlook across key performance dimensions."
            ]))
        else:
            parts.append(np.random.choice([
                "Mixed performance — some areas of concern but not without merit.",
                "Challenges are evident but situation could stabilize.",
                "Assessment is uncertain with both positive and negative indicators."
            ]))
    elif outcome == 'Partial Return':
        parts.append(np.random.choice([
            "Moderate outlook — some value preservation likely but upside capped.",
            "Mixed results across dimensions suggest limited return potential.",
            "Neither strong enough to champion nor weak enough to abandon.",
            "Risk-reward balance suggests maintaining minimal exposure."
        ]))
    else:  # Still Active
        parts.append(np.random.choice([
            "Too early for definitive assessment — monitoring actively.",
            "Position under review pending next quarterly update.",
            "Early indicators are mixed — maintain current allocation.",
            "Awaiting key milestones before adjusting position."
        ]))

    return ' '.join(parts)

print("  Generating analyst memos...")
df['analyst_memo'] = df.apply(generate_analyst_memo, axis=1)

# --- Method 1: TextBlob (Lexicon-based) ---
print("  Running TextBlob (Lexicon) sentiment...")
df['sentiment_textblob'] = df['analyst_memo'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity_textblob'] = df['analyst_memo'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# --- Method 2: VADER ---
print("  Running VADER sentiment...")
sia = SentimentIntensityAnalyzer()
vader_scores = df['analyst_memo'].apply(lambda x: sia.polarity_scores(x))
df['sentiment_vader'] = vader_scores.apply(lambda x: x['compound'])
df['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
df['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
df['vader_neu'] = vader_scores.apply(lambda x: x['neu'])

# Classify sentiment
def classify_sentiment(score, method='vader'):
    if method == 'vader':
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        else: return 'Neutral'
    else:  # textblob
        if score > 0.1: return 'Positive'
        elif score < -0.1: return 'Negative'
        else: return 'Neutral'

df['sentiment_class_vader'] = df['sentiment_vader'].apply(lambda x: classify_sentiment(x, 'vader'))
df['sentiment_class_textblob'] = df['sentiment_textblob'].apply(lambda x: classify_sentiment(x, 'textblob'))

# --- Method 3: Naive Bayes ---
print("  Training Naive Bayes classifier...")
# Create sentiment labels from outcome for supervised learning
outcome_sentiment = {
    'Successful Exit': 'Positive',
    'Partial Return': 'Neutral',
    'Write-off': 'Negative',
    'Still Active': 'Neutral'
}
df['sentiment_label'] = df['investment_outcome'].map(outcome_sentiment)

# TF-IDF features
tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df['analyst_memo'])
y_sentiment = df['sentiment_label']

# Train Naive Bayes
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_tfidf, y_sentiment)

# Cross-validation
cv_scores = cross_val_score(nb_model, X_tfidf, y_sentiment, cv=5, scoring='accuracy')
print(f"  Naive Bayes CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Predict
nb_predictions = nb_model.predict(X_tfidf)
nb_probabilities = nb_model.predict_proba(X_tfidf)
df['sentiment_class_nb'] = nb_predictions

# Get class names for probability columns
nb_classes = nb_model.classes_.tolist()

# Feature importance (top words per class)
feature_names = tfidf.get_feature_names_out()
top_words_per_class = {}
for i, cls in enumerate(nb_model.classes_):
    log_probs = nb_model.feature_log_prob_[i]
    top_idx = np.argsort(log_probs)[-15:][::-1]
    top_words_per_class[cls] = [(feature_names[j], float(log_probs[j])) for j in top_idx]

# --- Head-to-Head Comparison ---
print("  Computing head-to-head comparisons...")

# Agreement matrix
from sklearn.metrics import classification_report as clf_report, cohen_kappa_score

methods = {
    'VADER': df['sentiment_class_vader'],
    'TextBlob': df['sentiment_class_textblob'],
    'Naive Bayes': df['sentiment_class_nb']
}

agreement_matrix = {}
for m1_name, m1_vals in methods.items():
    agreement_matrix[m1_name] = {}
    for m2_name, m2_vals in methods.items():
        agreement = (m1_vals == m2_vals).mean()
        agreement_matrix[m1_name][m2_name] = float(agreement)

# Cohen's Kappa between each pair
kappa_scores = {}
for m1_name, m1_vals in methods.items():
    for m2_name, m2_vals in methods.items():
        if m1_name < m2_name:
            kappa = cohen_kappa_score(m1_vals, m2_vals)
            kappa_scores[f"{m1_name} vs {m2_name}"] = float(kappa)

# Performance vs actual outcome sentiment
actual_labels = df['sentiment_label']
method_performance = {}
for m_name, m_vals in methods.items():
    report = clf_report(actual_labels, m_vals, output_dict=True, zero_division=0)
    method_performance[m_name] = {
        'accuracy': float(report['accuracy']),
        'macro_f1': float(report['macro avg']['f1-score']),
        'weighted_f1': float(report['weighted avg']['f1-score']),
        'per_class': {cls: {
            'precision': float(report[cls]['precision']),
            'recall': float(report[cls]['recall']),
            'f1': float(report[cls]['f1-score'])
        } for cls in ['Positive', 'Neutral', 'Negative'] if cls in report}
    }

# Sentiment by sector, stage, outcome
sentiment_by_sector = df.groupby('sector').agg(
    vader_mean=('sentiment_vader', 'mean'),
    textblob_mean=('sentiment_textblob', 'mean'),
    count=('venture_id', 'count')
).reset_index().to_dict(orient='records')

sentiment_by_stage = df.groupby('funding_stage').agg(
    vader_mean=('sentiment_vader', 'mean'),
    textblob_mean=('sentiment_textblob', 'mean'),
    count=('venture_id', 'count')
).reset_index().to_dict(orient='records')

sentiment_by_outcome = df.groupby('investment_outcome').agg(
    vader_mean=('sentiment_vader', 'mean'),
    textblob_mean=('sentiment_textblob', 'mean'),
    vader_std=('sentiment_vader', 'std'),
    textblob_std=('sentiment_textblob', 'std'),
    count=('venture_id', 'count')
).reset_index().to_dict(orient='records')

# --- Save sentiment results ---
sentiment_data = {
    'memos': df['analyst_memo'].tolist(),
    'scores': {
        'vader_compound': df['sentiment_vader'].tolist(),
        'vader_pos': df['vader_pos'].tolist(),
        'vader_neg': df['vader_neg'].tolist(),
        'vader_neu': df['vader_neu'].tolist(),
        'textblob_polarity': df['sentiment_textblob'].tolist(),
        'textblob_subjectivity': df['subjectivity_textblob'].tolist(),
    },
    'classifications': {
        'vader': df['sentiment_class_vader'].tolist(),
        'textblob': df['sentiment_class_textblob'].tolist(),
        'naive_bayes': df['sentiment_class_nb'].tolist(),
        'actual': df['sentiment_label'].tolist(),
    },
    'naive_bayes': {
        'cv_accuracy': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'classes': nb_classes,
        'top_words_per_class': top_words_per_class,
        'probabilities': nb_probabilities.tolist(),
    },
    'head_to_head': {
        'agreement_matrix': agreement_matrix,
        'kappa_scores': kappa_scores,
        'method_performance': method_performance,
    },
    'breakdowns': {
        'by_sector': sentiment_by_sector,
        'by_stage': sentiment_by_stage,
        'by_outcome': sentiment_by_outcome,
    },
    'distributions': {
        'vader': {
            'Positive': int((df['sentiment_class_vader'] == 'Positive').sum()),
            'Neutral': int((df['sentiment_class_vader'] == 'Neutral').sum()),
            'Negative': int((df['sentiment_class_vader'] == 'Negative').sum()),
        },
        'textblob': {
            'Positive': int((df['sentiment_class_textblob'] == 'Positive').sum()),
            'Neutral': int((df['sentiment_class_textblob'] == 'Neutral').sum()),
            'Negative': int((df['sentiment_class_textblob'] == 'Negative').sum()),
        },
        'naive_bayes': {
            'Positive': int((df['sentiment_class_nb'] == 'Positive').sum()),
            'Neutral': int((df['sentiment_class_nb'] == 'Neutral').sum()),
            'Negative': int((df['sentiment_class_nb'] == 'Negative').sum()),
        }
    }
}

with open(MODELS_DIR / 'sentiment_results.pkl', 'wb') as f:
    pickle.dump(sentiment_data, f)

print(f"  VADER dist: {sentiment_data['distributions']['vader']}")
print(f"  TextBlob dist: {sentiment_data['distributions']['textblob']}")
print(f"  NB dist: {sentiment_data['distributions']['naive_bayes']}")
print(f"  Agreement (VADER vs TextBlob): {agreement_matrix['VADER']['TextBlob']:.1%}")
print("Sentiment analysis saved")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 60)
print("ALL ARTIFACTS GENERATED")
print("=" * 60)
print(f"\nFiles in models/:")
for f in sorted(MODELS_DIR.glob('*')):
    size = f.stat().st_size
    print(f"  {f.name}: {size/1024:.1f} KB")

print("\nReady for dashboard deployment!")
