import random
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, roc_curve

# ==========================================
# 0. CONFIGURATION & CONSTANTS
# ==========================================
TRAIN_DATA_PATH = 'data/ua.base'
TEST_DATA_PATH = 'data/ua.test'
ITEM_CF_PATH = 'data/pred_item_cf_test.csv'
USER_CF_PATH = 'data/pred_user_cf_test.csv'

SVD_ITEM_USER_PATH = 'data/svd_item_neighbor_user_embedding.csv'
SVD_ITEM_MOVIE_PATH = 'data/svd_item_neighbor_movie_embedding.csv'
SVD_ROW_USER_PATH = 'data/svd_row_mean_user_embedding.csv'
SVD_ROW_MOVIE_PATH = 'data/svd_row_mean_movie_embedding.csv'
UV_SGD_USER_PATH = 'data/uv_sgd_user_embedding.csv'
UV_SGD_MOVIE_PATH = 'data/uv_sgd_movie_embedding.csv'

USER_STATS_PATH = 'final_user_stats.csv'
MOVIE_STATS_PATH = 'final_movie_stats.csv'

LGB_MODEL_PATH = 'final_hybrid_lgb.pkl'
RF_MODEL_PATH = 'final_hybrid_rf.pkl'

ROC_PLOT_PATH = 'data/ensemble_test_roc_curve.png'
DIST_PLOT_PATH = 'data/ensemble_test_prediction_distribution.png'

RATING_THRESHOLD = 4.0
RANDOM_SEED = 42
DIVERSITY_SAMPLE_SIZE = 200

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
ua_test = pd.read_csv(
    TEST_DATA_PATH, 
    sep='\t', 
    names=['user_id', 'movie_id', 'actual_rating', 'timestamp']
)

cf_item_test = pd.read_csv(ITEM_CF_PATH).rename(
    columns={'predicted_rating': 'feat_item_cf', 'item_id': 'movie_id'}
)
cf_user_test = pd.read_csv(USER_CF_PATH).rename(
    columns={'predicted_rating': 'feat_user_cf', 'item_id': 'movie_id'}
)

svd_item_user = pd.read_csv(SVD_ITEM_USER_PATH)
svd_item_movie = pd.read_csv(SVD_ITEM_MOVIE_PATH)
svd_row_user = pd.read_csv(SVD_ROW_USER_PATH)
svd_row_movie = pd.read_csv(SVD_ROW_MOVIE_PATH)
uv_sgd_user = pd.read_csv(UV_SGD_USER_PATH)
uv_sgd_movie = pd.read_csv(UV_SGD_MOVIE_PATH)

svd_item_user.columns = ['user_id'] + [f'item_nb_u_{i}' for i in range(len(svd_item_user.columns) - 1)]
svd_item_movie.columns = ['movie_id'] + [f'item_nb_m_{i}' for i in range(len(svd_item_movie.columns) - 1)]
svd_row_user.columns = ['user_id'] + [f'row_mn_u_{i}' for i in range(len(svd_row_user.columns) - 1)]
svd_row_movie.columns = ['movie_id'] + [f'row_mn_m_{i}' for i in range(len(svd_row_movie.columns) - 1)]
uv_sgd_user.columns = ['user_id'] + [f'uv_sgd_u_{i}' for i in range(len(uv_sgd_user.columns) - 1)]
uv_sgd_movie.columns = ['movie_id'] + [f'uv_sgd_m_{i}' for i in range(len(uv_sgd_movie.columns) - 1)]

user_stats = pd.read_csv(USER_STATS_PATH)
movie_stats = pd.read_csv(MOVIE_STATS_PATH)

# ==========================================
# 2. FEATURE ENGINEERING & MATRIX BUILDING
# ==========================================
test_df = pd.merge(
    ua_test[['user_id', 'movie_id', 'actual_rating']], 
    cf_item_test[['user_id', 'movie_id', 'feat_item_cf']], 
    on=['user_id', 'movie_id'], 
    how='left'
)
test_df = pd.merge(test_df, cf_user_test[['user_id', 'movie_id', 'feat_user_cf']], on=['user_id', 'movie_id'], how='left')

test_df['feat_cf_diff'] = test_df['feat_user_cf'] - test_df['feat_item_cf']

test_df = pd.merge(test_df, svd_item_user, on='user_id', how='left')
test_df = pd.merge(test_df, svd_item_movie, on='movie_id', how='left')
test_df = pd.merge(test_df, svd_row_user, on='user_id', how='left')
test_df = pd.merge(test_df, svd_row_movie, on='movie_id', how='left')
test_df = pd.merge(test_df, uv_sgd_user, on='user_id', how='left')
test_df = pd.merge(test_df, uv_sgd_movie, on='movie_id', how='left')
test_df = pd.merge(test_df, user_stats, on='user_id', how='left')
test_df = pd.merge(test_df, movie_stats, on='movie_id', how='left')

test_df = test_df.fillna(0)

# ==========================================
# 3. MODEL INFERENCE & ENSEMBLE
# ==========================================
lgb_model = joblib.load(LGB_MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)

feature_names = list(lgb_model.feature_name_)
X_test = test_df[feature_names]
y_test = test_df['actual_rating']

lgb_preds = lgb_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

final_test_preds = np.clip((lgb_preds * 0.7 + rf_preds * 0.3), 1.0, 5.0)

# ==========================================
# 4. EVALUATION METRICS METRICS
# ==========================================
# Accuracy Metrics
rmse = np.sqrt(mean_squared_error(y_test, final_test_preds))
mae = mean_absolute_error(y_test, final_test_preds)

# Classification Metrics
actual_binary = (y_test >= RATING_THRESHOLD).astype(int)
pred_binary = (final_test_preds >= RATING_THRESHOLD).astype(int)

tp = np.sum((actual_binary == 1) & (pred_binary == 1))
fp = np.sum((actual_binary == 0) & (pred_binary == 1))
fn = np.sum((actual_binary == 1) & (pred_binary == 0))

global_precision = tp / (tp + fp + 1e-8)
global_recall = tp / (tp + fn + 1e-8)
global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall + 1e-8)

fpr, tpr, _ = roc_curve(actual_binary, final_test_preds)
auc_score = auc(fpr, tpr)

# Beyond-Accuracy Metrics
train_data = pd.read_csv(TRAIN_DATA_PATH, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
t0 = train_data['timestamp'].quantile(0.80)

test_eval_df = test_df.copy()
test_eval_df['pred_rating'] = final_test_preds
test_eval_df = test_eval_df.merge(ua_test[['user_id', 'movie_id', 'timestamp']], on=['user_id', 'movie_id'], how='left')
test_eval_df['is_future'] = (test_eval_df['timestamp'] > t0).astype(int)

all_recs = {}
novelty_scores = []

for user_id, group in test_eval_df.groupby('user_id'):
    true_items = set(group[group['actual_rating'] >= RATING_THRESHOLD]['movie_id'])
    recs_group = group[group['pred_rating'] >= RATING_THRESHOLD]
    
    if len(recs_group) == 0:
        continue
        
    recs = recs_group['movie_id'].tolist()
    all_recs[user_id] = recs
    
    user_novelty_score = 0
    for _, row in recs_group.iterrows():
        movie = row['movie_id']
        if movie in true_items:
            user_novelty_score += 1.0 if row['is_future'] == 1 else -1.0
    novelty_scores.append(user_novelty_score)

global_novelty = np.mean(novelty_scores) if novelty_scores else 0.0

users = list(all_recs.keys())
diversity_scores = []
if len(users) > DIVERSITY_SAMPLE_SIZE:
    random.seed(RANDOM_SEED)
    users = random.sample(users, DIVERSITY_SAMPLE_SIZE)

for i in range(len(users)):
    for j in range(i + 1, len(users)):
        diversity_scores.append(len(set(all_recs[users[i]]) & set(all_recs[users[j]])))
global_diversity = np.mean(diversity_scores) if diversity_scores else 0.0

recommended_items = set()
for u, recs in all_recs.items():
    recommended_items.update(recs)
global_coverage = len(recommended_items) / test_df['movie_id'].nunique()

# ==========================================
# 5. PRINT EVALUATION REPORT
# ==========================================
print("\n========== Test Evaluation (Global Baseline) ==========")
print(f"RMSE             : {rmse:.4f}")
print(f"MAE              : {mae:.4f}")
print(f"ROC-AUC          : {auc_score:.4f}")
print("\n--- [Classification Metrics (No Top-K)] ---")
print(f"Global Precision : {global_precision:.4f}")
print(f"Global Recall    : {global_recall:.4f}")
print(f"Global F1-Score  : {global_f1:.4f}")
print("\n===== Global Beyond-Accuracy Metrics =====")
print(f"Global Novelty   : {global_novelty:.4f}")
print(f"Global Diversity : {global_diversity:.4f}")
print(f"Global Coverage  : {global_coverage:.4f}")
print("=======================================================")

# ==========================================
# 6. VISUALIZATION & FEATURE IMPORTANCE
# ==========================================
# ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'Hybrid Ensemble (AUC={auc_score:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on MovieLens Set')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(ROC_PLOT_PATH, dpi=300, bbox_inches='tight')
plt.close()


# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n===== Top 20 Feature Importance (LightGBM) =====")
print(feature_importance.head(20).to_string(index=False))
print("================================================")