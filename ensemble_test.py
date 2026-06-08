import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_curve,
    auc
)

# ==========================================================
# 1. Evaluation Metric Functions
# ==========================================================
def precision_at_k(recs, true_items, k):
    return len(set(recs) & set(true_items)) / k


def recall_at_k(recs, true_items, k):
    return len(set(recs) & set(true_items)) / (len(true_items) + 1e-8)


def dcg(rel):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at_k(recs, true_items, k):
    rel = [1 if item in true_items else 0 for item in recs]
    ideal_rel = sorted(rel, reverse=True)
    return dcg(rel) / (dcg(ideal_rel) + 1e-8)


print("Hybrid Recommendation System Test Evaluation")


# ==========================================================
# 2. Load Test Data and Base Model Predictions
# ==========================================================
print("1. Loading test data")

ua_test = pd.read_csv(
    'data/ua.test',
    sep='\t',
    names=['user_id', 'movie_id', 'actual_rating', 'timestamp']
)

cf_item_test = pd.read_csv(
    'data/pred_item_cf_test.csv'
)

cf_user_test = pd.read_csv(
    'data/pred_user_cf_test.csv'
)

cf_item_test = cf_item_test.rename(columns={
    'predicted_rating': 'feat_item_cf',
    'item_id': 'movie_id'
})

cf_user_test = cf_user_test.rename(columns={
    'predicted_rating': 'feat_user_cf',
    'item_id': 'movie_id'
})


# ==========================================================
# 3. Load Embeddings and Meta Features
# ==========================================================
print("2. Loading embeddings and meta features")

svd_item_user = pd.read_csv('data/svd_item_neighbor_user_embedding.csv')
svd_item_movie = pd.read_csv('data/svd_item_neighbor_movie_embedding.csv')
svd_row_user = pd.read_csv('data/svd_row_mean_user_embedding.csv')
svd_row_movie = pd.read_csv('data/svd_row_mean_movie_embedding.csv')
uv_sgd_user = pd.read_csv('data/uv_sgd_user_embedding.csv')
uv_sgd_movie = pd.read_csv('data/uv_sgd_movie_embedding.csv')

# Feature Naming
svd_item_user.columns = ['user_id'] + [f'item_nb_u_{i}' for i in range(len(svd_item_user.columns) - 1)]
svd_item_movie.columns = ['movie_id'] + [f'item_nb_m_{i}' for i in range(len(svd_item_movie.columns) - 1)]
svd_row_user.columns = ['user_id'] + [f'row_mn_u_{i}' for i in range(len(svd_row_user.columns) - 1)]
svd_row_movie.columns = ['movie_id'] + [f'row_mn_m_{i}' for i in range(len(svd_row_movie.columns) - 1)]
uv_sgd_user.columns = ['user_id'] + [f'uv_sgd_u_{i}' for i in range(len(uv_sgd_user.columns) - 1)]
uv_sgd_movie.columns = ['movie_id'] + [f'uv_sgd_m_{i}' for i in range(len(uv_sgd_movie.columns) - 1)]

# Training Statistics
user_stats = pd.read_csv('data/final_user_stats.csv')
movie_stats = pd.read_csv('data/final_movie_stats.csv')


# ==========================================================
# 4. Build Test Feature Matrix
# ==========================================================
print("3. Building test feature matrix")

test_df = pd.merge(ua_test[['user_id', 'movie_id', 'actual_rating']], cf_item_test[['user_id', 'movie_id', 'feat_item_cf']], on=['user_id', 'movie_id'], how='left')
test_df = pd.merge(test_df, cf_user_test[['user_id', 'movie_id', 'feat_user_cf']], on=['user_id', 'movie_id'], how='left')
test_df = pd.merge(test_df, svd_item_user, on='user_id', how='left')
test_df = pd.merge(test_df, svd_item_movie, on='movie_id', how='left')
test_df = pd.merge(test_df, svd_row_user, on='user_id', how='left')
test_df = pd.merge(test_df, svd_row_movie, on='movie_id', how='left')
test_df = pd.merge(test_df, uv_sgd_user, on='user_id', how='left')
test_df = pd.merge(test_df, uv_sgd_movie, on='movie_id', how='left')
test_df = pd.merge(test_df, user_stats, on='user_id', how='left')
test_df = pd.merge(test_df, movie_stats, on='movie_id', how='left')
test_df = test_df.fillna(0)

# ==========================================================
# 5. Prepare Test Features
# ==========================================================
X_test = test_df.drop(
    columns=['user_id', 'movie_id', 'actual_rating']
)

y_test = test_df['actual_rating']


# ==========================================================
# 6. Load Final Meta Models
# ==========================================================
print("4. Loading trained ensemble models")

lgb_model = joblib.load(
    'data/final_hybrid_lgb.pkl'
)

rf_model = joblib.load(
    'data/final_hybrid_rf.pkl'
)


# ==========================================================
# 7. Ensemble Prediction
# ==========================================================
print("5. Running ensemble prediction")

lgb_preds = lgb_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

final_test_preds = (
    lgb_preds * 0.7
    + rf_preds * 0.3
)

final_test_preds = np.clip(
    final_test_preds,
    1.0,
    5.0
)

test_eval_df = test_df.copy()
test_eval_df['pred_rating'] = final_test_preds


# ==========================================================
# 8. Error-Based Evaluation
# ==========================================================
rmse = np.sqrt(mean_squared_error(y_test, final_test_preds))
mae = mean_absolute_error(y_test, final_test_preds)

print("\n========== Test Evaluation ==========")
print(f"RMSE    : {rmse:.4f}")
print(f"MAE     : {mae:.4f}")


# ==========================================================
# 9. ROC-AUC Evaluation
# ==========================================================
binary_label = (
    y_test >= 4
).astype(int)

fpr, tpr, _ = roc_curve(binary_label,final_test_preds)
auc_score = auc(fpr,tpr)

print(f"ROC-AUC : {auc_score:.4f}")
print("=====================================")


# ==========================================================
# 10. Top-K Recommendation Evaluation
# ==========================================================
K = 5

precisions = []
recalls = []
ndcgs = []

for user_id, group in test_eval_df.groupby('user_id'):

    true_items = set(
        group[
            group['actual_rating'] >= 4
        ]['movie_id']
    )

    if len(true_items) == 0:
        continue

    recs = (
        group
        .sort_values(
            by='pred_rating',
            ascending=False
        )['movie_id']
        .head(K)
        .tolist()
    )

    precisions.append(precision_at_k(recs, true_items, K))
    recalls.append(recall_at_k(recs, true_items, K))
    ndcgs.append(ndcg_at_k(recs, true_items, K))

print("\n===== Top-K Recommendation Metrics =====")
print(f"Precision@{K}: {np.mean(precisions):.4f}")
print(f"Recall@{K}:    {np.mean(recalls):.4f}")
print(f"NDCG@{K}:      {np.mean(ndcgs):.4f}")
print("========================================")

recommended_items = set()

for user_id, group in test_eval_df.groupby('user_id'):

    recs = (
        group
        .sort_values(
            by='pred_rating',
            ascending=False
        )['movie_id']
        .head(K)
        .tolist()
    )

    recommended_items.update(recs)

coverage = len(recommended_items) / test_df['movie_id'].nunique()

print(f"Item Coverage: {coverage:.4f}")


# ==========================================================
# 11. ROC Curve Visualization
# ==========================================================
plt.figure(figsize=(8, 6))

plt.plot(fpr,tpr,linewidth=2,label=f'Hybrid Ensemble (AUC={auc_score:.4f})')
plt.plot([0, 1],[0, 1],linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on MovieLens Test Set')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('data/ensemble_test_roc_curve.png',dpi=300,bbox_inches='tight')
plt.close()


# ==========================================================
# 12. Prediction Distribution Visualization
# ==========================================================
plt.figure(figsize=(10, 5))
sns.kdeplot(y_test,label='Actual Rating',color='green',fill=True)
sns.kdeplot(final_test_preds,label='Predicted Rating',color='purple',fill=True)
plt.title('Actual vs Predicted Rating Distribution')

plt.xlabel('Rating')
plt.ylabel('Density')

plt.legend()

plt.grid(
    linestyle='--',
    alpha=0.5
)

plt.savefig(
    'data/ensemble_test_prediction_distribution.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("\nSaved Files")
print("- final_test_roc_curve.png")
print("- final_test_prediction_distribution.png")
