import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc
import joblib


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

print("1. 협업 필터링 결과 및 잠재요인 임베딩 로드")

cf_item_base = pd.read_csv('data/pred_item_cf_base.csv')
cf_user_base = pd.read_csv('data/pred_user_cf_base.csv')

cf_item_base = cf_item_base.rename(columns={
    'item_id': 'movie_id',
    'true_rating': 'actual_rating',
    'predicted_rating': 'feat_item_cf'
})

cf_user_base = cf_user_base.rename(columns={
    'item_id': 'movie_id',
    'true_rating': 'actual_rating',
    'predicted_rating': 'feat_user_cf'
})

svd_item_user = pd.read_csv('data/svd_item_neighbor_user_embedding.csv')
svd_item_movie = pd.read_csv('data/svd_item_neighbor_movie_embedding.csv')
svd_row_user = pd.read_csv('data/svd_row_mean_user_embedding.csv')
svd_row_movie = pd.read_csv('data/svd_row_mean_movie_embedding.csv')
uv_sgd_user = pd.read_csv('data/uv_sgd_user_embedding.csv')
uv_sgd_movie = pd.read_csv('data/uv_sgd_movie_embedding.csv')

svd_item_user.columns = ['user_id'] + [f'item_nb_u_{i}' for i in range(len(svd_item_user.columns)-1)]
svd_item_movie.columns = ['movie_id'] + [f'item_nb_m_{i}' for i in range(len(svd_item_movie.columns)-1)]

svd_row_user.columns = ['user_id'] + [f'row_mn_u_{i}' for i in range(len(svd_row_user.columns)-1)]
svd_row_movie.columns = ['movie_id'] + [f'row_mn_m_{i}' for i in range(len(svd_row_movie.columns)-1)]

uv_sgd_user.columns = ['user_id'] + [f'uv_sgd_u_{i}' for i in range(len(uv_sgd_user.columns)-1)]
uv_sgd_movie.columns = ['movie_id'] + [f'uv_sgd_m_{i}' for i in range(len(uv_sgd_movie.columns)-1)]


print("2. 사용자 및 영화 통계 피처 생성")

ua_base = pd.read_csv(
    'data/ua.base',
    sep='\t',
    names=['user_id', 'movie_id', 'actual_rating', 'timestamp']
)

user_stats = ua_base.groupby('user_id').agg(
    user_mean_rating=('actual_rating', 'mean'),
    user_rating_count=('actual_rating', 'count')
).reset_index()

user_stats['user_rating_count_log'] = np.log1p(
    user_stats['user_rating_count']
)

movie_stats = ua_base.groupby('movie_id').agg(
    movie_mean_rating=('actual_rating', 'mean'),
    movie_rating_count=('actual_rating', 'count')
).reset_index()

movie_stats['movie_rating_count_log'] = np.log1p(
    movie_stats['movie_rating_count']
)


print("3. 하이브리드 피처 결합")

total_df = pd.merge(
    cf_item_base[['user_id', 'movie_id', 'actual_rating', 'feat_item_cf']],
    cf_user_base[['user_id', 'movie_id', 'feat_user_cf']],
    on=['user_id', 'movie_id'],
    how='left'
)

total_df = pd.merge(total_df, svd_item_user, on='user_id', how='left')
total_df = pd.merge(total_df, svd_item_movie, on='movie_id', how='left')

total_df = pd.merge(total_df, svd_row_user, on='user_id', how='left')
total_df = pd.merge(total_df, svd_row_movie, on='movie_id', how='left')

total_df = pd.merge(total_df, uv_sgd_user, on='user_id', how='left')
total_df = pd.merge(total_df, uv_sgd_movie, on='movie_id', how='left')

total_df = pd.merge(total_df, user_stats, on='user_id', how='left')
total_df = pd.merge(total_df, movie_stats, on='movie_id', how='left')

total_df['feat_cf_diff'] = total_df['feat_user_cf'] - total_df['feat_item_cf']

total_df = total_df.fillna(0)


print("4. 메타 모델 학습/검증 데이터 분할")

train_data, val_data = train_test_split(
    total_df,
    test_size=0.1,
    random_state=42
)

X_train = train_data.drop(columns=['user_id', 'movie_id', 'actual_rating'])
y_train = train_data['actual_rating']

X_val = val_data.drop(columns=['user_id', 'movie_id', 'actual_rating'])
y_val = val_data['actual_rating']


print("5. LightGBM 및 RandomForest 학습")

train_sample_weights = 1.0 / np.log1p(train_data['movie_rating_count'])

lgb_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.02,
    max_depth=3,
    num_leaves=7,
    min_child_samples=80,
    colsample_bytree=0.15,
    subsample=0.5,
    reg_alpha=15.0,
    reg_lambda=15.0,
    random_state=42,
    verbose=-1
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=40,
    max_features=0.15,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_train, y_train, sample_weight=train_sample_weights)
rf_model.fit(X_train, y_train)


print("6. 앙상블 예측 및 성능 평가")

lgb_preds = lgb_model.predict(X_val)
rf_preds = rf_model.predict(X_val)

final_preds = (0.7 * lgb_preds) + (0.3 * rf_preds)
final_preds = np.clip(final_preds, 1.0, 5.0)

val_data = val_data.copy()
val_data['pred_rating'] = final_preds

rmse = np.sqrt(mean_squared_error(y_val, final_preds))
mae = mean_absolute_error(y_val, final_preds)

print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

binary_label = (y_val >= 4).astype(int)

fpr, tpr, _ = roc_curve(binary_label, final_preds)
auc_score = auc(fpr, tpr)

print(f"ROC-AUC : {auc_score:.4f}")


print("7. Top-K 추천 성능 평가")

K = 5

precisions = []
recalls = []
ndcgs = []

for user_id, group in val_data.groupby('user_id'):

    true_items = set(
        group[group['actual_rating'] >= 4]['movie_id']
    )

    if len(true_items) == 0:
        continue

    recs = (
        group.sort_values(
            by='pred_rating',
            ascending=False
        )['movie_id']
        .head(K)
        .tolist()
    )

    precisions.append(
        precision_at_k(recs, true_items, K)
    )

    recalls.append(
        recall_at_k(recs, true_items, K)
    )

    ndcgs.append(
        ndcg_at_k(recs, true_items, K)
    )


print(f"Precision@{K}: {np.mean(precisions):.4f}")
print(f"Recall@{K}:    {np.mean(recalls):.4f}")
print(f"NDCG@{K}:      {np.mean(ndcgs):.4f}")


print("\n8. 전체 메타 피처 데이터로 최종 모델 재학습")

X_all = total_df.drop(
    columns=['user_id', 'movie_id', 'actual_rating']
)

y_all = total_df['actual_rating']

total_sample_weights = 1.0 / np.log1p(total_df['movie_rating_count'])

final_lgb_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.02,
    max_depth=3,
    num_leaves=7,
    min_child_samples=80,
    colsample_bytree=0.15,
    subsample=0.5,
    reg_alpha=15.0,
    reg_lambda=15.0,
    random_state=42,
    verbose=-1
)

final_rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=40,
    max_features=0.15,
    random_state=42,
    n_jobs=-1
)

final_lgb_model.fit(X_all, y_all, sample_weight=total_sample_weights)
final_rf_model.fit(X_all, y_all)


print("9. 최종 모델 및 메타 피처 저장")

joblib.dump(
    final_lgb_model,
    'final_hybrid_lgb.pkl'
)

joblib.dump(
    final_rf_model,
    'final_hybrid_rf.pkl'
)

user_stats.to_csv(
    'final_user_stats.csv',
    index=False
)

movie_stats.to_csv(
    'final_movie_stats.csv',
    index=False
)

print("\n================ Save Complete ================")
print("final_hybrid_lgb.pkl")
print("final_hybrid_rf.pkl")
print("final_user_stats.csv")
print("final_movie_stats.csv")
print("================================================")


print("ua.base rows :", len(ua_base))
print("total_df rows:", len(total_df))