import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc



#평가 지표 함수 정의   
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


# ==========================================
# data 폴더 내 파일 데이터 로드 
# ==========================================
print("2. data 폴더 내 실제 파일명 매핑하여 데이터 로드 중...")
cf_item_val = pd.read_csv(r'data/pred_item_cf_val.csv', sep=',') 
cf_user_val = pd.read_csv(r'data/pred_user_cf_val.csv', sep=',') 

# 컬럼명 앙상블 규격화
cf_item_val = cf_item_val.rename(columns={'predicted_rating': 'feat_item_cf', 'true_rating': 'actual_rating', 'item_id': 'movie_id'})
cf_user_val = cf_user_val.rename(columns={'predicted_rating': 'feat_user_cf', 'true_rating': 'actual_rating', 'item_id': 'movie_id'})

# SVD 3종 세트 유저/영화 10차원 임베딩 원본 로드 (수학 오차 유발하는 내적 연산 차단)
svd_item_user = pd.read_csv(r'data/svd_item_neighbor_user_embedding.csv', sep=',')
svd_item_movie = pd.read_csv(r'data/svd_item_neighbor_movie_embedding.csv', sep=',')
svd_row_user = pd.read_csv(r'data/svd_row_mean_user_embedding.csv', sep=',')
svd_row_movie = pd.read_csv(r'data/svd_row_mean_movie_embedding.csv', sep=',')
uv_sgd_user = pd.read_csv(r'data/uv_sgd_user_embedding.csv', sep=',')
uv_sgd_movie = pd.read_csv(r'data/uv_sgd_movie_embedding.csv', sep=',')

# 피처 중복 방지 식별자 부여
svd_item_user.columns = ['user_id'] + [f'item_nb_u_{i}' for i in range(len(svd_item_user.columns)-1)]
svd_item_movie.columns = ['movie_id'] + [f'item_nb_m_{i}' for i in range(len(svd_item_movie.columns)-1)]
svd_row_user.columns = ['user_id'] + [f'row_mn_u_{i}' for i in range(len(svd_row_user.columns)-1)]
svd_row_movie.columns = ['movie_id'] + [f'row_mn_m_{i}' for i in range(len(svd_row_movie.columns)-1)]
uv_sgd_user.columns = ['user_id'] + [f'uv_sgd_u_{i}' for i in range(len(uv_sgd_user.columns)-1)]
uv_sgd_movie.columns = ['movie_id'] + [f'uv_sgd_m_{i}' for i in range(len(uv_sgd_movie.columns)-1)]

# ==========================================
# 메타 레벨 피처 (Meta-Level Features) 생성
print("1. 원본 data/ua.base 기반 메타 레벨 통계 피처 생성 중...")
ua_base = pd.read_csv(r'data/ua.base', sep='\t', names=['user_id', 'movie_id', 'actual_rating', 'timestamp'])

# 각 유저 평점 수 로그 및 영화 리뷰 수 로그값 생성
user_stats = ua_base.groupby('user_id').agg(
    user_mean_rating=('actual_rating', 'mean'),
    user_rating_count=('actual_rating', 'count')
).reset_index()
user_stats['user_rating_count_log'] = np.log1p(user_stats['user_rating_count']) # 로그 변환으로 롱테일 분포 완화

movie_stats = ua_base.groupby('movie_id').agg(
    movie_mean_rating=('actual_rating', 'mean'),
    movie_rating_count=('actual_rating', 'count')
).reset_index()
movie_stats['movie_rating_count_log'] = np.log1p(movie_stats['movie_rating_count'])





# ==========================================
# [3단계] 모놀리식 피처 결합 하이브리드 (Feature Combination)
# ==========================================
print("3. 모든 협업 필터링 결과 및 6개 임베딩 피처 병합 중...")
total_df = pd.merge(cf_item_val, cf_user_val[['user_id', 'movie_id', 'feat_user_cf']], on=['user_id', 'movie_id'], how='left')

# 고차원 잠재 피처 결합 (Feature Combination)
total_df = pd.merge(total_df, svd_item_user, on='user_id', how='left')
total_df = pd.merge(total_df, svd_item_movie, on='movie_id', how='left')
total_df = pd.merge(total_df, svd_row_user, on='user_id', how='left')
total_df = pd.merge(total_df, svd_row_movie, on='movie_id', how='left')
total_df = pd.merge(total_df, uv_sgd_user, on='user_id', how='left')
total_df = pd.merge(total_df, uv_sgd_movie, on='movie_id', how='left')

# 메타 레벨 행동 패턴 피처 최종 결합
total_df = pd.merge(total_df, user_stats, on='user_id', how='left')
total_df = pd.merge(total_df, movie_stats, on='movie_id', how='left')
total_df = total_df.fillna(0)


# ==========================================
# [4단계] 데이터 분할 (Held-Out Split)
# ==========================================
print("4. 하이브리드 피처 매트릭스를 Train(9)과 Validation(1)로 분할 중...")
train_data, val_data = train_test_split(total_df, test_size=0.1, random_state=42)

X_train = train_data.drop(columns=['user_id', 'movie_id', 'actual_rating'])
y_train = train_data['actual_rating']
X_val = val_data.drop(columns=['user_id', 'movie_id', 'actual_rating'])
y_val = val_data['actual_rating']


# ==========================================
# [5단계] Bias-Variance 제어를 위한 부스팅 & 배깅 복합 학습
# ==========================================
print("5. 메타 모델들(LightGBM & RandomForest) 학습 시작...")
# 1) 부스팅 (LightGBM): 바이어스(Bias) 감소 전략
lgb_model = LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=7, num_leaves=63, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)

# 2) 배깅 (RandomForest): 오버피팅 분산(Variance) 감소 전략
rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)


# ==========================================
# [6단계] 최종 가중 블렌딩 결과 도출 및 다각도 평가
# ==========================================
print("6. 부스팅과 배깅 결과 최종 가중 블렌딩 및 오차 평가 중...")
lgb_preds = lgb_model.predict(X_val)
rf_preds = rf_model.predict(X_val)

# 부스팅 70% + 배깅 30% 황금 비율 결합
final_preds = (lgb_preds * 0.7) + (rf_preds * 0.3)
final_preds = np.clip(final_preds, 1.0, 5.0)

val_data = val_data.copy()
val_data['pred_rating'] = final_preds

# 평가지표
ensemble_rmse = np.sqrt(mean_squared_error(y_val, final_preds))
ensemble_mae = mean_absolute_error(y_val, final_preds)
print("\n=====================================================")
print(f"📉 [하이브리드 앙상블 평가]")
print(f"👉 최종 가중 블렌딩 앙상블 RMSE 지표: {ensemble_rmse:.4f}")
print(f"👉 최종 가중 블렌딩 앙상블 MAE 지표: {ensemble_mae:.4f}")
print("=====================================================")

# ROC-AUC (장표 5쪽 분류 관점 우회 증명 메트릭)
binary_label = (y_val >= 4).astype(int)
fpr, tpr, _ = roc_curve(binary_label, final_preds)
auc_score = auc(fpr, tpr)
print(f"👉 장표 5쪽 분류 기반 최종 ROC-AUC : {auc_score:.4f}")

# ==========================================
# Top-K 추천 성능 평가
# ==========================================

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

print("\n================ 추천 성능 평가 ================")
print(f"👉 Precision@{K}: {np.mean(precisions):.4f}")
print(f"👉 Recall@{K}:    {np.mean(recalls):.4f}")
print(f"👉 NDCG@{K}:      {np.mean(ndcgs):.4f}")
print("================================================")




# ua.base 전체 재학습

# ==========================================
# [8단계] 데이터 전체를 활용한 최종 재학습 (Full Re-training)
# ==========================================
print("\n[8단계] 성능 검증 완료. 전체 데이터(Total Data 100%)로 메타 모델 최종 재학습 시작...")

# 1. 전체 데이터셋에서 식별자와 정답 분리 (Leakage 방지)
X_all = total_df.drop(columns=['user_id', 'movie_id', 'actual_rating'])
y_all = total_df['actual_rating']

# 2. 최적화된 하이퍼파라미터로 최종 모델 인스턴스 생성
final_lgb_model = LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=7, num_leaves=63, random_state=42, verbose=-1)
final_rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1)

# 3. 100% 전체 데이터로 학습 진행
print("-> 최종 LightGBM 모델 학습 중...")
final_lgb_model.fit(X_all, y_all)

print("-> 최종 RandomForest 모델 학습 중...")
final_rf_model.fit(X_all, y_all)


# ==========================================
# [9단계] 실전 서빙(Serving)을 위한 파일 배포 및 저장
# ==========================================
print("\n[9단계] 실전 배포를 위한 최종 모델 및 메타 피처 저장 중...")

# 1. 학습된 메타 모델 파일 저장
joblib.dump(final_lgb_model, 'final_hybrid_lgb.pkl')
joblib.dump(final_rf_model, 'final_hybrid_rf.pkl')

# 2. 실전 예측 시 새로운 데이터에 조인할 '내가 만든 메타 통계 피처'도 함께 저장
user_stats.to_csv('final_user_stats.csv', index=False)
movie_stats.to_csv('final_movie_stats.csv', index=False)

print("🎉 모든 하이브리드 추천 파이프라인이 성공적으로 완료되었습니다!")
print("💾 저장된 파일: final_hybrid_lgb.pkl, final_hybrid_rf.pkl, final_user_stats.csv, final_movie_stats.csv")