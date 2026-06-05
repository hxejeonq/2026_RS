import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc

# ==========================================
# [1단계] Top-K 랭킹 평가 지표 함수 정의 
# ==========================================
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


print("🚀 [테스트 단계] 최종 하이브리드 추천 시스템 성능 평가를 시작합니다.")

# ==========================================
# [2단계] 기준이 되는 ua.test 및 CF 테스트 예측 파일 로드
# ==========================================
print("1. 테스트 기본 데이터 및 각 CF 모델의 테스트 예측 데이터 로드 중...")
# 기준이 되는 오리지널 테스트 파일 로드
ua_test = pd.read_csv(r'data/ua.test', sep='\t', names=['user_id', 'movie_id', 'actual_rating', 'timestamp'])

# 이웃 기반 CF 모델들이 ua.test에 대해 예측해 온 결과물 로드
cf_item_test = pd.read_csv(r'data/pred_item_cf_test.csv', sep=',') 
cf_user_test = pd.read_csv(r'data/pred_user_cf_test.csv', sep=',') 

# 컬럼명 앙상블 규격화 (predicted_rating 충돌 방지)
cf_item_test = cf_item_test.rename(columns={'predicted_rating': 'feat_item_cf', 'item_id': 'movie_id'})
cf_user_test = cf_user_test.rename(columns={'predicted_rating': 'feat_user_cf', 'item_id': 'movie_id'})


# ==========================================
# [3단계] 고차원 잠재 임베딩 및 저장된 메타 통계 피처 로드
# ==========================================
print("2. 잠재 임베딩 및 저장된 훈련 데이터 통계 피처(ua.base 기반) 결합 중...")
svd_item_user = pd.read_csv(r'data/svd_item_neighbor_user_embedding.csv', sep=',')
svd_item_movie = pd.read_csv(r'data/svd_item_neighbor_movie_embedding.csv', sep=',')
svd_row_user = pd.read_csv(r'data/svd_row_mean_user_embedding.csv', sep=',')
svd_row_movie = pd.read_csv(r'data/svd_row_mean_movie_embedding.csv', sep=',')
uv_sgd_user = pd.read_csv(r'data/uv_sgd_user_embedding.csv', sep=',')
uv_sgd_movie = pd.read_csv(r'data/uv_sgd_movie_embedding.csv', sep=',')

# 피처 중복 방지 식별자 부여 (학습 때와 완벽히 일치해야 함)
svd_item_user.columns = ['user_id'] + [f'item_nb_u_{i}' for i in range(len(svd_item_user.columns)-1)]
svd_item_movie.columns = ['movie_id'] + [f'item_nb_m_{i}' for i in range(len(svd_item_movie.columns)-1)]
svd_row_user.columns = ['user_id'] + [f'row_mn_u_{i}' for i in range(len(svd_row_user.columns)-1)]
svd_row_movie.columns = ['movie_id'] + [f'row_mn_m_{i}' for i in range(len(svd_row_movie.columns)-1)]
uv_sgd_user.columns = ['user_id'] + [f'uv_sgd_u_{i}' for i in range(len(uv_sgd_user.columns)-1)]
uv_sgd_movie.columns = ['movie_id'] + [f'uv_sgd_m_{i}' for i in range(len(uv_sgd_movie.columns)-1)]

# ⭐ [중요] 이전 단계에서 저장한 ua.base(훈련셋) 기준의 통계 피처 로드 (Leakage 차단!)
user_stats = pd.read_csv('data/final_user_stats.csv')
movie_stats = pd.read_csv('data/final_movie_stats.csv')


# ==========================================
# [4단계] 테스트 마스터 피처 매트릭스 조립 (Merge)
# ==========================================
# ua.test를 뼈대로 두고 모든 피처를 좌우로 조인
test_df = pd.merge(ua_test[['user_id', 'movie_id', 'actual_rating']], cf_item_test[['user_id', 'movie_id', 'feat_item_cf']], on=['user_id', 'movie_id'], how='left')
test_df = pd.merge(test_df, cf_user_test[['user_id', 'movie_id', 'feat_user_cf']], on=['user_id', 'movie_id'], how='left')

# 임베딩 결합
test_df = pd.merge(test_df, svd_item_user, on='user_id', how='left')
test_df = pd.merge(test_df, svd_item_movie, on='movie_id', how='left')
test_df = pd.merge(test_df, svd_row_user, on='user_id', how='left')
test_df = pd.merge(test_df, svd_row_movie, on='movie_id', how='left')
test_df = pd.merge(test_df, uv_sgd_user, on='user_id', how='left')
test_df = pd.merge(test_df, uv_sgd_movie, on='movie_id', how='left')

# 메타 통계 결합
test_df = pd.merge(test_df, user_stats, on='user_id', how='left')
test_df = pd.merge(test_df, movie_stats, on='movie_id', how='left')
test_df = test_df.fillna(0)


# ==========================================
# [5단계] 모델 투입용 문제지(X)와 정답지(y) 분리
# ==========================================
X_test = test_df.drop(columns=['user_id', 'movie_id', 'actual_rating'])
y_test = test_df['actual_rating']


# ==========================================
# [6단계] 저장된 최종 메타 모델 로드 및 예측 (70:30 블렌딩)
# ==========================================
print("3. 저장된 최종 하이브리드 메타 모델(LightGBM & RandomForest) 불러오는 중...")
lgb_model = joblib.load('data/final_hybrid_lgb.pkl')
rf_model = joblib.load('data/final_hybrid_rf.pkl')

print("4. 최종 앙상블 가중 블렌딩(LGBM 70% + RF 30%) 실전 예측 진행 중...")
lgb_preds = lgb_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# 황금 비율 결합 및 스케일 제약([1.0, 5.0])
final_test_preds = (lgb_preds * 0.7) + (rf_preds * 0.3)
final_test_preds = np.clip(final_test_preds, 1.0, 5.0)

# 평가용 데이터프레임 복사본 생성 후 예측치 병합
test_eval_df = test_df.copy()
test_eval_df['pred_rating'] = final_test_preds


# ==========================================
# [7단계] 최종 실전 성적표 출력 (오차, 분류, 랭킹 관점)
# ==========================================
test_rmse = np.sqrt(mean_squared_error(y_test, final_test_preds))
test_mae = mean_absolute_error(y_test, final_test_preds)

print("\n=====================================================")
print(f"📊 [최종 실전 TEST 데이터셋 평가 결과]")
print(f"📉 오차 관점  - 최종 하이브리드 RMSE: {test_rmse:.4f}")
print(f"📉 오차 관점  - 최종 하이브리드 MAE : {test_mae:.4f}")

# 분류 관점 (ROC-AUC)
binary_label = (y_test >= 4).astype(int)
fpr, tpr, _ = roc_curve(binary_label, final_test_preds)
test_auc_score = auc(fpr, tpr)
print(f"🎯 분류 관점  - 최종 실전 ROC-AUC  : {test_auc_score:.4f}")
print("=====================================================")

# Top-K 추천 성능 평가 (K=5)
K = 5
precisions, recalls, ndcgs = [], [], []

for user_id, group in test_eval_df.groupby('user_id'):
    # 유저가 실제로 좋아한 영화 (4점 이상)
    true_items = set(group[group['actual_rating'] >= 4]['movie_id'])
    if len(true_items) == 0:
        continue

    # 모델이 추천한 상위 5개 영화
    recs = group.sort_values(by='pred_rating', ascending=False)['movie_id'].head(K).tolist()

    precisions.append(precision_at_k(recs, true_items, K))
    recalls.append(recall_at_k(recs, true_items, K))
    ndcgs.append(ndcg_at_k(recs, true_items, K))

print("\n================ 🏆 실전 Top-K 추천 성능 ================")
print(f"👉 Precision@{K}: {np.mean(precisions):.4f}")
print(f"👉 Recall@{K}:    {np.mean(recalls):.4f}")
print(f"👉 NDCG@{K}:      {np.mean(ndcgs):.4f}")
print("=========================================================")


# ==========================================
# [8단계] 실전 테스트 결과 시각화 이미지 생성
# ==========================================
plt.figure(figsize=(10, 5))
sns.kdeplot(y_test, label='Actual Test Rating (실제 정답)', color='emerald' if 'emerald' in sns.colors.SEABORN_PALETTES else 'green', fill=True, linewidth=2)
sns.kdeplot(final_test_preds, label='Hybrid Ensemble Prediction (최종 예측)', color='purple', fill=True, linewidth=2)
plt.title('MovieLens TEST Set: Actual vs Hybrid Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Rating (평점)', fontsize=12)
plt.ylabel('Density (밀도)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.savefig('data/final_test_result.png', dpi=300, bbox_inches='tight')
print("\n📈 실전 테스트 결과 그래프가 'data/final_test_result.png'에 저장되었습니다.")