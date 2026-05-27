import numpy as np
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ---------------------------
# 데이터 로드 (user, item, rating)
# ---------------------------
def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            user, item, rating, _ = line.strip().split('\t')
            data.append((int(user), int(item), float(rating)))
    return data

# ---------------------------
# item-item matrix 생성
# ---------------------------
def build_item_matrix(data):
    item_dict = defaultdict(dict)

    for u, i, r in data:
        item_dict[i][u] = r

    return item_dict

# ---------------------------
# item similarity (Adjusted Cosine)
# ---------------------------
def compute_item_similarity(item_dict, lambda_shrink=10, min_common=5):
    items = list(item_dict.keys())
    item_sim = defaultdict(dict)

    # user mean 계산
    user_mean = {}

    for i in items:
        for u in item_dict[i]:
            if u not in user_mean:
                # user 전체 평균
                all_ratings = []
                for item in item_dict:
                    if u in item_dict[item]:
                        all_ratings.append(item_dict[item][u])
                user_mean[u] = np.mean(all_ratings)

    for i in items:
        for j in items:
            if i >= j:
                continue

            common = set(item_dict[i]) & set(item_dict[j])

            if len(common) < min_common:
                continue

            # adjusted cosine (user-based centering)
            num = sum(
                (item_dict[i][u] - user_mean[u]) *
                (item_dict[j][u] - user_mean[u])
                for u in common
            )

            denom_i = np.sqrt(sum(
                (item_dict[i][u] - user_mean[u]) ** 2
                for u in common
            ))

            denom_j = np.sqrt(sum(
                (item_dict[j][u] - user_mean[u]) ** 2
                for u in common
            ))

            sim = num / (denom_i * denom_j + 1e-8)

            # shrinkage
            sim = (len(common) / (len(common) + lambda_shrink)) * sim

            item_sim[i][j] = sim
            item_sim[j][i] = sim

    return item_sim

# ---------------------------
# 추천 함수 (item-based)
# ---------------------------
def recommend(user, item_sim, train_dict, k=20, top_n=10):
    scores = defaultdict(float)

    user_items = train_dict[user]

    for item in user_items:
        if item not in item_sim:
            continue

        sim_items = sorted(item_sim[item].items(), key=lambda x: x[1], reverse=True)[:k]

        for j, sim in sim_items:
            if j not in user_items:
                scores[j] += sim * user_items[item]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:top_n]]

# ---------------------------
# Precision@K
# ---------------------------
def precision_at_k(recs, true_items):
    if len(recs) == 0:
        return 0
    return len(set(recs) & set(true_items)) / len(recs)

# ---------------------------
# TRAIN
# ---------------------------
data = load_data(r"D:\HyeJeong\SungShin\3-1\RS\2026_RS\ml-100k\ml-100k\ua.base")

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_dict = build_item_matrix(train_data)
val_dict = build_item_matrix(val_data)

item_sim = compute_item_similarity(train_dict)

# ---------------------------
# k 튜닝
# ---------------------------
best_k = 0
best_score = 0

k_list = [5, 10, 20, 50, 100, 200]

for k in k_list:
    total = 0
    cnt = 0

    print(f"\n===== k = {k} =====")

    for user in val_dict:
        if user not in train_dict:
            continue

        recs = recommend(user, item_sim, train_dict, k=k)
        val_items = list(val_dict[user].keys())

        p = precision_at_k(recs, val_items)

        total += p
        cnt += 1

    score = total / cnt if cnt > 0 else 0

    print(f"Precision: {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print("\nBest k:", best_k)
print("Best Precision:", best_score)

# ---------------------------
# 전체 train으로 재학습
# ---------------------------
full_train_dict = build_item_matrix(data)
final_item_sim = compute_item_similarity(full_train_dict)

# ---------------------------
# 모델 저장
# ---------------------------
with open(r"D:\HyeJeong\SungShin\3-1\RS\2026_RS\model_item.pkl", "wb") as f:
    pickle.dump({
        "item_sim": final_item_sim,
        "train_dict": full_train_dict,
        "k": best_k
    }, f)