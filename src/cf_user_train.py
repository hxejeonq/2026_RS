import numpy as np
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split


# ---------------------------
# Data loading
# ---------------------------
def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            user, item, rating, _ = line.strip().split('\t')
            data.append((int(user), int(item), float(rating)))
    return data


# ---------------------------
# user-item matrix
# ---------------------------
def build_matrix(data):
    user_dict = defaultdict(dict)
    item_dict = defaultdict(set)

    for u, i, r in data:
        user_dict[u][i] = r
        item_dict[i].add(u)

    return user_dict, item_dict


# ---------------------------
# similarity (IUF + Significance Weighting)
# ---------------------------
def compute_user_similarity(user_dict, item_dict):
    users = list(user_dict.keys())
    user_sim = defaultdict(dict)

    # user mean
    user_mean = {
        u: np.mean(list(user_dict[u].values()))
        for u in users
    }

    N = len(users)

    # IUF
    def iuf(i):
        return np.log((N + 1) / (len(item_dict[i]) + 1))

    for u in users:
        items_u = set(user_dict[u])

        for v in users:
            if u == v:
                continue

            items_v = set(user_dict[v])
            common = items_u & items_v

            if len(common) == 0:
                user_sim[u][v] = 0
                continue

            num = 0
            denom_u = 0
            denom_v = 0

            for i in common:
                w = iuf(i)

                diff_u = user_dict[u][i] - user_mean[u]
                diff_v = user_dict[v][i] - user_mean[v]

                num += w * diff_u * diff_v
                denom_u += (w * diff_u) ** 2
                denom_v += (w * diff_v) ** 2

            denom = (np.sqrt(denom_u) * np.sqrt(denom_v)) + 1e-8
            sim = num / denom

            # Significance Weighting
            n = len(common)
            sig_weight = min(n, 50) / 50
            sim *= sig_weight

            user_sim[u][v] = sim

    return user_sim


# ---------------------------
# 추천 함수
# ---------------------------
def recommend(user, user_sim, train_dict, k=20, top_n=10):
    if user not in train_dict:
        return []

    sim_users = sorted(
        user_sim[user].items(),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    user_mean = np.mean(list(train_dict[user].values()))

    scores = defaultdict(float)
    sim_sum = defaultdict(float)

    for v, sim in sim_users:
        if v not in train_dict:
            continue

        v_mean = np.mean(list(train_dict[v].values()))

        for item, rating in train_dict[v].items():
            if item in train_dict[user]:
                continue

            scores[item] += sim * (rating - v_mean)
            sim_sum[item] += abs(sim)

    rankings = []

    for item in scores:
        if sim_sum[item] == 0:
            continue

        pred = user_mean + scores[item] / sim_sum[item]
        rankings.append((item, pred))

    rankings.sort(key=lambda x: x[1], reverse=True)

    return [i for i, _ in rankings[:top_n]]


# ---------------------------
# Precision@K
# ---------------------------
def precision_at_k(recs, true_items):
    if len(recs) == 0:
        return 0
    return len(set(recs) & set(true_items)) / len(recs)


# ---------------------------
# TRAIN / VALIDATION SPLIT
# ---------------------------
data = load_data(r"D:\HyeJeong\SungShin\3-1\RS\2026_RS\ml-100k\ml-100k\ua.base")

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_dict, train_item_dict = build_matrix(train_data)
val_dict, _ = build_matrix(val_data)


# ---------------------------
# similarity 계산
# ---------------------------
user_sim = compute_user_similarity(train_dict, train_item_dict)


# ---------------------------
# k 튜닝
# ---------------------------
k_list = [5, 10, 20, 50, 100, 200]

best_k = 0
best_score = 0

for k in k_list:
    total = 0
    cnt = 0

    print(f"\n===== k = {k} =====")

    for user in val_dict:
        if user not in train_dict:
            continue

        recs = recommend(user, user_sim, train_dict, k=k)
        true_items = list(val_dict[user].keys())

        total += precision_at_k(recs, true_items)
        cnt += 1

    score = total / cnt if cnt > 0 else 0
    print(f"Precision@{k}: {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k


print("\nBEST K:", best_k)
print("BEST PRECISION:", best_score)


# ---------------------------
# final model
# ---------------------------
full_user_dict, full_item_dict = build_matrix(data)

final_user_sim = compute_user_similarity(full_user_dict, full_item_dict)


# ---------------------------
# save model
# ---------------------------
with open(r"D:\HyeJeong\SungShin\3-1\RS\2026_RS\model_user.pkl", "wb") as f:
    pickle.dump({
        "user_sim": final_user_sim,
        "train_dict": full_user_dict,
        "k": best_k
    }, f)

print("model saved!")