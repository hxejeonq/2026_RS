import numpy as np
import pickle
import os
import csv
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# =========================
# DATA
# =========================
def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            u, i, r, _ = line.strip().split('\t')
            data.append((int(u), int(i), float(r)))
    return data


def build_matrix(data):
    user_dict = defaultdict(dict)
    item_dict = defaultdict(set)

    for u, i, r in data:
        user_dict[u][i] = r
        item_dict[i].add(u)

    return user_dict, item_dict


def build_item_matrix(data):
    item_dict = defaultdict(dict)
    for u, i, r in data:
        item_dict[i][u] = r
    return item_dict


# =========================
# USER SIM
# =========================
def compute_user_similarity(user_dict, item_dict):
    users = list(user_dict.keys())
    user_sim = defaultdict(dict)

    user_mean = {u: np.mean(list(user_dict[u].values())) for u in users}
    N = len(users)

    def iuf(i):
        return np.log((N + 1) / (len(item_dict[i]) + 1))

    for u in users:
        for v in users:
            if u == v:
                continue

            common = set(user_dict[u]) & set(user_dict[v])
            if not common:
                continue

            num = du = dv = 0

            for i in common:
                w = iuf(i)
                du_v = user_dict[u][i] - user_mean[u]
                dv_v = user_dict[v][i] - user_mean[v]

                num += w * du_v * dv_v
                du += (w * du_v) ** 2
                dv += (w * dv_v) ** 2

            sim = num / (np.sqrt(du) * np.sqrt(dv) + 1e-8)
            sig = min(len(common), 50) / 50
            user_sim[u][v] = sim * sig

    return user_sim


# =========================
# ITEM SIM
# =========================
def compute_item_similarity(item_dict, lambda_shrink=10, min_common=5):
    items = list(item_dict.keys())
    item_sim = defaultdict(dict)

    user_mean = {}

    for u in set(u for i in item_dict for u in item_dict[i]):
        ratings = []
        for it in item_dict:
            if u in item_dict[it]:
                ratings.append(item_dict[it][u])
        user_mean[u] = np.mean(ratings)

    for i in items:
        for j in items:
            if i >= j:
                continue

            common = set(item_dict[i]) & set(item_dict[j])
            if len(common) < min_common:
                continue

            num = di = dj = 0

            for u in common:
                num += (item_dict[i][u] - user_mean[u]) * (item_dict[j][u] - user_mean[u])
                di += (item_dict[i][u] - user_mean[u]) ** 2
                dj += (item_dict[j][u] - user_mean[u]) ** 2

            sim = num / (np.sqrt(di) * np.sqrt(dj) + 1e-8)
            sim *= len(common) / (len(common) + lambda_shrink)

            item_sim[i][j] = sim
            item_sim[j][i] = sim

    return item_sim


# =========================
# Rating Prediction
# ========================
def predict_user_rating(u, i, sim, train_dict):
    if u not in train_dict:
        return 0

    user_mean = np.mean(list(train_dict[u].values()))

    num, den = 0, 0

    for v, s in sim[u].items():
        if v in train_dict and i in train_dict[v]:
            v_mean = np.mean(list(train_dict[v].values()))
            num += s * (train_dict[v][i] - v_mean)
            den += abs(s)

    return user_mean + num / (den + 1e-8)


def predict_item_rating(u, i, sim, train_dict):
    if u not in train_dict:
        return 0

    user_mean = np.mean(list(train_dict[u].values()))

    num, den = 0, 0

    for j, r in train_dict[u].items():
        if j in sim and i in sim[j]:
            s = sim[j][i]
            num += s * (r - user_mean)
            den += abs(s)

    return user_mean + num / (den + 1e-8)


# =========================
# RECOMMEND
# =========================
def recommend_user(user, sim, train_dict, k):
    if user not in sim:
        return []

    sim_users = sorted(sim[user].items(), key=lambda x: x[1], reverse=True)[:k]

    user_mean = np.mean(list(train_dict[user].values()))
    scores = defaultdict(float)
    sim_sum = defaultdict(float)

    for v, s in sim_users:
        if v not in train_dict:
            continue

        v_mean = np.mean(list(train_dict[v].values()))

        for i, r in train_dict[v].items():
            if i in train_dict[user]:
                continue

            scores[i] += s * (r - v_mean)
            sim_sum[i] += abs(s)

    ranked = [(i, user_mean + scores[i] / (sim_sum[i] + 1e-8)) for i in scores]
    return sorted(ranked, key=lambda x: x[1], reverse=True)


def recommend_item(user, sim, train_dict, k):
    scores = defaultdict(float)

    if user not in train_dict:
        return []
    
    user_mean = np.mean(list(train_dict[user].values()))

    for i in train_dict[user]:
        if i not in sim:
            continue

        for j, s in sorted(sim[i].items(), key=lambda x: x[1], reverse=True)[:k]:
            if j not in train_dict[user]:
                scores[j] += s * (train_dict[user][i] - user_mean)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# =========================
# METRICS
# =========================
def precision_at_k(recs, true_items, k):
    rec_items = [i for i, _ in recs[:k]]
    return len(set(rec_items) & set(true_items)) / k


def dcg(rel):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at_k(recs, true_items, k):
    rec_items = [i for i, _ in recs[:k]]
    rel = [1 if i in true_items else 0 for i in rec_items]
    ideal = sorted(rel, reverse=True)
    return dcg(rel) / (dcg(ideal) + 1e-8)


def compute_rmse(predict_func, sim, train_dict, val_data):
    errors = []

    for u, i, r in val_data:
        if u not in train_dict:
            continue

        pred = predict_func(u, i, sim, train_dict)
        errors.append((r - pred) ** 2)

    return np.sqrt(np.mean(errors))


def evaluate(recommender, sim, train_dict, val_dict, k):
    p = n = 0
    cnt = 0

    for u in val_dict:
        if u not in train_dict:
            continue

        true_items = set(val_dict[u].keys())
        recs = recommender(u, sim, train_dict, k)

        p += precision_at_k(recs, true_items, k)
        n += ndcg_at_k(recs, true_items, k)
        cnt += 1

    return p / (cnt + 1e-8), n / (cnt + 1e-8)


# =========================
# RUN
# =========================
data = load_data("ml-100k/ua.base")

train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

train_user, train_item = build_matrix(train_data)
val_user, _ = build_matrix(val_data)

item_train = build_item_matrix(train_data)

user_sim = compute_user_similarity(train_user, train_item)
item_sim = compute_item_similarity(item_train)

k_list = [5, 10, 20, 50, 100]

user_p_list, user_n_list = [], []
item_p_list, item_n_list = [], []

# =========================
# SAVE VALIDATION PREDICTIONS
# =========================

# USER CF
with open("results/user_cf_val_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "item_id", "true_rating", "predicted_rating"])

    for u, i, r in val_data:
        if u not in train_user:
            continue

        pred = predict_user_rating(u, i, user_sim, train_user)
        writer.writerow([u, i, r, pred])


# ITEM CF
with open("results/item_cf_val_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "item_id", "true_rating", "predicted_rating"])

    for u, i, r in val_data:
        if u not in train_user:
            continue

        pred = predict_item_rating(u, i, item_sim, train_user)
        writer.writerow([u, i, r, pred])


print("validation prediction CSV 저장")

rmse_user = compute_rmse(predict_user_rating, user_sim, train_user, val_data)
rmse_item = compute_rmse(predict_item_rating, item_sim, train_user, val_data)

print("\n===== RMSE =====")
print("USER RMSE:", rmse_user)
print("ITEM RMSE:", rmse_item)

print("\n===== USER CF =====")
for k in k_list:
    p, n = evaluate(recommend_user, user_sim, train_user, val_user, k)
    user_p_list.append(p)
    user_n_list.append(n)
    print(f"k={k} P={p:.4f} NDCG={n:.4f}")

best_k_user = k_list[np.argmax(user_p_list)]
print(f"[BEST USER K] = {best_k_user}") 

print("\n===== ITEM CF =====")
for k in k_list:
    p, n = evaluate(recommend_item, item_sim, train_user, val_user, k)
    item_p_list.append(p)
    item_n_list.append(n)
    print(f"k={k} P={p:.4f} NDCG={n:.4f}")

best_k_item = k_list[np.argmax(item_p_list)]
print(f"[BEST ITEM K] = {best_k_item}")  

# =========================
# ROC
# =========================
def get_roc(predict_func, sim, train_dict, val_dict, all_items, sample_items=300, sample_users=300):
    scores, labels = [], []

    users = list(val_dict.keys())
    sampled_users = random.sample(users, min(sample_users, len(users)))

    for u in sampled_users:
        if u not in train_dict:
            continue

        true_items = set(val_dict[u].keys())

        # 아이템 샘플링
        sampled_items = random.sample(all_items, min(sample_items, len(all_items)))

        for i in sampled_items:
            if i in train_dict[u]:
                continue

            s = predict_func(u, i, sim, train_dict)
            scores.append(s)
            labels.append(1 if i in true_items else 0)

    return scores, labels


all_items = list(train_item.keys())

u_s, u_l = get_roc(predict_user_rating, user_sim, train_user, val_user, all_items)
i_s, i_l = get_roc(predict_item_rating, item_sim, train_user, val_user, all_items)

fpr_u, tpr_u, _ = roc_curve(u_l, u_s)
fpr_i, tpr_i, _ = roc_curve(i_l, i_s)

auc_u = auc(fpr_u, tpr_u)
auc_i = auc(fpr_i, tpr_i)

print("\nUSER AUC:", auc_u)
print("ITEM AUC:", auc_i)

plt.figure()
plt.plot(fpr_u, tpr_u, label=f"User CF AUC={auc_u:.4f}")
plt.plot(fpr_i, tpr_i, label=f"Item CF AUC={auc_i:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.legend()
plt.title("ROC Curve")
plt.savefig("results/roc.png", dpi=300)


# =========================
# SAVE
# =========================
with open("results/model_user.pkl", "wb") as f:
    pickle.dump({"sim": user_sim, "train": train_user, "k": best_k_user}, f)

with open("results/model_item.pkl", "wb") as f:
    pickle.dump({"sim": item_sim, "train": train_user, "k": best_k_item}, f)

plt.figure()
plt.plot(k_list, user_p_list, marker='o', label="User CF")
plt.plot(k_list, item_p_list, marker='o', label="Item CF")
plt.title("Precision@K Comparison")
plt.xlabel("K")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.savefig("results/precision.png", dpi=300)

plt.figure()
plt.plot(k_list, user_n_list, marker='o', label="User CF")
plt.plot(k_list, item_n_list, marker='o', label="Item CF")
plt.title("NDCG@K Comparison")
plt.xlabel("K")
plt.ylabel("NDCG")
plt.legend()
plt.grid(True)
plt.savefig("results/ndcg.png", dpi=300)

print("done")