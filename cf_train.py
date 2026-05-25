import numpy as np
import pickle
import os
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

    for i in items:
        for u in item_dict[i]:
            if u not in user_mean:
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

    for i in train_dict[user]:
        if i not in sim:
            continue

        for j, s in sorted(sim[i].items(), key=lambda x: x[1], reverse=True)[:k]:
            if j not in train_dict[user]:
                scores[j] += s * train_dict[user][i]

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# =========================
# METRICS
# =========================
def precision_at_k(recs, true_items):
    rec_items = [i for i, _ in recs]
    return len(set(rec_items) & set(true_items)) / (len(rec_items) + 1e-8)


def dcg(rel):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at_k(recs, true_items, k):
    rec_items = [i for i, _ in recs[:k]]
    rel = [1 if i in true_items else 0 for i in rec_items]
    ideal = sorted(rel, reverse=True)
    return dcg(rel) / (dcg(ideal) + 1e-8)


def evaluate(recommender, sim, train_dict, val_dict, k):
    p = n = 0
    cnt = 0

    for u in val_dict:
        if u not in train_dict:
            continue

        true_items = set(val_dict[u].keys())
        recs = recommender(u, sim, train_dict, k)

        p += precision_at_k(recs, true_items)
        n += ndcg_at_k(recs, true_items, k)
        cnt += 1

    return p / cnt, n / cnt


# =========================
# RUN
# =========================
data = load_data("D:/HyeJeong/SungShin/3-1/RS/2026_RS/ml-100k/ml-100k/ua.base")

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_user, train_item = build_matrix(train_data)
val_user, _ = build_matrix(val_data)

item_train = build_item_matrix(train_data)

user_sim = compute_user_similarity(train_user, train_item)
item_sim = compute_item_similarity(item_train)

k_list = [5, 10, 20, 50, 100]

user_p_list, user_n_list = [], []
item_p_list, item_n_list = [], []

best_k_user = best_k_item = 10

print("===== USER CF =====")
for k in k_list:
    p, n = evaluate(recommend_user, user_sim, train_user, val_user, k)
    user_p_list.append(p)
    user_n_list.append(n)
    print(f"k={k} P={p:.4f} NDCG={n:.4f}")

print("\n===== ITEM CF =====")
for k in k_list:
    p, n = evaluate(recommend_item, item_sim, train_user, val_user, k)
    item_p_list.append(p)
    item_n_list.append(n)
    print(f"k={k} P={p:.4f} NDCG={n:.4f}")

# =========================
# ROC (VALIDATION ONLY)
# =========================
def get_roc(recommender, sim, train_dict, val_dict, k):
    scores, labels = [], []

    for u in val_dict:
        if u not in train_dict:
            continue

        true_items = set(val_dict[u].keys())
        recs = recommender(u, sim, train_dict, k)

        for i, s in recs:
            scores.append(s)
            labels.append(1 if i in true_items else 0)

    return scores, labels


u_s, u_l = get_roc(recommend_user, user_sim, train_user, val_user, best_k_user)
i_s, i_l = get_roc(recommend_item, item_sim, train_user, val_user, best_k_item)

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