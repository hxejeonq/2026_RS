import pickle
import numpy as np
from collections import defaultdict

TOTAL_ITEM_COUNT = 1680


# =========================
# LOAD TEST
# =========================
def load_test(path):
    data = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            u, i, _, _ = line.strip().split('\t')
            data[int(u)].append(int(i))
    return data


# =========================
# USER CF
# =========================
def recommend_user(user, model):
    sim = model["sim"]
    train = model["train"]
    k = model["k"]

    if user not in sim:
        return []

    sim_users = sorted(sim[user].items(), key=lambda x: x[1], reverse=True)[:k]

    scores = defaultdict(float)

    for v, s in sim_users:
        if v not in train:
            continue
        for item, r in train[v].items():
            if item not in train[user]:
                scores[item] += s * r

    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]]


# =========================
# ITEM CF
# =========================
def recommend_item(user, model):
    sim = model["sim"]
    train = model["train"]
    k = model["k"]

    if user not in train:
        return []

    scores = defaultdict(float)

    for item, r in train[user].items():
        if item not in sim:
            continue

        sim_items = sorted(sim[item].items(), key=lambda x: x[1], reverse=True)[:k]

        for j, s in sim_items:
            if j not in train[user]:
                scores[j] += s * r

    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]]


# =========================
# METRICS
# =========================
def precision_at_k(recs, true):
    return len(set(recs) & set(true)) / (len(recs) + 1e-8)


def novelty(all_recs, item_cnt):
    scores = []
    for u in all_recs:
        for i in all_recs[u]:
            scores.append(-np.log(item_cnt[i] + 1))
    return np.mean(scores)


def diversity(all_recs):
    users = list(all_recs.keys())
    sims = []

    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            sims.append(len(set(all_recs[users[i]]) & set(all_recs[users[j]])))

    return np.mean(sims) if sims else 0


def coverage(all_recs, total_items):
    items = set()
    for u in all_recs:
        items.update(all_recs[u])
    return len(items) / total_items


def rmse(model, test, recommend_func):
    se = 0
    n = 0

    for u in test:
        recs = recommend_func(u, model)

        rec_set = set(recs)

        for item in test[u]:
            pred = 1.0 if item in rec_set else 0.0
            se += (1 - pred) ** 2
            n += 1

    return np.sqrt(se / (n + 1e-8))


# =========================
# EVALUATION
# =========================
def evaluate(model, test, recommend_func):
    all_recs = {}
    item_cnt = defaultdict(int)

    for u in test:
        recs = recommend_func(u, model)
        all_recs[u] = recs

        for i in recs:
            item_cnt[i] += 1

    precision = np.mean([
        precision_at_k(all_recs[u], test[u]) for u in test
    ])

    return {
        "precision": precision,
        "novelty": novelty(all_recs, item_cnt),
        "diversity": diversity(all_recs),
        "coverage": coverage(all_recs, TOTAL_ITEM_COUNT)
    }


# =========================
# RUN ONLY EVAL
# =========================
test = load_test(r"ml-100k\ua.test")

model_user = pickle.load(open("results/model_user.pkl", "rb"))
model_item = pickle.load(open("results/model_item.pkl", "rb"))

print("===== USER CF =====")
res_user = evaluate(model_user, test, recommend_user)
rmse_user = rmse(model_user, test, recommend_user)

print(res_user)
print("RMSE:", rmse_user)

print("\n===== ITEM CF =====")
res_item = evaluate(model_item, test, recommend_item)
rmse_item = rmse(model_item, test, recommend_item)

print(res_item)
print("RMSE:", rmse_item)