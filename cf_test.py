import pickle
import numpy as np
from collections import defaultdict


total_item_count = 1680
base_path = r"D:\HyeJeong\SungShin\3-1\RS\2026_RS"


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

    scores = defaultdict(float)

    if user not in train:
        return []

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
def precision(recs, true):
    return len(set(recs) & set(true)) / (len(recs) + 1e-8)


def novelty(all_recs, item_cnt):
    return np.mean([-np.log(item_cnt[i] + 1) for u in all_recs for i in all_recs[u]])


def diversity(all_recs):
    users = list(all_recs.keys())
    score = []
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            score.append(len(set(all_recs[users[i]]) & set(all_recs[users[j]])))
    return np.mean(score)


def coverage(all_recs, total_item_count):
    all_items = set()
    for user in all_recs:
        for item in all_recs[user]:
            all_items.add(item)
    return len(all_items) / total_item_count


# =========================
# 평가 함수 (공통)
# =========================
def evaluate(model, test, recommend_func):
    all_recs = {}
    item_cnt = defaultdict(int)

    p = 0

    for u in test:
        recs = recommend_func(u, model)
        all_recs[u] = recs

        p += precision(recs, test[u])

        for i in recs:
            item_cnt[i] += 1

    p /= len(test)

    return {
        "precision": p,
        "novelty": novelty(all_recs, item_cnt),
        "diversity": diversity(all_recs),
        "coverage": coverage(all_recs, total_item_count)
    }, all_recs


def save_recommendations(all_recs, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for user, recs in all_recs.items():
            rec_str = ", ".join(map(str, recs))
            f.write(f"User {user}: [{rec_str}]\n")

    print("Saved to:", filename)

    
# =========================
# LOAD
# =========================
test = load_test(r"D:\HyeJeong\SungShin\3-1\RS\2026_RS\ml-100k\ml-100k\ua.test")

model_user = pickle.load(open("results/model_user.pkl", "rb"))
model_item = pickle.load(open("results/model_item.pkl", "rb"))


# =========================
# RUN
# =========================
result_user, recs_user = evaluate(model_user, test, recommend_user)
result_item, recs_item = evaluate(model_item, test, recommend_item)


# =========================
# PRINT & SAVE
# =========================
print("===== USER CF =====")
print("Precision:", result_user["precision"])
print("Novelty:", result_user["novelty"])
print("Diversity:", result_user["diversity"])
print("Coverage:", result_user["coverage"])

print("\n===== ITEM CF =====")
print("Precision:", result_item["precision"])
print("Novelty:", result_item["novelty"])
print("Diversity:", result_item["diversity"])
print("Coverage:", result_item["coverage"])
print("\n")

save_recommendations(recs_user, base_path + r"\results" + r"\recommend_user.txt")
save_recommendations(recs_item, base_path + r"\results" + r"\recommend_item.txt")