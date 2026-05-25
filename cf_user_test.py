import pickle
from collections import defaultdict
import numpy as np

# =========================
# 데이터 로드
# =========================
def load_data(path):
    data = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            user, item, __, _ = line.strip().split('\t')
            data[int(user)].append(int(item))
    return data


# =========================
# 추천 함수
# =========================
def recommend(user, user_sim, train_dict, k=20, top_n=10):
    if user not in user_sim:
        return []

    sim_users = sorted(user_sim[user].items(), key=lambda x: x[1], reverse=True)[:k]

    scores = defaultdict(float)
    for v, sim in sim_users:
        for item, rating in train_dict[v].items():
            if item not in train_dict[user]:
                scores[item] += sim * rating

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:top_n]]


# =========================
# Precision@K
# =========================
def precision_at_k(recs, true_items):
    if len(recs) == 0:
        return 0
    return len(set(recs) & set(true_items)) / len(recs)


# =========================
# Diversity (User overlap)
# =========================
def diversity(all_recs):
    users = list(all_recs.keys())
    overlaps = []

    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u1 = set(all_recs[users[i]])
            u2 = set(all_recs[users[j]])
            overlaps.append(len(u1 & u2))

    return np.mean(overlaps)


# =========================
# Novelty
# =========================
def novelty(all_recs, item_counter):
    scores = []

    for user in all_recs:
        for item in all_recs[user]:
            pop = item_counter[item]
            scores.append(-np.log(pop + 1))

    return np.mean(scores)


# =========================
# TEST
# =========================
with open(r"D:\HyeJeong\SungShin\3-1\RS\2026_RS\model.pkl", "rb") as f:
    model = pickle.load(f)

user_sim = model["user_sim"]
train_dict = model["train_dict"]
k = model["k"]

test_dict = load_data(r"D:\HyeJeong\SungShin\3-1\RS\2026_RS\ml-100k\ml-100k\ua.test")

all_recs = {}
item_counter = defaultdict(int)

total_prec = 0
cnt = 0
all_items = set()

# =========================
# 추천 생성
# =========================
for user in test_dict:
    recs = recommend(user, user_sim, train_dict, k=k)

    all_recs[user] = recs

    total_prec += precision_at_k(recs, test_dict[user])

    all_items.update(recs)

    for item in recs:
        item_counter[item] += 1

    cnt += 1


# =========================
# 평가
# =========================
precision = total_prec / cnt
total_item_count = 1680
coverage = len(all_items)
coverage_ratio = coverage / total_item_count * 100
avg_overlap = diversity(all_recs)
novelty_score = novelty(all_recs, item_counter)


# =========================
# 출력
# =========================
print("Precision:", precision)
print("Coverage:", coverage)
print(f"Coverage ratio: {coverage_ratio:.2f}%")
print("Novelty:", novelty_score)
print("Diversity:", avg_overlap)


# =========================
# 결과 저장
# =========================
with open(r"D:\HyeJeong\SungShin\3-1\RS\2026_RS\submission_user.txt", "w") as f:
    for user, recs in all_recs.items():
        rec_str = ", ".join(map(str, recs))
        f.write(f"User {user}: [{rec_str}]\n")