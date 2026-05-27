from collections import Counter, defaultdict
import numpy as np
import re

# =========================
# 1. 파일 로드
# =========================
def load_recommendation(path):
    recs = defaultdict(list)

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("User"):
                user = int(re.findall(r'\d+', line.split(':')[0])[0])

                items_str = line.split(':')[1]
                items = re.findall(r'\d+', items_str)

                recs[user] = list(map(int, items))

    return recs


# =========================
# 2. 인기 아이템 분석 + popularity dict 반환
# =========================
def item_popularity(recs):
    counter = Counter()

    for user in recs:
        for item in recs[user]:
            counter[item] += 1

    print("\nTop 10 most recommended items:")
    for item, cnt in counter.most_common(10):
        print(f"Item {item}: {cnt} times")

    return counter


# =========================
# 3. 커버리지
# =========================
def coverage_analysis(recs, total_items=None):
    all_items = set()

    for user in recs:
        all_items.update(recs[user])

    coverage = len(all_items)

    print("\nCoverage")
    print(f"Unique items recommended: {coverage}")

    if total_items:
        ratio = coverage / total_items
        print(f"Coverage ratio: {ratio:.4f}")

    return coverage


# =========================
# 4. 참신성
# =========================
def novelty_analysis(recs, item_pop):
    novelty_scores = []

    for user in recs:
        for item in recs[user]:
            # popularity 기반 novelty
            pop = item_pop[item]
            novelty_scores.append(-np.log(pop + 1))

    avg_novelty = np.mean(novelty_scores)

    print("\nNovelty")
    print(f"Average novelty: {avg_novelty:.4f}")

    return avg_novelty


# =========================
# 5. 다양성
# =========================
def user_overlap(recs):
    users = list(recs.keys())
    overlaps = []

    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u1 = set(recs[users[i]])
            u2 = set(recs[users[j]])

            overlaps.append(len(u1 & u2))

    avg_overlap = np.mean(overlaps)

    print("\nDiversity")
    print(f"Average diversity: {avg_overlap:.2f}")

    if avg_overlap > 5:
        print("추천 리스트 유사도 높음 (다양성 낮음)")
    elif avg_overlap > 2:
        print("중간 수준 다양성")
    else:
        print("다양성 좋음")

    return avg_overlap


# =========================
# 6. 길이 체크
# =========================
def check_length(recs):
    lengths = [len(recs[u]) for u in recs]

    print("\nRecommendation length")
    print(f"min: {min(lengths)}")
    print(f"max: {max(lengths)}")
    print(f"mean: {np.mean(lengths):.2f}")


# =========================
# 7. 실행
# =========================
if __name__ == "__main__":
    path = r"D:\HyeJeong\SungShin\3-1\RS\2026_RS\recommend_user.txt"

    recs = load_recommendation(path)
    item_pop = item_popularity(recs)
    coverage_analysis(recs, total_items=1680)
    novelty_analysis(recs, item_pop)
    user_overlap(recs)
    check_length(recs)