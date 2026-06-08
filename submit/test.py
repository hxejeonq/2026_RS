#제출용_병합
# 20241172김나영_모델기반CF


# ================================================================================
# BEGIN: 00_baseline_test.py
# ================================================================================

from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd


# step0-1. 실험 설정
TOP_N = 10


# step0-2. 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "baseline_model.pkl"


def find_data_file(filename):
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "ml-100k" / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / "data" / "ml-100k" / filename,
        BASE_DIR / "raw" / "ml-100k" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{filename} not found. Searched paths:\n{searched}")


def load_ratings(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )


def build_test_dict(test_df):
    test = defaultdict(list)
    relevant = test_df[test_df["rating"] >= 4]
    for row in relevant.itertuples(index=False):
        test[int(row.userId)].append(int(row.movieId))
    return dict(test)


def predict_score(user_id, movie_id, model):
    return (
        model["global_mean"]
        + model["user_bias"].get(int(user_id), 0.0)
        + model["item_bias"].get(int(movie_id), 0.0)
    )


# step0-3. models/baseline_model.pkl 로드
model = joblib.load(MODEL_PATH)


# step0-4. ua.test 로드
ua_test_path = find_data_file("ua.test")
ua_test = load_ratings(ua_test_path)


# step0-5. test_dict 생성: rating >= 4만 relevant item
test = build_test_dict(ua_test)


# step0-6. baseline recommend_func 정의
def recommend_func(user_id, model):
    seen_items = model["ua_base_seen_items"].get(int(user_id), set())
    candidates = [
        int(movie_id)
        for movie_id in model["all_items"]
        if int(movie_id) not in seen_items
    ]
    scored_items = [
        (movie_id, predict_score(user_id, movie_id, model))
        for movie_id in candidates
    ]
    scored_items.sort(key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, _ in scored_items[:TOP_N]]


# step0-7. test 평가 함수 정의
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


def evaluate(model, test, recommend_func):
    all_recs = {}
    item_cnt = defaultdict(int)
    p = 0

    for user_id in test:
        recs = recommend_func(user_id, model)
        all_recs[user_id] = recs
        p += precision(recs, test[user_id])
        for movie_id in recs:
            item_cnt[movie_id] += 1

    p /= len(test)
    return {
        "precision": p,
        "novelty": novelty(all_recs, item_cnt),
        "diversity": diversity(all_recs),
        "coverage": coverage(all_recs, len(model["all_items"])),
    }


# step0-8. ua.test 기준 최종 성능 평가
result = evaluate(model, test, recommend_func)


# step0-9. 최종 test 성능 출력
print("\n[Baseline test result]")
print(f"Precision@10: {result['precision']:.6f}")
print(f"Novelty: {result['novelty']:.6f}")
print(f"Diversity: {result['diversity']:.6f}")
print(f"Coverage: {result['coverage']:.6f}")

# ================================================================================
# END: 00_baseline_test.py
# ================================================================================


# ================================================================================
# BEGIN: 01_uv_sgd_test.py
# ================================================================================

from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd


# step1-1. 실험 설정
TOP_N = 10


# step1-2. 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "uv_sgd_model.pkl"


def find_data_file(filename):
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "ml-100k" / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / "data" / "ml-100k" / filename,
        BASE_DIR / "raw" / "ml-100k" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{filename} not found. Searched paths:\n{searched}")


def load_ratings(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )


def build_test_dict(test_df):
    test = defaultdict(list)
    relevant = test_df[test_df["rating"] >= 4]
    for row in relevant.itertuples(index=False):
        test[int(row.userId)].append(int(row.movieId))
    return dict(test)


def clip_rating(pred):
    return float(np.clip(pred, 1.0, 5.0))


def predict_score(user_id, movie_id, model):
    user_to_idx = model["user_to_idx"]
    item_to_idx = model["item_to_idx"]
    user_id = int(user_id)
    movie_id = int(movie_id)

    pred = model["global_mean"]
    if user_id in user_to_idx:
        u_idx = user_to_idx[user_id]
        pred += model["user_bias"][u_idx]
    else:
        u_idx = None

    if movie_id in item_to_idx:
        i_idx = item_to_idx[movie_id]
        pred += model["item_bias"][i_idx]
    else:
        i_idx = None

    if u_idx is not None and i_idx is not None:
        pred += float(np.dot(model["U"][u_idx], model["V"][i_idx]))

    return clip_rating(pred)


# step1-3. 학습된 UV SGD 모델 로드
model = joblib.load(MODEL_PATH)


# step1-4. ua.test만 로드
ua_test_path = find_data_file("ua.test")
ua_test = load_ratings(ua_test_path)


# step1-5. rating >= 4를 정답 아이템으로 두고 test 딕셔너리 생성
test = build_test_dict(ua_test)


# step1-6. 추천 함수 정의
def recommend_func(user_id, model):
    seen_items = model["ua_base_seen_items"].get(int(user_id), set())
    candidates = [
        int(movie_id)
        for movie_id in model["all_items"]
        if int(movie_id) not in seen_items
    ]
    scored_items = [
        (movie_id, predict_score(user_id, movie_id, model))
        for movie_id in candidates
    ]
    scored_items.sort(key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, _ in scored_items[:TOP_N]]


# step1-7. test 공통 평가 지표
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


def evaluate(model, test, recommend_func):
    all_recs = {}
    item_cnt = defaultdict(int)
    p = 0

    for user_id in test:
        recs = recommend_func(user_id, model)
        all_recs[user_id] = recs
        p += precision(recs, test[user_id])
        for movie_id in recs:
            item_cnt[movie_id] += 1

    p /= len(test)
    return {
        "precision": p,
        "novelty": novelty(all_recs, item_cnt),
        "diversity": diversity(all_recs),
        "coverage": coverage(all_recs, len(model["all_items"])),
    }


# step1-8. ua.test 기준 최종 평가
result = evaluate(model, test, recommend_func)


# step1-9. 최종 test 결과 출력
print("\n[UV SGD test result]")
print(f"selected_latent_k: {model['latent_k']}")
print(f"best_epoch: {model['best_epoch']}")
print(f"Precision@10: {result['precision']:.6f}")
print(f"Novelty: {result['novelty']:.6f}")
print(f"Diversity: {result['diversity']:.6f}")
print(f"Coverage: {result['coverage']:.6f}")

# ================================================================================
# END: 01_uv_sgd_test.py
# ================================================================================


# ================================================================================
# BEGIN: 02_svd_row_mean_test.py
# ================================================================================

from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd


# step2-1. 실험 설정
TOP_N = 10


# step2-2. 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "svd_row_mean_model.pkl"


def find_data_file(filename):
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "ml-100k" / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / "data" / "ml-100k" / filename,
        BASE_DIR / "raw" / "ml-100k" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{filename} not found. Searched paths:\n{searched}")


def load_ratings(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )


def build_test_dict(test_df):
    test = defaultdict(list)
    relevant = test_df[test_df["rating"] >= 4]
    for row in relevant.itertuples(index=False):
        test[int(row.userId)].append(int(row.movieId))
    return dict(test)


def clip_rating(pred):
    return float(np.clip(pred, 1.0, 5.0))


def predict_score(user_id, movie_id, model):
    user_to_idx = model["user_to_idx"]
    item_to_idx = model["item_to_idx"]

    if int(user_id) in user_to_idx and int(movie_id) in item_to_idx:
        pred = model["prediction_matrix"][user_to_idx[int(user_id)], item_to_idx[int(movie_id)]]
    elif int(user_id) in user_to_idx:
        pred = model["row_means"][user_to_idx[int(user_id)]]
    else:
        pred = model["global_mean"]

    return clip_rating(pred)


# step2-3. 학습된 SVD 행 평균 초기화 모델 로드
model = joblib.load(MODEL_PATH)


# step2-4. ua.test만 로드
ua_test_path = find_data_file("ua.test")
ua_test = load_ratings(ua_test_path)


# step2-5. rating >= 4를 정답 아이템으로 두고 test 딕셔너리 생성
test = build_test_dict(ua_test)


# step2-6. 추천 함수 정의
def recommend_func(user_id, model):
    seen_items = model["ua_base_seen_items"].get(int(user_id), set())
    candidates = [
        int(movie_id)
        for movie_id in model["all_items"]
        if int(movie_id) not in seen_items
    ]
    scored_items = [
        (movie_id, predict_score(user_id, movie_id, model))
        for movie_id in candidates
    ]
    scored_items.sort(key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, _ in scored_items[:TOP_N]]


# step2-7. test 공통 평가 지표
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


def evaluate(model, test, recommend_func):
    all_recs = {}
    item_cnt = defaultdict(int)
    p = 0

    for user_id in test:
        recs = recommend_func(user_id, model)
        all_recs[user_id] = recs
        p += precision(recs, test[user_id])
        for movie_id in recs:
            item_cnt[movie_id] += 1

    p /= len(test)
    return {
        "precision": p,
        "novelty": novelty(all_recs, item_cnt),
        "diversity": diversity(all_recs),
        "coverage": coverage(all_recs, len(model["all_items"])),
    }


# step2-8. ua.test 기준 최종 평가
result = evaluate(model, test, recommend_func)


# step2-9. 최종 test 결과 출력
print("\n[SVD row-mean test result]")
print(f"selected_rank_k: {model['rank_k']}")
print(f"Precision@10: {result['precision']:.6f}")
print(f"Novelty: {result['novelty']:.6f}")
print(f"Diversity: {result['diversity']:.6f}")
print(f"Coverage: {result['coverage']:.6f}")

# ================================================================================
# END: 02_svd_row_mean_test.py
# ================================================================================


# ================================================================================
# BEGIN: 03_svd_item_neighbor_test.py
# ================================================================================

from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd


# step3-1. 실험 설정
TOP_N = 10


# step3-2. 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "svd_item_neighbor_model.pkl"


def find_data_file(filename):
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "ml-100k" / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / "data" / "ml-100k" / filename,
        BASE_DIR / "raw" / "ml-100k" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{filename} not found. Searched paths:\n{searched}")


def load_ratings(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )


def build_test_dict(test_df):
    test = defaultdict(list)
    relevant = test_df[test_df["rating"] >= 4]
    for row in relevant.itertuples(index=False):
        test[int(row.userId)].append(int(row.movieId))
    return dict(test)


def clip_rating(pred):
    return float(np.clip(pred, 1.0, 5.0))


def predict_score(user_id, movie_id, model):
    user_to_idx = model["user_to_idx"]
    item_to_idx = model["item_to_idx"]

    if int(user_id) in user_to_idx and int(movie_id) in item_to_idx:
        pred = model["prediction_matrix"][user_to_idx[int(user_id)], item_to_idx[int(movie_id)]]
    elif int(user_id) in user_to_idx:
        pred = model["user_means"][user_to_idx[int(user_id)]]
    else:
        pred = model["global_mean"]

    return clip_rating(pred)


# step3-3. 학습된 SVD 아이템 이웃 초기화 모델 로드
model = joblib.load(MODEL_PATH)


# step3-4. ua.test만 로드
ua_test_path = find_data_file("ua.test")
ua_test = load_ratings(ua_test_path)


# step3-5. rating >= 4를 정답 아이템으로 두고 test 딕셔너리 생성
test = build_test_dict(ua_test)


# step3-6. 추천 함수 정의
def recommend_func(user_id, model):
    seen_items = model["ua_base_seen_items"].get(int(user_id), set())
    candidates = [
        int(movie_id)
        for movie_id in model["all_items"]
        if int(movie_id) not in seen_items
    ]
    scored_items = [
        (movie_id, predict_score(user_id, movie_id, model))
        for movie_id in candidates
    ]
    scored_items.sort(key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, _ in scored_items[:TOP_N]]


# step3-7. test 공통 평가 지표
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


def evaluate(model, test, recommend_func):
    all_recs = {}
    item_cnt = defaultdict(int)
    p = 0

    for user_id in test:
        recs = recommend_func(user_id, model)
        all_recs[user_id] = recs
        p += precision(recs, test[user_id])
        for movie_id in recs:
            item_cnt[movie_id] += 1

    p /= len(test)
    return {
        "precision": p,
        "novelty": novelty(all_recs, item_cnt),
        "diversity": diversity(all_recs),
        "coverage": coverage(all_recs, len(model["all_items"])),
    }


# step3-8. ua.test 기준 최종 평가
result = evaluate(model, test, recommend_func)


# step3-9. 최종 test 결과 출력
print("\n[SVD item-neighbor test result]")
print(f"selected_rank_k: {model['rank_k']}")
print(f"neighbor_k: {model['neighbor_k']}")
print(f"Precision@10: {result['precision']:.6f}")
print(f"Novelty: {result['novelty']:.6f}")
print(f"Diversity: {result['diversity']:.6f}")
print(f"Coverage: {result['coverage']:.6f}")

# ================================================================================
# END: 03_svd_item_neighbor_test.py
# ================================================================================


