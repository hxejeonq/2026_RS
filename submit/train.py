#제출용_병합
# 20241172김나영_모델기반CF

# ================================================================================
# BEGIN: 00_baseline_train.py
# ================================================================================

from pathlib import Path
from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split


# step0-1. 실험 설정
RANDOM_STATE = 42
TEST_SIZE_FOR_VALIDATION = 0.1
TOP_N = 10

LEARNING_RATE = 0.005
REGULARIZATION = 0.02
N_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 3


# step0-2. 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
MODEL_PATH = MODEL_DIR / "baseline_model.pkl"


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


def build_user_item_dict(df, relevant_only=False):
    data = defaultdict(dict)
    source = df[df["rating"] >= 4] if relevant_only else df
    for row in source.itertuples(index=False):
        data[int(row.userId)][int(row.movieId)] = float(row.rating)
    return dict(data)


def predict_score(user_id, movie_id, global_mean, user_bias, item_bias):
    return (
        global_mean
        + user_bias.get(int(user_id), 0.0)
        + item_bias.get(int(movie_id), 0.0)
    )


# =========================
# 평가 지표
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


def rmse_on_validation(val_df, global_mean, user_bias, item_bias):
    preds = [
        predict_score(row.userId, row.movieId, global_mean, user_bias, item_bias)
        for row in val_df.itertuples(index=False)
    ]
    y_true = val_df["rating"].to_numpy(dtype=float)
    y_pred = np.array(preds, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def recommend_baseline(user_id, model, seen_items, k):
    candidates = [i for i in model["all_items"] if i not in seen_items.get(user_id, set())]
    recs = [
        (
            int(movie_id),
            predict_score(
                user_id,
                movie_id,
                model["global_mean"],
                model["user_bias"],
                model["item_bias"],
            ),
        )
        for movie_id in candidates
    ]
    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:k]


def evaluate_validation(model, train_seen_items, val_dict, k):
    p = n = 0.0
    cnt = 0
    for user_id in val_dict:
        if user_id not in train_seen_items:
            continue
        true_items = set(val_dict[user_id].keys())
        recs = recommend_baseline(user_id, model, train_seen_items, k)
        p += precision_at_k(recs, true_items)
        n += ndcg_at_k(recs, true_items, k)
        cnt += 1
    if cnt == 0:
        return 0.0, 0.0
    return p / cnt, n / cnt


# =========================
# ROC 계산(validation 전용)
# =========================
def get_roc_validation(model, train_seen_items, val_dict, k):
    scores, labels = [], []
    for user_id in val_dict:
        if user_id not in train_seen_items:
            continue
        true_items = set(val_dict[user_id].keys())
        recs = recommend_baseline(user_id, model, train_seen_items, k)
        for movie_id, score in recs:
            scores.append(score)
            labels.append(1 if movie_id in true_items else 0)
    return scores, labels


# step0-3. ua.base 로드
ua_base_path = find_data_file("ua.base")
ua_base = load_ratings(ua_base_path)


# step0-4. ua.base를 train/validation = 9:1로 랜덤 분할
train_df, val_df = train_test_split(
    ua_base,
    test_size=TEST_SIZE_FOR_VALIDATION,
    random_state=RANDOM_STATE,
    shuffle=True,
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


# step0-5. train 데이터만 사용해 global_mean 계산
global_mean = float(train_df["rating"].mean())


# step0-6. baseline bias 반복 업데이트 준비
user_bias = {int(user_id): 0.0 for user_id in train_df["userId"].unique()}
item_bias = {int(movie_id): 0.0 for movie_id in train_df["movieId"].unique()}
rng = np.random.default_rng(RANDOM_STATE)

best_rmse = float("inf")
best_epoch = 0
best_user_bias = user_bias.copy()
best_item_bias = item_bias.copy()
patience_count = 0
history = []


# step0-7. train 데이터만 사용해 user_bias, item_bias 반복 학습
for epoch in range(1, N_EPOCHS + 1):
    shuffled_idx = rng.permutation(len(train_df))
    shuffled_train = train_df.iloc[shuffled_idx]

    for row in shuffled_train.itertuples(index=False):
        user_id = int(row.userId)
        movie_id = int(row.movieId)
        rating = float(row.rating)

        pred = predict_score(user_id, movie_id, global_mean, user_bias, item_bias)
        error = rating - pred

        user_bias[user_id] += LEARNING_RATE * (
            error - REGULARIZATION * user_bias[user_id]
        )
        item_bias[movie_id] += LEARNING_RATE * (
            error - REGULARIZATION * item_bias[movie_id]
        )

    val_rmse = rmse_on_validation(val_df, global_mean, user_bias, item_bias)
    history.append({"epoch": epoch, "validation_rmse": val_rmse})
    print(f"epoch={epoch:02d}, validation_rmse={val_rmse:.6f}")

    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_epoch = epoch
        best_user_bias = user_bias.copy()
        best_item_bias = item_bias.copy()
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= EARLY_STOPPING_PATIENCE:
            print(f"early stopping at epoch={epoch}")
            break


# step0-8. validation 평가용 데이터 구조 생성
train_user = build_user_item_dict(train_df, relevant_only=False)
val_user = build_user_item_dict(val_df, relevant_only=True)
train_seen_items = {
    user_id: set(items.keys()) for user_id, items in train_user.items()
}
ua_base_user = build_user_item_dict(ua_base, relevant_only=False)
ua_base_seen_items = {
    user_id: set(items.keys()) for user_id, items in ua_base_user.items()
}
all_items = sorted(int(movie_id) for movie_id in ua_base["movieId"].unique())


# step0-9. best bias 기준 모델 구성
model = {
    "model_name": "baseline_bias",
    "global_mean": global_mean,
    "user_bias": best_user_bias,
    "item_bias": best_item_bias,
    "all_items": all_items,
    "train_seen_items": train_seen_items,
    "ua_base_seen_items": ua_base_seen_items,
    "random_state": RANDOM_STATE,
    "learning_rate": LEARNING_RATE,
    "regularization": REGULARIZATION,
    "n_epochs": N_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "best_epoch": best_epoch,
    "best_validation_rmse": best_rmse,
    "history": history,
}


# step0-10. validation 기준 Precision@10, NDCG@10 계산
precision_10, ndcg_10 = evaluate_validation(model, train_seen_items, val_user, TOP_N)


# step0-11. validation 기준 ROC/AUC 계산 및 저장
RESULT_DIR.mkdir(parents=True, exist_ok=True)
roc_scores, roc_labels = get_roc_validation(model, train_seen_items, val_user, TOP_N)
if len(set(roc_labels)) == 2:
    fpr, tpr, _ = roc_curve(roc_labels, roc_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"Baseline AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("Baseline ROC Curve")
    plt.savefig(RESULT_DIR / "baseline_roc.png", dpi=300)
    plt.close()
else:
    roc_auc = float("nan")
    print("validation ROC/AUC skipped because labels contain only one class.")


# step0-12. 학습 완료 모델 저장
model["validation_precision_at_10"] = precision_10
model["validation_ndcg_at_10"] = ndcg_10
model["validation_auc"] = roc_auc
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)


# step0-13. validation 결과 출력
print("\n[Baseline validation result]")
print(f"best_epoch: {best_epoch}")
print(f"RMSE: {best_rmse:.6f}")
print(f"Precision@10: {precision_10:.6f}")
print(f"NDCG@10: {ndcg_10:.6f}")
print(f"AUC: {roc_auc:.6f}")
print(f"model saved to: {MODEL_PATH}")

# ================================================================================
# END: 00_baseline_train.py
# ================================================================================


# ================================================================================
# BEGIN: 01_uv_sgd_train.py
# ================================================================================

from pathlib import Path
from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, roc_curve
from sklearn.model_selection import train_test_split


# step1-1. 실험 설정
RANDOM_STATE = 42
TEST_SIZE_FOR_VALIDATION = 0.1
TOP_N = 10
K_CANDIDATES = [10, 20, 50, 100]
LEARNING_RATE = 0.005
REGULARIZATION = 0.02
N_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 3


# step1-2.경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
MODEL_PATH = MODEL_DIR / "uv_sgd_model.pkl"
PREDICTION_CSV_PATH = RESULT_DIR / "uv_sgd_validation_predictions.csv"
K_RESULT_CSV_PATH = RESULT_DIR / "uv_sgd_k_validation_results.csv"
K_METRIC_PLOT_PATH = RESULT_DIR / "uv_sgd_k_metrics.png"
K_ERROR_PLOT_PATH = RESULT_DIR / "uv_sgd_k_errors.png"
ROC_PLOT_PATH = RESULT_DIR / "uv_sgd_roc.png"


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


def build_user_item_dict(df, relevant_only=False):
    data = defaultdict(dict)
    source = df[df["rating"] >= 4] if relevant_only else df
    for row in source.itertuples(index=False):
        data[int(row.userId)][int(row.movieId)] = float(row.rating)
    return dict(data)


def make_mappings(ua_base):
    user_ids = sorted(int(user_id) for user_id in ua_base["userId"].unique())
    movie_ids = sorted(int(movie_id) for movie_id in ua_base["movieId"].unique())
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    return user_ids, movie_ids, user_to_idx, item_to_idx


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


def evaluate_rmse_mae(model, val_df):
    preds = [
        predict_score(row.userId, row.movieId, model)
        for row in val_df.itertuples(index=False)
    ]
    y_true = val_df["rating"].to_numpy(dtype=float)
    y_pred = np.array(preds, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def train_uv_sgd(train_df, val_df, user_ids, movie_ids, user_to_idx, item_to_idx, latent_k):
    rng = np.random.default_rng(RANDOM_STATE + latent_k)
    n_users = len(user_ids)
    n_items = len(movie_ids)

    global_mean = float(train_df["rating"].mean())
    U = rng.normal(0.0, 0.1, size=(n_users, latent_k))
    V = rng.normal(0.0, 0.1, size=(n_items, latent_k))
    user_bias = np.zeros(n_users, dtype=float)
    item_bias = np.zeros(n_items, dtype=float)

    train_rows = [
        (
            user_to_idx[int(row.userId)],
            item_to_idx[int(row.movieId)],
            float(row.rating),
        )
        for row in train_df.itertuples(index=False)
    ]

    best_rmse = float("inf")
    best_state = None
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, N_EPOCHS + 1):
        order = rng.permutation(len(train_rows))

        for pos in order:
            u_idx, i_idx, rating = train_rows[pos]
            u_vec = U[u_idx].copy()
            i_vec = V[i_idx].copy()

            pred = global_mean + user_bias[u_idx] + item_bias[i_idx] + np.dot(u_vec, i_vec)
            error = rating - pred

            user_bias[u_idx] += LEARNING_RATE * (
                error - REGULARIZATION * user_bias[u_idx]
            )
            item_bias[i_idx] += LEARNING_RATE * (
                error - REGULARIZATION * item_bias[i_idx]
            )
            U[u_idx] += LEARNING_RATE * (error * i_vec - REGULARIZATION * u_vec)
            V[i_idx] += LEARNING_RATE * (error * u_vec - REGULARIZATION * i_vec)

        model = {
            "model_name": "uv_sgd",
            "latent_k": latent_k,
            "global_mean": global_mean,
            "user_bias": user_bias,
            "item_bias": item_bias,
            "U": U,
            "V": V,
            "user_ids": user_ids,
            "movie_ids": movie_ids,
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
        }
        val_rmse, val_mae = evaluate_rmse_mae(model, val_df)
        history.append({"epoch": epoch, "validation_rmse": val_rmse, "validation_mae": val_mae})
        print(
            f"latent_k={latent_k}, epoch={epoch:02d}, "
            f"validation_rmse={val_rmse:.6f}, validation_mae={val_mae:.6f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {
                "U": U.copy(),
                "V": V.copy(),
                "user_bias": user_bias.copy(),
                "item_bias": item_bias.copy(),
                "rmse": val_rmse,
                "mae": val_mae,
            }
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= EARLY_STOPPING_PATIENCE:
                print(f"early stopping latent_k={latent_k} at epoch={epoch}")
                break

    return {
        "global_mean": global_mean,
        "U": best_state["U"],
        "V": best_state["V"],
        "user_bias": best_state["user_bias"],
        "item_bias": best_state["item_bias"],
        "best_epoch": best_epoch,
        "best_validation_rmse": best_state["rmse"],
        "best_validation_mae": best_state["mae"],
        "history": history,
    }


# =========================
# 평가 지표
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


def recommend_uv(user_id, model, seen_items, k):
    candidates = [
        movie_id
        for movie_id in model["all_items"]
        if movie_id not in seen_items.get(int(user_id), set())
    ]
    recs = [
        (int(movie_id), predict_score(user_id, movie_id, model))
        for movie_id in candidates
    ]
    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:k]


def evaluate_validation(model, train_seen_items, val_dict, k):
    p = n = 0.0
    cnt = 0
    for user_id in val_dict:
        if user_id not in train_seen_items:
            continue
        true_items = set(val_dict[user_id].keys())
        recs = recommend_uv(user_id, model, train_seen_items, k)
        p += precision_at_k(recs, true_items)
        n += ndcg_at_k(recs, true_items, k)
        cnt += 1
    if cnt == 0:
        return 0.0, 0.0
    return p / cnt, n / cnt


def get_roc_validation(model, train_seen_items, val_dict, k):
    scores, labels = [], []
    for user_id in val_dict:
        if user_id not in train_seen_items:
            continue
        true_items = set(val_dict[user_id].keys())
        recs = recommend_uv(user_id, model, train_seen_items, k)
        for movie_id, score in recs:
            scores.append(score)
            labels.append(1 if movie_id in true_items else 0)
    return scores, labels


def save_validation_predictions(model, val_df, output_path):
    rows = []
    for row in val_df.itertuples(index=False):
        rows.append(
            {
                "userId": int(row.userId),
                "movieId": int(row.movieId),
                "actual_rating": float(row.rating),
                "predicted_rating": predict_score(row.userId, row.movieId, model),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_k_plots(k_result_df):
    plt.figure()
    plt.plot(k_result_df["latent_k"], k_result_df["precision_at_10"], marker="o", label="Precision@10")
    plt.plot(k_result_df["latent_k"], k_result_df["ndcg_at_10"], marker="o", label="NDCG@10")
    plt.plot(k_result_df["latent_k"], k_result_df["auc"], marker="o", label="AUC")
    plt.xlabel("latent_k")
    plt.ylabel("Validation score")
    plt.title("UV SGD Validation Ranking Metrics by latent_k")
    plt.xticks(k_result_df["latent_k"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(K_METRIC_PLOT_PATH, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(k_result_df["latent_k"], k_result_df["rmse"], marker="o", label="RMSE")
    plt.plot(k_result_df["latent_k"], k_result_df["mae"], marker="o", label="MAE")
    plt.xlabel("latent_k")
    plt.ylabel("Validation error")
    plt.title("UV SGD Validation Error Metrics by latent_k")
    plt.xticks(k_result_df["latent_k"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(K_ERROR_PLOT_PATH, dpi=300)
    plt.close()


# step1-3. ua.base 로드
ua_base_path = find_data_file("ua.base")
ua_base = load_ratings(ua_base_path)


# step1-4. ua.base > train/validation = 9:1
train_df, val_df = train_test_split(
    ua_base,
    test_size=TEST_SIZE_FOR_VALIDATION,
    random_state=RANDOM_STATE,
    shuffle=True,
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


# step1-5. train 데이터만 사용해 user/item 매핑 생성 (9:1)
user_ids, movie_ids, user_to_idx, item_to_idx = make_mappings(ua_base)
train_user = build_user_item_dict(train_df, relevant_only=False)
val_user = build_user_item_dict(val_df, relevant_only=True)
train_seen_items = {
    user_id: set(items.keys()) for user_id, items in train_user.items()
}

ua_base_user = build_user_item_dict(ua_base, relevant_only=False)
ua_base_seen_items = {
    user_id: set(items.keys()) for user_id, items in ua_base_user.items()
}
all_items = sorted(int(movie_id) for movie_id in ua_base["movieId"].unique())


# step1-6. UV SGD를 train 데이터만 이용해 latent_k별로 학습
RESULT_DIR.mkdir(parents=True, exist_ok=True)
k_results = []
best_model = None
best_selection_key = None

for latent_k in K_CANDIDATES:
    print(f"\ntraining UV SGD model: latent_k={latent_k}")
    learned = train_uv_sgd(
        train_df,
        val_df,
        user_ids,
        movie_ids,
        user_to_idx,
        item_to_idx,
        latent_k,
    )

    model = {
        "model_name": "uv_sgd",
        "latent_k": latent_k,
        "global_mean": learned["global_mean"],
        "user_bias": learned["user_bias"],
        "item_bias": learned["item_bias"],
        "U": learned["U"],
        "V": learned["V"],
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "all_items": all_items,
        "train_seen_items": train_seen_items,
        "ua_base_seen_items": ua_base_seen_items,
        "learning_rate": LEARNING_RATE,
        "regularization": REGULARIZATION,
        "n_epochs": N_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "best_epoch": learned["best_epoch"],
        "history": learned["history"],
        "random_state": RANDOM_STATE,
    }

    precision_10, ndcg_10 = evaluate_validation(model, train_seen_items, val_user, TOP_N)
    rmse, mae = evaluate_rmse_mae(model, val_df)
    roc_scores, roc_labels = get_roc_validation(model, train_seen_items, val_user, TOP_N)

    if len(set(roc_labels)) == 2:
        fpr, tpr, _ = roc_curve(roc_labels, roc_scores)
        roc_auc = float(auc(fpr, tpr))
    else:
        roc_auc = float("nan")

    k_result = {
        "latent_k": latent_k,
        "precision_at_10": precision_10,
        "ndcg_at_10": ndcg_10,
        "auc": roc_auc,
        "rmse": rmse,
        "mae": mae,
        "best_epoch": learned["best_epoch"],
    }
    k_results.append(k_result)

    print(
        "latent_k={latent_k}, Precision@10={precision:.6f}, "
        "NDCG@10={ndcg:.6f}, AUC={auc_value:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}"
        .format(
            latent_k=latent_k,
            precision=precision_10,
            ndcg=ndcg_10,
            auc_value=roc_auc,
            rmse=rmse,
            mae=mae,
        )
    )

    selection_key = (ndcg_10, precision_10)
    if best_selection_key is None or selection_key > best_selection_key:
        best_selection_key = selection_key
        best_model = model
        best_model["validation_precision_at_10"] = precision_10
        best_model["validation_ndcg_at_10"] = ndcg_10
        best_model["validation_auc"] = roc_auc
        best_model["validation_rmse"] = rmse
        best_model["validation_mae"] = mae


# step1-7. validation 결과 표와 그래프 저장
k_result_df = pd.DataFrame(k_results)
k_result_df.to_csv(K_RESULT_CSV_PATH, index=False, encoding="utf-8-sig")
save_k_plots(k_result_df)


# step1-8. 선택된 latent_k의 ROC curve 저장
best_roc_scores, best_roc_labels = get_roc_validation(
    best_model, train_seen_items, val_user, TOP_N
)
if len(set(best_roc_labels)) == 2:
    fpr, tpr, _ = roc_curve(best_roc_labels, best_roc_scores)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        label=f"UV SGD k={best_model['latent_k']} AUC={best_model['validation_auc']:.4f}",
    )
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("UV SGD ROC Curve")
    plt.savefig(ROC_PLOT_PATH, dpi=300)
    plt.close()


# step1-9. validation 예측 평점을 CSV로 저장
save_validation_predictions(best_model, val_df, PREDICTION_CSV_PATH)


# step1-10. 선택된 모델 저장
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_PATH)


# step1-11. 최종 선택 모델의 validation 결과 출력
print("\n[UV SGD validation result]")
print(f"selected_latent_k: {best_model['latent_k']}")
print(f"best_epoch: {best_model['best_epoch']}")
print(f"Precision@10: {best_model['validation_precision_at_10']:.6f}")
print(f"NDCG@10: {best_model['validation_ndcg_at_10']:.6f}")
print(f"AUC: {best_model['validation_auc']:.6f}")
print(f"RMSE: {best_model['validation_rmse']:.6f}")
print(f"MAE: {best_model['validation_mae']:.6f}")
print(f"model saved to: {MODEL_PATH}")
print(f"validation predictions saved to: {PREDICTION_CSV_PATH}")
print(f"k validation results saved to: {K_RESULT_CSV_PATH}")
print(f"k metric plot saved to: {K_METRIC_PLOT_PATH}")
print(f"k error plot saved to: {K_ERROR_PLOT_PATH}")

# ================================================================================
# END: 01_uv_sgd_train.py
# ================================================================================


# ================================================================================
# BEGIN: 02_svd_row_mean_train.py
# ================================================================================

from pathlib import Path
from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, roc_curve
from sklearn.model_selection import train_test_split


# step2-1. 실험 설정
RANDOM_STATE = 42
TEST_SIZE_FOR_VALIDATION = 0.1
TOP_N = 10
RANK_K_CANDIDATES = [10, 20, 50, 100]
MAX_ITER = 20
TOL = 1e-4


# step2-2. 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
MODEL_PATH = MODEL_DIR / "svd_row_mean_model.pkl"
PREDICTION_CSV_PATH = RESULT_DIR / "svd_row_mean_validation_predictions.csv"
RANK_RESULT_CSV_PATH = RESULT_DIR / "svd_row_mean_rank_validation_results.csv"
RANK_METRIC_PLOT_PATH = RESULT_DIR / "svd_row_mean_rank_metrics.png"
RANK_ERROR_PLOT_PATH = RESULT_DIR / "svd_row_mean_rank_errors.png"


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


def build_user_item_dict(df, relevant_only=False):
    data = defaultdict(dict)
    source = df[df["rating"] >= 4] if relevant_only else df
    for row in source.itertuples(index=False):
        data[int(row.userId)][int(row.movieId)] = float(row.rating)
    return dict(data)


def make_mappings(ua_base):
    user_ids = sorted(int(user_id) for user_id in ua_base["userId"].unique())
    movie_ids = sorted(int(movie_id) for movie_id in ua_base["movieId"].unique())
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    return user_ids, movie_ids, user_to_idx, item_to_idx


def build_train_matrix(train_df, user_ids, movie_ids, user_to_idx, item_to_idx):
    matrix = np.full((len(user_ids), len(movie_ids)), np.nan, dtype=float)
    observed_mask = np.zeros(matrix.shape, dtype=bool)

    for row in train_df.itertuples(index=False):
        u_idx = user_to_idx[int(row.userId)]
        i_idx = item_to_idx[int(row.movieId)]
        matrix[u_idx, i_idx] = float(row.rating)
        observed_mask[u_idx, i_idx] = True

    return matrix, observed_mask


def initialize_missing_with_row_mean(train_matrix):
    initialized = train_matrix.copy()
    global_mean = float(np.nanmean(train_matrix))
    row_means = np.nanmean(train_matrix, axis=1)
    row_means = np.where(np.isnan(row_means), global_mean, row_means)

    missing_rows, missing_cols = np.where(np.isnan(initialized))
    initialized[missing_rows, missing_cols] = row_means[missing_rows]
    return initialized, row_means, global_mean


def clip_rating(pred):
    return float(np.clip(pred, 1.0, 5.0))


def iterative_svd(train_matrix, observed_mask, rank_k):
    current_matrix, row_means, global_mean = initialize_missing_with_row_mean(train_matrix)
    previous_matrix = current_matrix.copy()
    history = []

    for iteration in range(1, MAX_ITER + 1):
        u, s, vt = svds(current_matrix, k=rank_k)
        order = np.argsort(s)[::-1]
        u = u[:, order]
        s = s[order]
        vt = vt[order, :]

        reconstructed = (u * s) @ vt
        reconstructed = np.clip(reconstructed, 1.0, 5.0)

        next_matrix = reconstructed.copy()
        next_matrix[observed_mask] = train_matrix[observed_mask]

        change = float(
            np.linalg.norm(next_matrix - previous_matrix)
            / (np.linalg.norm(previous_matrix) + 1e-8)
        )
        history.append({"iteration": iteration, "change": change})

        current_matrix = next_matrix
        if change <= TOL:
            break
        previous_matrix = current_matrix.copy()

    return {
        "prediction_matrix": current_matrix,
        "row_means": row_means,
        "global_mean": global_mean,
        "rank_k": rank_k,
        "iterations_run": len(history),
        "converged": history[-1]["change"] <= TOL if history else False,
        "history": history,
    }


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


# =========================
# 평가 지표
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


def recommend_svd(user_id, model, seen_items, k):
    candidates = [
        movie_id
        for movie_id in model["all_items"]
        if movie_id not in seen_items.get(int(user_id), set())
    ]
    recs = [
        (int(movie_id), predict_score(user_id, movie_id, model))
        for movie_id in candidates
    ]
    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:k]


def evaluate_validation(model, train_seen_items, val_dict, k):
    p = n = 0.0
    cnt = 0
    for user_id in val_dict:
        if user_id not in train_seen_items:
            continue
        true_items = set(val_dict[user_id].keys())
        recs = recommend_svd(user_id, model, train_seen_items, k)
        p += precision_at_k(recs, true_items)
        n += ndcg_at_k(recs, true_items, k)
        cnt += 1
    if cnt == 0:
        return 0.0, 0.0
    return p / cnt, n / cnt


def get_roc_validation(model, train_seen_items, val_dict, k):
    scores, labels = [], []
    for user_id in val_dict:
        if user_id not in train_seen_items:
            continue
        true_items = set(val_dict[user_id].keys())
        recs = recommend_svd(user_id, model, train_seen_items, k)
        for movie_id, score in recs:
            scores.append(score)
            labels.append(1 if movie_id in true_items else 0)
    return scores, labels


def evaluate_rmse_mae(model, val_df):
    preds = [
        predict_score(row.userId, row.movieId, model)
        for row in val_df.itertuples(index=False)
    ]
    y_true = val_df["rating"].to_numpy(dtype=float)
    y_pred = np.array(preds, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def save_validation_predictions(model, val_df, output_path):
    rows = []
    for row in val_df.itertuples(index=False):
        rows.append(
            {
                "userId": int(row.userId),
                "movieId": int(row.movieId),
                "actual_rating": float(row.rating),
                "predicted_rating": predict_score(row.userId, row.movieId, model),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


# step2-3. ua.base만 로드
ua_base_path = find_data_file("ua.base")
ua_base = load_ratings(ua_base_path)


# step2-4. ua.base를 train/validation = 9:1로 분할
train_df, val_df = train_test_split(
    ua_base,
    test_size=TEST_SIZE_FOR_VALIDATION,
    random_state=RANDOM_STATE,
    shuffle=True,
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


# step2-5. 매핑 생성 및 train 데이터만으로 평점 행렬 생성
user_ids, movie_ids, user_to_idx, item_to_idx = make_mappings(ua_base)
train_matrix, observed_mask = build_train_matrix(
    train_df, user_ids, movie_ids, user_to_idx, item_to_idx
)


# step2-6. validation 평가용 딕셔너리 생성
train_user = build_user_item_dict(train_df, relevant_only=False)
val_user = build_user_item_dict(val_df, relevant_only=True)
train_seen_items = {
    user_id: set(items.keys()) for user_id, items in train_user.items()
}

ua_base_user = build_user_item_dict(ua_base, relevant_only=False)
ua_base_seen_items = {
    user_id: set(items.keys()) for user_id, items in ua_base_user.items()
}
all_items = sorted(int(movie_id) for movie_id in ua_base["movieId"].unique())


# step2-7. Iterative SVD를 train 데이터만 이용해 rank_k별로 학습
RESULT_DIR.mkdir(parents=True, exist_ok=True)
rank_results = []
best_model = None
best_selection_key = None

for rank_k in RANK_K_CANDIDATES:
    print(f"\ntraining Iterative SVD row-mean model: rank_k={rank_k}")
    learned = iterative_svd(train_matrix, observed_mask, rank_k)

    model = {
        "model_name": "iterative_svd_row_mean",
        "rank_k": rank_k,
        "prediction_matrix": learned["prediction_matrix"],
        "row_means": learned["row_means"],
        "global_mean": learned["global_mean"],
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "all_items": all_items,
        "train_seen_items": train_seen_items,
        "ua_base_seen_items": ua_base_seen_items,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "iterations_run": learned["iterations_run"],
        "converged": learned["converged"],
        "history": learned["history"],
        "random_state": RANDOM_STATE,
        "initialization": "row_mean",
    }

    precision_10, ndcg_10 = evaluate_validation(model, train_seen_items, val_user, TOP_N)
    rmse, mae = evaluate_rmse_mae(model, val_df)
    roc_scores, roc_labels = get_roc_validation(model, train_seen_items, val_user, TOP_N)

    if len(set(roc_labels)) == 2:
        fpr, tpr, _ = roc_curve(roc_labels, roc_scores)
        roc_auc = float(auc(fpr, tpr))
    else:
        roc_auc = float("nan")

    rank_result = {
        "rank_k": rank_k,
        "precision_at_10": precision_10,
        "ndcg_at_10": ndcg_10,
        "auc": roc_auc,
        "rmse": rmse,
        "mae": mae,
        "iterations_run": learned["iterations_run"],
        "converged": learned["converged"],
    }
    rank_results.append(rank_result)

    print(
        "rank_k={rank_k}, Precision@10={precision:.6f}, "
        "NDCG@10={ndcg:.6f}, AUC={auc_value:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}"
        .format(
            rank_k=rank_k,
            precision=precision_10,
            ndcg=ndcg_10,
            auc_value=roc_auc,
            rmse=rmse,
            mae=mae,
        )
    )

    selection_key = (ndcg_10, precision_10)
    if best_selection_key is None or selection_key > best_selection_key:
        best_selection_key = selection_key
        best_model = model
        best_model["validation_precision_at_10"] = precision_10
        best_model["validation_ndcg_at_10"] = ndcg_10
        best_model["validation_auc"] = roc_auc
        best_model["validation_rmse"] = rmse
        best_model["validation_mae"] = mae


# step2-8. validation 결과 표 저장
rank_result_df = pd.DataFrame(rank_results)
rank_result_df.to_csv(RANK_RESULT_CSV_PATH, index=False, encoding="utf-8-sig")


# step2-9. rank_k별 validation 성능 시각화
plt.figure()
plt.plot(
    rank_result_df["rank_k"],
    rank_result_df["precision_at_10"],
    marker="o",
    label="Precision@10",
)
plt.plot(
    rank_result_df["rank_k"],
    rank_result_df["ndcg_at_10"],
    marker="o",
    label="NDCG@10",
)
plt.plot(
    rank_result_df["rank_k"],
    rank_result_df["auc"],
    marker="o",
    label="AUC",
)
plt.xlabel("rank_k")
plt.ylabel("Validation score")
plt.title("SVD Row Mean Validation Ranking Metrics by rank_k")
plt.xticks(rank_result_df["rank_k"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RANK_METRIC_PLOT_PATH, dpi=300)
plt.close()

plt.figure()
plt.plot(
    rank_result_df["rank_k"],
    rank_result_df["rmse"],
    marker="o",
    label="RMSE",
)
plt.plot(
    rank_result_df["rank_k"],
    rank_result_df["mae"],
    marker="o",
    label="MAE",
)
plt.xlabel("rank_k")
plt.ylabel("Validation error")
plt.title("SVD Row Mean Validation Error Metrics by rank_k")
plt.xticks(rank_result_df["rank_k"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RANK_ERROR_PLOT_PATH, dpi=300)
plt.close()


# step2-10. 선택된 rank_k의 ROC curve 저장
best_roc_scores, best_roc_labels = get_roc_validation(
    best_model, train_seen_items, val_user, TOP_N
)
if len(set(best_roc_labels)) == 2:
    fpr, tpr, _ = roc_curve(best_roc_labels, best_roc_scores)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        label=f"SVD Row Mean rank={best_model['rank_k']} AUC={best_model['validation_auc']:.4f}",
    )
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("SVD Row Mean ROC Curve")
    plt.savefig(RESULT_DIR / "svd_row_mean_roc.png", dpi=300)
    plt.close()


# step2-11. validation 예측 평점을 CSV로 저장
save_validation_predictions(best_model, val_df, PREDICTION_CSV_PATH)


# step2-12. 선택된 모델 저장
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_PATH)


# step2-13. 최종 선택 모델의 validation 결과 출력
print("\n[SVD row-mean validation result]")
print(f"selected_rank_k: {best_model['rank_k']}")
print(f"Precision@10: {best_model['validation_precision_at_10']:.6f}")
print(f"NDCG@10: {best_model['validation_ndcg_at_10']:.6f}")
print(f"AUC: {best_model['validation_auc']:.6f}")
print(f"RMSE: {best_model['validation_rmse']:.6f}")
print(f"MAE: {best_model['validation_mae']:.6f}")
print(f"model saved to: {MODEL_PATH}")
print(f"validation predictions saved to: {PREDICTION_CSV_PATH}")
print(f"rank validation results saved to: {RANK_RESULT_CSV_PATH}")
print(f"rank metric plot saved to: {RANK_METRIC_PLOT_PATH}")
print(f"rank error plot saved to: {RANK_ERROR_PLOT_PATH}")

# ================================================================================
# END: 02_svd_row_mean_train.py
# ================================================================================


# ================================================================================
# BEGIN: 03_svd_item_neighbor_train.py
# ================================================================================

from pathlib import Path
from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, roc_curve
from sklearn.model_selection import train_test_split


# step3-1. 실험 설정
RANDOM_STATE = 42
TEST_SIZE_FOR_VALIDATION = 0.1
TOP_N = 10
RANK_K_CANDIDATES = [10, 20, 50, 100]
NEIGHBOR_K = 20
MAX_ITER = 20
TOL = 1e-4


# step3-2. 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
MODEL_PATH = MODEL_DIR / "svd_item_neighbor_model.pkl"
PREDICTION_CSV_PATH = RESULT_DIR / "svd_item_neighbor_validation_predictions.csv"
RANK_RESULT_CSV_PATH = RESULT_DIR / "svd_item_neighbor_rank_validation_results.csv"
RANK_METRIC_PLOT_PATH = RESULT_DIR / "svd_item_neighbor_rank_metrics.png"
RANK_ERROR_PLOT_PATH = RESULT_DIR / "svd_item_neighbor_rank_errors.png"
ROC_PLOT_PATH = RESULT_DIR / "svd_item_neighbor_roc.png"


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


def build_user_item_dict(df, relevant_only=False):
    data = defaultdict(dict)
    source = df[df["rating"] >= 4] if relevant_only else df
    for row in source.itertuples(index=False):
        data[int(row.userId)][int(row.movieId)] = float(row.rating)
    return dict(data)


def make_mappings(ua_base):
    user_ids = sorted(int(user_id) for user_id in ua_base["userId"].unique())
    movie_ids = sorted(int(movie_id) for movie_id in ua_base["movieId"].unique())
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    return user_ids, movie_ids, user_to_idx, item_to_idx


def build_train_matrix(train_df, user_ids, movie_ids, user_to_idx, item_to_idx):
    matrix = np.full((len(user_ids), len(movie_ids)), np.nan, dtype=float)
    observed_mask = np.zeros(matrix.shape, dtype=bool)

    for row in train_df.itertuples(index=False):
        u_idx = user_to_idx[int(row.userId)]
        i_idx = item_to_idx[int(row.movieId)]
        matrix[u_idx, i_idx] = float(row.rating)
        observed_mask[u_idx, i_idx] = True

    return matrix, observed_mask


def clip_rating(pred):
    return float(np.clip(pred, 1.0, 5.0))


def compute_item_cosine_similarity(train_matrix):
    filled = np.nan_to_num(train_matrix, nan=0.0)
    item_norms = np.linalg.norm(filled, axis=0)
    item_norms[item_norms == 0] = 1e-8
    normalized = filled / item_norms
    item_sim = normalized.T @ normalized
    np.fill_diagonal(item_sim, 0.0)
    return item_sim


def initialize_missing_with_item_neighbors(train_matrix, observed_mask):
    initialized = train_matrix.copy()
    global_mean = float(np.nanmean(train_matrix))
    user_means = np.nanmean(train_matrix, axis=1)
    user_means = np.where(np.isnan(user_means), global_mean, user_means)
    item_sim = compute_item_cosine_similarity(train_matrix)

    n_users, n_items = train_matrix.shape
    for u_idx in range(n_users):
        rated_items = np.where(observed_mask[u_idx])[0]
        missing_items = np.where(~observed_mask[u_idx])[0]

        if len(rated_items) == 0:
            initialized[u_idx, missing_items] = global_mean
            continue

        rated_values = train_matrix[u_idx, rated_items]
        for i_idx in missing_items:
            sims = item_sim[i_idx, rated_items]
            if len(sims) > NEIGHBOR_K:
                top_pos = np.argpartition(sims, -NEIGHBOR_K)[-NEIGHBOR_K:]
                sims = sims[top_pos]
                neighbor_values = rated_values[top_pos]
            else:
                neighbor_values = rated_values

            denom = np.sum(np.abs(sims))
            if denom > 1e-8:
                pred = np.sum(sims * neighbor_values) / denom
            else:
                pred = user_means[u_idx]
            initialized[u_idx, i_idx] = clip_rating(pred)

    return initialized, user_means, global_mean, item_sim


def iterative_svd(train_matrix, observed_mask, rank_k):
    current_matrix, user_means, global_mean, item_sim = initialize_missing_with_item_neighbors(
        train_matrix, observed_mask
    )
    previous_matrix = current_matrix.copy()
    history = []

    for iteration in range(1, MAX_ITER + 1):
        u, s, vt = svds(current_matrix, k=rank_k)
        order = np.argsort(s)[::-1]
        u = u[:, order]
        s = s[order]
        vt = vt[order, :]

        reconstructed = (u * s) @ vt
        reconstructed = np.clip(reconstructed, 1.0, 5.0)

        next_matrix = reconstructed.copy()
        next_matrix[observed_mask] = train_matrix[observed_mask]

        change = float(
            np.linalg.norm(next_matrix - previous_matrix)
            / (np.linalg.norm(previous_matrix) + 1e-8)
        )
        history.append({"iteration": iteration, "change": change})

        current_matrix = next_matrix
        if change <= TOL:
            break
        previous_matrix = current_matrix.copy()

    return {
        "prediction_matrix": current_matrix,
        "user_means": user_means,
        "global_mean": global_mean,
        "item_similarity": item_sim,
        "rank_k": rank_k,
        "iterations_run": len(history),
        "converged": history[-1]["change"] <= TOL if history else False,
        "history": history,
    }


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


# =========================
# 평가 지표
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


def recommend_svd(user_id, model, seen_items, k):
    candidates = [
        movie_id
        for movie_id in model["all_items"]
        if movie_id not in seen_items.get(int(user_id), set())
    ]
    recs = [
        (int(movie_id), predict_score(user_id, movie_id, model))
        for movie_id in candidates
    ]
    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:k]


def evaluate_validation(model, train_seen_items, val_dict, k):
    p = n = 0.0
    cnt = 0
    for user_id in val_dict:
        if user_id not in train_seen_items:
            continue
        true_items = set(val_dict[user_id].keys())
        recs = recommend_svd(user_id, model, train_seen_items, k)
        p += precision_at_k(recs, true_items)
        n += ndcg_at_k(recs, true_items, k)
        cnt += 1
    if cnt == 0:
        return 0.0, 0.0
    return p / cnt, n / cnt


def get_roc_validation(model, train_seen_items, val_dict, k):
    scores, labels = [], []
    for user_id in val_dict:
        if user_id not in train_seen_items:
            continue
        true_items = set(val_dict[user_id].keys())
        recs = recommend_svd(user_id, model, train_seen_items, k)
        for movie_id, score in recs:
            scores.append(score)
            labels.append(1 if movie_id in true_items else 0)
    return scores, labels


def evaluate_rmse_mae(model, val_df):
    preds = [
        predict_score(row.userId, row.movieId, model)
        for row in val_df.itertuples(index=False)
    ]
    y_true = val_df["rating"].to_numpy(dtype=float)
    y_pred = np.array(preds, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def save_validation_predictions(model, val_df, output_path):
    rows = []
    for row in val_df.itertuples(index=False):
        rows.append(
            {
                "userId": int(row.userId),
                "movieId": int(row.movieId),
                "actual_rating": float(row.rating),
                "predicted_rating": predict_score(row.userId, row.movieId, model),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_rank_plots(rank_result_df):
    plt.figure()
    plt.plot(rank_result_df["rank_k"], rank_result_df["precision_at_10"], marker="o", label="Precision@10")
    plt.plot(rank_result_df["rank_k"], rank_result_df["ndcg_at_10"], marker="o", label="NDCG@10")
    plt.plot(rank_result_df["rank_k"], rank_result_df["auc"], marker="o", label="AUC")
    plt.xlabel("rank_k")
    plt.ylabel("Validation score")
    plt.title("SVD Item Neighbor Validation Ranking Metrics by rank_k")
    plt.xticks(rank_result_df["rank_k"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RANK_METRIC_PLOT_PATH, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(rank_result_df["rank_k"], rank_result_df["rmse"], marker="o", label="RMSE")
    plt.plot(rank_result_df["rank_k"], rank_result_df["mae"], marker="o", label="MAE")
    plt.xlabel("rank_k")
    plt.ylabel("Validation error")
    plt.title("SVD Item Neighbor Validation Error Metrics by rank_k")
    plt.xticks(rank_result_df["rank_k"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RANK_ERROR_PLOT_PATH, dpi=300)
    plt.close()


# step3-3. ua.base만 로드
ua_base_path = find_data_file("ua.base")
ua_base = load_ratings(ua_base_path)


# step3-4. ua.base를 train/validation = 9:1로 분할
train_df, val_df = train_test_split(
    ua_base,
    test_size=TEST_SIZE_FOR_VALIDATION,
    random_state=RANDOM_STATE,
    shuffle=True,
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


# step3-5. 매핑 생성 및 train 데이터만으로 평점 행렬 생성
user_ids, movie_ids, user_to_idx, item_to_idx = make_mappings(ua_base)
train_matrix, observed_mask = build_train_matrix(
    train_df, user_ids, movie_ids, user_to_idx, item_to_idx
)


# step3-6. validation 평가용 딕셔너리 생성
train_user = build_user_item_dict(train_df, relevant_only=False)
val_user = build_user_item_dict(val_df, relevant_only=True)
train_seen_items = {
    user_id: set(items.keys()) for user_id, items in train_user.items()
}

ua_base_user = build_user_item_dict(ua_base, relevant_only=False)
ua_base_seen_items = {
    user_id: set(items.keys()) for user_id, items in ua_base_user.items()
}
all_items = sorted(int(movie_id) for movie_id in ua_base["movieId"].unique())


# step3-7. Iterative SVD를 train 데이터만 이용해 rank_k별로 학습
RESULT_DIR.mkdir(parents=True, exist_ok=True)
rank_results = []
best_model = None
best_selection_key = None

for rank_k in RANK_K_CANDIDATES:
    print(f"\ntraining Iterative SVD item-neighbor model: rank_k={rank_k}")
    learned = iterative_svd(train_matrix, observed_mask, rank_k)

    model = {
        "model_name": "iterative_svd_item_neighbor",
        "rank_k": rank_k,
        "neighbor_k": NEIGHBOR_K,
        "prediction_matrix": learned["prediction_matrix"],
        "user_means": learned["user_means"],
        "global_mean": learned["global_mean"],
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "all_items": all_items,
        "train_seen_items": train_seen_items,
        "ua_base_seen_items": ua_base_seen_items,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "iterations_run": learned["iterations_run"],
        "converged": learned["converged"],
        "history": learned["history"],
        "random_state": RANDOM_STATE,
        "initialization": "item_neighbor_cosine",
    }

    precision_10, ndcg_10 = evaluate_validation(model, train_seen_items, val_user, TOP_N)
    rmse, mae = evaluate_rmse_mae(model, val_df)
    roc_scores, roc_labels = get_roc_validation(model, train_seen_items, val_user, TOP_N)

    if len(set(roc_labels)) == 2:
        fpr, tpr, _ = roc_curve(roc_labels, roc_scores)
        roc_auc = float(auc(fpr, tpr))
    else:
        roc_auc = float("nan")

    rank_result = {
        "rank_k": rank_k,
        "precision_at_10": precision_10,
        "ndcg_at_10": ndcg_10,
        "auc": roc_auc,
        "rmse": rmse,
        "mae": mae,
        "iterations_run": learned["iterations_run"],
        "converged": learned["converged"],
    }
    rank_results.append(rank_result)

    print(
        "rank_k={rank_k}, Precision@10={precision:.6f}, "
        "NDCG@10={ndcg:.6f}, AUC={auc_value:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}"
        .format(
            rank_k=rank_k,
            precision=precision_10,
            ndcg=ndcg_10,
            auc_value=roc_auc,
            rmse=rmse,
            mae=mae,
        )
    )

    selection_key = (ndcg_10, precision_10)
    if best_selection_key is None or selection_key > best_selection_key:
        best_selection_key = selection_key
        best_model = model
        best_model["validation_precision_at_10"] = precision_10
        best_model["validation_ndcg_at_10"] = ndcg_10
        best_model["validation_auc"] = roc_auc
        best_model["validation_rmse"] = rmse
        best_model["validation_mae"] = mae


# step3-8. validation 결과 표와 그래프 저장
rank_result_df = pd.DataFrame(rank_results)
rank_result_df.to_csv(RANK_RESULT_CSV_PATH, index=False, encoding="utf-8-sig")
save_rank_plots(rank_result_df)


# step3-9. 선택된 rank_k의 ROC curve 저장
best_roc_scores, best_roc_labels = get_roc_validation(
    best_model, train_seen_items, val_user, TOP_N
)
if len(set(best_roc_labels)) == 2:
    fpr, tpr, _ = roc_curve(best_roc_labels, best_roc_scores)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        label=f"SVD Item Neighbor rank={best_model['rank_k']} AUC={best_model['validation_auc']:.4f}",
    )
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("SVD Item Neighbor ROC Curve")
    plt.savefig(ROC_PLOT_PATH, dpi=300)
    plt.close()


# step3-10. validation 예측 평점을 CSV로 저장
save_validation_predictions(best_model, val_df, PREDICTION_CSV_PATH)


# step3-11. 선택된 모델 저장
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_PATH)


# step3-12. 최종 선택 모델의 validation 결과 출력
print("\n[SVD item-neighbor validation result]")
print(f"selected_rank_k: {best_model['rank_k']}")
print(f"neighbor_k: {best_model['neighbor_k']}")
print(f"Precision@10: {best_model['validation_precision_at_10']:.6f}")
print(f"NDCG@10: {best_model['validation_ndcg_at_10']:.6f}")
print(f"AUC: {best_model['validation_auc']:.6f}")
print(f"RMSE: {best_model['validation_rmse']:.6f}")
print(f"MAE: {best_model['validation_mae']:.6f}")
print(f"model saved to: {MODEL_PATH}")
print(f"validation predictions saved to: {PREDICTION_CSV_PATH}")
print(f"rank validation results saved to: {RANK_RESULT_CSV_PATH}")
print(f"rank metric plot saved to: {RANK_METRIC_PLOT_PATH}")
print(f"rank error plot saved to: {RANK_ERROR_PLOT_PATH}")

# ================================================================================
# END: 03_svd_item_neighbor_train.py
# ================================================================================
