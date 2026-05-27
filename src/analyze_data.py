import pandas as pd

# 데이터 로드
cols = ['user_id', 'item_id', 'rating', 'timestamp']
train = pd.read_csv(r"D:\HyeJeong\SungShin\3-1\RS\2026_RS\ml-100k\ml-100k\ua.base", sep='\t', names=cols)
test = pd.read_csv(r"D:\HyeJeong\SungShin\3-1\RS\2026_RS\ml-100k\ml-100k\ua.test", sep='\t', names=cols)

# 기본 정보
print("Train size:", train.shape)
print("Test size:", test.shape)

print("유저 수:", train['user_id'].nunique())
print("아이템 수:", train['item_id'].nunique())

# 평점 분포
print("\n평점 분포")
print(train['rating'].value_counts().sort_index())

# sparsity 계산
num_users = train['user_id'].nunique()
num_items = train['item_id'].nunique()
num_ratings = len(train)

sparsity = 1 - (num_ratings / (num_users * num_items))
print("\nSparsity:", sparsity)

# 유저별 rating 개수
user_counts = train.groupby('user_id')['rating'].count()
print("\n유저별 rating 개수 통계")
print(user_counts.describe())