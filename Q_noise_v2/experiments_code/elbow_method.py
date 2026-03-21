from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.model_selection import train_test_split

DATA_PATH = [
    "../output/doc0_hidden_representation.json", 
    "../output/doc1_hidden_representation.json"
]

X = []
Y = []

for data in DATA_PATH:

    with open(data, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

        X.extend([item["hidden_representation"] for item in doc_data])
        Y.extend([item["doc_id"] for item in doc_data])

print(f"총 데이터 개수(X) : {len(X)}")
print(f"총 label 개수(Y) : {len(Y)}")

# 2) 데이터 분할(train / test)------
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f"Train 데이터 개수: {len(X_train)}")
print(f"Test 데이터 개수: {len(X_test)}")

# 1. 스케일링 (RobustScaler 적용)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train) # 훈련 데이터만 사용!

# 2. PCA 피팅 (차원 지정 없이 전체 피팅)
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

# 3. 누적 설명 분산 비율 계산
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 4. 90%의 정보를 보존하려면 몇 차원이 필요한지 확인
target_variance = 0.90
num_components_90 = np.argmax(cumulative_variance >= target_variance) + 1
print(f"원본 데이터 정보량의 {target_variance*100}%를 보존하기 위한 최소 차원 수: {num_components_90}차원")

# 5. 그래프 시각화 (논문용)
plt.figure(figsize=(10, 5))
plt.plot(cumulative_variance, linewidth=2)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Explained Variance')
plt.axvline(x=num_components_90, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.legend()
plt.grid(True)

plt.tight_layout()
save_path = "pca_elbow_method.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"그래프가 '{save_path}' 파일로 성공적으로 저장되었습니다!")