import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import RobustScaler  # StandardScaler 대신 이상치에 강한 RobustScaler!
from matplotlib.lines import Line2D

# 1) 데이터 불러오기 (기존과 동일)
DATA_PATH = [
    "../output/doc0_hidden_representation.json", 
    "../output/doc1_hidden_representation.json"
]

X = []
y = []
print("데이터를 불러오는 중...")
for data in DATA_PATH:
    with open(data, "r", encoding="utf-8") as f:
        doc_data = json.load(f)
        X.extend([item["hidden_representation"] for item in doc_data])
        y.extend([item["doc_id"] for item in doc_data])

# 2) 전처리 : RobustScaler (PCA 생략하고 2048차원 원본 구조를 유지)
print("RobustScaler 적용 중...")
X_scaled = RobustScaler().fit_transform(X)

# 3) UMAP 파라미터 그리드 탐색
print("UMAP 2차원 변환 및 시각화 진행 중...")
neighbors_list = [5, 15, 30]        # 줌 렌즈 (Local vs Global)
min_dist_list = [0.01, 0.1, 0.5]    # 점들의 뭉침 정도 (밀집도)

fig, axes = plt.subplots(3, 3, figsize=(18, 15), dpi=150)
fig.suptitle("UMAP Projection without PCA (Metric: Cosine)", fontsize=22, fontweight="bold", y=0.95)

for i, n_neighbors in enumerate(neighbors_list):
    for j, min_dist in enumerate(min_dist_list):
        
        # UMAP 모델 생성 및 학습
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            min_dist=min_dist, 
            n_components=2,         # 2차원 축소
            metric='cosine',        # LLM 벡터의 방향성 보존
            random_state=42
        )
        
        # 2048차원을 바로 2차원으로 변환
        X_umap = reducer.fit_transform(X_scaled)
        
        # 서브플롯 그리기
        ax = axes[i, j]
        sns.scatterplot(
            x=X_umap[:, 0], 
            y=X_umap[:, 1], 
            hue=y, 
            palette={0: "royalblue", 1: "tomato"}, # 시각적으로 편안한 색상 대조
            s=80, 
            alpha=0.8, 
            edgecolor="w",
            ax=ax,
            legend=False 
        )
        
        # 각 그래프의 파라미터 표기
        ax.set_title(f"n_neighbors = {n_neighbors} | min_dist = {min_dist}", fontsize=13, pad=10)
        ax.set_xticks([]) # 불필요한 x축 숫자 제거
        ax.set_yticks([]) # 불필요한 y축 숫자 제거

# 전체 범례(Legend) 설정
custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markersize=12),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', markersize=12)]
fig.legend(custom_lines, ['Doc 0 (LoRA)', 'Doc 1 (LoRA)'], loc='upper right', 
           fontsize=14, title="Document Class", title_fontsize=15, bbox_to_anchor=(0.95, 0.95))

# 레이아웃 조정 및 저장
plt.tight_layout(rect=[0, 0, 1, 0.92]) # suptitle과 범례 공간 확보
save_path = "umap_grid_search_direct.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 3x3 UMAP 시각화 완료! '{save_path}' 파일이 생성되었습니다.")