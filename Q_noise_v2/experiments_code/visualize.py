import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

# 1) 데이터 불러오기
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

# 2) 전처리 : 스케일링 -> PCA(50) -> UMAP(2D)
print("스케일링 및 PCA 압축 중...")
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=50,random_state=42).fit_transform(X_scaled)

print("UMAP 2차원 변환 중...")
neighbors_list = [5, 15, 30]        # 고차원 공간에서 하나의 점이 '몇 개의 가까운 이웃'까지를 자신의 동네(Local) 구조로 인정할 것인지 결정
min_dist_list = [0.01, 0.1, 0.5]    # 저차원(2차원) 도화지로 점들을 옮겨 그릴 때, '점들끼리 최소한 얼마나 거리를 띄워놓을 것인가'를 정함

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("UMAP (Metric: Cosine)", fontsize=20, fontweight="bold", y=0.95)

for i, n_neighbors in enumerate(neighbors_list):
    for j, min_dist in enumerate(min_dist_list):
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,    # 최종적으로 몇 차원의 공간으로 압축할 것인가
            min_dist=min_dist, 
            metric='cosine', 
            random_state=42
        )
        X_umap = reducer.fit_transform(X_pca)
        
        ax = axes[i, j]
        sns.scatterplot(
            x=X_umap[:, 0], 
            y=X_umap[:, 1], 
            hue=y, 
            palette={0: "#FF5733", 1: "#3380FF"}, 
            s=80, 
            alpha=0.8, 
            edgecolor="w",
            ax=ax,
            legend=False 
        )
        
        ax.set_title(f"n_neighbors={n_neighbors}, min_dist={min_dist}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF5733', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#3380FF', markersize=10)]
fig.legend(custom_lines, ['doc0_LoRA', 'doc1_LoRA'], loc='upper right', fontsize=12, title="Models")

save_path = "umap_grid_search.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 3x3 UMAP 시각화 완료! '{save_path}' 파일이 생성되었습니다.")

# # 3) 시각화 및 이미지 파일로 저장
# print("그래프를 그리는 중...")
# plt.figure(figsize=(10, 8))

# sns.scatterplot(
#     x=X_umap[:, 0], 
#     y=X_umap[:, 1], 
#     hue=y, 
#     palette={0: "#FF5733", 1: "#3380FF"}, # 빨간색과 파란색
#     s=100, # 점 크기
#     alpha=0.8, # 투명도
#     edgecolor="w"
# )

# plt.title("UMAP Visualization of LoRA Hidden Representations", fontsize=16, fontweight="bold")
# plt.xlabel("UMAP Dimension 1", fontsize=12)
# plt.ylabel("UMAP Dimension 2", fontsize=12)
# plt.legend(title="LoRA Models", title_fontsize='13', fontsize='11')
# plt.grid(True, linestyle='--', alpha=0.5)

# save_path = "umap_visualization.png"
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# print(f"✅ 시각화 완료! '{save_path}' 파일이 생성되었습니다.")
