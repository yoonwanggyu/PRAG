import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import random

DATA_PATH = [
    "../output/pre/doc0_hidden_representation.json", 
    "../output/pre/doc1_hidden_representation.json"
]

X_doc0 = []
X_doc1 = []

# doc0 데이터 로드
with open(DATA_PATH[0], "r", encoding="utf-8") as f:
    doc0_data = json.load(f)
    X_doc0.extend([item["hidden_representation"] for item in doc0_data])

# doc1 데이터 로드
with open(DATA_PATH[1], "r", encoding="utf-8") as f:
    doc1_data = json.load(f)
    X_doc1.extend([item["hidden_representation"] for item in doc1_data])

print(f"doc0 총 데이터 개수: {len(X_doc0)}")
print(f"doc1 총 데이터 개수: {len(X_doc1)}")

X_doc0_np = np.array(X_doc0)
X_doc1_np = np.array(X_doc1)

# 각 문서에서 랜덤으로 5개씩 인덱스 추출
random.seed(42)
sample_indices_doc0 = random.sample(range(len(X_doc0_np)), 5)
sample_indices_doc1 = random.sample(range(len(X_doc1_np)), 5)

print(f"doc0 추출 샘플 인덱스: {sample_indices_doc0}")
print(f"doc1 추출 샘플 인덱스: {sample_indices_doc1}")

# 추출된 데이터 가져오기 (각각 5개씩)
sampled_doc0 = X_doc0_np[sample_indices_doc0]
sampled_doc1 = X_doc1_np[sample_indices_doc1]

# 시각화를 위해 10개를 하나로 합치기 (순서: doc0 5개 -> doc1 5개)
X_sampled = np.vstack((sampled_doc0, sampled_doc1))

# x축에 표시할 라벨 생성 (어느 문서의 몇 번째 데이터인지 명시)
labels = [f"Doc0 (idx {i})" for i in sample_indices_doc0] + \
         [f"Doc1 (idx {i})" for i in sample_indices_doc1]

# 시각적 구분을 위한 색상 설정 (Doc0: 파란색 계열, Doc1: 붉은색 계열)
colors = ['royalblue'] * 5 + ['tomato'] * 5

# 플롯 설정
fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=150)

# 그래프 A (좌측): Boxplot
df_box = pd.DataFrame(X_sampled.T, columns=labels)

# palette에 지정한 색상을 입혀서 문서별로 확연히 구분되게 합니다.
sns.boxplot(data=df_box, ax=axes[0], palette=colors, orient="v")
axes[0].set_title("Value Distribution: Doc0 vs Doc1 (5 Samples Each)", fontsize=15)
axes[0].set_ylabel("Activation Value", fontsize=12)
axes[0].set_xlabel("Sample Source", fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# 그래프 B (우측): Line Plot
doc0_labeled = False
doc1_labeled = False

for i in range(10):
    if i < 5: 
        # Doc0 데이터 그리기
        axes[1].plot(X_sampled[i], alpha=0.6, linewidth=1.0, color='royalblue', 
                     label='Doc 0' if not doc0_labeled else "")
        doc0_labeled = True
    else:     
        # Doc1 데이터 그리기
        axes[1].plot(X_sampled[i], alpha=0.6, linewidth=1.0, color='tomato', 
                     label='Doc 1' if not doc1_labeled else "")
        doc1_labeled = True

axes[1].set_title("Activation Values across 2048 Dimensions", fontsize=15)
axes[1].set_ylabel("Activation Value", fontsize=12)
axes[1].set_xlabel("Dimension (0 to 2047)", fontsize=12)

# 문서 구분을 위한 범례(Legend) 추가
axes[1].legend(loc='upper right', fontsize=12)

# 저장 및 출력
plt.tight_layout()
save_path = "doc_comparison_distribution.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"그래프가 '{save_path}' 파일로 성공적으로 저장되었습니다!")