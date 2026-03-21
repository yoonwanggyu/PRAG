import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import wandb
from tqdm import tqdm

wandb.init(project="KNN", name="KNN_GridSearch")

# KNN (K-Nearest Neighbors) 알고리즘이란?
# KNN(K-최근접 이웃)은 이름 그대로 "새로운 데이터가 주어졌을 때, 벡터 공간상에서 가장 가까이 있는 K개의 이웃(데이터)들을 확인하여, 다수결로 해당 데이터의 클래스를 판별하는 알고리즘"
# K-Means 클러스터링과 마찬가지로 데이터 간의 '거리' 기반
# K-Means가 중심점을 찾아가는 비지도 학습이라면, KNN은 이미 정답(LoRA A에서 추출됨 vs LoRA B에서 추출됨)을 알고 있는 훈련 데이터를 공간에 뿌려둔 뒤, 새로운 테스트 벡터가 들어오면 주변에 어느 LoRA의 벡터가 더 많이 포진해 있는지 확인하는 지도 학습

# 1) 데이터 가져오기-----
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

# 3) KNN 학습-------

# n_neighbors (기본값 5): 몇 개의 이웃을 보고 다수결을 진행할지 결정
#   - 최적의 K를 찾는 경험 법칙(Rule of Thumb) 중 하나는 학습 데이터 개수의 제곱근(sqrt{N})을 기준으로 삼는 것
#   - sqrt(160) ~ 12.6
# weights (기본값 'uniform'): 다수결을 할 때 이웃들의 투표권에 가중치를 줄지 결정
#   - uniform : 거리에 상관없이 K개의 이웃이 모두 똑같은 1표씩 행사
#   - distance : 더 가까이 있는 이웃의 표에 더 높은 가중치를 줌. 즉, 거리가 가까울수록 더 강한 영향을 미치게 함
# metric (기본값 metric='minkowski', p=2로 되어 있어 사실상 기본값은 유클리디안 거리로 작동): 벡터 간의 거리를 어떻게 잴 것인가를 결정

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17],
    'weights': ['uniform', 'distance'],
    'metric': ['cosine', 'euclidean']
}

grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,   # 학습 데이터를 다시 K개(보통 5개)의 폴드(Fold)로 나누어 학습과 검증을 반복
    # scoring='accuracy',
    scoring='f1_macro',
    n_jobs=-1
)

print("학습 시작----------------------------")
grid_search.fit(X_train,y_train)
print("학습 끝-----------------------------")
# 4) 결과 정리-----
results_df = pd.DataFrame(grid_search.cv_results_)

results_df = results_df[['param_n_neighbors', 'param_weights', 'param_metric', 'mean_test_score', 'rank_test_score']]
results_df.columns = ['K', 'Weights', 'Metric', 'Mean_Validation_F1_Macro', 'Rank']

results_df_sorted = results_df.sort_values(by='Rank').reset_index(drop=True)

    # DataFrame을 WandB Table로 변환하여 기록 (웹에서 인터랙티브 그래프 생성 가능)
wandb.log({"PCA_GridSearch_Results_Table": wandb.Table(dataframe=results_df_sorted)})

# 5) 최적의 파라미터로 Test 데이터 최종 평가
best_knn = grid_search.best_estimator_
test_pred = best_knn.predict(X_test)

test_acc = accuracy_score(y_test, test_pred)
test_f1_macro = f1_score(y_test, test_pred, average='macro')

class_report = classification_report(y_test, test_pred)

    # 핵심 지표(Metrics)를 WandB에 기록
wandb.log({
    "Best_Mean_Validation_F1_Macro": grid_search.best_score_,
    "Test_Accuracy": test_acc,
    "Test_F1_macro_Score": test_f1_macro,   # 두 클래스의 F1 점수를 각각 구한 뒤 평균을 내는 방식(지금 계속 1로만 찍어서)
})                                          # 기본 f1 score은 1 클래스에 대해서만 결과 반환

    # 예측값 / 실제값
y_test_str = str(y_test)
y_pred_str = str(test_pred.tolist())

html_content = f"""
<pre>
y_test    : {y_test_str}
y_predict : {y_pred_str}
</pre>
"""

wandb.log({"Raw_Predictions": wandb.Html(html_content)})

wandb.finish()

# 6) txt 파일로 저장
# filename = "knn_results.txt"

# with open(filename, 'w', encoding='utf-8') as f:
#     print("[GridSearchCV 교차 검증 결과 - 모든 경우의 수]", file=f)
#     print("-" * 65, file=f)
#     print(results_df_sorted.to_string(), file=f)
#     print("-" * 65, file=f)
    
#     print(f"\n[최적의 파라미터 조합]", file=f)
#     print("-" * 50, file=f)
#     print(f"최적의 파라미터: {grid_search.best_params_}", file=f)
#     print(f"최고 Validation 평균 정확도: {grid_search.best_score_:.4f}", file=f)
#     print("-" * 50, file=f)
    
#     print("\n[최종 Test 데이터 평가 결과]", file=f)
#     print("-" * 50, file=f)
#     print(f"y_test            : {y_test}",file=f)
#     print(f"y_predict         : {test_pred.tolist()}",file=f)
#     print(f"Test Accuracy     : {test_acc:.4f}", file=f)
#     print(f"Test F1           : {test_f1:.4f}", file=f)
#     print("-" * 50, file=f)
    
#     print("\n[상세 분류 리포트 (Classification Report)]", file=f)
#     print(class_report, file=f)

# print(f"✅ 모든 실험 결과가 '{filename}' 파일에 성공적으로 저장되었습니다.")