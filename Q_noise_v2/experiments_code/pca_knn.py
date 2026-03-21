import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import wandb

wandb.init(project="KNN", name="PCA_KNN_GridSearch")

# 왜 모델링에는 UMAP 대신 PCA를 쓸까요?
# 1. 거리 정보의 보존 (가장 중요)
#   - PCA (선형): 데이터의 전체적인 분산(퍼진 정도)을 기준으로 차원을 줄임
#                즉, 원래 고차원에서 멀리 있던 애들은 저차원에서도 멀리 있고, 가까웠던 애들은 가깝게 유지되는 '절대적인 거리(Global Distance)'가 상대적으로 잘 보존됨
#                KNN은 철저하게 거리를 재는 모델이기 때문에 이 점이 매우 중요
#   - UMAP (비선형): 군집(Cluster)을 예쁘게 뭉쳐서 보여주기 위해, 데이터 간의 거리를 인위적으로 왜곡(구부리고 당김)
#                 눈으로 보기엔(시각화) 최고지만, 이 왜곡된 거리를 KNN에게 주면 모델이 잘못된 판단을 내릴 위험이 높음
# 2. 테스트 데이터 적용의 안정성
#   - PCA는 한 번 변환 기준(수식)을 만들어두면 새로운 Test 데이터가 들어왔을 때 아주 빠르고 일관되게 똑같은 방식으로 50차원으로 줄일 수 있음
#   - 반면 UMAP은 새로운 데이터를 기존 공간에 끼워 넣는 과정이 복잡하고 불안정할 수 있음
# 결론: UMAP/t-SNE는 사람이 눈으로 확인하는 시각화용으로 쓰고, 실제 KNN 모델에게 먹여줄 데이터를 만들 때는 PCA 50차원을 쓰는 것

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

# 3) 파이프라인 
pipeline = Pipeline([
    ('scaler', RobustScaler()),                      # 1단계: 각 차원의 크기(Scale)를 균일하게 맞춤
    ('pca', PCA(n_components=55, random_state=42)),  # 2단계: 55차원으로 핵심 정보만 압축
    ('knn', KNeighborsClassifier())                  # 3단계: KNN 분류기
])

# 4) knn 학습
param_grid = {
    # 'pca__n_components': [10, 30, 50, 80, 120],
    'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['cosine', 'euclidean']
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,   
    # scoring='accuracy',
    scoring='f1_macro',
    n_jobs=-1
)

print("데이터 스케일링, PCA 차원 축소 및 GridSearchCV 탐색 중...")
grid_search.fit(X_train,y_train)

# 5) 결과 정리-----
results_df = pd.DataFrame(grid_search.cv_results_)

results_df = results_df[['param_knn__n_neighbors', 'param_knn__weights', 'param_knn__metric', 'mean_test_score', 'rank_test_score']]
results_df.columns = ['K', 'Weights', 'Metric', 'Mean_Validation_F1_Macro', 'Rank']

results_df_sorted = results_df.sort_values(by='Rank').reset_index(drop=True)

wandb.log({"PCA_GridSearch_Results_Table": wandb.Table(dataframe=results_df_sorted)})

# 5) 최적의 파라미터로 Test 데이터 최종 평가
best_knn = grid_search.best_estimator_
test_pred = best_knn.predict(X_test)

test_acc = accuracy_score(y_test, test_pred)
test_f1_macro = f1_score(y_test, test_pred, average='macro')

class_report = classification_report(y_test, test_pred)

wandb.log({
    "Best_Mean_Validation_F1_Macro": grid_search.best_score_,
    "Test_Accuracy": test_acc,
    "Test_F1_macro_Score": test_f1_macro,  
})         

y_test_str = str(y_test)
y_pred_str = str(test_pred.tolist())

html_content = f"""
<pre>
y_test    : {y_test_str}
y_predict : {y_pred_str}
</pre>
"""

wandb.log({"PCA_Raw_Predictions": wandb.Html(html_content)})

wandb.finish()

# 6) txt 파일로 저장
# filename = "pca_knn_results.txt"

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