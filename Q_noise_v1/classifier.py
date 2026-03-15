import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.decomposition import PCA

# SEED = 859
# SEED = 46
SEED = 635
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# 1. 아주 단순한 단일 Linear 모델 정의
# ==========================================
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        # 하나의 Linear 레이어만 사용
        self.layer = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.layer(x)

# ==========================================
# 2. 메인 실행부
# ==========================================
if __name__ == "__main__":
    # 추출해둔 JSON 파일 경로
    DATA_PATH = "output/doc1_extracted_representations.json"
    
    print("### 1. 데이터 로드 및 전처리 ###")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # X(벡터)와 y(정답 라벨) 분리
    X = [item["hidden_representation"] for item in data]
    y = [item["doc_id"] for item in data]

    # 혹시나 차원이 다를 수 있으므로 첫 번째 데이터에서 차원 크기 유추 (2048 예상)
    INPUT_DIM = len(X[0])
    NUM_CLASSES = len(set(y)) # 문서가 2개면 2클래스
    
    print(f"총 데이터 개수: {len(X)}개")
    print(f"입력 벡터 차원: {INPUT_DIM}")
    print(f"클래스 개수: {NUM_CLASSES}")

    # ==========================================
    # 3. 데이터 8:2 분할 및 텐서 변환
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y 
        # stratify=y로 학습/테스트 셋에 문서 0과 1의 비율을 동일하게 유지
    )

    print(f"\n### 2. 학습(8) / 테스트(2) 데이터 분할 ###")
    print(f"학습용 데이터: {len(X_train)}개")
    print(f"테스트용 데이터: {len(X_test)}개")

    # print(f"\n### 2. PCA 차원 축소 (2048 -> 50) ###")
    # pca = PCA(n_components=50) 
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca = pca.transform(X_test)

    # # 모델의 입력 차원을 PCA 결과값인 50으로 변경
    # INPUT_DIM_PCA = X_train_pca.shape[1] 
    # print(f"PCA 적용 후 입력 벡터 차원: {INPUT_DIM_PCA}")

    # 원본 X_train 대신 X_train_pca를 텐서로 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 배치 처리를 위한 DataLoader 생성
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # ==========================================
    # 4. 모델 세팅 (입력 차원 수정)
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LinearClassifier(INPUT_DIM, NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 50 
    
    print("\n### 3. 학습 시작 ###")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 10 에포크마다 로그 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

    # ==========================================
    # 6. 분류기 평가 (Test)
    # ==========================================
    print("\n### 4. 테스트 평가 시작 ###")
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        
        outputs = model(X_test_tensor)
        # 로짓(Logit) 중 가장 값이 큰 인덱스를 예측 클래스로 선택
        _, predicted = torch.max(outputs, 1)
        
        y_true = y_test_tensor.cpu().numpy()
        print(f"y_true : {y_true}")
        y_pred = predicted.cpu().numpy()
        print(f"y_pred : {y_pred}")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("====================================")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")
    print("====================================")