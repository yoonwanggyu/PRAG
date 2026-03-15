import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.decomposition import PCA
import wandb
from tqdm import tqdm
from torchinfo import summary

SEED = 45
torch.manual_seed(SEED)
np.random.seed(SEED)

# 비선형 딥러닝이 되려면 모델이 데이터 공간을 구부리고 휘게 만들 수 있어야 함
# 그러기 위해서는 반드시 은닉층과 그 사이에 들어가는 비선형 활성화 함수(예: ReLU, Tanh 등)가 필요

# 1. 모델 정의
class NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NonLinearClassifier, self).__init__()

        # 첫 번째 레이어 (입력층 -> 은닉층) / 2048차원의 데이터를 512차원으로 압축하며 특징을 추출
        self.layer1 = nn.Linear(input_dim, hidden_dim)

        # 비선형 활성화 함수 (ReLU)
        self.relu = nn.ReLU()

        # 두 번째 레이어 (은닉층 -> 출력층) / 512차원의 은닉 표현을 받아 최종적으로 1개의 Logit 값(0 또는 1 판별)을 출력
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self,x):
        # 데이터가 순차적으로 통과하는 흐름 (Forward Pass)
        x = self.layer1(x)
        x = self.relu(x)
        out = self.layer2(x) # 마지막엔 Sigmoid 없이 Logit 그대로 반환 (BCEWithLogitsLoss 사용을 위해)
        return out

# 2. 데이터 준비
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
print("-" * 50)

X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

print(f"Train 데이터 개수: {len(X_train)}")
print("-" * 50)

X_val, X_test, y_val, y_test= train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Val 데이터 개수: {len(X_val)}")
print(f"Test 데이터 개수: {len(X_test)}")
print("-" * 50)

    # BCEWithLogitsLoss를 사용하기 위해서는 y 텐서를 실수형(Float)으로 만들고, 모델 출력과 형태를 맞추기 위해 2차원 [데이터 수, 1] 형태로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

print(f"X_train_tensor shape: {X_train_tensor.shape}")  
print(f"y_train_tensor shape: {y_train_tensor.shape}")  
print("-" * 50)

# 3. 하이퍼파라미터 및 모델 초기화
hidden_dim_size = 512 # 통상적으로 256, 512 등을 많이 사용

input_dim = len(X[0])
print(f"input_dim : {input_dim}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")
print("-" * 50)

model = NonLinearClassifier(input_dim=input_dim, hidden_dim=hidden_dim_size).to(device)

# 4. 손실 함수 & 최적화(Optimizer) 함수

criterion = nn.BCEWithLogitsLoss()  # 클래스 1일 확률 하나만 구함

lr = 0.001
weight_decay = 1e-4
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 5. DataLoader 생성
batch_size = 32
summary(model, input_size=(batch_size,input_dim))

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

# 6. Wandb 초기화 
epochs = 50

wandb.init(
    project="Neural-Network", 
    name=f"2layer-epochs_{epochs}-lr_{lr}-opti_AdamW-wd_{weight_decay}",     
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
    }
)

# 7. 학습
print("학습을 시작합니다...")

# 1 epoch 당 5 steps으로 학습하고 train loss는 5번 steps의 평균 loss 기록
for epoch in tqdm(range(epochs)):
    # --- [Train Phase] ---
    model.train()
    train_loss = 0.0
    
    # 평가 지표 계산을 위해 한 에포크의 모든 예측과 정답을 담을 리스트
    all_train_preds = []
    all_train_targets = []
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()               # 1단계: 기울기 초기화
        logits = model(batch_x)             # 2단계: 순전파(Forward)
        loss = criterion(logits, batch_y)   # 3단계: 오차 계산
        loss.backward()                     # 4단계: 역전파(Backward)
        optimizer.step()                    # 5단계: 가중치 업데이트
        
        train_loss += loss.item() * batch_x.size(0)
            # 파이토치의 BCEWithLogitsLoss()는 미니 배치(32개)가 들어오면, 내부적으로 32개 각각의 오차를 구한 뒤 그걸 모두 더해서 32로 나눈 '평균값' 딱 1개만 뱉어냄
            # 즉, loss.item()으로 나오는 숫자는 '32개짜리 묶음의 평균 점수'
            # 근데 미니 배치마다 데이터 수가 다르다면 틀린 계산 -> 그래서 각 배치의 평균 오차에 해당 배치의 데이터 개수를 다시 곱해서 그 배치의 오차 총합으로 되돌림
        
        # 확률 변환 및 0, 1 예측
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        # CPU로 내리고 Numpy 배열로 변환하여 리스트에 추가
        all_train_preds.extend(preds.cpu().numpy())
        all_train_targets.extend(batch_y.cpu().numpy())
        
    epoch_train_loss = train_loss / len(train_dataset)
        # 위 train_loss 계산 부분에서 batch_x.size(0)을 곱했음으로 여기서는 전체 데이터 개수로 나눠줌

    epoch_train_acc = accuracy_score(all_train_targets, all_train_preds) * 100
    epoch_train_f1 = f1_score(all_train_targets, all_train_preds, average='binary')

    # --- [Validation Phase] ---
    model.eval()
    val_loss = 0.0
    
    all_val_preds = []
    all_val_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            val_loss += loss.item() * batch_x.size(0)
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            all_val_preds.extend(preds.cpu().numpy())
            all_val_targets.extend(batch_y.cpu().numpy())
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = accuracy_score(all_val_targets, all_val_preds) * 100
    epoch_val_f1 = f1_score(all_val_targets, all_val_preds, average='binary')

    # --- [Wandb Logging] ---
    # ★ F1 Score 기록 추가
    wandb.log({
        "epoch": epoch + 1,
        
        "Loss/Train": epoch_train_loss,
        "Loss/Val": epoch_val_loss,
        
        "Accuracy/Train": epoch_train_acc,
        "Accuracy/Val": epoch_val_acc,
        
        "F1_Score/Train": epoch_train_f1,
        "F1_Score/Val": epoch_val_f1
    })

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] "
              f"| Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.1f}% F1: {epoch_train_f1:.4f} "
              f"| Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.1f}% F1: {epoch_val_f1:.4f}")

# 8. 테스트
print("\n--- 최종 테스트(Test) 평가를 시작합니다 ---")

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

model.eval() 
test_loss = 0.0

all_test_preds = []
all_test_targets = []

with torch.no_grad(): 
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        test_loss += loss.item() * batch_x.size(0)
        
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        all_test_preds.extend(preds.cpu().numpy())
        all_test_targets.extend(batch_y.cpu().numpy())

print(f"test 모델 예측값 : {all_test_preds}")
print(f"test 정답       : {all_test_targets}")
print("-" * 50)

    # 최종 평가지표 계산
final_test_loss = test_loss / len(test_dataset)
final_test_acc = accuracy_score(all_test_targets, all_test_preds) * 100
final_test_f1 = f1_score(all_test_targets, all_test_preds, average='binary')

print(f"최종 Test Loss: {final_test_loss:.4f}")
print(f"최종 Test Acc:  {final_test_acc:.1f}%")
print(f"최종 Test F1:   {final_test_f1:.4f}")

wandb.log({
    "Loss/Test": final_test_loss,

    "Accuracy/Test": final_test_acc,

    "F1_Score/Test": final_test_f1
})

wandb.finish()
print("모든 실험과 기록이 성공적으로 완료되었습니다!")

