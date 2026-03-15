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

# 1. 단일 Linear 모델 정의 (은닉층 없이 입력에서 출력으로 바로 연결 (단일 레이어)) -> 입력된 데이터 X에 가중치 W를 곱하고 편향 b를 더하는 Y = XW + b 단일 레이어 연산
    # 단일 레이어(Single Layer) 모델은 완벽한 선형 모델 = 엄밀히 말하면 전통적인 통계학의 로지스틱 회귀
class LinearClassifier(nn.Module):

    def __init__(self, input_dim):
        super(LinearClassifier, self).__init__()
        
        # 연결 구조: 입력층 -> [가중치 연산 1번] -> 출력층
        # 이진 분류이므로 출력 노드는 1개. 출력된 1개의 값이 '정답(클래스 1)일 확률'을 의미
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # 활성화 함수 없이 날것의 값(Logit)을 그대로 반환
        # Q. 왜 모델 안에서 Sigmoid를 안 쓰고 밖에서 쓰나요? (BCEWithLogitsLoss)
        #   - 이 부분이 파이토치로 분류 모델을 짤 때 가장 중요한 팁
        #   - 초보자들은 보통 모델 forward 안에 nn.Sigmoid()를 넣고, 밖에서 nn.BCELoss()를 씀
        #   - 하지만 이 둘을 합쳐놓은 nn.BCEWithLogitsLoss()를 사용하는 것이 수학적으로 훨씬 안정적(수치적 안정성)이고 학습도 잘 됨
        #   - 그래서 학습할 때는 모델이 날것의 값(Logit)을 출력하게 놔두고, Loss 함수가 알아서 Sigmoid를 적용해 오차를 계산하도록 위임하는 방식 사용
        out = self.linear(x)
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
hidden_dim = len(X[0])
print(f"hidden_dim : {hidden_dim}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")
print("-" * 50)

model = LinearClassifier(input_dim=hidden_dim).to(device)

# 4. 손실 함수 & 최적화(Optimizer) 함수

    # PyTorch에서 CrossEntropyLoss는 기본적으로 다중 클래스 분류를 위해 설계된 함수
    # 따라서 이진 분류(Binary Classification)를 할 때도 이를 '클래스가 2개인 다중 클래스 분류'로 취급해서 접근해야 함
    # 클래스 0일 점수'와 '클래스 1일 점수' 총 2개의 값을 출력 (출력 노드가 2개 (클래스 0에 대한 Logit, 클래스 1에 대한 Logit))
    # 내부에 LogSoftmax 연산을 포함하고 있음. 따라서 두 개의 출력값(Logit)의 합이 1이 되도록 확률로 변환하여 오차를 계산
    # 정답 클래스의 인덱스 번호인 [1, 0] 처럼 1차원 정수 배열로 넣어줘야함
# criterion = nn.CrossEntropyLoss()

    # 수학적으로 이진 분류 상황에서 1개의 노드 + Sigmoid(BCE) 조합과 2개의 노드 + Softmax(CrossEntropy) 조합은 본질적으로 완전히 동일한 결과
    # 하지만 실무에서는 이진 분류를 할 때 앞서 보여드렸던 BCEWithLogitsLoss를 사용하는 것을 더 선호.
    # 왜냐하면 노드를 1개만 쓰기 때문에 연산해야 할 가중치 파라미터 수가 절반으로 줄어들어(예: 4096 * 1 vs 4096 * 2), 메모리와 계산 속도 측면에서 아주 약간 더 효율적이기 때문
criterion = nn.BCEWithLogitsLoss()  # 클래스 1일 확률 하나만 구함

    # 차원이 데이터 수보다 압도적으로 크므로 weight_decay(L2 정규화)를 주어 과적합 방지
lr = 0.001
weight_decay = 1e-4
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 5. DataLoader 생성
batch_size = 32
    # 160개의 데이터 -> 5번의 스텝(Step): 전체 학습 데이터가 160개이므로, 160을 32로 나누면 딱 5묶음이 나옴.
    # 즉, 모델은 한 에포크(Epoch, 전체 데이터를 한 번 다 보는 것) 동안 총 5번에 걸쳐 가중치(Weight)를 업데이트하며 학습하게 됨

summary(model, input_size=(batch_size,hidden_dim))

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

# 6. Wandb 초기화 
epochs = 50

wandb.init(
    project="Neural-Network", 
    name=f"single-epochs_{epochs}-lr_{lr}-opti_AdamW-wd_{weight_decay}",     
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
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
