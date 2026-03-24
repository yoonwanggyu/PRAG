import numpy as np
import json

def fit(in_domain_data_path, epsilon=1e-8):
    """
    문서 A(In-domain)의 학습 데이터 100개를 바탕으로 차원별 평균과 표준편차를 계산
    """
    with open(in_domain_data_path, "r", encoding="utf-8") as f:
        in_domain_data = json.load(f)

    hidden_representations = []
    questions = []

    hidden_representations.extend([item["hidden_representation"] for item in in_domain_data])
    questions.extend([item["question"] for item in in_domain_data])
    
    # (100, 2048) 형태의 2D Numpy 배열로 변환
    data_matrix = np.array(hidden_representations)

    # 각 차원(axis=0) 축을 기준으로 통계량 계산
    mean = np.mean(data_matrix, axis=0)
    std = np.std(data_matrix, axis=0) + epsilon # 특정 차원의 분산이 0일 경우 ZeroDivisionError를 방지하기 위한 아주 작은 값
    return mean, std

def compute_anomaly_score(out_domain_data_path, mean, std, method='l2_norm', threshold=50):
    """
    새로운 hidden representation(2048차원)에 대한 Z-Score 기반 이상치 점수를 계산
    """

    with open(out_domain_data_path, "r", encoding="utf-8") as f:
        out_domain_data = json.load(f)

    hidden_representations = []
    questions = []

    hidden_representations.extend([item["hidden_representation"] for item in out_domain_data])
    questions.extend([item["question"] for item in out_domain_data])

    out_domain_data_list = np.array(hidden_representations)

    is_outlier = []
    for x in out_domain_data_list:
        
        # 2048개 차원 각각의 Z-Score 계산: z = (x - μ) / σ
        z_scores = (x - mean) / std
        
    
        # 2048개의 Z-Score를 하나의 통계적 점수로 집계(Aggregation)
        if method == 'l2_norm':
            # Z-Score 벡터의 유클리디안 거리 (가장 추천하는 방식)
            score = np.linalg.norm(z_scores)
        elif method == 'mean_abs':
            # Z-Score 절대값의 평균
            score = np.mean(np.abs(z_scores))
        elif method == 'count_extreme':
            # 절대값이 3을 넘는(3시그마 밖) 차원의 개수
            score = np.sum(np.abs(z_scores) > 3.0)
        else:
            raise ValueError("지원하지 않는 집계 방식입니다.")
        
        if score > threshold:
            is_outlier.append(1)
        else:
            is_outlier.append(0)

    print(f"Test out domain 데이터 {len(hidden_representations)}개 중에 outlier로 판별된 개수는 {is_outlier.count(1)}개 입니다.")
    return is_outlier

# Z-score가 3을 넘어가면 이상치로 본다는 것은 통계학의 가장 기본적이고 유명한 규칙. 
#  그런데 여기서 THRESHOLD = 50.0이 된 이유는, 우리가 비교하려는 값이 단일 차원(1차원)의 Z-score가 아니라 
#  2048개의 Z-score를 하나로 뭉친 '다차원 거리 점수(L2 Norm)'이기 때문
# l2_norm 방식을 사용해 2048개의 Z-score를 하나의 점수로 압축 : 2048개 차원의 값들을 모두 제곱해서 더한 뒤 루트를 씌움
# 정상 데이터라도 2048개 차원의 값들을 모두 제곱해서 더한 뒤 루트를 씌우면, sqrt{1^2 * 2048} = 45.25가 됨
#  완벽하게 정상인 분포의 데이터조차도 2048차원 공간에서는 그 중심(평균)으로부터 약 45만큼의 거리에 모여 있게 됨
# 추가적으로 표준정규분포를 따르는 독립 변수들의 제곱합은 카이제곱 분포를 따름
# 즉, 차원이 2048개이므로 자유도가 2048인 카이제곱 분포를 형성하게 됨. 자유도가 k인 카이제곱 분포의 기댓값은 k이므로, 
#  점수의 제곱합 기댓값은 2048이고 여기에 루트를 씌운 값이 약 45가 되는 것
#  2048차원을 L2 Norm으로 압축한 점수 체계에서는 정상 데이터들이 45 근처에 거대한 산(분포)을 이루게 됨
#  그러므로 THRESHOLD = 50.0이라는 숫자는 "정상 데이터들이 모여 있는 45 근처를 벗어나, 
#  점수가 50 이상으로 비정상적으로 튀었을 때 이를 이상치(OOD)로 판별하겠다"는 의미로 설정한 예시 값

    
        



