import matplotlib.pyplot as plt

# 1. 실험 결과 데이터 (미리 입력해 두었습니다)
w_popqa = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
em_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
f1_scores = [0.2582, 0.1734, 0.1448, 0.1133, 0.1219, 0.1552, 0.1267, 0.0933, 0.0933, 0.0333, 0.0]

# 2. 그래프 크기 및 스타일 설정
plt.figure(figsize=(10, 6))

# 3. 데이터 플로팅 (선 굵기, 마커 스타일 지정)
plt.plot(w_popqa, em_scores, marker='o', linestyle='-', linewidth=2.5, color='tab:blue', label='Exact Match (EM)')
plt.plot(w_popqa, f1_scores, marker='s', linestyle='--', linewidth=2.5, color='tab:orange', label='F1 Score')

# 4. 논문용 제목 및 라벨 세팅
plt.title('Target : PopQA / Noise : HotpotQA', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Noise LoRA Weight ($w_{noise}$)', fontsize=14)
plt.ylabel('Evaluation Score', fontsize=14)

# 5. 축 눈금 및 그리드 설정
plt.xticks(w_popqa, fontsize=11)
plt.yticks(fontsize=11)
# Y축 최대값을 새로운 데이터에 맞게 0.35로 축소 조정
plt.ylim(-0.02, 0.35) 
plt.grid(True, linestyle=':', alpha=0.7)

# 7. 범례 표시
plt.legend(loc='upper right', fontsize=12)

# 8. 여백 정리 및 고화질 이미지로 저장
plt.tight_layout()
plt.savefig('interpolation_graph.png', dpi=300, bbox_inches='tight')
print("그래프가 'interpolation_graph.png'로 고화질 저장되었습니다!")
