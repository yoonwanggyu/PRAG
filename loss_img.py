import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# ================================
# 1. 전체 offline 루트
# ================================
model = "llama3.2-1b-instruct"
OFFLINE_ROOT = f"/root/PRAG/offline/{model}/rank=2_alpha=32"
SAVE_ROOT = f"/root/PRAG/{model}_loss_img"

os.makedirs(SAVE_ROOT, exist_ok=True)

AUG_DIR = "aug_model=llama3.2-1b-instruct"

# ================================
# 2. dataset 단위로 순회
# ================================
for dataset_name in tqdm(os.listdir(OFFLINE_ROOT)):
    print(f"1. dataset_name : {dataset_name}")
    dataset_path = os.path.join(OFFLINE_ROOT, dataset_name)
    if not os.path.isdir(dataset_path):
        continue

    # 예: lr=0.0003_epoch=5_direct
    for exp_name in os.listdir(dataset_path):
        print(f"2. exp_name : {exp_name}")
        exp_path = os.path.join(dataset_path, exp_name)
        if not os.path.isdir(exp_path):
            continue

        aug_path = os.path.join(exp_path, AUG_DIR)
        if not os.path.isdir(aug_path):
            continue
        print(f"3. aug_path : {aug_path}")

        # ================================
        # 3. 결과 타입 단위 (inference, bridge, total, ...)
        # ================================
        for result_type in os.listdir(aug_path):
            print(f"4. result_type : {result_type}")
            result_path = os.path.join(aug_path, result_type)
            if not os.path.isdir(result_path):
                continue

            epoch_losses = defaultdict(list)

            # ================================
            # 4. loss.json 수집
            # ================================
            for root, _, files in os.walk(result_path):
                if "loss.json" in files:
                    loss_path = os.path.join(root, "loss.json")
                    with open(loss_path, "r") as f:
                        data = json.load(f)
                        loss_dict = data["loss"]

                        for epoch, values in loss_dict.items():
                            # values = [loss]
                            epoch_losses[epoch].append(values[0])

            if not epoch_losses:
                print(f"[WARN] No loss found: {dataset_name}/{result_type}")
                continue

            # epoch 정렬
            epochs = sorted(
                epoch_losses.keys(),
                key=lambda x: int(x.split("_")[1])
            )
            data = [epoch_losses[e] for e in epochs]

            print(f"  -> passages: {len(data[0])}, epochs: {epochs}")

            # ================================
            # 5. Boxplot
            # ================================
            plt.figure(figsize=(10, 6))
            plt.boxplot(
                data,
                labels=[e.replace("epoch_", "") for e in epochs],
                showfliers=True
            )

            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title(
                f"Training Loss Distribution\n{dataset_name} - {result_type}"
            )
            plt.grid(axis="y", linestyle="--", alpha=0.5)

            # ================================
            # 6. 저장
            # ================================
            save_name = f"{dataset_name}_{result_type}.png"
            save_path = os.path.join(SAVE_ROOT, save_name)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

            print(f"  -> Saved: {save_path}")