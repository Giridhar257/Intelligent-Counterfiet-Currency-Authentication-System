# import cv2
# import numpy as np
# import os

# # -------- INPUT IMAGE PATH --------
# image_path = "./dataset/India/real/2000/12.jpg"   # change this
# output_dir = "output_images"

# os.makedirs(output_dir, exist_ok=True)

# # -------- LOAD IMAGE --------
# image = cv2.imread(image_path)

# # -------- 1. RESIZE --------
# resized = cv2.resize(image, (224, 224))
# cv2.imwrite(os.path.join(output_dir, "resized.jpg"), resized)

# # -------- 2. NORMALIZATION --------
# # convert to float and normalize [0,1]
# normalized = resized / 255.0

# # Save normalized (convert back to 0-255 for saving)
# normalized_save = (normalized * 255).astype(np.uint8)
# cv2.imwrite(os.path.join(output_dir, "normalized.jpg"), normalized_save)

# # -------- 3. GRAYSCALE (optional) --------
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# cv2.imwrite(os.path.join(output_dir, "grayscale.jpg"), gray)

# # -------- 4. CONTRAST ENHANCEMENT --------
# # Using CLAHE (best for papers)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# enhanced = clahe.apply(gray)
# cv2.imwrite(os.path.join(output_dir, "enhanced.jpg"), enhanced)

# # -------- 5. NOISE REDUCTION --------
# denoised = cv2.GaussianBlur(resized, (5,5), 0)
# cv2.imwrite(os.path.join(output_dir, "denoised.jpg"), denoised)

# print("✅ All preprocessing images saved in:", output_dir)

# import matplotlib.pyplot as plt
# import numpy as np

# # =========================
# # 🔹 Example Data (replace with your actual values)
# # =========================

# epochs = list(range(1, 11))

# train_acc = [0.72, 0.78, 0.82, 0.86, 0.89, 0.91, 0.93, 0.94, 0.95, 0.96]
# val_acc   = [0.70, 0.75, 0.80, 0.83, 0.87, 0.88, 0.90, 0.91, 0.92, 0.93]

# train_loss = [0.8, 0.6, 0.5, 0.42, 0.35, 0.30, 0.25, 0.22, 0.20, 0.18]
# val_loss   = [0.85, 0.65, 0.55, 0.48, 0.40, 0.38, 0.34, 0.32, 0.30, 0.28]

# precision = 0.94
# recall    = 0.92
# f1        = 0.93


# # =========================
# # 🔹 1. Accuracy vs Epochs
# # =========================
# plt.figure()
# plt.plot(epochs, train_acc, marker='o', label="Train Accuracy")
# plt.plot(epochs, val_acc, marker='o', linestyle='--', label="Validation Accuracy")

# plt.title("Training vs Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()

# plt.savefig("accuracy_curve.png")
# plt.show()


# # =========================
# # 🔹 2. Loss vs Epochs
# # =========================
# plt.figure()
# plt.plot(epochs, train_loss, marker='o', label="Train Loss")
# plt.plot(epochs, val_loss, marker='o', linestyle='--', label="Validation Loss")

# plt.title("Training vs Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()

# plt.savefig("loss_curve.png")
# plt.show()


# # =========================
# # 🔹 3. Metrics Comparison Bar Graph
# # =========================
# labels = ["Precision", "Recall", "F1 Score"]
# values = [precision, recall, f1]

# plt.figure()
# bars = plt.bar(labels, values)

# plt.title("Performance Metrics Comparison")
# plt.ylabel("Score")
# plt.ylim(0, 1)

# # Add values on bars
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center')

# plt.savefig("metrics_bar.png")
# plt.show()
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # =========================
# # 🔹 Replace with your actual values
# # =========================
# y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]   # Actual labels
# y_pred = [0, 1, 0, 0, 0, 1, 0, 0, 1, 1]   # Predicted labels

# # 0 = Real, 1 = Fake
# class_names = ["Real", "Fake"]

# # =========================
# # 🔹 Confusion Matrix
# # =========================
# cm = confusion_matrix(y_true, y_pred)

# plt.figure()
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# disp.plot()

# plt.title("Confusion Matrix for Currency Classification")

# # Save image for paper
# plt.savefig("confusion_matrix.png")

# plt.show()
# import matplotlib.pyplot as plt

# # =========================
# # 🔹 Replace with your values
# # =========================

# methods = ["Traditional ML", "CNN", "Hybrid DL", "Proposed Model"]

# accuracy = [0.82, 0.88, 0.91, 0.96]
# complexity = [0.3, 0.6, 0.8, 0.5]   # lower is better
# interpretability = [0.7, 0.4, 0.3, 0.8]

# # =========================
# # 🔹 Plot Line Graph
# # =========================

# plt.figure()

# plt.plot(methods, accuracy, marker='o', label="Accuracy")
# plt.plot(methods, complexity, marker='o', linestyle='--', label="Complexity")
# plt.plot(methods, interpretability, marker='o', linestyle='-.', label="Interpretability")

# plt.title("Comparison of Proposed Method with Existing Approaches")
# plt.xlabel("Methods")
# plt.ylabel("Score")
# plt.grid()
# plt.legend()

# plt.savefig("comparison_graph.png")
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt

# # =========================
# # 🔹 Metrics
# # =========================

# labels = ["Accuracy", "Precision", "Recall", "F1 Score", "Efficiency"]

# proposed = [0.96, 0.94, 0.92, 0.93, 0.90]
# cnn_model = [0.90, 0.88, 0.87, 0.88, 0.80]

# # =========================
# # 🔹 Radar Setup
# # =========================

# angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# # close the loop
# proposed += proposed[:1]
# cnn_model += cnn_model[:1]
# angles += angles[:1]

# # =========================
# # 🔹 Plot
# # =========================

# plt.figure()

# ax = plt.subplot(111, polar=True)

# ax.plot(angles, proposed, label="Proposed Model")
# ax.plot(angles, cnn_model, linestyle='--', label="CNN Model")

# ax.fill(angles, proposed, alpha=0.1)
# ax.fill(angles, cnn_model, alpha=0.1)

# ax.set_thetagrids(np.degrees(angles[:-1]), labels)

# plt.title("Radar Chart Comparison")
# plt.legend(loc="upper right")

# plt.savefig("radar_chart.png")
# plt.show()
# import matplotlib.pyplot as plt

# # =========================
# # 🔹 Replace with your actual results
# # =========================
# models = [
#     "CNN Only",
#     "CNN + Texture",
#     "CNN + Texture + OCR",
#     "Full Model"
# ]

# accuracy = [0.85, 0.89, 0.92, 0.96]

# # =========================
# # 🔹 Plot
# # =========================
# plt.figure()

# bars = plt.bar(models, accuracy)

# plt.title("Ablation Study of Proposed Model")
# plt.xlabel("Model Variants")
# plt.ylabel("Accuracy")
# plt.ylim(0, 1)

# # rotate labels (important for readability)
# plt.xticks(rotation=20)

# # add values on top
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01,
#              f"{yval:.2f}", ha='center')

# plt.savefig("ablation_study.png")
# plt.show()
# import numpy as np

# models = ["CNN", "CNN+Tex", "CNN+Tex+OCR", "Full"]

# accuracy = [0.85, 0.89, 0.92, 0.96]
# f1 = [0.83, 0.88, 0.91, 0.95]

# x = np.arange(len(models))

# plt.figure()

# plt.bar(x - 0.2, accuracy, 0.4, label="Accuracy")
# plt.bar(x + 0.2, f1, 0.4, label="F1 Score")

# plt.xticks(x, models)
# plt.ylabel("Score")
# plt.title("Ablation Study Comparison")
# plt.legend()
# plt.ylim(0, 1)

# plt.savefig("ablation_grouped.png")
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# =========================
# 🔹 Replace with YOUR actual results
# Format: [Accuracy, Precision, Recall, F1]
# =========================
without_norm = [0.85, 0.82, 0.80, 0.81]
norm_only     = [0.90, 0.88, 0.87, 0.88]
aug_norm      = [0.96, 0.94, 0.92, 0.93]

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

x = np.arange(len(metrics))
width = 0.25

# =========================
# 🔹 Plot
# =========================
plt.figure(figsize=(10, 6))

bars1 = plt.bar(x - width, without_norm, width, label="Without Normalization")
bars2 = plt.bar(x, norm_only, width, label="Normalization Only")
bars3 = plt.bar(x + width, aug_norm, width, label="Augmentation + Normalization")

# =========================
# 🔹 Styling
# =========================
plt.title("Impact of Normalization and Data Augmentation on Model Performance")
plt.xlabel("Performance Metrics")
plt.ylabel("Score")
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()

# =========================
# 🔹 Add value labels
# =========================
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f"{height:.2f}",
            ha='center',
            fontsize=10
        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# =========================
# 🔹 Grid for better readability
# =========================
plt.grid(axis='y', linestyle='--', alpha=0.5)

# =========================
# 🔹 Save (High Quality for paper)
# =========================
plt.savefig("augmentation_comparison.png", dpi=300, bbox_inches='tight')

plt.show()