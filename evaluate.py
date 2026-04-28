import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from models.multitask_model import MultiTaskModel
from utils.dataset_loader import CurrencyDataset
from utils.preprocessing import get_transforms
from utils.texture_features import extract_texture_features

device = torch.device("cpu")

if __name__ == "__main__":

    print("Loading dataset...")
    dataset = CurrencyDataset("dataset", transform=get_transforms())

    # 🔥 Use only 20% for validation (FAST)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    _, val_dataset = random_split(dataset, [train_size, val_size])

    loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Validation samples: {len(val_dataset)}")

    print("Loading model...")
    model = MultiTaskModel().to(device)
    model.load_state_dict(torch.load("cpu_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating...")

    with torch.no_grad():
        for images, auth_labels, denom_labels in loader:

            texture_batch = []

            for img in images:
                img_np = img.permute(1,2,0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                texture = extract_texture_features(img_np)
                texture_batch.append(texture)

            texture_batch = torch.tensor(np.array(texture_batch)).float()

            auth_out, denom_out = model(images, texture_batch)

            preds = torch.argmax(auth_out, dim=1)

            all_preds.extend(preds.numpy())
            all_labels.extend(auth_labels.numpy())

    # ================= RESULTS =================
    acc = accuracy_score(all_labels, all_preds) * 100

    print("\n================ RESULTS ================")
    print(f"Accuracy: {acc:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\n✅ Evaluation Completed!")
