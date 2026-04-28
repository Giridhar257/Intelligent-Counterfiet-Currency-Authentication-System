import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import pytesseract
from utils.text_features import extract_text_features
from models.multitask_model import MultiTaskModel
from utils.dataset_loader import CurrencyDataset
from utils.preprocessing import get_train_transforms, get_val_transforms
from utils.texture_features import extract_texture_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    train_dataset_full = CurrencyDataset("../dataset", transform=get_train_transforms())
    val_dataset_full   = CurrencyDataset("../dataset", transform=get_val_transforms())

    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    generator = torch.Generator().manual_seed(42)
    
    train_dataset, _ = random_split(train_dataset_full, [train_size, val_size], generator=generator)
    _, val_dataset   = random_split(val_dataset_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    num_denoms = len(train_dataset_full.denom_map)
    num_countries = len(train_dataset_full.country_map)

    model = MultiTaskModel(
        num_denoms=num_denoms,
        num_countries=num_countries
    ).to(device)

    criterion_auth = nn.CrossEntropyLoss()
    criterion_denom = nn.CrossEntropyLoss()
    criterion_country = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10

    for epoch in range(epochs):

        # ================= TRAIN =================
        model.train()
        train_loss = 0

        for images, auth_labels, denom_labels, country_labels in tqdm(train_loader):

            images = images.to(device)
            auth_labels = auth_labels.to(device)
            denom_labels = denom_labels.to(device)
            country_labels = country_labels.to(device)

            texture_batch = []
            text_batch = []
            for img in images:
                img_np = img.permute(1,2,0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)

                texture = extract_texture_features(img_np)
                texture_batch.append(texture)

                text = pytesseract.image_to_string(img_np)
                text_feat = extract_text_features(text)
                text_batch.append(text_feat)

            texture_batch = torch.tensor(np.array(texture_batch)).float().to(device)
            text_batch = torch.tensor(np.array(text_batch)).float().to(device)
            optimizer.zero_grad()

            auth_out, denom_out, country_out = model(images, texture_batch,text_batch)

            loss_auth = criterion_auth(auth_out, auth_labels)
            loss_denom = criterion_denom(denom_out, denom_labels)
            loss_country = criterion_country(country_out, country_labels)

            loss = loss_auth + 0.5 * loss_denom + 0.3 * loss_country

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"\nEpoch {epoch+1} | Train Loss: {train_loss:.4f}")

        # ================= VALIDATION =================
        model.eval()

        auth_correct = 0
        auth_total = 0

        denom_correct = 0
        denom_total = 0

        country_correct = 0
        country_total = 0

        with torch.no_grad():
            for images, auth_labels, denom_labels, country_labels in val_loader:

                images = images.to(device)
                auth_labels = auth_labels.to(device)
                denom_labels = denom_labels.to(device)
                country_labels = country_labels.to(device)

                texture_batch = []
                text_batch = []
                for img in images:
                    img_np = img.permute(1,2,0).cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)

                    texture = extract_texture_features(img_np)
                    texture_batch.append(texture)

                    # 🔥 OCR
                    text = pytesseract.image_to_string(img_np)
                    text_feat = extract_text_features(text)
                    text_batch.append(text_feat)


                texture_batch = torch.tensor(np.array(texture_batch)).float().to(device)
                text_batch = torch.tensor(np.array(text_batch)).float().to(device)
                auth_out, denom_out, country_out = model(images, texture_batch,text_batch)

                # Auth
                auth_preds = torch.argmax(auth_out, dim=1)
                auth_correct += (auth_preds == auth_labels).sum().item()
                auth_total += auth_labels.size(0)

                # Denomination
                denom_preds = torch.argmax(denom_out, dim=1)
                denom_correct += (denom_preds == denom_labels).sum().item()
                denom_total += denom_labels.size(0)

                # Country
                country_preds = torch.argmax(country_out, dim=1)
                country_correct += (country_preds == country_labels).sum().item()
                country_total += country_labels.size(0)

        auth_acc = 100 * auth_correct / auth_total
        denom_acc = 100 * denom_correct / denom_total
        country_acc = 100 * country_correct / country_total

        print(f"Validation → Auth: {auth_acc:.2f}% | Denom: {denom_acc:.2f}% | Country: {country_acc:.2f}%")

    torch.save(model.state_dict(), "cpu_model.pth")
    print("\n✅ Training Completed and Model Saved!")