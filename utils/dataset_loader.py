import os
from PIL import Image
from torch.utils.data import Dataset

class CurrencyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.transform = transform

        self.auth_map = {"real": 0, "fake": 1}
        self.country_map = {"India": 0, "Indonesia": 1}

        self.denom_map = {}
        denom_counter = 0

        for country in sorted(os.listdir(root_dir)):
            country_path = os.path.join(root_dir, country)

            if not os.path.isdir(country_path):
                continue

            for auth in os.listdir(country_path):
                if auth not in ["real", "fake"]:
                    continue
                auth_path = os.path.join(country_path, auth)
                

                for denom in os.listdir(auth_path):
                    denom_path = os.path.join(auth_path, denom)

                    if denom not in self.denom_map:
                        self.denom_map[denom] = denom_counter
                        denom_counter += 1

                    for img in os.listdir(denom_path):
                        if not img.lower().endswith((".jpg", ".png", ".jpeg")):
                            continue

                        img_path = os.path.join(denom_path, img)

                        self.data.append((
                            img_path,
                            self.auth_map[auth],
                            self.denom_map[denom],
                            self.country_map[country]
                        ))

        print("Denomination Mapping:", self.denom_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, auth, denom, country = self.data[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, auth, denom, country