import os
import shutil

image_dir = "dataset/Indonesia/train/images"
label_dir = "dataset/Indonesia/train/labels"
output_dir = "dataset_indonesia/train"

os.makedirs(output_dir, exist_ok=True)

# YOUR mapping from data.yaml
class_map = {
    0: "real_100k",
    1: "real_50k",
    2: "fake_100k",
    3: "fake_50k",
    4: "real_100k",
    5: "real_50k",
    6: "fake_100k",
    7: "fake_50k"
}

for file in os.listdir(image_dir):
    if file.endswith(".jpg"):
        img_path = os.path.join(image_dir, file)
        label_path = os.path.join(label_dir, file.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            line = f.readline()
            class_id = int(line.split()[0])

        class_name = class_map[class_id]

        class_folder = os.path.join(output_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)

        shutil.copy(img_path, os.path.join(class_folder, file))

print("✅ Dataset ready!")