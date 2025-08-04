import os
import shutil

# Source folders
image_src = "images"
label_src = "labels"

# Destination folders (same files go to both train and val)
image_train_dst = "data/images/train"
image_val_dst   = "data/images/val"
label_train_dst = "data/labels/train"
label_val_dst   = "data/labels/val"

# Make sure output folders exist
os.makedirs(image_train_dst, exist_ok=True)
os.makedirs(image_val_dst, exist_ok=True)
os.makedirs(label_train_dst, exist_ok=True)
os.makedirs(label_val_dst, exist_ok=True)

# Copy all .jpg files into train and val
for file in os.listdir(image_src):
    if file.endswith(".jpg"):
        shutil.copy(os.path.join(image_src, file), os.path.join(image_train_dst, file))
        shutil.copy(os.path.join(image_src, file), os.path.join(image_val_dst, file))

# Copy all .txt files into train and val
for file in os.listdir(label_src):
    if file.endswith(".txt"):
        shutil.copy(os.path.join(label_src, file), os.path.join(label_train_dst, file))
        shutil.copy(os.path.join(label_src, file), os.path.join(label_val_dst, file))

print("✅ Copied all data into train/ and val/ folders.")
