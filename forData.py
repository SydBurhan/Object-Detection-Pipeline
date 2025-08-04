import os
import shutil

# Source folders
src_images_train = "raw_dataset/images/train"
src_images_val = "raw_dataset/images/val"
src_labels_train = "raw_dataset/labels/train"
src_labels_val = "raw_dataset/labels/val"

# Destination folders
dst_images_train = "datasets/data/images/train"
dst_images_val = "datasets/data/images/val"
dst_labels_train = "datasets/data/labels/train"
dst_labels_val = "datasets/data/labels/val"

def copy_files(src, dst):
    os.makedirs(dst, exist_ok=True)
    for filename in os.listdir(src):
        src_path = os.path.join(src, filename)
        dst_path = os.path.join(dst, filename)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
    print(f"✅ Copied {len(os.listdir(dst))} files to {dst}")

if __name__ == "__main__":
    copy_files(src_images_train, dst_images_train)
    copy_files(src_images_val, dst_images_val)
    copy_files(src_labels_train, dst_labels_train)
    copy_files(src_labels_val, dst_labels_val)
