import os
import random
import shutil


base_dir = os.path.dirname(__file__)
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "annotations")


train_images = os.path.join(base_dir, "images/train")
val_images = os.path.join(base_dir, "images/val")
train_labels = os.path.join(base_dir, "labels/train")
val_labels = os.path.join(base_dir, "labels/val")

# Creating dirs
for d in [train_images, val_images, train_labels, val_labels]:
    os.makedirs(d, exist_ok=True)


image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")]
random.shuffle(image_files)

# Split 80/20
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def copy_files(file_list, dest_img, dest_lbl):
    for fname in file_list:
        name_no_ext = os.path.splitext(fname)[0]
        label_file = name_no_ext + ".txt"

        shutil.copy(os.path.join(images_dir, fname), os.path.join(dest_img, fname))
        shutil.copy(os.path.join(labels_dir, label_file), os.path.join(dest_lbl, label_file))

copy_files(train_files, train_images, train_labels)
copy_files(val_files, val_images, val_labels)

print(f"Done! {len(train_files)} train, {len(val_files)} val images copied.")
