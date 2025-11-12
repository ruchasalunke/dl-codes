import os
import shutil
import random

# Set these paths as needed
src_dir = r"./101_ObjectCategories/101_ObjectCategories"   # Folder with original class folders
dst_dir = "."                         # Destination folder for train/valid/test

# Split definition: 50% train, 25% valid, 25% test
splits = ['train', 'valid', 'test']
split_ratio = [0.5, 0.25, 0.25]

# Make sure random is reproducible
random.seed(42)

for class_name in os.listdir(src_dir):
    class_path = os.path.join(src_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        continue  # skip empty folders or annotation dirs

    random.shuffle(images)
    n_total = len(images)
    n_train = int(split_ratio[0] * n_total)
    n_valid = int(split_ratio[1] * n_total)
    n_test = n_total - n_train - n_valid  # Ensure all images used

    split_idx = {
        'train': (0, n_train),
        'valid': (n_train, n_train + n_valid),
        'test': (n_train + n_valid, n_total)
    }

    for split in splits:
        split_dir = os.path.join(dst_dir, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        start, end = split_idx[split]
        for img_name in images[start:end]:
            src_img = os.path.join(class_path, img_name)
            dst_img = os.path.join(split_dir, img_name)
            shutil.copy2(src_img, dst_img)

    print(f'Split done for class: {class_name}')
import os
print(os.listdir('.'))
import os
print("Current directory files:")
print(os.listdir('.'))
print("Classes in src_dir:")
print(os.listdir('./101_ObjectCategories'))

print('Dataset splitting complete! You can now use the folders for PyTorch training.')
