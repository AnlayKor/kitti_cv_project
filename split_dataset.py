import os
import random
import shutil

src_images = "kitti-8/valid/images"
src_labels = "kitti-8/valid/labels"

base_dist = "data/processed"

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(base_dist, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dist, split, "labels"), exist_ok=True)

images = [f for f in os.listdir(src_images) if f.endswith(('.jpg', '.jpeg', '.png'))]

random.seed(42)
random.shuffle(images)

total = len(images)
train_end = int(total * 0.70)
val_end = train_end + int(total * 0.15)

train_imgs = images[:train_end]
val_imgs = images[train_end:val_end]
test_imgs = images[val_end:]


def move_files(file_list, split_name):
    for img_name in file_list:
        base_name = os.path.splitext(img_name)[0]
        lbl_name = f"{base_name}.txt"

        img_src_path = os.path.join(src_images, img_name)
        lbl_src_path = os.path.join(src_labels, lbl_name)

        img_dst_path = os.path.join(base_dist, split_name, "images", img_name)
        lbl_dst_path = os.path.join(base_dist, split_name, "labels", lbl_name)

        shutil.copy(img_src_path, img_dst_path)

        if os.path.exists(lbl_src_path):
            shutil.copy(lbl_src_path, lbl_dst_path)


print(f"Всего найдено исходных кадров: {total}")
print("Раскладываю данные по пропорциям 70/15/15...")
move_files(train_imgs, "train")
move_files(val_imgs, "val")
move_files(test_imgs, "test")

print("🎉 Готово! Проверь папку data/processed — теперь там идеальный порядок!")
