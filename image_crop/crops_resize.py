import os
from PIL import Image
from tqdm import tqdm

def resize_imagefolder(src_root, dst_root, size=(256, 256)):
    """
    Перегоняет датасет формата ImageFolder (папки = классы)
    в новую директорию, приводя все картинки к size.
    """
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    for split in ["train", "valid", "test"]:  # можно поменять под свой датасет
        split_src = os.path.join(src_root, split)
        split_dst = os.path.join(dst_root, split)

        if not os.path.exists(split_src):
            continue  # вдруг этого сплита нет

        print(f"Обрабатываю {split}...")
        for class_name in tqdm(os.listdir(split_src)):
            class_src = os.path.join(split_src, class_name)
            class_dst = os.path.join(split_dst, class_name)

            os.makedirs(class_dst, exist_ok=True)

            for fname in os.listdir(class_src):
                fsrc = os.path.join(class_src, fname)
                fdst = os.path.join(class_dst, fname)

                try:
                    with Image.open(fsrc) as img:
                        img = img.convert("RGB")
                        img = img.resize(size, Image.BILINEAR)
                        img.save(fdst, format="JPEG", quality=95)
                except Exception as e:
                    print(f"⚠️ Ошибка на {fsrc}: {e}")


if __name__ == "__main__":
    src_root = "dataset_crops"       # путь к исходному датасету
    dst_root = "dataset_crops_256"   # путь для сохранения нового датасета
    resize_imagefolder(src_root, dst_root, size=(256, 256))