import os
import cv2
import numpy as np

PATCH_SIZE = 256


def extract_patches(image, mask=None):
    h, w, _ = image.shape
    patches = []
    mask_patches = []

    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            img_patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patches.append(img_patch)

            if mask is not None:
                mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                mask_patches.append(mask_patch)

    return patches, mask_patches


def process_folder(image_dir, mask_dir, out_img, out_mask):
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    for name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, name)
        mask_path = os.path.join(mask_dir, name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img_patches, mask_patches = extract_patches(image, mask)

        for i, (ip, mp) in enumerate(zip(img_patches, mask_patches)):
            cv2.imwrite(f"{out_img}/{name}_{i}.png", ip)
            cv2.imwrite(f"{out_mask}/{name}_{i}.png", mp)


if __name__ == "__main__":
    process_folder(
        image_dir="data/LEVIR-CD/train/images",
        mask_dir="data/LEVIR-CD/train/masks",
        out_img="data/LEVIR-CD/patches/images",
        out_mask="data/LEVIR-CD/patches/masks"
    )
