# Preprocessing pipeline - Day 1
"""
preprocessing.py
Author: Muhammad Yasir
Handles dataset loading, balancing, and preprocessing
"""

import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def augment_one_image(img):
    """Create one augmented image from existing image"""
    image = img.astype(np.uint8)

    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    angle = random.uniform(-10, 10)
    h, w  = image.shape[:2]
    M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h),
                           borderMode=cv2.BORDER_REFLECT)

    factor = random.uniform(0.8, 1.2)
    image  = np.clip(image.astype(np.float32) * factor,
                     0, 255).astype(np.uint8)
    return image


def balance_dataset(images, labels, names, target=60):
    """
    Balance dataset using undersampling and oversampling
    Target: equal images per class
    """
    balanced_images = []
    balanced_labels = []

    for class_id in range(len(names)):
        mask           = labels == class_id
        class_imgs     = images[mask]
        original_count = len(class_imgs)

        if original_count >= target:
            selected   = np.random.choice(original_count,
                                          target,
                                          replace=False)
            final_imgs = class_imgs[selected]
        else:
            final_imgs = list(class_imgs)
            needed     = target - original_count
            for _ in range(needed):
                src = class_imgs[random.randint(0, original_count-1)]
                aug = augment_one_image(src)
                final_imgs.append(aug)
            final_imgs = np.array(final_imgs)

        balanced_images.append(final_imgs)
        balanced_labels.extend([class_id] * target)

    balanced_images = np.concatenate(balanced_images, axis=0)
    balanced_labels = np.array(balanced_labels)
    return balanced_images, balanced_labels


def preprocess_images(images, target_size=(112, 112)):
    """Resize images and keep float normalization"""
    processed = []
    for img in images:
        img_resized = cv2.resize(img, target_size)
        processed.append(img_resized)
    return np.array(processed, dtype=np.float32)


def create_splits(images, labels):
    """Create stratified train/val/test splits"""

    X_tv, X_test, y_tv, y_test = train_test_split(
        images, labels,
        test_size=0.15,
        random_state=42,
        stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=0.176,
        random_state=42,
        stratify=y_tv
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
