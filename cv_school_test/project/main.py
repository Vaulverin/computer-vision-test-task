import cv2
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

annotations_dir = '../annotations/'
images_dir = '../images/'
fragments_dir = '../fragments/'
gray_dir = '../fragments_greyscale/'
flip_dir = '../fragments_flip/'
norm_dir = '../fragments_norm/'
noise_dir = '../fragments_noise/'


def process_image(image_name, image, ann, index):
    # Вырезаем нужную часть картинки - основное задание 1
    crop_img = image[ann[1]:ann[3], ann[0]:ann[2]]
    image_name = ''.join([image_name, '_', str(index)])
    cv2.imwrite(''.join([fragments_dir, image_name, '.png']), crop_img)

    # Сохраняем в серых тонах - доп. задание 1
    gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(''.join([gray_dir, image_name, '_gray.png']), gray_image)

    # Переворачиваем картинку - доп. задание 2
    flipped_image = cv2.flip(crop_img, 0)
    cv2.imwrite(''.join([flip_dir, image_name, '_flip.png']), flipped_image)

    # Нормализуем картинку - доп. задание 3
    norm_image = crop_img.copy()
    cv2.normalize(crop_img, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite(''.join([norm_dir, image_name, '_norm.png']), norm_image)

    # Добавляем помехи - доп. задание 4
    mean = 0.0  # some constant
    std = 1.0  # some constant (standard deviation)
    noisy_img = gray_image + np.random.normal(mean, std, gray_image.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)  # we might get out of bounds due to noise
    cv2.imwrite(''.join([noise_dir, image_name, '_noise.png']), noisy_img_clipped)

# Разбираем текстовые файлы с частями картинок
for f in listdir(annotations_dir):
    file_path = join(annotations_dir, f)
    if not isfile(file_path):
        continue
    annotations = pd.read_csv(file_path, sep=",", header=None)
    name = f.split('.')
    img = cv2.imread(images_dir + name[0] + '.png')

    # Каждая строка - это часть картинки, нам нужно обработать каждую
    for i, row in annotations.iterrows():
        process_image(name[0], img, row, i)
