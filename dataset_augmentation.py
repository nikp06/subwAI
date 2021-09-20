from PIL import Image
import os

"""
Script for flipping all training images horizontally in order to double the size of the training set.
Changes labels accordingly if necessary (left flipped -> right).
"""

PATH_TO_IMAGES = 'images\\training2'
images = []
labels = []

total = 0
for folder in os.listdir(PATH_TO_IMAGES):
    folder_path = os.path.join(PATH_TO_IMAGES, str(folder))
    total += len(os.listdir(folder_path))

counter = 0
for folder in os.listdir(PATH_TO_IMAGES):
    folder_path = os.path.join(PATH_TO_IMAGES, str(folder))
    for image in os.listdir(folder_path):
        counter += 1
        progress = counter/total*100

        print(f"[{int(round(progress)/10+1)*'='}>{int(10-round(progress)/10+1)*'.'}] {round(progress, 2)}% done "
              f"({counter}/{total} images flipped!)")

        # open the original image
        original_img = Image.open(os.path.join(folder_path, image))

        if not os.path.exists(os.path.join(PATH_TO_IMAGES, str(folder+'_flipped'))):
            os.makedirs(os.path.join(PATH_TO_IMAGES, str(folder+'_flipped')))

        # Flip the original image horizontally
        horz_img = original_img.transpose(method=Image.FLIP_LEFT_RIGHT)
        horz_img.save(os.path.join(PATH_TO_IMAGES, str(folder+'_flipped'), image))

        # close all our files object
        original_img.close()
        horz_img.close()
