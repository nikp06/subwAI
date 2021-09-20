import os
import copy
import cv2
import sys
import tensorflow as tf
import numpy as np

"""
Script for checking what a trained model is predicting for individual images from the training data along with
the respective certainty of the prediction.
Also for double-ckecking in case player made wrong actions in certain situations during gathering of training data.
Press 'y' to move on to next image.
Press 'n' to delete image.
Press 'w' to move image to 'up' folder.
Press 'a' to move image to 'left' folder.
Press 's' to move image to 'down' folder.
Press 'd' to move image to 'right' folder.
"""

PATH_TO_IMAGES = 'images\\training2'

# keys
esc = 27
y = 121
n = 110
w = 119
a = 97
s = 115
d = 100
keys = [115, 97, 0, 100, 119]  # ['down', 'left', 'noop', 'right', 'up'] -> order in which it is read

path = os.path.join('models', 'Sequential')
model = tf.keras.models.load_model(path)
actions = ['left', 'right', 'up', 'down', 'noop']
skip = ['noop', 'left', 'down', 'right']
check = ['left', 'right', 'up', 'down', 'noop']

for i, folder in enumerate(os.listdir(PATH_TO_IMAGES)):
	folder_path = os.path.join(PATH_TO_IMAGES, str(folder))
	images = os.listdir(folder_path)
	if folder not in check:
		continue
	for j, image in enumerate(copy.deepcopy(images)):
		if j == 1:
			break
		im = cv2.imread(os.path.join(folder_path, image))
		im_small = cv2.resize(im, (96, 96), interpolation=cv2.INTER_AREA)
		im_small = tf.keras.preprocessing.image.img_to_array(im_small)[:, :, :3]  # convert image to array
		im_small = tf.expand_dims(im_small, 0)  # create a batch
		predictions = model.predict(im_small)
		prediction = actions[np.argmax(predictions)]
		certainty = round(np.amax(predictions)*100, 2)
		while True:
			f = keys[i]
			cv2.putText(im, 'Actual: '+folder, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
			color = (0, 255, 0) if prediction == folder else (0, 0, 255)
			cv2.putText(im, 'Prediction: '+prediction, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
			cv2.putText(im, 'Certainty: '+str(certainty)+'%', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
			cv2.putText(im, str(j+1)+'/'+str(len(images)+1), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
			cv2.imshow('image check', im)
			k = cv2.waitKey(33)
			if k == esc:
				sys.exit("Terminating Program")
			elif k == y and k != f:
				print("all clear")
				break
			elif k == n and k != f:
				print("delete image")
				os.remove(os.path.join(folder_path, image))
				break
			elif k == w and k != f:
				print("move image up")
				path = os.path.join(PATH_TO_IMAGES, 'up')
				os.rename(os.path.join(folder_path, image), os.path.join(path, str(len(os.listdir(path))+2)+'.png'))
				break
			elif k == a and k != f:
				print("move image to left")
				path = os.path.join(PATH_TO_IMAGES, 'left')
				os.rename(os.path.join(folder_path, image), os.path.join(path, str(len(os.listdir(path))+2)+'.png'))
				break
			elif k == s and k != f:
				print("move image to down")
				path = os.path.join(PATH_TO_IMAGES, 'down')
				os.rename(os.path.join(folder_path, image), os.path.join(path, str(len(os.listdir(path))+2)+'.png'))
				break
			elif k == d and k != f:
				path = os.path.join(PATH_TO_IMAGES, 'right')
				os.rename(os.path.join(folder_path, image), os.path.join(path, str(len(os.listdir(path))+2)+'.png'))
				print("move image to right")
				break
			elif k == -1:
				continue

print("done")
