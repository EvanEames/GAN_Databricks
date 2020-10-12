# Databricks notebook source
from PIL import Image
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import os
import os.path
import cv2

min_ratio = 0.5 #Smallest height ratio between the image and the figure
max_ratio = 0.7 #Largest height ratio between the image and the figure

#Put background image filenames and people image filenames into lists
files = [name for name in os.listdir('./Off') if os.path.isfile('./Off/'+name)]
people = [name for name in os.listdir('./people') if os.path.isfile('./people/'+name)]
'''
for name in os.listdir('./people'):
	filename = str('./people/'+name)
	img2 = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
	print(name, np.shape(img2))
#	print(np.shape(img2)[2] == 4):
#		print(img2[0:3,0:3,3])
'''

for name in os.listdir('./Off'):
	#Read in background image
	filename = str('./Off/'+name)
	img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)[:,:,0:3]
	#Read in person image
	filename2 = str("./people/"+people[random.randint(0,len(people)-1)])
	img2 = cv2.imread(filename2, cv2.IMREAD_UNCHANGED)
	#Resize person to fit the larger image
	ratio = random.uniform(min_ratio,max_ratio)
	img2 = cv2.resize(img2, (int(ratio*np.shape(img2)[1]*np.shape(img)[0]/np.shape(img2)[0]), int(ratio*np.shape(img)[0])), interpolation = cv2.INTER_AREA)
	#Superimpose the person onto the background image
	x_pos = random.randint(0,np.shape(img)[0]-np.shape(img2)[0])
	y_pos = random.randint(0,np.shape(img)[1]-np.shape(img2)[1])
	filter = np.stack((img2[:,:,3], img2[:,:,3], img2[:,:,3]),axis = 2)
	img_tmp = img[x_pos:x_pos+np.shape(img2)[0], y_pos:y_pos+np.shape(img2)[1]]
	combined_img = np.where(filter == 0, img_tmp, img2[:,:,0:3])
	img[x_pos:x_pos+np.shape(img2)[0], y_pos:y_pos+np.shape(img2)[1]] = combined_img
	#write out the combined image
	cv2.imwrite(str('./On/'+name), img)