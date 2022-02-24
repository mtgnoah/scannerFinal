from PIL import Image
from PIL import ImageFilter
import utils.logger as logger
import time


from torchvision import transforms
#from utils.rotate import rotate
# config import *
from typing import Tuple, List
import sys


import math
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from utils.ocr import OCR


# i = 0
def crop_image(image, area):
	''' Uses PIL to crop an image, given its area.
	Input:
		image - PIL opened image
		Area - Coordinates in tuple (xmin, ymax, xmax, ymin) format '''
	assert image.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(image) to crop_image() input image.'
	im = Image.fromarray(image)
	imgFixer = np.asarray(im)
	# print("IMAGE DATA:")
	# print(*image)
	#im = transforms.ToPILImage()(image)
	#im = Image.fromarray(image)
	img = imgFixer.crop(area)
	basewidth = 200
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	cropped_image = img.resize((basewidth,hsize), Image.ANTIALIAS)
	# global i
	# cropped_image.save("r" + str(i) + ".jpg", "JPEG",dpi=(300,300))
	# i += 1
	
	
	return cropped_image

def locate_asset(image, xyxyCoords) -> List:
	''' Determines where an asset is in the picture, returning
	 a set of coordinates, for the top left, top right, bottom
	 left, and bottom right of the tag
	 Returns:
	 [(area, image)]
	 	Area is the coordinates of the bounding box
	 	Image is the image, opened by PIL.'''
	images = []
	#print(lines)
	# for line in str(lines).split('\n'):
	# 	if "sign" in line:
	# 		continue
	# 	if "photo" in line:
	# 		continue
	# 	#print(line)
	# 	if  "left_x" in line:
	# 		#if 'photo' or 'sign' in line:
	# 		# Extract the nameplate info
	# 		#print(line)
	# 		#area = classifier.extract_info(line)
	# 		# Open image
	#images.append((xyxyCoords, crop_image(image, xyxyCoords)))
	images.append((xyxyCoords, crop_image(image, xyxyCoords)))
	# if images == []:
	# 	#logger.bad("No label found in image.")
	# else:
	# 	logger.good("Found " + str(len(images)) + " label(s) in image.")

	return images
def ocrScan(cropped_images) -> List:
	ocr_results = None
		
	if cropped_images == []:
		#logger.bad("No text found!")
		return None 	 
	else:
		#logger.good("Performing OCR")
		ocr_results = OCR.ocr(cropped_images)
		#print(ocr_results)
		k=[]
		v=[]
			
			
			#fil=filename+'-ocr'
			#with open(fil, 'w+') as f:
		for i in range(len(ocr_results)):			
			
						v.append(ocr_results[i][0])
						k.append(ocr_results[i][1])
						#k.append(inf[i][0][:-1])
							
			#k.insert(0,'Filename')
			#v.insert(0,filename)
		# print("here is v")
		# print(v)
		# print("here is k")
		# print(k)
		t=dict(zip(k, v))
			

		
		#time3 = time.time()
		#print("OCR Time: " + str(time3-time2))

		#end = time.time()
		#logger.good("Elapsed: " + str(end-start))
		# print("here is t")
		# print(t)
		return t
		
