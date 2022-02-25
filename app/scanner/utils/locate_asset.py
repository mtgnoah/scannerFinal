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



def crop_image(image, area):
	''' Uses PIL to crop an image, given its area.
	Input:
		image - PIL opened image
		Area - Coordinates in tuple (xmin, ymax, xmax, ymin) format '''
	assert image.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(image) to crop_image() input image.'
	im = Image.fromarray(image)
	imgFixer = np.asarray(im)
	img = imgFixer.crop(area)
	basewidth = 200
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	cropped_image = img.resize((basewidth,hsize), Image.ANTIALIAS)

	
	
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
	images.append((xyxyCoords, crop_image(image, xyxyCoords)))
	
	return images
def ocrScan(cropped_images) -> List:
	ocr_results = None
		
	if cropped_images == []:
		return None 	 
	else:
		ocr_results = OCR.ocr(cropped_images)
		
		k=[]
		v=[]	
		
		for i in range(len(ocr_results)):				
						v.append(ocr_results[i][0])
						k.append(ocr_results[i][1])

		t=dict(zip(k, v))
			
		return t
		
