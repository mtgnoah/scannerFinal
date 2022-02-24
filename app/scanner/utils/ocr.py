from typing import List
import threading

import cv2
import google.cloud.vision as vision
import os, io
import sys
from PIL import Image
from typing import Tuple

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'eliteseat-4fbd46832f71.env'

class OCR():
	def initialize(self):
		''' Initialize the OCR '''
		pass

	def ocr_one_image(area, image, imgLabel):
		img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		(thresh, blackAndWhiteImage) = cv2.threshold(img_rgb, 127, 255, cv2.THRESH_BINARY)
		
		client = vision.ImageAnnotatorClient()
		imageGoogle = vision.Image(content=cv2.imencode('.jpg', blackAndWhiteImage)[1].tostring())
		
		response = client.text_detection(image=imageGoogle)
		texts = response.text_annotations

		txt = texts[0].description


		return (txt, imgLabel)

	def ocr(images:List) -> List:
		'''Sends an opened image to Google Cloud Vision API
		Input: images List
		Returns the results from Vision API.'''
		results = []
		for image in images:
			results.append(OCR.ocr_one_image(image[0], image[1], image[2]))

		return results

	def __init__(self):
		self.initialize()
