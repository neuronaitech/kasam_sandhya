# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:29:51 2018

@author: user
"""

import cvlib as cv
import cv2
image = cv2.imread('D:\ML\objdet\dog.jpg')
bbox, label, conf = cv.detect_common_objects(image)
print(label, conf)











