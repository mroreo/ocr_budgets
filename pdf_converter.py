# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:12:51 2019

@author: bcheung
"""

import re
import cv2
import imutils
import numpy as np
from imutils import contours
from pdf2image import convert_from_path

RESAVE_DATA = False

def pdfImageConversion(pdf,input_folder='./budget_pdfs',output_folder='./images'):
    pages = convert_from_path('{}/{}'.format(input_folder,pdf), 500)
    outputfile = re.sub('.pdf','',pdf)

    for idx, page in enumerate(pages):
        page.save('{}/{}/page_{}.jpg'.format(output_folder,outputfile,idx), 'JPEG')
        
def resizeResolution(img,w=1200,h=1800,rgb=True):
    if rgb:
        img_w,img_h,img_d = img.shape
    else:
        img_w,img_h = img.shape
    return(cv2.resize(img,(0,0),fx=round(1/(img_w/w),2),fy=round(1/(img_h/h),2)))
    
def identifyBoundingBoxes(img):
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayimg = cv2.threshold(grayimg, 10, 255, cv2.THRESH_BINARY_INV)[1]
    resized_grayimg = resizeResolution(grayimg,rgb=False)
    refCnts = cv2.findContours(resized_grayimg.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    refCnts = imutils.grab_contours(refCnts)
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
    
    contours_dict = {}
    # loop over the OCR-A reference contours
    for (i, c) in enumerate(refCnts):
        contours_dict[i] = cv2.boundingRect(c)
    return(contours_dict)
    
def drawRedRectangles(img,contours_dict,line_width=2):
    for i in contours_dict:
        dim = contours_dict[i]
        img[dim[1]-line_width:dim[1]+dim[3]+line_width,dim[0]-line_width:dim[0]+line_width,:] = [0,0,255]
        img[dim[1]-line_width:dim[1]+dim[3]+line_width,dim[0]+dim[2]-line_width:dim[0]+dim[2]+line_width,:] = [0,0,255]
        img[dim[1]-line_width:dim[1]+line_width,dim[0]-line_width:dim[0]+dim[2]+line_width,:] = [0,0,255]
        img[dim[1]+dim[3]-line_width:dim[1]+dim[3]+line_width,dim[0]:dim[0]-line_width+dim[2]+line_width,:] = [0,0,255]
    return(img)
        
budget = 'weehawken_annual_budget_2018'
if RESAVE_DATA:
    pdfImageConversion('{}.pdf'.format(budget))
img = cv2.imread('./images/{}/page_3.jpg'.format(budget))
contours_dict = identifyBoundingBoxes(img)

resized_img = resizeResolution(img,rgb=True)
cv2.imshow('boundingboxes',drawRedRectangles(resized_img,contours_dict))
