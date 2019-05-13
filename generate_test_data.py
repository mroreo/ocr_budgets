# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:12:51 2019

@author: bcheung
"""

import re
import cv2
import imutils
import os
import numpy as np
import uuid
import glob as glob
import pandas as pd
from imutils import contours
from pdf2image import convert_from_path

RESAVE_DATA = False
TRAIN_PIXELS = 64
BUDGET = 'weehawken_annual_budget_2018'

def pdfImageConversion(pdf,input_folder='./budget_pdfs',output_folder='./images'):
    """
    The function converts a pdf file into a bunch of jpg files.
    """
    pages = convert_from_path('{}/{}'.format(input_folder,pdf), 500)
    outputfile = re.sub('.pdf','',pdf)
    
    #Create the directory if it does not exist
    if not(os.path.exists('{}/{}'.format(output_folder,outputfile))):
        os.makedirs('{}/{}'.format(output_folder,outputfile))

    #Save each page of the pdf as a jpg file
    for idx, page in enumerate(pages):
        page.save('{}/{}/page_{}.jpg'.format(output_folder,outputfile,idx), 'JPEG')
        
def resizeResolution(img,w=1200,h=1800,rgb=True):
    """
    Standardize the size of the jpg files
    """
    if rgb:
        img_w,img_h,img_d = img.shape
    else:
        img_w,img_h = img.shape
    return(cv2.resize(img,(0,0),fx=round(1/(img_w/w),2),fy=round(1/(img_h/h),2)))
    
def identifyBoundingBoxes(img):
    """
    Find the bounding boxes after converting the images to negatives
    """
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
    
def drawRedRectangles(img,contours_dict,line_width=2,batch=True):
    """
    Draw a red rectangle over each bounding box identified.
    """
    img_copy = img.copy()
    
    if batch:
        for i in contours_dict:
            dim = contours_dict[i]
            img_copy[dim[1]-line_width:dim[1]+dim[3]+line_width,dim[0]-line_width:dim[0]+line_width,:] = [0,0,255]
            img_copy[dim[1]-line_width:dim[1]+dim[3]+line_width,dim[0]+dim[2]-line_width:dim[0]+dim[2]+line_width,:] = [0,0,255]
            img_copy[dim[1]-line_width:dim[1]+line_width,dim[0]-line_width:dim[0]+dim[2]+line_width,:] = [0,0,255]
            img_copy[dim[1]+dim[3]-line_width:dim[1]+dim[3]+line_width,dim[0]:dim[0]-line_width+dim[2]+line_width,:] = [0,0,255]
    else:
        dim = contours_dict
        img_copy[dim[1]-line_width:dim[1]+dim[3]+line_width,dim[0]-line_width:dim[0]+line_width,:] = [0,0,255]
        img_copy[dim[1]-line_width:dim[1]+dim[3]+line_width,dim[0]+dim[2]-line_width:dim[0]+dim[2]+line_width,:] = [0,0,255]
        img_copy[dim[1]-line_width:dim[1]+line_width,dim[0]-line_width:dim[0]+dim[2]+line_width,:] = [0,0,255]
        img_copy[dim[1]+dim[3]-line_width:dim[1]+dim[3]+line_width,dim[0]:dim[0]-line_width+dim[2]+line_width,:] = [0,0,255]
    return(img_copy)
    
def padImgZeros(img,pixels=TRAIN_PIXELS):
    
    #IF the image exceeds the pixels, trim the image
    if img.shape[0] > TRAIN_PIXELS or img.shape[1] > TRAIN_PIXELS:
        img = img[:min(TRAIN_PIXELS,img.shape[0]),:min(TRAIN_PIXELS,img.shape[1])]
    
    img_zeros = np.zeros([TRAIN_PIXELS,TRAIN_PIXELS])
        
    x_offset = int(img.shape[0]/2)
    y_offset = int(img.shape[1]/2)
    
    x_center = int(img_zeros.shape[0]/2)
    y_center = int(img_zeros.shape[1]/2)
    
    img_zeros[x_center-x_offset:x_center-x_offset+img.shape[0],
              y_center-y_offset:y_center-y_offset+img.shape[1]] = img
    return(img_zeros)
    
if __name__ == '__main__':
    
    if RESAVE_DATA:
        pdfImageConversion('{}.pdf'.format(BUDGET))
    
    jpg_files = glob.glob('./images/{}/*.jpg'.format(BUDGET))
    train_ids = []
    for jpg in jpg_files:
        img = cv2.imread(jpg)
        contours_dict = identifyBoundingBoxes(img)
        
        #Check the bounding boxes to see if they make sense
        resized_img = resizeResolution(img,rgb=True)
        #cv2.imshow('boundingboxes',drawRedRectangles(resized_img,contours_dict))
        
        #Save the train labels. Save the images as a gray scale image
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayimg = cv2.threshold(grayimg, 10, 255, cv2.THRESH_BINARY_INV)[1]
        resized_grayimg = resizeResolution(grayimg,rgb=False)
        #cv2.imshow('grayscaleimg',resized_grayimg)
        
        
        for i in contours_dict:
            id_key = uuid.uuid4()
            dim = contours_dict[i]
            train_img = resized_grayimg[dim[1]:dim[1]+dim[3],dim[0]:dim[0]+dim[2]]
            #cv2.imshow('image_check',train_img)
            #cv2.imshow('image_check_bounding_box',drawRedRectangles(resized_img,dim,batch=False))
            train_img_padded = padImgZeros(train_img)
            train_img_bounded = drawRedRectangles(resized_img,dim,batch=False)
            cv2.imwrite('./test_labels/{}.jpg'.format(id_key), train_img_padded, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            cv2.imwrite('./test_labels/image_boxed/{}_bounding_box.jpg'.format(id_key), train_img_bounded, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            train_ids.append(id_key)
            
    train_ids_df = pd.DataFrame({'id_key':train_ids})
    train_ids_df.to_csv('target.csv',index=False)
