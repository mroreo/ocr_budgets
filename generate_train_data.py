# -*- coding: utf-8 -*-
"""
Created on Thu May  9 00:03:01 2019

@author: bcheung
"""
import uuid
import pandas as pd
import glob as glob
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

CHARS ='AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789.:,;"(!?)+-*/='

font_list = glob.glob('./fonts/*.ttf')

train_labels = {}
for f in font_list:
    font_name = f.split('\\')[1].split('.')[0]
    font = ImageFont.truetype(f,14)
    for c in CHARS:
        id_key = uuid.uuid4()
        img=Image.new("L", (64,64))
        draw = ImageDraw.Draw(img)
        draw.text((32, 32),c,(255),font=font)
        draw = ImageDraw.Draw(img)
        img.save("./train_labels/{}.jpg".format(id_key), "JPEG")
        label_description = {'font':font_name,
                             'char':c}
        train_labels[id_key] = label_description
            
train_labels_df = pd.DataFrame.from_dict(train_labels,orient='index').reset_index()
train_labels_df.columns = ['id_key','font','target']
train_labels_df.to_csv('train_labels.csv',index=False)