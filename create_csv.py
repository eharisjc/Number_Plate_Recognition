import os
import re
import urllib.request
import simplejson
from PIL import Image
import pandas as pd


def download_image_from_url(url, dataset_save_dir, imag_name):
    fullname = dataset_save_dir + '/' + url.split('/')[-1]



    flags = re.VERBOSE | re.MULTILINE | re.DOTALL
    whitespace = re.compile(r'[ \t\n\r]*', flags)
    decoder = simplejson.JSONDecoder()

    f_obj = open('Indian_Number_plates.json', 'r')
    f_lines = f_obj.readlines()
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    csv_list = []
    for line in f_lines:
        obj, end = decoder.raw_decode(line)
        end = whitespace.match(line, end).end()
        img_url = obj['content']
        print(img_url)
        image_name = 'images/' + img_url.split('/')[-1].split('.')[0] + '.jpg'
        urllib.request.urlretrieve(img_url, image_name)
        im = Image.open(fullname)
        os.remove(fullname)
        im.convert('RGB').save(imag_name, 'JPEG')
        annotation_list = obj['annotation']
        points_list = annotation_list[0]['points']
        points_min = points_list[0]
        points_max = points_list[1]
        img_width = annotation_list[0]['imageWidth']
        img_height = annotation_list[0]['imageHeight']
        x_min = int(points_min['x'] * img_width)
        y_min = int(points_min['y'] * img_height)
        x_max = int(points_max['x'] * img_width)
        y_max = int(points_max['y'] * img_height)
        csv_row = (img_url.split('/')[-1].split('.')[0] + '.jpg',
                   img_width,
                   img_height,
                   'number_plate',
                   x_min, y_min, x_max, y_max)
        csv_list.append(csv_row)

    df = pd.DataFrame(csv_list, columns=column_name)
    df.to_csv('data.csv', index=None)
