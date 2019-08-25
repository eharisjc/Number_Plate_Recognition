import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from pytesseract import pytesseract
from PIL import Image, ImageFont, ImageDraw
from object_detection.utils import label_map_util


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
PATH_TO_CKPT = 'trained_model/frozen_inference_graph.pb'
PATH_TO_LABELS = 'dataset/class_labels.pbtxt'
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        i = 0
        test_img_path = sys.argv[1]
        for image_name in os.listdir(test_img_path):
            image = Image.open(sys.argv[1] + '/' + image_name)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            ymin = boxes[0, 0, 0]
            xmin = boxes[0, 0, 1]
            ymax = boxes[0, 0, 2]
            xmax = boxes[0, 0, 3]
            (im_width, im_height) = image.size
            (xminn, xmaxx, yminn, ymaxx) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))

            cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn), int(ymaxx - yminn),
                                                          int(xmaxx - xminn))

            img_data = sess.run(cropped_image)

            tes_text = pytesseract.image_to_string(img_data, lang='eng')

            punctuations = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
            text = ""
            for char in tes_text:
                if char in punctuations:
                    text = text + char

            print(text)

            im_wid, im_hei = image.size
            font = ImageFont.truetype('Montserrat-Regular.ttf', size=16)
            draw = ImageDraw.Draw(image)
            draw.rectangle(((xminn, yminn), (xmaxx, ymaxx)), outline="Chartreuse")
            text_width, text_height = font.getsize(text)

            draw.rectangle([(xminn, yminn-text_height-4), (xmaxx, yminn)], fill='Chartreuse')
            cen = ((xmaxx - xminn) - text_width)/2 + xminn
            draw.text((cen, yminn - (text_height+2)), text, font=font, fill='black')

            image.save(sys.argv[2]+'/'+image_name, "JPEG")

