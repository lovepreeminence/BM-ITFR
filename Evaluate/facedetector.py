import os 
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

#face detector model load
interpreter = tf.lite.Interpreter(model_path='./face_detect_model/thermal_face_automl_edge_fast.tflite')
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32
input_height   = input_details[0]['shape'][1]
input_width    = input_details[0]['shape'][2]



def detector(file_img, rectangle_save):
    img = Image.open(file_img).resize((input_width, input_height))
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes       = interpreter.get_tensor(output_details[0]['index'])
    labels      = interpreter.get_tensor(output_details[1]['index'])
    confidences = interpreter.get_tensor(output_details[2]['index'])
    nboxes      = interpreter.get_tensor(output_details[3]['index'])

    maxindex = np.argmax(confidences[0])
    box = boxes[0][maxindex] * 192 # width == height == 192px
    box = [box[1], box[0], box[3], box[2]]

    if rectangle_save==True: # (optional) draw image with rectangle
        draw = ImageDraw.Draw(img)
        draw.rectangle(box, outline="red")

        if not os.path.exists("./face_detec_img"):
           os.mkdir("./face_detec_img")
        img.save("./face_detec_img/{}".format(file_img.split('/')[-1]))

    cropped = img.crop(box)
    #cropped.save("./temp/image_save_202005011.jpg")
    return cropped

detector(file_img, rectangle_save)