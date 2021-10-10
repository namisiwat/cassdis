import os
import tensorflow as tf 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
  # No GPU found
import numpy as np 
from PIL import Image, ImageOps	
import cv2
def process_image(img, img_size=(224, 224)):

  # reshapes the image
  image = ImageOps.fit(img, img_size, Image.ANTIALIAS)
  
  # converts the image into numpy array
  image = np.asarray(image)
  
  # converts image from BGR color space to RGB
  img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  img_resize = (cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC))/255.

  img_reshape = img_resize[np.newaxis,...]
  
  return img_reshape


def prediction_result(model, image_data):
 
  # Mapping prediction results to the cassava leaf disease  type
  classes = {0: "แอนแทรคโนส", 
             1: "ใบไหม้",
             2: "อาการใบด่าง",
             3: "ใบจุดสีน้ำตาล",
             4: "ปกติ"}
  
  pred = model.predict(image_data)
  pred = pred.round(2)
  result = np.argmax(pred)
  
  prediction = {"class": classes[result],
                "accuracy": np.round(np.max(pred) * 100, 2)}
  
  return prediction
