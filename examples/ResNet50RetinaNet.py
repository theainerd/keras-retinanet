#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


# ## Load RetinaNet model

# In[2]:


# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'snapshots', 'resnet50_csv_09.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'Anorak',
 1: 'Blazer',
 2: 'Blouse',
 3: 'Bomber',
 4: 'Button-Down',
 5: 'Cardigan',
 6: 'Flannel',
 7: 'Halter',
 8: 'Henley',
 9: 'Hoodie',
 10: 'Jacket',
 11: 'Jersey',
 12: 'Parka',
 13: 'Peacoat',
 14: 'Poncho',
 15: 'Sweater',
 16: 'Tank',
 17: 'Tee',
 18: 'Top',
 19: 'Turtleneck',
 20: 'Capris',
 21: 'Chinos',
 22: 'Culottes',
 23: 'Cutoffs',
 24: 'Gauchos',
 25: 'Jeans',
 26: 'Jeggings',
 27: 'Jodhpurs',
 28: 'Joggers',
 29: 'Leggings',
 30: 'Sarong',
 31: 'Shorts',
 32: 'Skirt',
 33: 'Sweatpants',
 34: 'Sweatshorts',
 35: 'Trunks',
 36: 'Caftan',
 37: 'Cape',
 38: 'Coat',
 39: 'Coverup',
 40: 'Dress',
 41: 'Jumpsuit',
 42: 'Kaftan',
 43: 'Kimono',
 44: 'Nightdress',
 45: 'Onesie',
 46: 'Robe',
 47: 'Romper',
 48: 'Shirtdress',
 49: 'Sundress'}

# ## Run detection on example

# In[3]:


# load image
image = read_image_bgr('img_00000004.jpg')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break

    color = label_color(label)

    b = box.astype(int)
    draw_box(draw, b, color=color)

    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)

plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()
