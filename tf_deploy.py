import time
import pickle
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import keras
import tensorflow
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model

# load dictionary (class names + class indeces)
file = open("file.dict",'rb')
object_file = pickle.load(file)
file.close()

img_width, img_height = 224, 224
def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor
  
# test images   

#img_path = 'left_01.jpg'
#img_path = 'left_02.jpg'
#img_path = './test/pra_somdet_01.jpg'
#img_path = './test/01/21.jpg'
#img_path = './test/pra_somdet_01.jpg'
img_path = './test/01/02.jpg'


# load the trained model
# ------------------------ tf 1.5.0 and keras 2.1.4 ------------------------------------------- #
from keras.utils.generic_utils import CustomObjectScope
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model('model.h5')
#-------------------------------------------------------------------------------------------------
#model = load_model('model.h5')

new_image = load_image(img_path, show=False)
start = time.time()
pred = model.predict(new_image)
end = time.time()
print(end - start)

print('Probabilities of each class:')
print(pred[0])
print(object_file)
print('Predicted class : %d'%pred.argmax(axis=-1))

for k in object_file:
    if object_file[k] == (pred.argmax(axis=-1)):
        print('predicted class name: %s' %k)
        #print(object_file[k])