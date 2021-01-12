import pickle
import pandas as pd
import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "-1"   
import keras
#import tensorflow
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
# load and evaluate a saved model
from keras.models import load_model


train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input) #included in our dependencies
train_generator = train_datagen.flow_from_directory('./train/', # this is where you specify the path to the main data folder
                                                 target_size = (224,224),
                                                 color_mode = 'rgb',
                                                 batch_size = 8,
                                                 class_mode = 'categorical',
                                                 shuffle = True,
                                                 seed=42)

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input) #included in our dependencies
test_generator = test_datagen.flow_from_directory('./test/', # this is where you specify the path to the main data folder
                                                 target_size = (224,224),
                                                 color_mode = 'rgb',
                                                 batch_size = 8,
                                                 class_mode = 'categorical',
                                                 shuffle = True,
                                                 seed=42)



# ตรวจนับจำนวน classes ทั้งหมด โดยนับจากจำนวน folders ที่เจอ
num_of_classes = len(train_generator.class_indices)
print('number of classes : %d ' %num_of_classes) 

base_model = MobileNet(weights='imagenet', include_top=False,  input_shape=(224, 224,3)) #imports the mobilenet model and discards the last 1000 neuron layer.

# mobilenet มี 87 layer (+1 output layer ที่ถูกเอาออกไป) 

cnt = 0
for layer in base_model.layers[:]:
    cnt = cnt + 1
print('number of layers of base_model : %d' %cnt) 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation = 'relu')(x) # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(512, activation = 'relu')(x) # dense layer 2 used to be 1024
x = Dense(512, activation = 'relu')(x) # dense layer 3 used to be 1024
preds = Dense(num_of_classes, activation = 'softmax')(x) #final layer with softmax activation

model = Model(inputs = base_model.input, outputs = preds)

print(model.summary())

#specify the inputs
#specify the outputs
#now a model has been created based on our architecture

# for layer in model.layers[:20]:
#     layer.trainable = False
# for layer in model.layers[20:]:
#     layer.trainable = True

for layer in model.layers[:50]:
    layer.trainable = False
for layer in model.layers[50:]:
    layer.trainable = True

model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n//train_generator.batch_size
model.fit_generator(generator = train_generator,
                    steps_per_epoch = step_size_train,
                    epochs = 20,
                    verbose=1)

print('End of the training session')


_, acc = model.evaluate_generator(test_generator, steps=len(test_generator))
#_, acc = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=1)


print('> %.3f' % (acc * 100.0))

# save the model to disk
model.save("model.h5")

print('Model has been saved')

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

filehandler = open("file.dict","wb")
pickle.dump(train_generator.class_indices,filehandler)
filehandler.close()