# -*- coding: utf-8 -*-
"""
@author: Drayang
@Supervisor : Dr Soon Foo Chong
Created on :  Sun Dec 27 13:27:03 2020
Code_1
"""


# Import required module 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models,losses
from scipy import io

import random

from contextlib import redirect_stdout

'''
Load .mat file text
'''
### color_list.mat
color_list_mat = io.loadmat('color_list.mat')
color_list = color_list_mat['color_list']
image_path = np.array(color_list[:,0])
color_value = np.array(color_list[:,1])

color =['black','white','red','yellow','blue','green','purple','brown','magenta','silver','aqua'] 

### sv_make_model_name.mat
sv_make_model_name_mat = io.loadmat('sv_make_model_name.mat')
sv_make_model_name = sv_make_model_name_mat['sv_make_model_name']
make_name = np.array(sv_make_model_name[:,0])
model_name = np.array(sv_make_model_name[:,1])
model_id_data = np.array(sv_make_model_name[:,2])





'''
Comment : make_data and model_id_data are not necessary here, but just leave it there
          color_list.mat: haven't find out the purpose of color yet, maybe use for plotting or segmentation purpose
'''

### read .txt file to acess the image
#output is 1-D np.array list
test_list = np.loadtxt("test_surveillance.txt", comments="#", delimiter=",",dtype = 'str', unpack=False)
train_list = np.loadtxt('train_surveillance.txt',comments="#", delimiter=",",dtype = 'str', unpack=False)


'''
Load data set
'''

dataset_file = 'Car.npz'

## load numpy array from .npz file
def load_data(file):
    # load numpy array from .npz file
    # load dict of arrays
    dict_data = np.load(file)
    
    # extract the array
    x_test, y_test = dict_data['x_test'],dict_data['y_test']
    x_train , y_train = dict_data['x_train'], dict_data['y_train']
    
    #return np.array
    return (x_train,y_train), (x_test,y_test)

(x_train,y_train), (x_test,y_test) = load_data(dataset_file)
 
#Normalize the pixel value
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') /255.0

#Identify the number of class
class_num = max(y_test)+1

#Identify the number of testing images
image_num = len(y_test)

## np.argmax use to find the index of max value, hence use to identify the model name
#print(model_data[np.argmax(y_train)])

'''
Comment: X_train and x_test is a 4D np.array
'''

'''
Verify the dataset
'''

# plt.figure(figsize=(10,10))
# for i in range(25):
#     index= i*i
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[index], cmap=plt.cm.binary)
#     plt.xlabel(model_name[y_train[index]])
# plt.show()


'''
Designing the model
'''

model = models.Sequential()
#Number of filer = 32,filter size = 3x3
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))


model.add(layers.Flatten())
# model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(class_num))


model.summary()

# Compile and train model

model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])





'''
Training data
'''
epochs = 25

history = model.fit(
    x_train,
    y_train,
    #batch_size=64,
    epochs=epochs,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_test, y_test),
)

## Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='upper right')

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)




#%%

# Store data into a txt file
with open('model_summary.txt', 'a+') as f:
    f.write("\n\n_________________________________________________________________\n\n")
    with redirect_stdout(f):
        model.summary()
    f.write("\nNumber of class used: {}".format(class_num))
    f.write("\nNumber of epochs: {}".format(epochs))   
    f.write("\nText accuracy: {}".format(test_acc))
    f.write("\nTest lost: {}".format(test_loss))    



#%%
'''
Prediction 
'''

probability_model = tf.keras.Sequential([model, layers.Softmax()])


rand = random.randint(0,image_num)

#prediction is an array of "confidence" to the class
predictions = probability_model.predict(x_test)
print(predictions[rand])
max_prob = np.amax(predictions)

prediction_index = np.argmax(predictions[rand])
prediction_model_name = model_name[prediction_index]
correct_model_name = model_name[y_test[rand]]
print("The random vehicle we select:{}".format(rand))
print('The prediction car model:{}'.format(prediction_model_name))
print('The correct car model:{}'.format(correct_model_name))

#%%
