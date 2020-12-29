# -*- coding: utf-8 -*-
"""

@author: Drayang
@Supervisor : Dr Soon Foo Chong
Created on : Thu Dec 17 13:54:44 2020
Trial_1

"""
'''
Setup

'''
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Softmax
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.utils import to_categorical
from time import time
from contextlib import redirect_stdout
from PIL import Image
from scipy import io

'''
Load .mat file text

'''

### color_list.mat
color_list_mat = io.loadmat('color_list.mat')
color_list = color_list_mat['color_list']
image_path = np.array(color_list[:,0])
color_value = np.array(color_list[:,1])

color =['black','white','red','yellow','blue','green','purple','brown','magenta','silver','aqua'] 


#color_value = np.array(color_value)
# print(color_value[:])
# print(type(color_value[:]))

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


### One hote encode output
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# # define number of class
# class_num = y_test.shape[1]

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
Create the model
'''

model = Sequential()

#Number of filer = 32, size = 3x3 ,padding = 'same'(no changing size of image)
model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = x_train.shape[1:], padding = 'same'))
# Dropout layer to prevent overfitting
model.add(Dropout(0.2))
# Batch normalization normalizes the inputs heading into next layer, ensuring that the network always creates activations with the same distribution that we desire
model.add(BatchNormalization())


#Convolution layer, filter size increase so the network can learn more complex representation
model.add(Conv2D(64,(3,3), padding = 'same'))
model.add(Activation('relu')) 
# Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())


# Flatten data for classification purpose
model.add(Flatten())
model.add(Dropout(0.2))


# Create densely connected layer
# We need specify the number of neurons(256,128) and it decrease as deeper layer,eventually same as the classes size (class_num)
# Kernel constraint regularize the data as it learns and help prevent overfitting
model.add(Dense(256, kernel_constraint = MaxNorm(3),activation ='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())    
model.add(Dense(128, kernel_constraint = MaxNorm(3),activation ='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
# Final Layer
model.add(Dense(class_num,activation = 'softmax'))

# Compile the model
optimizer = 'Adam'
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer , metrics = ['accuracy'])

model.summary()

'''

Comment: Increase filters as you go on and it's advised to make them powers of 2 
          which can grant a slight benefit when training on a GPU.
          Add convolutional layers you typically increase their number of filters so the model 
          can learn more complex representations.( but increase computation expenses)

          Do not have too many pooling layer as it discards some data
'''


'''

Training data

'''

# Train the model
epochs = 10

# Calculate the training time
start = time()


history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_test), 
                    epochs=epochs, 
                    batch_size=64)


# Determine training time and convert into minute
train_time = (time()-start)/60

## Evaluate the model
#Plot Accuracy during training
plt.subplot(2,1,1)
plt.title("Accuracy and Loss")
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

#Plot Loss during training
plt.subplot(2,1,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc = 'lower right')

# Final evaluation of the model
test_loss,test_acc = model.evaluate(x_test, y_test, verbose=0)

print("\nTraining time: {:.2f} minutes".format(train_time))
print("\nTest Accuracy: {:.2f}%" .format(test_acc*100))
print("\nTest Loss:{:.2f}%" .format(test_loss*100))

#%%
# Store data into a txt file
with open('summary_Trial_1.txt', 'a+') as f:
    f.write("\n\n_________________________________________________________________\n\n")
    with redirect_stdout(f):
        model.summary()
    f.write("\nNumber of class used: {}".format(class_num))
    f.write("\nNumber of epochs: {}".format(epochs))   
    f.write("\nTest Accuracy:{:.2f}%" .format(test_acc*100))
    f.write("\nTest Loss:{:.2f}%" .format(test_loss*100))  
    f.write("\nTotal training time:{:.2f} minutes".format(train_time))


#%%
'''
Prediction 
'''

probability_model = tf.keras.Sequential([model, Softmax()])


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
