"""
@author: Drayang
@Supervisor : Dr Soon Foo Chong
Created on :  Thu Jan 21 10:23:21 2021
Updated on : 
Code_4

"""

'''
Setup
'''
# Import required module
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
import os

#You can disable all debugging logs using os.environ:
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Dropout,Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Softmax, Lambda
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet import preprocess_input



from time import time, strftime, gmtime
from contextlib import redirect_stdout
from scipy import io


'''
Load .mat file text
'''

# sv_make_model_name.mat
sv_make_model_name_mat = io.loadmat('sv_make_model_name.mat')
sv_make_model_name = sv_make_model_name_mat['sv_make_model_name']
make_name = np.array(sv_make_model_name[:,0])
model_name = np.array(sv_make_model_name[:,1])
model_id_data = np.array(sv_make_model_name[:,2])

# read .txt file to acess the image
test_list = np.loadtxt("test_surveillance.txt", comments="#", delimiter=",",dtype = 'str', unpack=False)
train_list = np.loadtxt('train_surveillance.txt',comments="#", delimiter=",",dtype = 'str', unpack=False)


'''
Load data set
'''


dataset_file = 'CarApp224_100c.npz'

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


### We can also use the below function to preprocess the image input if we
### only have the original image dataset that havent be preprocess(e.g Car.npz)
def preprocess_data(X,Y):
    
    ## X and Y is the x_train and y_train data respectively
    ## we will preprocess the x_data( if we didnt do so in image processing)
    ## we also to_categorical y_data here( mean we no need to do it again below)
    
    
    X_data = preprocess_input(X)
    Y_data = to_categorical(Y)
    
    return X_data,Y_data
    
# #Normalize the pixel value
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') /255.0



### One hote encode output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# define number of class
class_num = y_test.shape[1]

#Identify the number of testing images
image_num = y_test.shape[0]


'''
Create the model
'''

# input_t = tf.keras.Input(shape =(224,224,3))

basemodel = VGG16(include_top=False, 
            weights= "imagenet", 
            input_shape=x_train.shape[1:], 
            # input_tensor = input_t,
            classes=class_num
            )

# basemodel.summary()

###To freeze the layer, Once a layer is frozen, its weights are not updated while training.
###use [:1] to freeze all layer except last block of VGG16
for layer in basemodel.layers[:15]:
    layer.trainable = False

'''
Comment: Reason to only freeze infront layer is beause since we oncorporated the feature extraction, the deeper
         layer is more important. And actually if we just use a flatten and dense layer already enough to the classification
         
                                                                                                

'''

# ### First way to check whether the layer freeze
# for layer in model.layers:
#     sp = ' '[len(layer.name)-9:]
#     print(layer.name, sp, layer.trainable)

## Second way to check whether the layer freeze correctly    
# for i, layer in enumerate(basemodel.layers):
#     print(i, layer.name, "-" , layer.trainable)

# basemodel.summary()



model = Sequential()

# ### use to resize the input image( use for cifar10)
# to_res = (224,224)    
# model.add(Lambda(lambda image: tf.image.resize(image, to_res)))

model.add(basemodel)

# Flatten data for classification purpose
model.add(Flatten(name = "Flatten_1"))
model.add(Dropout(0.2 , name = "DropOut_1"))

model.add(Dense(512,activation ='relu',  name = "Dense_1"))
model.add(Dropout(0.2,  name = "DropOut_2"))
model.add(BatchNormalization( name = "Batch_normalization_1"))    
model.add(Dense(256,activation ='relu', name = "Dense_2"))
model.add(Dropout(0.2,  name = "DropOut_3"))
model.add(BatchNormalization(name = "Batch_normalization_2"))
model.add(Dense(128,activation ='relu', name = "Dense_3"))
model.add(Dropout(0.2,  name = "DropOut_4"))
model.add(BatchNormalization(name = "Batch_normalization_3"))
  
# Final Layer
model.add(Dense(class_num,activation = 'softmax', name = "Softmax"))

### Example online use optimizer = optimizers.RMSprop(lr = 2e-5), lr is learning rate
# Compile the model
# Optimizer = Adam or RMSprop
optimizer = 'Adam'
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer , metrics = ['accuracy'])



################################ REMINDER ################################

model._name = 'VGG16_1'

################################ REMINDER ################################
# model.summary()



'''
Training data
'''
# Save the model or model weights 
model_filepath =  model._name + '.h5'
checkpointer = ModelCheckpoint(filepath=model_filepath, 
                                monitor='val_accuracy', 
                                verbose = 1, 
                                save_best_only=True, 
                                save_weights_only=False, 
                                mode='auto', 
                                save_freq = 'epoch')

# Early stopping callback function
early_stopping = EarlyStopping(monitor='val_loss',  
                               patience = 4, 
                               verbose = 1,
                               mode='auto', 
                               baseline=None, 
                               restore_best_weights=False)


# Calculate the training time
start = time()

# Define number of epochs
epochs = 10

### Example online use batch_size =32
# Train the model
history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_test), 
                    epochs=epochs, 
                    verbose = 1,
                    batch_size = 64,
                    callbacks = [checkpointer,early_stopping]
                    #callbacks = [checkpointer]
                    )


# Determine training time and convert into minute
train_time = strftime("%H hour %M min %S seconds", gmtime(time()-start))


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
plt.ylim([0, 3])
plt.legend(loc = 'upper right')

# Final evaluation of the model
test_loss,test_acc = model.evaluate(x_test, y_test, verbose=0)
all_acc =history.history['val_accuracy']
all_loss =history.history['val_loss']

print("\nTraining time: {}".format(train_time))
print("\nTest Accuracy: {:.2f}%" .format(np.max(all_acc)*100))
print("\nThe epochs of highest accuracy : {}" .format(np.argmax(all_acc)+1))
print("\nMean Loss:{:.5f}" .format(np.mean(all_loss)))
print("\nMin Loss:{:.5f}" .format(np.min(all_loss)))



#%%

# Store data into a txt file
with open('summary_Code_3.txt', 'a+') as f:
    f.write("\n\n_________________________________________________________________\n\n")
    with redirect_stdout(f):
        model.summary()
    f.write("\nNumber of class used: {}".format(class_num))
    f.write("\nNumber of epochs: {}".format(len(all_loss)))   
    f.write("\nTest Accuracy:{:.2f}%" .format(np.max(all_acc)*100))
    f.write("\nThe epochs of highest accuracy : {}" .format(np.argmax(all_acc)+1))    
    # f.write("\nTest Loss:{:.2f}%" .format(test_loss*100)) 
    f.write("\nMean Loss:{:.5f}" .format(np.mean(all_loss))) 
    f.write("\nMin Loss:{:.5f}" .format(np.min(all_loss)))
    f.write("\nTotal training time: {}".format(train_time))
    
    f.write("\nAcc_1: {:.2f}%" .format(all_acc[0]*100))
    f.write("\nAcc_2: {:.2f}%" .format(all_acc[1]*100))
    f.write("\nAcc_3: {:.2f}%" .format(all_acc[2]*100))
    f.write("\nAcc_4: {:.2f}%" .format(all_acc[3]*100))
    f.write("\nAcc_5: {:.2f}%" .format(all_acc[4]*100))
    f.write("\nAcc_6: {:.2f}%" .format(all_acc[5]*100))    
    f.write("\nAcc_7: {:.2f}%" .format(all_acc[6]*100))
    f.write("\nAcc_8: {:.2f}%" .format(all_acc[7]*100))
    f.write("\nAcc_9: {:.2f}%" .format(all_acc[8]*100))
    f.write("\nAcc_10: {:.2f}%" .format(all_acc[9]*100))



#%%
'''
Prediction 
'''

probability_model = Sequential([model, Softmax()])
# prediction_acc = 0
n= 1
# x=0

for i in range(n):
    rand = random.randint(0,image_num)
    
    #prediction is an array of "confidence" to the class
    predictions = probability_model.predict(x_test)
    #print(predictions[rand])
    
    prediction_index = np.argmax(predictions[rand])
    prediction_model_name = model_name[prediction_index]
    #np.argmax use to find the index of max value, hence use to identify the model name
    correct_model_name = model_name[np.argmax(y_test[rand])]
    print("The random vehicle we select:{}".format(rand))
    print('The prediction car model:{}'.format(prediction_model_name[0]))
    print('The correct car model:{}\n'.format(correct_model_name[0]))
    





