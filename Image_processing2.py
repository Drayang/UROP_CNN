
# -*- coding: utf-8 -*-
"""
@author: Drayang
@Supervisor : Dr Soon Foo Chong
Created on : Mon Jan 18 20:35:36 2021
Updated on : Wed Jan 20 19:51:14 2021
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import io
from tensorflow.keras.utils import to_categorical
from numpy import savez_compressed
from tensorflow.keras.applications.resnet import preprocess_input
# from keras.preprocessing import image

from time import time, strftime, gmtime

from tensorflow.keras.preprocessing.image import load_img 
import warnings 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.preprocessing.image import array_to_img 
  


'''

Load important txt file for image converting purpose
'''


### sv_make_model_name.mat
sv_make_model_name_mat = io.loadmat('sv_make_model_name.mat')
sv_make_model_name = sv_make_model_name_mat['sv_make_model_name']
model_data = np.array(sv_make_model_name[:,1])

### read .txt file to acess the image
#output is 1-D np.array list
test_list = np.loadtxt("test_surveillance.txt", comments="#", delimiter=",",dtype = 'str', unpack=False)
train_list = np.loadtxt('train_surveillance.txt',comments="#", delimiter=",",dtype = 'str', unpack=False)


'''
comment: sv_make_model_name.mat no need here, but i can try to match the model name with the vehicle  
'''


'''
convert images into data set
'''

### Load all images in a directory
x_train = list()
y_train = list()

x_test = list()
y_test = list()


# Argument i is to control the number of image, remove it to convert all images
def creating_data(path_list,x_data,y_data,i):
    for img in path_list:
        
        # load image 1st way-the image pixel will be altered, no original 
        # image = load_img(path='image/'+ img,target_size=(64,64))
        
        #load image 2nd way - the image pixel remain the same
        image = Image.open('image/'+ img)
        image = image.resize((224,224),Image.ANTIALIAS)

        work_data = img_to_array(image, dtype='uint8')
        work_data = preprocess_input(work_data)
        
        #     # image.show()
        #     # print(work_data)
        #     print(type(work_data))

        #### Need to try out see preprocess one by one better or
        ### preprocess after we process all data
        # preprocess_input(work_data)
        
        
        #To get the Make of vehicle indices
        index = img.split('/')
        y_data.append(int(index[0])-1)
        #return index that indicate which vehicle make it belong to
        
        #plt.imshow(work_data)
        x_data.append(work_data)
        # return 3D array matrix
        print('> loaded {} {}' .format(i, img))
        
        i = i- 1
    
        if i == 0:
            return(np.array(x_data),y_data)
            break
      
        

# Calculate the training time
start = time()
        
## Creating train data
# 897:10 class;5060:50 class 12036:100 class;31148: 281 class
i = 12036
(x_train,y_train)=creating_data(train_list,x_train,y_train,i)
print("x_train done")
# savez_compressed('Car_App_224_train.npz',x_train = x_train, y_train = y_train)


# # ## Creating test data
# # # 386:10 class; 2166 : 50 class; 5154:100 class;13333:281 class
i= 5154
(x_test,y_test)=creating_data(test_list,x_test,y_test,i)
print("x_test done")
# savez_compressed('Car_App_224_test.npz',x_test = x_test,y_test = y_test)



savez_compressed('CarApp224_10c.npz',x_train = x_train, y_train = y_train,x_test = x_test,y_test = y_test)
print('Saving done')


# Determine training time and convert into minute
train_time = strftime("%H hour %M min %S seconds", gmtime(time()-start))


print("\nProcessing time: {}".format(train_time))

'''
extracting data set(for checking purpose only)

'''

#savez_compressed('Comar.npz', x_train = x_train,y_train = y_train)

# load numpy array from .npz file
# load dict of arrays

# dict_data = np.load('Car_trial_10.npz')

# # extract the first array
# datax = dict_data['x_train']
# datay = dict_data['y_train']
# # print the array
# print(datax[2])
# print(datay[2])
# plt.imshow(datax[2])

# #check model name correct or not
# datay = to_categorical(datay)
# print(model_data[np.argmax(datay[2])])








  
# # load the image via load_img function 
# img = load_img('sample.png') 
  
# # details about the image printed below 
# print(type(img)) 
# print(img.format) 
# print(img.mode) 
# print(img.size) 
  
# # convert the given image into  numpy array 
# img_numpy_array = img_to_array(img) 
# print("Image is converted and NumPy array information :") 
  
# # <class 'numpy.ndarray'> 
# print(type(img_numpy_array)) 
  

# # type: float32 
# print("type:", img_numpy_array.dtype) 
  
# # shape: (200, 400, 3) 
# print("shape:", img_numpy_array.shape) 
  
# # convert back to image 
# img_pil_from_numpy_array = array_to_img(img_numpy_array) 
