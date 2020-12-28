# -*- coding: utf-8 -*-
"""
@author: Drayang
@Supervisor : Dr Soon Foo Chong
Created on Wed Dec 23 15:12:36 2020
Image_processing : To process the enormous amount of CompCar images into dataset form
                   Return: CompCar.npz file consist of x_test,y_test,x_train and y_train dictionary in np.array form 

"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import io
from tensorflow.keras.utils import to_categorical
from numpy import savez_compressed


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
        #load image
        image = Image.open('image/'+ img)
        
        ## resize image
        #image.thumbnail((32,32),Image.ANTIALIAS)
        image = image.resize((32,32),Image.ANTIALIAS)
        work_data = np.asarray(image)
        #work_data = work_data/255.0  #normalization
        
        
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
            break

'''
Comment : Can remove /255.0 so data is in normal form so it is exactly the same 
          format as online data set source
          Besides, normalization causing data store as float64 which use out too much memory (cant even process
          100 class)
'''


## Creating train data
# 897:10 class;5060:50 class 12036:100 class;31148: 281 class
i = 31148
creating_data(train_list,x_train,y_train,i)
x_train = np.array(x_train)
print("x_train done")

## Creating test data
# 386:10 class; 2166 : 50 class; 5154:100 class;13333:281 class
i = 13333
creating_data(test_list,x_test,y_test,i)
x_test = np.array(x_test)

print("x_test done")
savez_compressed('Car.npz',x_test = x_test,y_test = y_test, x_train = x_train, y_train = y_train)

print('Saving done')
'''

Comment: y_train and y_test no necessary to convert into np.array form (can in list data type)
         We will do to_categorical when load out the y data and y data will become 2D np.array 
ytainisa2dD
'''


'''
extracting data set(for checking purpose only)

'''

#savez_compressed('Comar.npz', x_train = x_train,y_train = y_train)

# # load numpy array from .npz file
# # load dict of arrays
# dict_data = np.load('CompCar.npz')

# # extract the first array
# datax = dict_data['x_train']
# datay = dict_data['y_train']
# # print the array
# print(datax[449])
# print(datay[449])
# plt.imshow(datax[449])

# #check model name correct or not
# datay = to_categorical(datay)
# print(model_data[np.argmax(datay[449])])
