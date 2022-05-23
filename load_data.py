from keras.engine import training
import numpy as np
import pandas as pd
import os
import cv2 as cv
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split

def load_data(dataset_root_folder, no_of_classes = 10):

  dataset_x = []
  dataset_y = []
  for i in range (0,no_of_classes):
      all_folders_list = os.listdir(r""+dataset_root_folder+"" +"/"+str(i))
      for j in all_folders_list:
          pic = cv.imread(r""+dataset_root_folder +"/"+str(i)+"/"+j)
          pic = cv.resize(pic,(32,32))
          dataset_x.append(pic)
          dataset_y.append(i)

          
  print("Dataset Loaded !")

  return np.array(dataset_x), np.array(dataset_y)

  # r value

  #possible modifications
  # #Augmentation
# datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
# datagen.fit(train_X)


def preprocess( dataset):
    print("Hello")
    if(len(dataset) <= 0 ):
      print("No dataset recieved")
      return
  
    training_data = dataset['train']
    validation_data = dataset['validation']
    testing_data = dataset['test']

    print(training_data.shape)
    # training_data =[lambda x : ((cv.equalizeHist(cv.cvtColor(x,cv.COLOR_BGR2GRAY)))/255) for x in training_data]
    # validation_data =[lambda x : cv.equalizeHist(cv.cvtColor(x,cv.COLOR_BGR2GRAY))/255 for x in validation_data]
    # testing_data =[lambda x : cv.equalizeHist(cv.cvtColor(x,cv.COLOR_BGR2GRAY))/255 for x in testing_data]
    def Prepare(img):
      img = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #making image grayscale
      img = cv.equalizeHist(img) #Histogram equalization to enhance contrast
      img = img/255
      return img

    # training_data = np.array(training_data)
    # validation_data = np.array(validation_data)
    # testing_data = np.array(testing_data)

    print(training_data.shape)
    # print(training_data.shape)
    train_X = np.array(list(map(Prepare, training_data)))
    test_X = np.array(list(map(Prepare, testing_data)))
    valid_X= np.array(list(map(Prepare, validation_data)))

    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2],1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2],1)
    valid_X = valid_X.reshape(valid_X.shape[0], valid_X.shape[1], valid_X.shape[2],1)
    # return training_data.reshape(training_data.shape[0], training_data.shape[1], training_data.shape[2], 1), \
    #        validation_data.reshape(validation_data.shape[0], validation_data.shape[1], validation_data.shape[2], 1), \
    #        testing_data.reshape(testing_data.shape[0], testing_data.shape[1], testing_data.shape[2], 1)
    
    return train_X, valid_X, test_X


def load_sudoku_data(file): 

    data = pd.read_csv(file)

    raw_features = data['quizzes']
    raw_labels = data['solutions']

    features = []
    labels = []

    for i in raw_features:
    
        each_feature = np.array([int(j) for j in i]).reshape((9,9,1))
        features.append(each_feature)
    
    features = np.array(features)
    features = features/9
    features -= .5    
    
    for i in raw_labels:
    
        each_label = np.array([int(j) for j in i]).reshape((81,1)) - 1
        labels.append(each_label)   
    
    labels = np.array(labels)
    
    del(raw_features)
    del(raw_labels)    

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test
