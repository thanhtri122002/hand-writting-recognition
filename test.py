import numpy as np 
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import  load_model , Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import  BatchNormalization , Concatenate , Conv2D , Dense , Dropout , Flatten , GlobalAveragePooling2D , Input , Lambda, ZeroPadding2D , MaxPooling2D

path_train = r'C:\Users\ASUS\Desktop\computer-vision-project\emnist-balanced-train.csv'
path_test = r'C:\Users\ASUS\Desktop\computer-vision-project\emnist-balanced-test.csv'
list_paths = [path_train, path_test]

path_map = r'C:\Users\ASUS\Desktop\computer-vision-project\emnist-balanced-mapping.txt'
label_map = pd.read_csv(path_map, delimiter = ' ', index_col=0, header=None, squeeze=True) 

label_dictionary = {}
for index, label in enumerate(label_map):
    label_dictionary[index] = chr(label)

def extract_dataset(list_paths):
    list_datasets = []
    X_set = []
    y_set = []
    for i in range(len(list_paths)):
            dataset = pd.read_csv(list_paths[i], header = None , delimiter=',')
            X = dataset.iloc[:, 1:]
            y = dataset.iloc[:, 0]
            X_set.append(X)
            y_set.append(y)
            list_datasets.append(dataset)
    return list_datasets, X_set, y_set 

HEIGHT = WIDTH = 28

def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

def flip_rotate(X_set):
    for i in range(0,2):
        X_set[i] = np.asarray(X_set[i])
        X_set[i] = np.apply_along_axis(rotate, 1 , X_set[i])
        X_set[i] = X_set[i].astype('float32')
        X_set[i] /=255
    return X_set

def one_hot_encoding(y_set,no_classes):
    for i in range(len(y_set)):
        y_set[i] = to_categorical(y_set[i],no_classes)
    return y_set



def define_model_alexnet():
    n_outputs = 47 
    input = Input(shape  = (HEIGHT, WIDTH , 1))

    #first layer 
    x = Conv2D(filters = 96 , kernel_size=11, strides=4, name = 'conv1' , activation='relu')(input)
    x = MaxPooling2D(pool_size= 3, strides = 2,name = 'pool1' )(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(2)(x)
    
    #second layer
    x = Conv2D(filters = 256 , kernel_size = 3 , strides = 1,  name ="conv2" , activation  = 'relu')(x)
    x = MaxPooling2D(pool_size = 3 , strides= 2 , name = "pool2" )(x)
    x = BatchNormalization()(x)
    x = ZeroPadding2D(1)(x)

    #third layer
    x = Conv2D(filters = 384 , kernel_size = 3, strides= 1 , name = 'conv3' , activation= 'relu' )(x)
    x = ZeroPadding2D(1)(x)

    #fourth layer 
    x = Conv2D(filters= 384, kernel_size = 3 , strides=1 , name = 'conv4' , activation = 'relu')(x)
    x = ZeroPadding2D(1)(x)

    #fifth layer
    x = Conv2D(filters= 256 , kernel_size= 3, strides=1 , name = 'conv5', activation = 'relu')(x)
   
    x = Flatten()(x)

    x = Dense(4096, activation = 'relu', name = 'fc6')(x)
    x = Dropout(0.5 , name = 'dropout_6')(x)

    x = Dense(4096 , activation='relu',  name = 'fc7')(x)
    x = Dropout(0.5, name = 'dropout_7')(x)

    x = Dense(n_outputs , activation = 'softmax', name = 'fc8')(x)
    
    model = Model(inputs = input, outputs = x)
    return model

def confusion_matrix_plot(y_pred,y_test):
    cm = confusion_matrix(y_pred,y_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


if __name__ == "__main__":
    ist_datasets, X_set, y_set = extract_dataset(list_paths)

    number_classes = y_set[0].nunique()

    #flip the image
    X_set = flip_rotate(X_set)
    y_set = one_hot_encoding(y_set, number_classes)

    X_train = X_set[0].reshape(-1,HEIGHT,WIDTH, 1)
    X_test = X_set[1].reshape(-1,HEIGHT,WIDTH , 1)

    y_train = y_set[0]
    y_test = y_set[1]

    