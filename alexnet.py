import numpy as np 
import pandas as pd
import cv2
import matplotlib.pyplot as plt
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



def define_model():
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

def plotgraph(epochs, acc, val_acc):
    #Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    ist_datasets, X_set, y_set = extract_dataset(list_paths)
    number_classes = y_set[0].nunique()
    X_set = flip_rotate(X_set)
    y_set = one_hot_encoding(y_set, number_classes)

    X_train = X_set[0].reshape(-1,HEIGHT,WIDTH, 1)
    X_test = X_set[1].reshape(-1,HEIGHT,WIDTH , 1)

    y_train = y_set[0]
    y_test = y_set[1]

    X_train , X_test , y_train , y_test = train_test_split(X_train, y_train , test_size= 0.2, random_state= 42)
    """model = define_model()
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    train = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=callbacks)"""
    model = load_model('best_model.h5')
    scores = model.evaluate(X_test,y_test, verbose = 0)
    
    print("Accuracy: %.2f%%"%(scores[1]*100))
    


"""dataset_train = pd.read_csv(path_train, header = None)
print(dataset_train.shape) #(112800, 785) 112800 data with each has 785 features (pixel values)
print(dataset_train.head())

X  = dataset_train.loc[: , 1:]
Y = dataset_train.loc[:,0]

label_map = pd.read_csv(path_map, delimiter = ' ', index_col=0, header=None) 

#dictionary of labels
#dict  ={"label" :"charecters"}
label_dictionary = {}
for index, label in enumerate(label_map):
    label_dictionary[index] = chr(label)
print(len(label_dictionary))


print(y_set[0].shape)
print(type(y_set))
print(type(y_set[0]))
for i in range(100, 109):
    plt.subplot(330 + (i+1))
    plt.imshow(X_set[0][i], cmap=plt.get_cmap('gray'))
    plt.title(label_dictionary[y_set[0][i]])
plt.show()

https://www.kaggle.com/code/ashwani07/emnist-using-keras-cnn
"""







