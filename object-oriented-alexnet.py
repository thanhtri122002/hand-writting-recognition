import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, ZeroPadding2D, MaxPooling2D
from sklearn.svm import SVC
import joblib
HEIGHT = WIDTH = 28
path_train = r'.\dataset\emnist-balanced-train.csv'
path_test = r'.\dataset\emnist-balanced-test.csv'
path_map = r'.\dataset\emnist-balanced-mapping.txt'

class Preprocess:
    def __init__(self, list_paths, path_map):
        self.list_paths = list_paths
        self.path_map = path_map
        self.label_map = pd.read_csv(self.path_map, delimiter=' ', index_col=0, header=None, squeeze=True)
        label_dictionary = {}
        for index, label in enumerate(self.label_map):
            label_dictionary[index] = chr(label)

    def extract_dataset(self):
        
        X_set = []
        y_set = []
        for i in range(len(self.list_paths)):
            dataset = pd.read_csv(self.list_paths[i], header=None, delimiter=',')
            X = dataset.iloc[:, 1:]
            y = dataset.iloc[:, 0]
            X_set.append(X)
            y_set.append(y)

        return X_set , y_set

    def flip_rotate(self,X_set):
       
        for i in range(0, 2):
            X_set[i] = np.asarray(X_set[i])
            X_set[i] = np.apply_along_axis(self.rotate, 1, X_set[i])
            X_set[i] = X_set[i].astype('float32')
            X_set[i] /= 255
        return X_set

    def rotate(self, image):
        image = image.reshape([HEIGHT, WIDTH])
        image = np.fliplr(image)
        image = np.rot90(image)
        return image

    def one_hot_encoding(self, y_set):
        no_classes =  y_set[0].nunique()  # 47 classes
        for i in range(len(y_set)):
            y_set[i] = to_categorical(y_set[i], no_classes)
        return y_set , no_classes

class AlexNet:
    def __init__(self, HEIGHT, WIDTH, n_outputs):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.n_outputs = n_outputs

    def define_model(self):
        input = Input(shape=(self.HEIGHT, self.WIDTH, 1))

        # first layer
        x = Conv2D(filters=96, kernel_size=11, strides=4, name='conv1', activation='relu')(input)
        x = MaxPooling2D(pool_size=3, strides=2, name='pool1')(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D(2)(x)

        # second layer
        x = Conv2D(filters=256, kernel_size=3, strides=1, name="conv2", activation='relu')(x)
        x = MaxPooling2D(pool_size=3, strides=2, name="pool2")(x)
        x = BatchNormalization()(x)
        x = ZeroPadding2D(1)(x)

        # third layer
        x = Conv2D(filters=384, kernel_size=3, strides=1, name='conv3', activation='relu')(x)
        x = ZeroPadding2D(1)(x)

        # fourth layer
        x = Conv2D(filters=384, kernel_size=3, strides=1, name='conv4' , activation = 'relu')(x)
        x = ZeroPadding2D(1)(x)

    #fifth layer
        x = Conv2D(filters= 256 , kernel_size= 3, strides=1 , name = 'conv5', activation = 'relu')(x)
        
        x = Flatten()(x)

        x = Dense(4096, activation = 'relu', name = 'fc6')(x)
        x = Dropout(0.5 , name = 'dropout_6')(x)

        x = Dense(4096 , activation='relu',  name = 'fc7')(x)
        x = Dropout(0.5, name = 'dropout_7')(x)

        x = Dense(self.n_outputs , activation = 'softmax', name = 'fc8')(x)
        
        model = Model(inputs = input, outputs = x)
        return model
    def train_model(self):
        model = self.define_model()
        model.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])
        callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                    ModelCheckpoint(filepath='alexnet_emnist.h5', monitor='val_loss', save_best_only=True)]
        #train the model
        train = model.fit(X_train, y_train, epochs=20, validation_data=(
            X_test, y_test), callbacks=callbacks)
    def predict_model(self):
        pass
   
    
class Lenet:
    def __init__(self, HEIGHT,WIDTH, n_outputs):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.n_outputs = n_outputs
    def define_model(self):
        pass



class Evaluation_model:
    def __init__(self, *args, **kwargs):
        pass
    pass
if __name__ =="__main__":
    list_paths = [path_train, path_test]
    #preprocess data
    preprocessing = Preprocess(list_paths= list_paths, path_map= path_map)
    X_set , y_set = preprocessing.extract_dataset()
    print(type(X_set))
    X_set = preprocessing.flip_rotate(X_set)
    print(X_set[0].shape)
    y_set , no_classes = preprocessing.one_hot_encoding(y_set)

    X_train = X_set[0].reshape(-1,HEIGHT,WIDTH, 1)
    X_test = X_set[1].reshape(-1,HEIGHT,WIDTH , 1)

    y_train = y_set[0]
    y_test = y_set[1]
    while True:
        print("menu choose the model")
        print("1.Alexnet")
        print("2.svm with kernel linear")
        choice = int(input("choose model to train: "))
        if choice ==1:
    #train the model
            model = AlexNet(HEIGHT= HEIGHT , WIDTH= WIDTH , n_outputs= no_classes)
            model.train_model()
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
        if choice == 2:
            
            pass
        else:
            break





"""
class DataVisualizer:
    def __init__(self, X_train, y_train, X_test, y_test, label_map):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.label_map = label_map

    def plot_images(self, X, y, n_rows=4, n_cols=4):
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10,10))
        for i, ax in enumerate(axes.flat):
            if i < len(X):
                ax.imshow(X[i].reshape(HEIGHT, WIDTH), cmap='gray')
                ax.set_title(f"Label: {self.label_map[np.argmax(y[i])]}", fontsize=12)
            ax.axis('off')
        plt.show()

    def plot_class_distribution(self, y, title=None):
        labels = [self.label_map[i] for i in range(len(self.label_map))]
        class_counts = [sum(y[:, i]) for i in range(y.shape[1])]
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=labels, y=class_counts, ax=ax)
        ax.set_title(title or 'Class Distribution', fontsize=14)
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.tick_params(axis='x', labelrotation=90)
        plt.show()

    def visualize_data(self, n_rows=4, n_cols=4):
        print("Visualizing training data")
        self.plot_images(self.X_train, self.y_train, n_rows, n_cols)
        self.plot_class_distribution(self.y_train)

        print("Visualizing test data")
        self.plot_images(self.X_test, self.y_test, n_rows, n_cols)
        self.plot_class_distribution(self.y_test)

        

"""
