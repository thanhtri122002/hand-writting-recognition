import numpy as np 
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix , f1_score , recall_score , precision_score , accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


path_train = r'/Users/tuanthanhnguyen/hand-writting-recognition/emnist-balanced-train.csv'
path_test = r'/Users/tuanthanhnguyen/hand-writting-recognition/emnist-balanced-test.csv'

list_paths = [path_train, path_test]
path_map = r'/Users/tuanthanhnguyen/hand-writting-recognition/emnist-balanced-mapping.txt'
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

# Load the data
list_datasets, X_set, y_set = extract_dataset(list_paths)
X_train, y_train = X_set[0], y_set[0]

# Standardize the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

# Apply PCA
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_std)

print("x_train_pca.shape: ", X_train_pca.shape)

# Visualize the data in 2D
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_train_pca[:, 0],y= X_train_pca[:, 1],hue=y_train, palette='bright', style=y_train, legend='full')
plt.title('PCA Visualization (2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., markerscale=0.5)  # Move legend to the right side and make it smaller
plt.show()


# Visualize the data in 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train, cmap='tab10')
ax.set_title('PCA Visualization (3D)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.view_init(elev=20, azim=45) # adjust viewing angle
plt.show()