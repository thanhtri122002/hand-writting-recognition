import tensorflow as tf
from tensorflow import keras
from keras import layers

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the dat
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Define the model architecture
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28)),
        layers.Reshape(target_shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(train_images, train_labels, batch_size=128,
          epochs=15, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy:", test_acc)

# Predict the digits in a new image
# open image

image = tf.io.read_file("digits.png")
image = tf.expand_dims(image, axis=0)
prediction = model.predict(image)
print("Predicted digit:", tf.argmax(prediction, axis=1).numpy()[0])
