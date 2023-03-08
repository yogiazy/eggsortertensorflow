import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D

images = []
class_no = []
class_img = ["fertil", "infertil", "null"]

test_ratio = 0.2
val_ratio = 0.2

for x in class_img:
    img_path = os.listdir("imgTrain/citra_" + x)
    for y in img_path:
        cur_img = cv2.imread("imgTrain/citra_" + x + "/" + y)
        cur_img = cv2.resize(cur_img, (32,32))
        images.append(cur_img)
        class_no.append(x)
        #print(len(images), end=" ")

images = np.array(images)
class_no = np.array(class_no)
class_img = np.array(class_img)

x_train, x_test, y_train, y_test = train_test_split(images, class_no, test_size=test_ratio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=val_ratio)

print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

num_of_samples = []
for x in class_img:
    num_of_samples.append(len(np.where(y_train == x)[0]))

print(num_of_samples)

plt.figure(figsize=(10,5))
plt.bar(class_img, num_of_samples)
plt.title("Number of Images for each class")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.savefig('graphs/number_of_images.png')
plt.show()

def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

x_train = np.array(list(map(pre_processing, x_train)))
x_test = np.array(list(map(pre_processing, x_test)))
x_validation = np.array(list(map(pre_processing, x_validation)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 3)

data_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
data_gen.fit(x_train)

encoder = LabelEncoder()
encoder.fit(y_train)
encoder.fit(y_test)
encoder.fit(y_validation)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)
y_validation = encoder.transform(y_validation)
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)
y_validation = to_categorical(y_validation, num_classes=3)

def my_model():
    number_filters = 64
    size_filters1 = (5,5)
    size_filters2 = (3,3)
    size_pool = (2,2)
    number_node = 512

    model = Sequential()
    model.add((Conv2D(number_filters, size_filters1, input_shape=(32,32,3), activation='relu')))
    model.add((Conv2D(number_filters, size_filters1, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add((Conv2D(number_filters//2, size_filters2, activation='relu')))
    model.add((Conv2D(number_filters//2, size_filters2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(number_node, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = my_model()
model.save('./model_cnn.h5')
print(model.summary())

history = model.fit(data_gen.flow(x_train, y_train), batch_size=16, epochs=50, validation_data=(x_validation, y_validation), shuffle=1)
df = pd.DataFrame(history.history)
df.to_excel('summary/cnn_summary.xlsx', sheet_name='Sheet1')

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.savefig('graphs/loss.png')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.savefig('graphs/accuracy.png')
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score: ', score[0])
print('Test Accuracy: ', score[1])