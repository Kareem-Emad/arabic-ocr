import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from ast import literal_eval
from utils import convert_to_binary, display_image # noqa
import matplotlib.pyplot as plt


def build_model():
    try:
        model = load_model('son_of_anton.h5')
        return model
    except Exception:
        model = Sequential()
        model.add(Dense(50, input_shape=(100, 50), activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        return model


def eliminate_extra_padding(img):
    horz_sum = np.sum(img, axis=1)
    ver_sum = np.sum(img, axis=0)
    upper_x = -1
    upper_y = -1
    lower_x = -1
    lower_y = -1
    for i in range(0, horz_sum.shape[0]):
        if (horz_sum[i] != 0):
            if (upper_x == -1):
                upper_x = i
            else:
                lower_x = i

    for i in range(0, ver_sum.shape[0]):
        if (ver_sum[i] != 0):
            if (upper_y == -1):
                upper_y = i
            else:
                lower_y = i
    return img[upper_x:lower_x + 1, upper_y:lower_y + 1], upper_y


def pad_image(img):
    h, w = img.shape
    padh = 50 - h
    padw = 100 - w
    if(padw):
        padding_columns = np.zeros((h, padw))
        img = np.hstack((padding_columns, img))
    h, w = img.shape
    if(padh):
        padding_rows = np.zeros((padh, w))
        img = np.vstack((padding_rows, img))
    return img, padw


data_path = 'train_set'
label_path = 'label_set'
files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

X = np.zeros((10000, 100, 50))
y = np.zeros((10000, 100, 1))
for i in range(10000):
    f = files[i]
    image = cv2.imread(os.path.join(data_path, f), 0)
    image = convert_to_binary(image)
    image, neg_padw = eliminate_extra_padding(image)
    image, padw = pad_image(image)
    padw = padw - neg_padw
    X[i] = image.reshape((100, 50))
    with open(f'{label_path}/{f.replace("png","txt")}') as f:
        list_text = f.read()
        cuts = literal_eval(list_text)

        cuts_array = np.zeros((image.shape[1], 1))
        cuts = np.asarray(cuts)

        cuts = cuts + padw
        cuts -= max(np.max(cuts) - 99, 0)
        cuts_array[cuts] = 1
        y[i] = cuts_array
        f.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = build_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

model.save('son_of_anton.h5')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
