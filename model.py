import csv
import cv2
import numpy as np

lines = []
# Read all the lines from the csv log file
with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

#Array to store image data
images = []
#Array to store measurement data
measurements = []

#Read all the data from each line of log file, the model.py file, driving_log.csv and IMG file were stored in the same directory
for line in lines:
	source_path = line[0]
	image = cv2.imread(source_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

## The following flipping data augmentation method was used during testing but didn't turn out better result, so it was discarded.

# augmented_images, augmented_measurements = [], []
# for image, measurement in zip(images, measurements):
# 	augmented_images.append(image)
# 	augmented_measurements.append(measurement)
# 	augmented_images.append(cv2.flip(image, 1))
# 	augmented_measurements.append(measurement*-1.0)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

model = Sequential()
#Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#Cropped the top 60 pixels and bottom 20 pixels
model.add(Cropping2D(cropping=((60,20), (0,0))))
# Used the CNN from nVidia
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
#Use the history object to produce the visualization.
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')