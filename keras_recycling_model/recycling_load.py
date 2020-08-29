from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy import argmax

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('test2/', target_size=(24,24), batch_size=3, class_mode='categorical')

from keras.models import load_model
model = load_model('recycling.h5')

print "-- Predict --"
output = model.predict_generator(test_generator, steps=4)
np.set_printoptions(formatter={'float' : lambda x: "{0:0.3f}".format(x)})
print test_generator.class_indices
print output