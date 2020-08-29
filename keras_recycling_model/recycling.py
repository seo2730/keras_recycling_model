import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# fix random.seed
np.random.seed(3)

train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range = 10,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.5,  
                                  zoom_range = [0.9, 2.2],
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  fill_mode='nearest')

train_generator = train_datagen.flow_from_directory('train/', target_size=(24,24), batch_size=3, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory('test/', target_size=(24,24), batch_size=3, class_mode='categorical')

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(24, 24, 3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=398,
        epochs=1000,
        validation_data=test_generator,
        validation_steps=4) # test sample = 12 / batch size = 3

print "-- Evaluate --"
scores = model.evaluate_generator(test_generator,  steps=4)
print "%s: %.2f%%" %(model.metrics_names[1], scores[1]*100)

print "-- Predict --"
output = model.predict_generator(test_generator, steps=4)
np.set_printoptions(formatter={'float' : lambda x: "{0:0.3f}".format(x)})
print test_generator.class_indices
print output

from keras.models import load_model
model.save('recycling.h5')