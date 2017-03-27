from keras.layers.noise import GaussianNoise
import numpy as np
from keras.datasets import cifar100, mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.metrics import categorical_accuracy
import matplotlib.pyplot as plt


batch_size = 128
nb_classes = 10
nb_epoch = 25
sigma = 0.01


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(GaussianNoise(sigma, input_shape=(784,)))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(GaussianNoise(sigma))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(GaussianNoise(sigma))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(GaussianNoise(sigma))
model.add(Dense(10))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print(score)
print(history.history.keys())
print history.history['loss']
print history.history['val_loss']
print history.history['acc']

preds = model.predict(X_test)
print categorical_accuracy(y_test, preds)


plt.plot(np.linspace(1, nb_epoch, nb_epoch), history.history['loss'])
plt.plot(np.linspace(1, nb_epoch, nb_epoch), history.history['val_loss'])
plt.show()


