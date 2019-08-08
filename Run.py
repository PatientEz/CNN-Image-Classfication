import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import  numpy as np
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

#设定为自增长
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

#15类分类
num_classes = 15
nppath = "./DataSet/"
# 加载数据
x_train = np.load(nppath + "x_train.npy")
x_test = np.load(nppath + "x_test.npy")
y_train = np.load(nppath + "y_train.npy")
y_test = np.load(nppath + "y_test.npy")

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
#

x_train = x_train.astype('float64')/255
x_test = x_test.astype('float64')/255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('relu'))

model.summary()


# train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=3, shuffle=True)
model.save("./cnnmodel.h5")

# evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(loss, accuracy)
