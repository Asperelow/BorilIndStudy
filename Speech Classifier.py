import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from keras.backend.tensorflow_backend import clear_session
import gc
from keras import backend as K
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.33
# session = InteractiveSession(config=config)


def shuffle_in_unison(first_mat, second_mat):
    assert first_mat.shape[0] == second_mat.shape[0]
    p = np.random.permutation(first_mat.shape[0])
    return first_mat[p], second_mat[p]


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

interval = "0.05s"
num_samples = "80"
X = np.load("{} RGB Data len{} Train.npy".format(interval, num_samples))
y = np.load("{} Gender Data len{} Trainst.npy".format(interval, num_samples))
X_CV = np.load("{} RGB Data len{} CV.npy".format(interval, num_samples))
y_CV = np.load("{} Gender Data len{} CVst.npy".format(interval, num_samples))
X_test = np.load("{} RGB Data len{} Test.npy".format(interval, num_samples))
y_test = np.load("{} Gender Data len{} Test.npy".format(interval, num_samples))

#X = np.load("{} Train Log RGB Data.npy".format(interval))
#y = np.load("{} Train Gen Data.npy".format(interval))
#X_val = np.load("{} Val Log RGB Data.npy".format(interval))
#y_val = np.load("{} Val Gen Data.npy".format(interval))
#X_test = np.load("{} Test Log RGB Data.npy".format(interval))
#y_test = np.load("{} Test Gen Data.npy".format(interval))

print(X.shape[0])

conv1_sizes = [6, 10]
dense1_sizes = [24, 32, 40]
kern1_sizes = [8, 10, 12]
#kern2_sizes = [4, 6, 8]
hist = {}
top_performances = {};
for conv1 in conv1_sizes:
    for dense1 in dense1_sizes:
        for kern1 in kern1_sizes:
            for kern2 in kern1_sizes:
                for i in range(4):
                    NAME = "GendRec--conv1{}-{}kern1-{}kern2-dense1{}".format(conv1, kern1, kern2, dense1)
                    model = Sequential()
                    model.add(Conv2D(conv1, (kern1, kern2), strides=(2, 2), input_shape=X.shape[1:], activation='relu'))
                    model.add(MaxPooling2D((6, 6)))
                    model.add(Flatten())
                    model.add(Dense(dense1, activation='relu'))
                    model.add(Dropout(0.35))

                    model.add(Dense(1, activation='sigmoid'))
                    model.compile(loss='binary_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'])
                    history = model.fit(X, y, batch_size=8, verbose=0, epochs=10, validation_data=(X_CV, y_CV))
                    # plt.plot(history.history['loss'])
                    # plt.plot(history.history['val_loss'])
                    # plt.ylabel('loss')
                    # plt.xlabel('epoch')
                    # plt.legend(['train', 'test'], loc='upper left')
                    # plt.show()
                    val_loss = history.history['val_loss']
                    val_acc = history.history['val_accuracy']
                    if i == 0:
                        top_performances[NAME] = val_acc
                    else:
                        if val_acc > top_performances[NAME]:
                            top_performances[NAME] = val_acc
                    # model.save('{}-{}.model'.format(NAME, i))
                    # result = model.evaluate(X_test, y_test, batch_size=64)
                    gc.collect()
                    K.clear_session()
                    del model
                    #print("conv1:", conv1, "dense1", dense1, "kern1:", kern1, "kern2:", kern2)
                    #print("val loss, acc:", val_loss, val_acc)
print(top_performances)