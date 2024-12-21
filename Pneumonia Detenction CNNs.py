import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import (
    Activation, MaxPooling2D, Dropout, Flatten, InputLayer,
    Reshape, Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization
)
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, RMSprop

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121


class DataHandler:
    @staticmethod
    def get_metadata(metadata_path, which_splits=['train', 'test']):
        metadata = pd.read_csv(metadata_path)
        return metadata[metadata['split'].isin(which_splits)]

    @staticmethod
    def get_data_split(split_name, flatten, all_data, metadata, image_shape):
        sub_df = metadata[metadata['split'] == split_name]
        indices = sub_df['index'].values
        labels = sub_df['class'].values
        data = all_data[indices]

        if flatten:
            data = data.reshape(-1, np.prod(image_shape))
        
        return data, labels

    @staticmethod
    def get_train_data(flatten, all_data, metadata, image_shape):
        return DataHandler.get_data_split('train', flatten, all_data, metadata, image_shape)

    @staticmethod
    def get_test_data(flatten, all_data, metadata, image_shape):
        return DataHandler.get_data_split('test', flatten, all_data, metadata, image_shape)


class Visualization:
    @staticmethod
    def plot_one_image(data, labels=None, index=None, image_shape=(64, 64, 3)):
        if labels is None:
            labels = []

        if data.ndim == 1:
            data = data.reshape(image_shape)

        if data.ndim == 2:
            data = data.reshape(-1, *image_shape)

        if data.ndim == 3:
            image = data
            label = labels[0] if labels else ""
        elif data.ndim == 4:
            image = data[index]
            label = labels[index]

        print(f"Label: {label}")
        plt.imshow(image)
        plt.show()

    @staticmethod
    def plot_acc(history, xlabel='Epoch #'):
        history_df = pd.DataFrame(history.history)
        history_df['epoch'] = range(len(history_df['val_accuracy']))

        best_epoch = history_df.loc[history_df['val_accuracy'].idxmax(), 'epoch']

        plt.figure()
        sns.lineplot(x='epoch', y='val_accuracy', data=history_df, label='Validation')
        sns.lineplot(x='epoch', y='accuracy', data=history_df, label='Training')
        plt.axhline(0.5, linestyle='--', color='red', label='Chance')
        plt.axvline(best_epoch, linestyle='--', color='green', label='Best Epoch')
        plt.legend(loc='lower right')
        plt.xlabel(xlabel)
        plt.ylabel('Accuracy')
        plt.show()


class ModelBuilder:
    @staticmethod
    def DenseClassifier(hidden_layer_sizes, nn_params):
        model = Sequential([Flatten(input_shape=nn_params['input_shape']), Dropout(0.5)])
        
        for size in hidden_layer_sizes:
            model.add(Dense(size, activation='relu'))
            model.add(Dropout(0.5))
        
        model.add(Dense(nn_params['output_neurons'], activation=nn_params['output_activation']))
        model.compile(loss=nn_params['loss'], optimizer=SGD(learning_rate=1e-4, momentum=0.95), metrics=['accuracy'])
        return model

    @staticmethod
    def CNNClassifier(num_hidden_layers, nn_params):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01), input_shape=nn_params['input_shape']))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        for _ in range(num_hidden_layers - 1):
            model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(nn_params['output_neurons'], activation=nn_params['output_activation']))
        
        opt = RMSprop(learning_rate=1e-5, decay=1e-6)
        model.compile(loss=nn_params['loss'], optimizer=opt, metrics=['accuracy'])
        return model

    @staticmethod
    def TransferClassifier(name, nn_params, trainable=False):
        expert_dict = {'VGG16': VGG16, 'VGG19': VGG19, 'ResNet50': ResNet50, 'DenseNet121': DenseNet121}
        expert_conv = expert_dict[name](weights='imagenet', include_top=False, input_shape=nn_params['input_shape'])

        for layer in expert_conv.layers:
            layer.trainable = trainable

        model = Sequential([
            expert_conv,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(nn_params['output_neurons'], activation=nn_params['output_activation'])
        ])

        model.compile(loss=nn_params['loss'], optimizer=SGD(learning_rate=1e-4, momentum=0.9), metrics=['accuracy'])
        return model
# Defining project variables
metadata_url = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/metadata.csv"
image_data_url = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20A)%20Pneumonia/image_data.npy"
image_data_path = "./image_data.npy"
metadata_path = "./metadata.csv"
image_shape = (64, 64, 3)

# Neural net parameters
nn_params = {
    "input_shape": image_shape,
    "output_neurons": 1,
    "loss": "binary_crossentropy",
    "output_activation": "sigmoid",
}

# Downloading data
!wget -q --show-progress $metadata_url
!wget -q --show-progress $image_data_url

# Pre-loading data
_all_data = np.load("image_data.npy")
_metadata = DataHandler.get_metadata(metadata_path, ["train", "test", "field"])

# Preparing definitions
get_data_split = DataHandler.get_data_split
get_metadata = lambda: DataHandler.get_metadata(metadata_path, ["train", "test"])
get_train_data = lambda flatten=False: DataHandler.get_train_data(
    flatten=flatten, all_data=_all_data, metadata=_metadata, image_shape=image_shape
)
get_test_data = lambda flatten=False: DataHandler.get_test_data(
    flatten=flatten, all_data=_all_data, metadata=_metadata, image_shape=image_shape
)

# Plotting functions
plot_one_image = lambda data, labels=[], index=None: Visualization.plot_one_image(
    data=data, labels=labels, index=index, image_shape=image_shape
)
plot_acc = lambda history: Visualization.plot_acc(history)

# Model definitions
DenseClassifier = lambda hidden_layer_sizes: ModelBuilder.DenseClassifier(
    hidden_layer_sizes=hidden_layer_sizes, nn_params=nn_params
)
CNNClassifier = lambda num_hidden_layers: ModelBuilder.CNNClassifier(
    num_hidden_layers=num_hidden_layers, nn_params=nn_params
)
TransferClassifier = lambda name: ModelBuilder.TransferClassifier(
    name=name, nn_params=nn_params
)

monitor = ModelCheckpoint(
    "./model.h5",
    monitor="val_accuracy",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
)

# Model example
model_1 = Sequential()
model_1.add(InputLayer(input_shape=(3,)))
model_1.add(Dense(4, activation="relu"))
model_1.add(Dense(2, activation="softmax"))
model_1.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

input_data = [[14, 54, 2]]
print(model_1.predict(input_data))
print((model_1.predict(input_data) > 0.5).astype("int32"))

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

X_train, y_train = get_train_data(flatten=True)
X_test, y_test = get_test_data(flatten=True)

mlp = MLPClassifier(alpha=1, max_iter=100, random_state=1).fit(X_train, y_train)
y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

X_train, y_train = get_train_data()
X_test, y_test = get_test_data()

cnn = CNNClassifier(num_hidden_layers=5)
cnn.add(Dropout(0.1))

dense = DenseClassifier(hidden_layer_sizes=(64, 32, 16))
dense.add(Dropout(0.1))

dense_history = dense.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), shuffle=True, callbacks=[monitor])
cnn_history = cnn.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), shuffle=True, callbacks=[monitor])

plot_acc(cnn_history)
plot_acc(dense_history)

transfer = TransferClassifier(name='ResNet50')
transfer.add(Dropout(0.2))
transfer.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), shuffle=True, callbacks=[monitor])

best_model = cnn

prediction = (best_model.predict(X_test) > 0.5).astype("int32")
accuracy = best_model.evaluate(X_test, y_test)

confusion = confusion_matrix(y_test, y_pred)
print(confusion)

tp = confusion[1][1]
tn = confusion[0][0]
fp = confusion[0][1]
fn = confusion[1][0]

print('True positive:', tp)
print('True negative:', tn)
print('False positive:', fp)
print('False negative:', fn)

sns.heatmap(confusion, annot=True, fmt='d', cbar_kws={'label': 'count'})
plt.ylabel('Actual')
plt.xlabel('Predicted')
