from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
class SiameseGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, n_classes, batch_size=32, shape=(224, 224, 3), shuffle=True):
        self.data = pd.DataFrame(data)
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.data.shape[0] / float(self.batch_size)))
   

    def __getitem__(self, index):
        indices = np.arange(len(self.data))[index*self.batch_size:(index+1)*self.batch_size]
        X1, X2, y = self.__data_generation(indices)
        return [X1, X2], y
    
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        X1 = np.empty((self.batch_size, *self.shape))
        X2 = np.empty((self.batch_size, *self.shape))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        for i, index in enumerate(indices):
            item = self.data.iloc[index]
            X1[i,] = self._load_image(item[0])
            X2[i,] = self._load_image(item[1])
            y[i] = item[2]

        return X1, X2, y

    def _load_image(self, path):
        img = load_img(path, target_size=self.shape)
        img = img_to_array(img)
        img = img / 255.0  # Normalize pixel values to [0, 1]
        return img
