import glob
import os

import numpy as np

import keras
import tensorflow as tf
from keras.callbacks import Callback

from data_preprocessing import EPOCHS

from model import transformer
from dataset import train_ds, val_ds

MODEL_CHECKPOINT_DIR = "model_chkpts/"
if not os.path.isdir(MODEL_CHECKPOINT_DIR):
    os.makedirs(MODEL_CHECKPOINT_DIR)

transformer.compile(
    "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                MODEL_CHECKPOINT_DIR,
                                monitor="val_loss",
                                verbose=1,
                                save_best_only=False,
                                save_weights_only=False,
                                mode="auto",
                                save_freq=10,
                                initial_value_threshold=None,
)

class SaveModelsCallback(Callback):

    def on_train_begin(self, logs=None):        
        self.best_loss = [-1, np.inf]
        self.best_model = None
        
        best_model_paths = glob.glob(os.path.join(MODEL_CHECKPOINT_DIR, "model_best_epoch*"))
        if len(best_model_paths) >= 1:
            best_model_paths = sorted(best_model_paths, key=lambda x: float(x.split('.h5')[0].split('_')[-1]))[0]
            self._model = keras.saving.load_model(best_model_paths)

    def on_train_end(self, logs=None):
        if not self.best_loss[0] == -1:
            keras.save(self.best_model, os.path.join(MODEL_CHECKPOINT_DIR, "model_best_epoch%d_%f.h5" % (self.best_loss[0], self.best_loss[1]))) 

    def on_epoch_end(self, epoch, logs=None):
        
        if self.val_loss < self.best_loss[1]:
            self.best_loss[1] = self.val_loss
            self.best_loss[0] = epoch
            self.best_model = self.model()

transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[model_checkpoint_callback]) #SaveModelsCallback(monitor="val_loss")])
#        tf.keras.callbacks.ModelCheckpoint(filepath='%s/model.{epoch:02d}-{val_loss:.2f}.h5' % MODEL_CHECKPOINT_DIR)
#    ])
