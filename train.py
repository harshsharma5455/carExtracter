import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model, Model
# from model import build_u2net_lite, build_u2net

""" Global parameters """
H = 512
W = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.1):
    train_x = sorted(glob(os.path.join(path, "train", "all_images", "*.JPG")))
    train_y = sorted(glob(os.path.join(path, "train", "modified_masks", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "train", "test","images", "*.JPG")))
    valid_y = sorted(glob(os.path.join(path, "train", "test", "modified_masks", "*.png")))

    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    x = cv2.resize(x, (W, H))
    x = x[:,:,3]
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(buffer_size=50)
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-4
    num_epochs = 30
    model_path = os.path.join("/content/drive/MyDrive/imageModel/", "experiment_shuffle.h5")
    new_model_path = os.path.join("/content/drive/MyDrive/imageModel/", "experiment_alpha_only.h5")
    csv_path = os.path.join("/content/drive/MyDrive/imageModel/", "log.csv")

    """ Dataset """
    dataset_path = "/content/drive/MyDrive/imageModel/P3M-10k"
    (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    pretrained_model = load_model(model_path)

    # Create a new model with the added layer
    model = Model(inputs=pretrained_model.input, outputs=pretrained_model.output)
    model.compile(loss="mean_squared_error", optimizer=Adam(lr), metrics=['accuracy'])


    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-8, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )

    # Save the trained model
    model.save(new_model_path)