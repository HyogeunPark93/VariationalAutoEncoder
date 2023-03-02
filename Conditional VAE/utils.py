import PIL
import PIL.Image
import tensorflow as tf
import numpy as np

class Datasetmanager():
    def create_tf_dataset(*args):
        return tf.data.Dataset.from_tensor_slices(args)

    def shuffle_dataset(dataset,buffer_size,seed = None):
        return dataset.shuffle(buffer_size, seed)

    def batch_dataset(dataset,batch_size=32):
        return dataset.batch(batch_size)


train , test = tf.keras.datasets.mnist.load_data()
train_img, train_lbl = train
train_lbl_onehot = tf.one_hot(train_lbl, 10)
BUFFER_SIZE, IMAGE_X_AXIS, IMAGE_Y_AXIS = train_img.shape
dataset = Datasetmanager.create_tf_dataset(train_img, train_lbl_onehot)
dataset = Datasetmanager.shuffle_dataset(dataset, BUFFER_SIZE, 1)
dataset = Datasetmanager.batch_dataset(dataset)
iterator = iter(dataset)
train_data, label_data = iterator.get_next()
print(label_data)