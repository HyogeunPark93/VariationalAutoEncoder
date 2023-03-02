import tensorflow as tf
import numpy as np

class DatasetManager():
        
    def reshape_dataset(dataset):
        '''reshape dataset'''
        return dataset.reshape((dataset.shape[0],dataset.shape[1],dataset.shape[2], 1))/ 255.
    
    def binarize_dataset(dataset):
        return np.where(dataset > .5, 1.0, 0.0).astype('float32')
    
    def shuffle_dataset(dataset):
        dataset_size = dataset.shape[0]
        return tf.data.Dataset.from_tensor_slices(dataset).shuffle(dataset_size)
    
    def batch_dataset(dataset, batch_size):
        return dataset.batch(batch_size)
    