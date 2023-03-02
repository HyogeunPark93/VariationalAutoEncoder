import tensorflow as tf
import numpy as np

class DatasetManager():
    def __init__(self,dataset_name):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.dataset_dict = {'mnist' : tf.keras.datasets.mnist.load_data}
        self.load_data(dataset_name)
    
    def load_data(self, dataset_name):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset_dict[dataset_name]()
        
    def normalize_and_reshape_dataset(self, dataset):
        return dataset.reshape((dataset.shape[0], dataset.shape[1], dataset.shape[2], 1))/ 255.
    
    def binarize_dataset(self, dataset):
        return np.where(dataset > .5, 1.0, 0.0).astype('float32')
    
    def shuffle_dataset(self, dataset):
        dataset_size = dataset.shape[0]
        return tf.data.Dataset.from_tensor_slices(dataset).shuffle(dataset_size)
    
    def batch_dataset(self, dataset, batch_size):
        return dataset.batch(batch_size)
    
    def process_all(self, batch_size = 32):
        self.x_train = self.normalize_and_reshape_dataset(self.x_train)
        self.x_train = self.binarize_dataset(self.x_train)
        self.x_train = self.shuffle_dataset(self.x_train)
        self.x_train = self.batch_dataset(self.x_train, batch_size)
        
        self.x_test = self.normalize_and_reshape_dataset(self.x_test)
        self.x_test = self.binarize_dataset(self.x_test)
        self.x_test = self.shuffle_dataset(self.x_test)
        self.x_test = self.batch_dataset(self.x_test, batch_size)
        
        self.y_train = self.normalize_and_reshape_dataset(self.y_train)
        self.y_train = self.binarize_dataset(self.y_train)
        self.y_train = self.shuffle_dataset(self.y_train)
        self.y_train = self.batch_dataset(self.y_train, batch_size)
        
        self.y_test = self.normalize_and_reshape_dataset(self.y_test)
        self.y_test = self.binarize_dataset(self.y_test)
        self.y_test = self.shuffle_dataset(self.y_test)
        self.y_test = self.batch_dataset(self.y_test, batch_size)

        
        
        