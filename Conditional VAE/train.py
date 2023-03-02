from IPython import display
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from nets import *
import time
import imageio
import PIL
import glob

def train_nets(model,
               train_dataset,
               test_dataset,
               batch_size,
               optimizer=tf.keras.optimizers.Adam(1e-4),
               epochs= 10):
    test_sample = create_test_sample(test_dataset,batch_size)    
    
    for epoch in range(1, epochs +1):
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
        end_time = time.time()
    
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
        generate_and_save_images(model, epoch, test_sample)

def create_test_sample(test_dataset,batch_size,num_examples_to_generate = 16):
    assert batch_size >= num_examples_to_generate
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    return test_sample

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
     with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
     gradients = tape.gradient(loss, model.trainable_variables)
     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))