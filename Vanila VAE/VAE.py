import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
    

def get_encoder(latent_dim):
    inputs = tf.keras.Input(shape = (28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name = "z_mean")(x)
    z_logvar = tf.keras.layers.Dense(latent_dim, name = "z_logvar")(x)
    outputs = [z_mean, z_logvar]
    encoder = tf.keras.Model(inputs, outputs, name = "VAE_Encoder")
    return encoder

class Sampling_layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Sampling_layer, self).__init__(**kwargs)
    
    def call(self, inputs):
        means, logvars = inputs
        batch_size = tf.shape(means)[0]
        latent_dim = tf.shape(means)[1]
        eps = tf.random.normal(shape=(batch_size, latent_dim))
        stds = tf.exp(.5 * logvars)
        return means + stds*eps

def get_decoder(latent_dim):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")(inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid",padding="same")(x)
    decoder = tf.keras.Model(inputs, outputs,name = 'VAE_Decoder')
    return decoder

class VAE(tf.keras.Model):
    def __init__(self,latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = get_encoder(self.latent_dim)
        self.sampler = Sampling_layer(name = "VAE_Sampler")
        self.decoder = get_decoder(self.latent_dim)
        
        #loss tracker
        self.elbo_tracker = tf.keras.metrics.Mean(name = "ELBO")
        self.recon_tracker = tf.keras.metrics.Mean(name = "Reconstruction Error")
        self.dkl_tracker = tf.keras.metrics.Mean(name = "Dkl(qz_x|pz)")


        
    @property
    def metrics(self):
        return[
            self.elbo_tracker,
            self.recon_tracker,
            self.dkl_tracker,
        ]
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_means, z_logvars = self.encoder(data)
            z = self.sampler([z_means, z_logvars])
            recon = self.decoder(z)
            
            logpx_z = -tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, recon), axis=(1,2)))
            logpz = self.log_normal_pdf(z, 0.0, 0.0)
            logqz_x = self.log_normal_pdf(z, z_means, z_logvars)

            dkl_qz_x_pz = tf.reduce_mean(tf.reduce_sum(logqz_x - logpz, axis= 1))
            elbo = logpx_z - dkl_qz_x_pz
            loss = -logpx_z + dkl_qz_x_pz
        
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.elbo_tracker.update_state(elbo)
        self.recon_tracker.update_state(-logpx_z)
        self.dkl_tracker.update_state(dkl_qz_x_pz)
        return {
            "ELBO" : self.elbo_tracker.result(),
            "Reconstruction Error" : self.recon_tracker.result(),
            "KL Divergence(q(z|x)|p(z))" : self.dkl_tracker.result(),
        }
    
    def log_normal_pdf(self, sample, mean, logvar):
        log2pi = tf.math.log(2. * np.pi)
        return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)

def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 4.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

if __name__ =="__main__":
    # LATENT_DIM = 2
    # BATCH_SIZE = 32
    # EPOCH = 30
    
    (x_train,_),(x_test,_) = tf.keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255.
    
    train_dataset = tf.data.Dataset.from_tensor_slices(mnist_digits).shuffle(mnist_digits.shape[0]).batch(128)
    
    # vae = VAE(LATENT_DIM)
    # vae.compile(optimizer =tf.keras.optimizers.Adam(1e-4))
    # vae.fit(train_dataset, epochs= EPOCH)
    # vae.save_weights("model\\VAE_model")
    vae = VAE(2)
    vae.compile(optimizer =tf.keras.optimizers.Adam(1e-4))
    vae.load_weights("model\\VAE_model")
    # vae.fit(train_dataset, epochs= 10)
    # vae.save_weights("model\\VAE_model")

    plot_latent_space(vae)