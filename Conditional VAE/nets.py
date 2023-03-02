import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, input_shape):
        super(Encoder, self).__init__()
        self.inputlayer = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.convlayer01 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')
        self.convlayer02 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        self.flattenlayer = tf.keras.layers.Flatten()
        self.outputlayer = tf.keras.layers.Dense(latent_dim + latent_dim)
    
    def call(self, x):
        x = self.inputlayer(x)
        x = self.convlayer01(x)
        x = self.convlayer02(x)
        x = self.flattenlayer(x)
        return self.outputlayer(x)
    
class Decoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Decoder,self).__init__()
        self.inputlayer = tf.keras.layers.InputLayer(input_shape = (latent_dim,))
        self.hiddenlayer = tf.keras.layers.Dense(units =(7*7*32), activation = 'relu')
        self.reshapelayer = tf.keras.layers.Reshape(target_shape=(7,7,32))
        self.convtranslayer01 = tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu')
        self.convtranslayer02 = tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu')
        self.outputlayer = tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same')
        
    def call(self, x):
        x = self.inputlayer(x)
        x = self.hiddenlayer(x)
        x = self.reshapelayer(x)
        x = self.convtranslayer01(x)
        x = self.convtranslayer02(x)
        return self.outputlayer(x)

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim= self.latent_dim, input_shape= (28,28,1))
        self.decoder = Decoder(latent_dim= self.latent_dim)
    
    def sample(self, eps = None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim)) 
        return self.decode(eps, apply_sigmoid=True)
            
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape = mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits