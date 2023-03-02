
from tensorflow.python import keras
from tensorflow.python.keras import layers
import tensorflow as tf

LATENT_DIM = 16
NUM_OF_EMBEDDINGS = 64

class Vector_Quantizer(layers.Layer):
    def __init__(self, num_of_embeddings, embedding_dim, beta = 0.25,**kwargs):
        super().__init__(**kwargs)
        self.num_of_embeddings = num_of_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_of_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_of_vqvae",
        )
        
    def call(self, _inputs):
        input_shape = tf.shape(_inputs)
        flattened = tf.reshape(_inputs, [-1, self.embedding_dim])
        indices = self.get_indices_from_codebook(flattened)
        indices = tf.one_hot(indices, self.num_of_embeddings)
        embeddings_tr = tf.transpose(self.embeddings, perm=[1, 0])
        e_k_flattend = tf.matmul(indices, embeddings_tr)
        e_k = tf.reshape(e_k_flattend, input_shape)
        
        
        codebook_loss = tf.reduce_mean((tf.stop_gradient(_inputs) - e_k)**2)
        commitment_loss = tf.reduce_mean((_inputs - tf.stop_gradient(e_k))**2)
        self.add_loss(codebook_loss + self.beta* commitment_loss)
        
        e_k = _inputs + tf.stop_gradient(e_k - _inputs)
        return e_k
    
    def get_indices_from_codebook(self, flattened_inputs):
        cross_terms = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * cross_terms
        )
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

def get_encoder(latent_dim = LATENT_DIM):
    _inputs = layers.Input(shape=(28,28,1))
    conv01 = layers.Conv2D(32, 3, strides=(2,2), activation='relu', padding='same')(_inputs)
    conv02 = layers.Conv2D(64, 3, strides=(2,2), activation='relu', padding='same')(conv01)
    _outputs = layers.Conv2D(latent_dim, 1, padding="same")(conv02)
    return keras.Model(_inputs, _outputs, name="Encoder")

def get_decoder(latent_dim = LATENT_DIM):
    _inputs = layers.Input(shape = get_encoder(latent_dim).output.shape[1:])
    convtr01  = layers.Conv2DTranspose(64, 3, strides=(2,2) ,activation='relu', padding='same')(_inputs)
    convtr02  = layers.Conv2DTranspose(32, 3, strides=(2,2) ,activation='relu', padding='same')(convtr01)
    _outputs = layers.Conv2DTranspose(1, 3, padding="same")(convtr02)
    return keras.Model(_inputs, _outputs, name = "Decoder")

def get_vq_vae(latent_dim = LATENT_DIM, num_of_embeddings = NUM_OF_EMBEDDINGS):
    vq_layer = Vector_Quantizer(num_of_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    _inputs = keras.Input(shape=(28, 28, 1))
    encoder_outputs = encoder(_inputs)
    quantized_latents = vq_layer(encoder_outputs)
    _outputs = decoder(quantized_latents)
    return keras.Model(_inputs, _outputs, name="vq_vae")


    
get_vq_vae().summary()