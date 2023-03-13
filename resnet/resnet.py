import tensorflow as tf

def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu'):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation:
        x = tf.keras.layers.Activation(activation)(x)
    return x

def identity_block(x, filters, kernel_size):
    shortcut = x
    x = conv2d_bn(x, filters=filters, kernel_size=kernel_size)
    x = conv2d_bn(x, filters=filters, kernel_size=kernel_size, activation=None)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def conv_block(x, filters, kernel_size, strides=2):
    shortcut = x
    x = conv2d_bn(x, filters=filters, kernel_size=kernel_size, strides=strides)
    x = conv2d_bn(x, filters=filters, kernel_size=kernel_size, activation=None)
    shortcut = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                      strides=strides, use_bias=False)(shortcut)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ResNet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = conv2d_bn(inputs, filters=64, kernel_size=7, strides=2)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = conv_block(x, filters=64, kernel_size=3)
    x = identity_block(x, filters=64, kernel_size=3)
    x = identity_block(x, filters=64, kernel_size=3)

    x = conv_block(x, filters=128, kernel_size=3)
    x = identity_block(x, filters=128, kernel_size=3)
    x = identity_block(x, filters=128, kernel_size=3)
    x = identity_block(x, filters=128, kernel_size=3)

    x = conv_block(x, filters=256, kernel_size=3)
    x = identity_block(x, filters=256, kernel_size=3)
    x = identity_block(x, filters=256, kernel_size=3)
    x = identity_block(x, filters=256, kernel_size=3)
    x = identity_block(x, filters=256, kernel_size=3)
    x = identity_block(x, filters=256, kernel_size=3)

    x = conv_block(x, filters=512, kernel_size=3)
    x = identity_block(x, filters=512, kernel_size=3)
    x = identity_block(x, filters=512, kernel_size=3)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
