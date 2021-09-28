import tensorflow as tf
from tensorflow.keras import layers, models



def simple_model(input_shape):

    height, width, channels = input_shape
    image = layers.Input(input_shape)
    x = layers.Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu')(image)
    x = layers.Conv2D(64, 5, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 1, padding='same', activation=None)(x)
    # resize back into same size as regularization mask
    x = tf.image.resize(x, [height, width])
    x = tf.keras.activations.sigmoid(x)

    model = models.Model(inputs=image, outputs=x)

    return model


def conv2d_3x3(filters):
    conv = layers.Conv2D(
        filters, kernel_size=(3, 3), activation='relu', padding='same'
    )
    return conv


def max_pool():
    return layers.MaxPooling2D((2, 2), strides=2, padding='same')


def unet(input_shape):

    height, width, channels = input_shape
    image = layers.Input(shape=input_shape)

    # first double layer
    c0 = layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(image)
    c1 = layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(c0)
    p1 = max_pool()(c1)

    # second double layer
    c2 = layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(p1)
    c3 = layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(c2)
    p2 = max_pool()(c3)

    # third double layer
    c4 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(p2)
    c5 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(c4)
    #c5_padded = layers.ZeroPadding2D(padding=((1,0), (1,0)))(c5)
    p3 = max_pool()(c5)

    # fourth double layer
    c6 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(p3)
    c7 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(c6)
    p4 = max_pool()(c7)

    # the bottom module
    c8 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(p4)
    c9 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(c8)
    up1 = layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="valid")(c9)

    # first up
    concatenated_data_1 = layers.Concatenate(axis=-1)([c7, up1])
    c10 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(concatenated_data_1)
    c11 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(c10)
    up2 = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding="valid")(c11)

    # second up
    #up2_ = layers.Cropping2D(cropping=((1, 0), (1, 0)))(up2)
    concatenated_data_2 = layers.Concatenate(axis=-1)([c5, up2]) #up2_
    c12 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(concatenated_data_2)  # plus
    c13 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(c12)
    up3 = layers.Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding="valid")(c13)

    # third up
    concatenated_data_3 = layers.Concatenate(axis=-1)([c3, up3])
    c14 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(concatenated_data_3)  # plus
    c15 = layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(c14)
    up4 = layers.Conv2DTranspose(8, kernel_size=(2, 2), strides=(2, 2), padding="valid")(c15)

    # output layer
    concatenated_data_4 = layers.Concatenate(axis=-1)([c1, up4])
    c16 = layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(concatenated_data_4)  # plus
    c17 = layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(c16)
    c18 = layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(c17)
    probs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(c18)

    model = models.Model(inputs=image, outputs=probs)
    return model

#model = unet([512,512,3])
#model.summary()
