from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute

     

def unet_Model(image_size, learning_rate, weight_decay):
    # Build U-Net model
    inputs = Input((image_size[0], image_size[1], 3))
    kernels = 32
    
    c1 = Conv2D(kernels, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(kernels, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(kernels*16, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(kernels*16, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(kernels*8, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(kernels*4, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(kernels*2, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(kernels, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(kernels, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(kernels, (3, 3), activation='relu', padding='same', name='check') (c9)
    

    c9 = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    

    model_ = Model(inputs=[inputs], outputs=[c9])
    
    model_.compile(loss="binary_crossentropy", optimizer= Adam(lr=learning_rate, decay=weight_decay) , metrics=['accuracy'])
    
    return model_


def unet_Model_LT(image_size, learning_rate, weight_decay):
    # Build U-Net model
    inputs = Input((image_size[0], image_size[1], 3))
    kernels = 32

    c1 = Conv2D(kernels, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(kernels, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(kernels*16, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(kernels*16, (3, 3), activation='relu', padding='same', name='LSpace') (c5)

    u6 = Conv2DTranspose(kernels*8, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(kernels*4, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(kernels*2, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(kernels, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(kernels, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(kernels, (3, 3), activation='relu', padding='same', name='check') (c9)
    

    c9 = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    

    model_ = Model(inputs=[inputs], outputs=[c9, c5])
    model_.compile(loss="binary_crossentropy", optimizer= Adam(lr=learning_rate, decay=weight_decay), metrics=['accuracy'])
#     model.compile(loss="binary_crossentropy", optimizer= Adam(lr=1e-4, decay=1e-6))
    
    return model_


def unet_Model_CLS(image_size, learning_rate, weight_decay):
    
    # Build U-Net model
    inputs = Input((image_size[0], image_size[1], 3))
    kernel = 3

    inp0 = inputs
    
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same') (inp0)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(512, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same', name='lastLayer') (c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid', name='semSeg') (c9)
    
    
    
    c50 = Conv2D(64, (1, 1), activation='relu', padding='same') (c5)
    c50 = UpSampling2D(size=(2, 2), data_format=None) (c50)

    
    c90 = Conv2D(64, (1, 1), activation='relu', padding='same') (u9)
    p90 = AveragePooling2D(pool_size=(8, 8)) (c90)
    
    
    c91 = concatenate([c50, p90], axis=3)
    c91 = Conv2D(32, (1, 1), activation='relu', padding='same') (c91)
    
    d0 = Flatten()(c91)
    d0 = Dropout(0.5)(d0)    
    d1 = Dense(64, activation='relu')(d0)  
    d1 = Dropout(0.5)(d1)
    d2 = Dense(16, activation='relu')(d1)
    outPut2 = Dense(1, activation='sigmoid', name='binCls')(d2)

    model = Model(inputs=[inputs], outputs=[outputs, outPut2])
    
    model.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer= Adam(lr=learning_rate, decay=weight_decay) , metrics=['accuracy'],loss_weights=[1.0, 0.01])
    
    return model


def unet_Model_LT_WAN(image_size, learning_rate, weight_decay):
    # Build U-Net model
    inputs = Input((image_size[0], image_size[1], 3))
    kernels = 32
    inp0 = inputs
    
    c1 = Conv2D(kernels, (3, 3), activation='relu', padding='same') (inp0)
    c1 = Conv2D(kernels, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(kernels*16, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(kernels*16, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(kernels*8, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(kernels*8, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(kernels*4, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(kernels*4, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(kernels*2, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(kernels*2, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(kernels, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(kernels, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(kernels, (3, 3), activation='relu', padding='same', name='check') (c9)
    

    output1 = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    c50 = Conv2D(64, (1, 1), activation='relu', padding='same') (c5)
    c50 = UpSampling2D(size=(2, 2), data_format=None) (c50)
    
    c90 = Conv2D(64, (1, 1), activation='relu', padding='same') (u9)
    p90 = AveragePooling2D(pool_size=(8, 8)) (c90)
    
    
    c91 = concatenate([c50, p90], axis=3)
    c91 = Conv2D(32, (1, 1), activation='relu', padding='same') (c91)
    
    d0 = Flatten()(c91)
    d0 = Dropout(0.5)(d0)    
    d1 = Dense(64, activation='relu')(d0)  
    d1 = Dropout(0.5)(d1)
    d2 = Dense(16, activation='relu')(d1)
    output2 = Dense(1, activation='sigmoid', name='binCls')(d2)
    

    model = Model(inputs=[inputs], outputs=[output1, output2, c5])
    
    model.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], optimizer= Adam(lr=learning_rate, decay=weight_decay) , metrics=['accuracy'],loss_weights=[1.0, 0.01,0.1])
    
    
    return model


def discriminator_OSA(image_size):

    inputs = Input((image_size[0], image_size[1], 1))
    kernel = 4
    x = inputs
    # Encoder
    
    x = Conv2D(64, (kernel, kernel), strides=(2,2), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)


    x = Conv2D(128, (kernel, kernel), strides=(2,2), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)


    x = Conv2D(256, (kernel, kernel), strides=(2,2), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)


    x = Conv2D(512, (kernel, kernel), strides=(2,2), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Decoder
    x = Conv2D(1, (kernel, kernel), strides=(2,2), padding="same")(x)
    x = UpSampling2D(size=(32, 32))(x)
    outputs = Activation("sigmoid")(x)
 

    model_d = Model(inputs=[inputs], outputs=[outputs])

    model_d.compile(loss="binary_crossentropy", optimizer= Adam(lr=1e-5, decay=1e-6) , metrics=['accuracy'] )

    return model_d


def discriminator_LTA(image_size):
    image_size = 16
    inputs = Input((image_size, image_size, 512))
    kernel = 4
    x = inputs
    # Encoder
    
    x = Conv2D(256, (kernel, kernel), strides=(1,1), padding="same")(x)

    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (kernel, kernel), strides=(1,1), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)


    x = Conv2D(64, (kernel, kernel), strides=(1,1), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Decoder
    x = Conv2D(1, (kernel, kernel), strides=(1,1), padding="same")(x)
    x = UpSampling2D(size=(16,16))(x)
    outputs = Activation("sigmoid")(x)
 

    model_d = Model(inputs=[inputs], outputs=[outputs])

    model_d.compile(loss="binary_crossentropy", optimizer= Adam(lr=1e-5, decay=1e-6) , metrics=['accuracy'] )

    return model_d


if __name__ == '__main__':
    model = unet_Model([256,256], 1e-4, 1e-6)
