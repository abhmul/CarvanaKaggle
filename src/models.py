import tensorflow as tf  # Needed for norm method
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, MaxPool2D, \
    UpSampling2D, Concatenate
from keras.models import Model
from keras.losses import binary_crossentropy
import keras.backend as K

MODEL_FILE = "../models/{}.h5"


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true = K.round(K.flatten(y_true))
    y_pred = K.round(K.flatten(y_pred))

    isct = K.sum(y_true * y_pred)

    return (2 * isct + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def log_dice_loss(y_true, y_pred):
    return -K.log(dice_loss(y_true, y_pred))


def jacard_loss(y_true, y_pred):
    smooth = 1.
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    isct_norm = tf.norm(y_true * y_pred)

    y_true_dot = K.dot(K.expand_dims(y_true, axis=0), K.expand_dims(y_true, axis=-1))[0, 0]
    y_pred_dot = K.dot(K.expand_dims(y_pred, axis=0), K.expand_dims(y_pred, axis=-1))[0, 0]

    return 1 - (isct_norm / (y_true_dot + y_pred_dot - isct_norm))


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))


def simple_model(img_size, optimizer=None, loss=bce_dice_loss, train=True):
    input_layer = Input(img_size + (3,))
    x = BatchNormalization()(input_layer)

    x = Conv2D(16, 3, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32, 3, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(1, 5, activation='linear', padding='same')(x)
    x = Activation('sigmoid')(x)
    out = Reshape(img_size)(x)

    model = Model(input_layer, out)
    if train:
        model.compile(optimizer, loss=loss, metrics=['accuracy', dice_coef])

    return model


def down(input_layer, filters, pool=True):
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='linear')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    residual = Conv2D(filters, (3, 3), padding='same', activation='linear')(conv1)
    residual = BatchNormalization()(residual)
    residual = Activation('relu')(residual)

    if pool:
        max_pool = MaxPool2D()(residual)
        return max_pool, residual
    else:
        return residual


def up(input_layer, residual, filters):
    filters = int(filters)
    upsample = UpSampling2D()(input_layer)

    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
    # upconv = BatchNormalization()(upconv)
    # upconv = Activation('relu')(upconv)

    concat = Concatenate(axis=3)([residual, upconv])

    conv1 = Conv2D(filters, (3, 3), padding='same', activation='linear')(concat)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(filters, (3, 3), padding='same', activation='linear')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    return conv2


def u_net_small(img_size, filters=32, optimizer=None, loss=bce_dice_loss, train=True):
    # Make a custom U-nets implementation.
    input_layer = Input(img_size + (3,))
    layers = [input_layer]
    residuals = []
    # Down 1, 128
    d1, res1 = down(input_layer, filters)
    residuals.append(res1)
    filters *= 2
    # Down 2, 64
    d2, res2 = down(d1, filters)
    residuals.append(res2)
    filters *= 2
    # Down 3, 32
    d3 = down(d2, filters, pool=False)
    # Up 1, 64
    up1 = up(d3, residual=residuals[-1], filters=filters / 2)
    filters /= 2
    # Up 2, 128
    up2 = up(up1, residual=residuals[-2], filters=filters / 2)

    out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up2)
    out = Reshape(img_size)(out)

    model = Model(input_layer, out)
    model.summary()
    if train:
        model.compile(optimizer, loss=loss, metrics=['accuracy', dice_coef])

    return model


def u_net_big(img_size, filters=32, optimizer=None, loss=bce_dice_loss, train=True):
    # Make a custom U-nets implementation.
    input_layer = Input(img_size + (3,))
    layers = [input_layer]
    residuals = []
    # Down 1, 128
    d1, res1 = down(input_layer, filters)
    residuals.append(res1)
    filters *= 2
    # Down 2, 64
    d2, res2 = down(d1, filters)
    residuals.append(res2)
    filters *= 2
    # Down 3, 32
    d3, res3 = down(d2, filters)
    residuals.append(res3)
    filters *= 2
    # Down 4, 16
    d4, res4 = down(d3, filters)
    residuals.append(res4)
    filters *= 2
    # Down 5, 8
    d5 = down(d4, filters, pool=False)
    # Up 1, 16
    up1 = up(d5, residual=residuals[-1], filters=filters / 2)
    filters /= 2
    # Up 2,  32
    up2 = up(up1, residual=residuals[-2], filters=filters / 2)
    filters /= 2
    # Up 3, 64
    up3 = up(up2, residual=residuals[-3], filters=filters / 2)
    filters /= 2
    # Up 4, 128
    up4 = up(up3, residual=residuals[-4], filters=filters / 2)

    out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)
    out = Reshape(img_size)(out)

    model = Model(input_layer, out)
    model.summary()
    if train:
        model.compile(optimizer, loss=loss, metrics=['accuracy', dice_coef])

    return model


def u_net(img_size, down_layers, filters=8, optimizer=None, loss=bce_dice_loss, train=True):
    """down samples dimensions by 2^(down_layers-1)"""
    # Make a custom U-nets implementation.
    input_layer = Input(img_size + (3,))
    layers = [input_layer]
    residuals = []
    d = input_layer
    # Down layers
    for i in range(down_layers - 1):
        d, res = down(d, filters)
        residuals.append(res)
        filters *= 2
    # Final non-res down layer
    d = down(d, filters, pool=False)
    # Up layers
    u = d
    for i in range(down_layers - 1):
        u = up(u, residual=residuals[-(i + 1)], filters=filters / 2)
        filters /= 2

    # Output layer
    out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(u)
    out = Reshape(img_size)(out)

    model = Model(input_layer, out)
    model.summary()
    if train:
        print("Training with ", loss)
        model.compile(optimizer, loss=loss, metrics=[dice_loss, dice_coef])

    return model


def u_net_big_img(img_size=(320, 480), filters=8, optimizer=None, loss=bce_dice_loss, train=True):
    return u_net(img_size, 6, filters=filters, optimizer=optimizer, loss=loss, train=train)


def u_net_full_img(img_size=(1280, 1920), filters=2, optimizer=None, loss=bce_dice_loss, train=True):
    return u_net(img_size, 8, filters=filters, optimizer=optimizer, loss=loss, train=train)


def u_net_512(img_size=(512, 512), filters=8, optimizer=None, loss=bce_dice_loss, train=True):
    return u_net(img_size, 7, filters=filters, optimizer=optimizer, loss=loss, train=train)


def u_net_1024(img_size=(1024, 1024), filters=4, optimizer=None, loss=bce_dice_loss, train=True):
    return u_net(img_size, 8, filters=filters, optimizer=optimizer, loss=loss, train=train)
