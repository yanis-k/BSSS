import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, \
    BatchNormalization, Input
from tensorflow.keras.layers import LSTM, Concatenate, LeakyReLU
from keras.losses import mean_squared_error


def custom_mse(x):
    """
    Permutation Invariant Training MSE custom function loss.
    Keras Implementation.
    Used in PIT with the CRNN.
    """

    def pit_loss(y_true, y_pred):
        cost1 = mean_squared_error(y_pred[x], y_true[x])

        def c1(): return tf.reduce_mean(cost1)

        cost2 = mean_squared_error(y_pred[x - 1], y_true[x])

        def c2(): return tf.reduce_mean(cost2)

        result = tf.cond(tf.less(tf.reduce_mean(cost1), tf.reduce_mean(cost2)), c1, c2)
        return result

    return pit_loss


def bl_dnn_mimo(bins, hl_nodes):
    """
    Baseline DNN model
    Used as reference
    [Speaker1, Speaker2] Output

    :param bins: number of frequency bins
    :param hl_nodes: number of hidden nodes
    """
    inp_x = Input(shape=(bins,))
    x = inp_x
    x = Dense(hl_nodes)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)
    x = Dense(hl_nodes)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)
    x = Dense(hl_nodes)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    out_l = Dense(bins, activation='linear', name="Out1")(x)
    out_r = Dense(bins, activation='linear', name="Out2")(x)

    model = Model(inputs=inp_x, outputs=[out_l, out_r])
    model.compile(optimizer='adam', loss=keras.losses.mean_squared_error)
    return model


def cnn_twin(bins, time_frames):
    """
    CNN Model
    "Twin" Implementation: Implemented w/ dilation rate (2)
    Used as reference to twin-CRNN
    [Speaker1, Speaker2] Output

    :param bins: number of frequency bins
    :param time_frames: number of consecutive time frames
    """
    in_l = Input(shape=(bins, time_frames, 1))
    x = in_l
    filters = 8
    for i in range(2):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(1, 2))(x)
        x = Dropout(0.25)(x)
        filters *= 2

    in_r = Input(shape=(bins, time_frames, 1))
    y = in_r
    filters = 8
    for i in range(2):
        y = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=2, data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=2, data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = MaxPooling2D(pool_size=(1, 2))(y)
        y = Dropout(0.25)(y)
        filters *= 2

    y = Concatenate(axis=3)([x, y])

    filters = 64
    for i in range(1):
        y = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = MaxPooling2D(pool_size=(1, 2))(y)
        y = Dropout(0.25)(y)
        filters *= 2

    y = Flatten()(y)
    y = Dense(2048, activation=LeakyReLU())(y)  # 2048
    y = BatchNormalization()(y)
    y = Dropout(0.4)(y)

    out_l = Dense(bins * time_frames, activation='linear')(y)
    out_l = Reshape((bins, time_frames, 1))(out_l)
    out_r = Dense(bins * time_frames, activation='linear')(y)
    out_r = Reshape((bins, time_frames, 1))(out_r)

    model = Model(inputs=[in_l, in_r], outputs=[out_l, out_r])
    model.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

    return model


def crnn_mimo(bins, time_frames):
    """
    Twin CRNN (branch w/ dilation_rate = 2)
    Multiple, yet common, Input
    Multiple Outputs ([Speaker1, Speaker2])

    :param bins: Num. of freq. bins
    :param time_frames: Num. of consecutive Time Frames
    """
    in_l = Input(shape=(bins, time_frames, 1))
    x = in_l
    filters = 64
    for i in range(2):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=filters // 4, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 1))(x)
        x = Dropout(0.25)(x)
        filters //= 2

    in_r = Input(shape=(bins, time_frames, 1))
    y = in_r
    filters = 64
    for i in range(2):
        y = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=2, data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv2D(filters=filters // 4, kernel_size=(3, 3), padding='same', dilation_rate=2,
                   data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = MaxPooling2D(pool_size=(2, 1))(y)
        y = Dropout(0.25)(y)
        filters //= 2

    y = Concatenate(axis=3)([x, y])

    filters = 32
    for i in range(1):
        y = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv2D(filters=filters // 2, kernel_size=(3, 3), padding='same', data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = MaxPooling2D(pool_size=(2, 1))(y)
        y = Dropout(0.3)(y)
        filters *= 2

    y = Reshape((time_frames, 512))(y)  # hardcoded, ((time_frames, bins/8*last_conv_filter)
    y = LSTM(bins, return_sequences=True)(y)
    y = Dropout(0.3)(y)
    y = Flatten()(y)

    out_l = Dense(bins * time_frames, activation='linear')(y)
    out_l = Reshape((bins, time_frames, 1))(out_l)
    out_r = Dense(bins * time_frames, activation='linear')(y)
    out_r = Reshape((bins, time_frames, 1))(out_r)

    model = Model([in_l, in_r], [out_l, out_r])
    model.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

    return model


def crnn_mimo_PIT(bins, time_frames):
    """
    Twin-CRNN
    MIMO
    Designed for Permutation Invariant Training
    Makes use of custom_mse()

    :param bins: Num of freq. bins
    :param time_frames: Num of consecutive time frames
    """
    in_l = Input(shape=(bins, time_frames, 1))
    x = in_l
    filters = 64
    for i in range(2):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=filters // 4, kernel_size=(3, 3), padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 1))(x)
        x = Dropout(0.25)(x)
        filters //= 2

    in_r = Input(shape=(bins, time_frames, 1))
    y = in_r
    filters = 64
    for i in range(2):
        y = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=2, data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv2D(filters=filters // 4, kernel_size=(3, 3), padding='same', dilation_rate=2,
                   data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = MaxPooling2D(pool_size=(2, 1))(y)
        y = Dropout(0.25)(y)
        filters //= 2

    y = Concatenate(axis=3)([x, y])

    filters = 32
    for i in range(1):
        y = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv2D(filters=filters // 2, kernel_size=(3, 3), padding='same', data_format='channels_last')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = MaxPooling2D(pool_size=(2, 1))(y)
        y = Dropout(0.3)(y)
        filters *= 2

    y = Reshape((time_frames, 512))(y)  # hardcoded, ((time_frames, bins/8*last_conv_filter)
    y = LSTM(bins, return_sequences=True)(y)
    y = Dropout(0.3)(y)
    y = Flatten()(y)

    out_l = Dense(bins * time_frames, activation='linear')(y)
    out_l = Reshape((bins, time_frames, 1), name='out_l')(out_l)
    out_r = Dense(bins * time_frames, activation='linear')(y)
    out_r = Reshape((bins, time_frames, 1), name='out_r')(out_r)

    model = Model(inputs=[in_l, in_r], outputs=[out_l, out_r])

    losses = {'out_l': custom_mse(0), 'out_r': custom_mse(1)}

    model.compile(loss=losses, optimizer='adam')

    return model
