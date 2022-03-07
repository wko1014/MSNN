import tensorflow as tf, numpy as np

tf.keras.backend.set_floatx("float64")

class MSNN(tf.keras.Model):
    tf.keras.backend.set_floatx("float32")
    def __init__(self):
        super(MSNN, self).__init__()
        self.n_channels = 20 # number of electrodes
        self.f_freq = 100 # sampling frequency

        # Regularizer
        self.regularizer = tf.keras.regularizers.L1L2(l1=.001, l2=.01)

        # Activation functions
        self.activation = tf.keras.layers.LeakyReLU()
        self.softmax = tf.keras.layers.Softmax()

        # Spectral convolution
        self.conv0 = tf.keras.layers.Conv2D(4, (1, int(self.f_freq/2)), kernel_regularizer=self.regularizer)
        # Spatio-temporal convolution
        self.conv1t = tf.keras.layers.SeparableConv2D(16, (1, 25), padding="same",
                                                     depthwise_regularizer=self.regularizer,
                                                     pointwise_regularizer=self.regularizer)
        self.conv1s = tf.keras.layers.Conv2D(16, (self.n_channels, 1), kernel_regularizer=self.regularizer)

        self.conv2t = tf.keras.layers.SeparableConv2D(32, (1, 15), padding="same",
                                                     depthwise_regularizer=self.regularizer,
                                                     pointwise_regularizer=self.regularizer)
        self.conv2s = tf.keras.layers.Conv2D(32, (self.n_channels, 1), kernel_regularizer=self.regularizer)

        self.conv3t = tf.keras.layers.SeparableConv2D(64, (1, 6), padding="same",
                                                     depthwise_regularizer=self.regularizer,
                                                     pointwise_regularizer=self.regularizer)
        self.conv3s = tf.keras.layers.Conv2D(64, (self.n_channels, 1), kernel_regularizer=self.regularizer)


        # Flatteninig
        self.flatten = tf.keras.layers.Flatten()

        # Dropout
        self.dropout = tf.keras.layers.Dropout(0.5)

        # Decision making
        self.dense = tf.keras.layers.Dense(2, activation=None, kernel_regularizer=self.regularizer)

    def embedding(self, x, random_mask=False):
        x = self.activation(self.conv0(x))

        x = self.activation(self.conv1t(x))
        f1 = self.activation(self.conv1s(x))

        x = self.activation(self.conv2t(x))
        f2 = self.activation(self.conv2s(x))

        x = self.activation(self.conv3t(x))
        f3 = self.activation(self.conv3s(x))

        # multi-scale feature representation by exploiting intermediate features
        feature = tf.concat((f1, f2, f3), -1)

        return feature

    def classifier(self, feature):
        # Flattening, dropout, mapping into the decision nodes
        feature = self.flatten(feature)
        feature = self.dropout(feature)
        y_hat = self.softmax(self.dense(feature))
        return y_hat

    def GAP(self, feature):
        return tf.reduce_mean(feature, -2)

    def call(self, x):
        # Extract feature using MSNN encoder
        feature = self.embedding(x)

        # Global Average Pooling (MSNN)
        feature = self.GAP(feature)

        # Decision making
        y_hat = self.classifier(feature)
        return y_hat
