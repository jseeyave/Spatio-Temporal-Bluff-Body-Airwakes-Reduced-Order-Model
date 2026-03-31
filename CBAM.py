import tensorflow as tf
from tensorflow.keras.layers import Conv3D, GlobalAveragePooling3D, GlobalMaxPooling3D, Dense, Reshape, Multiply, Concatenate, Add, Activation

class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # For Channel Attention
        channel_dim = input_shape[-1]
        self.shared_mlp = tf.keras.Sequential([
            Dense(channel_dim // self.reduction_ratio, activation='relu', use_bias=True),
            Dense(channel_dim, activation=None, use_bias=True)
        ])
        # For Spatial Attention
        self.spatial_conv = Conv3D(filters=1, kernel_size=self.kernel_size, strides=1, padding='same', activation='sigmoid')

    def call(self, inputs, **kwargs):
        # Channel Attention Module
        avg_pool = GlobalAveragePooling3D()(inputs)
        max_pool = GlobalMaxPooling3D()(inputs)
        avg_pool = self.shared_mlp(Reshape((1, 1, 1, -1))(avg_pool))
        max_pool = self.shared_mlp(Reshape((1, 1, 1, -1))(max_pool))
        channel_attention = Add()([avg_pool, max_pool])
        channel_attention = Activation('sigmoid')(channel_attention)
        refined_features = Multiply()([inputs, channel_attention])

        # Spatial Attention Module
        avg_pool = tf.reduce_mean(refined_features, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(refined_features, axis=-1, keepdims=True)
        spatial_attention = Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = self.spatial_conv(spatial_attention)
        refined_features = Multiply()([refined_features, spatial_attention])

        return refined_features
