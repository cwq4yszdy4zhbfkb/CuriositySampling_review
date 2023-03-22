import tensorflow as tf
import tensorflow_addons as tfa

from ..core.base_loger import logger

# relative imports to avoid tf import by RAY
from ..utils.mltools import UAF, Psnake, ftswishplus


# a custom initializer, that returns uniform distributed
# weights with weights shifted from zero
# to prevent small weights
class NonZeroHeUniform(tf.keras.initializers.HeUniform):
    """ """

    def __init__(self, seed=None, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, shape, dtype=None):
        weights = super().__call__(shape, dtype)
        negative_mask = tf.cast(weights < 0, dtype)
        positive_mask = tf.cast(weights >= 0, dtype)
        # for all negative weights add negative epsilon
        # and for all positive, add positive epsilon
        weights += -self.epsilon * negative_mask + self.epsilon * positive_mask
        return weights


class Orthonormal(tf.keras.constraints.Constraint):
    """approximate Orthonormal weight constraint.
    Constrains the weights incident to each hidden unit
    to be approximately orthonormal

    # Arguments
        beta: the strength of the constraint

    # References
        https://arxiv.org/pdf/1710.04087.pdf
    """

    def __init__(self, beta=0.01):
        self.beta = beta

    def __call__(self, w):
        eye = tf.linalg.matmul(w, w, transpose_b=True)
        return (1 + self.beta) * w - self.beta * tf.linalg.matmul(eye, w)

    def get_config(self):
        return {"beta": self.beta}


# Thanks for the code
# https://gist.github.com/aeftimia/a5249168c84bc541ace2fc4e1d22a13e
class Orthogonal(tf.keras.constraints.Constraint):
    """Orthogonal weight constraint.
    Constrains the weights incident to each hidden unit
    to be orthogonal when there are more inputs than hidden units.
    When there are more hidden units than there are inputs,
    the rows of the layer's weight matrix are constrainted
    to be orthogonal.
    # Arguments
        axis: Axis or axes along which to calculate weight norms.
            `None` to use all but the last (output) axis.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
        orthonormal: If `True`, the weight matrix is further
            constrained to be orthonormal along the appropriate axis.
    """

    def __init__(self, axis=None, orthonormal=False):
        if axis is not None:
            self.axis = axis
        else:
            self.axis = None
        self.orthonormal = orthonormal

    def __call__(self, w):
        # Python block for permutating axis
        w_ndim_minus_1 = len(w.shape) - 1
        if self.axis is None:
            self.axis = tf.range(w_ndim_minus_1)
        elif isinstance(self.axis, int):
            self.axis = [self.axis]

        axis_shape = []
        for a in self.axis:
            w_shape = w.shape[a]
            axis_shape.append(w_shape)

        perm = []
        for i in range(w_ndim_minus_1):
            if i not in self.axis:
                perm.append(i)
        perm.extend(self.axis)
        perm.append(w_ndim_minus_1)

        w = tf.transpose(w, perm=perm)
        shape = w.shape
        new_shape = [-1] + axis_shape + [shape[-1]]
        w = tf.reshape(w, new_shape)
        w = tf.map_fn(self.orthogonalize, w)
        w = tf.reshape(w, shape)
        w = tf.transpose(w, perm=tf.argsort(perm))
        return w

    def orthogonalize(self, w):
        shape = w.shape
        output_shape = tf.convert_to_tensor(shape[-1], dtype=tf.int32)
        input_shape = tf.math.reduce_prod(shape[:-1])
        final_shape = tf.math.maximum(input_shape, output_shape)
        w_matrix = tf.reshape(w, (output_shape, input_shape))
        zero_int = tf.constant(0, dtype=tf.int32)
        paddings = tf.convert_to_tensor(
            [
                [zero_int, final_shape - output_shape],
                [zero_int, final_shape - input_shape],
            ]
        )
        w_matrix = tf.pad(w_matrix, paddings)
        upper_triangular = tf.linalg.band_part(w_matrix, 1, -1)
        antisymmetric = upper_triangular - tf.transpose(upper_triangular)
        rotation = tf.linalg.expm(antisymmetric)
        w_final = tf.slice(
            rotation,
            [
                0,
            ]
            * 2,
            [output_shape, input_shape],
        )
        if not self.orthonormal:
            w_final = tf.cond(
                tf.math.greater_equal(input_shape, output_shape),
                lambda: tf.linalg.matmul(
                    w_final,
                    tf.linalg.band_part(
                        tf.slice(w_matrix, [0, 0], [input_shape, input_shape]), 0, 0
                    ),
                ),
                lambda: tf.linalg.matmul(
                    tf.linalg.band_part(
                        tf.slice(w_matrix, [0, 0], [output_shape, output_shape]), 0, 0
                    ),
                    w_final,
                ),
            )
        return tf.reshape(w_final, w.shape)

    def get_config(self):
        return {"axis": self.axis, "orthonormal": self.orthonormal}


def orthogonality(w):
    """
    Penalty for deviation from orthogonality:
    Orthogonalize column vectors
    ||dot(x, x.T) - I||_1
    """
    wTw = tf.matmul(w, w, transpose_b=True)
    return tf.norm(wTw - tf.eye(*wTw.shape), ord=1)


class MemoryUnit(tf.keras.layers.Layer):
    def __init__(
        self,
        units=512,
        sparsity_mul=1,
        alpha=0.001,
        eps=1e-8,
        kernel_initializer="lacun_normal",
    ):
        super(MemoryUnit, self).__init__()
        self.units = units
        # self.norm = tf.math.sqrt(tf.cast(self.units, tf.float32))
        # Should be between 1/N and 3/N
        # where N is number of memory units
        self.sparsity_treshold = sparsity_mul / units
        self.alpha = alpha
        self.epsilon = eps
        self.kernel_initializer = kernel_initializer
        logger.warning("Using memory units")

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.M = self.add_weight(
            "kernel",
            shape=(self.units, input_shape[-1]),
            initializer=self.kernel_initializer,
            dtype="float32",
            trainable=True,
        )
        self.bias = self.add_weight(
            "bias",
            shape=(1, input_shape[-1]),
            initializer=self.kernel_initializer,
            dtype="float32",
            trainable=True,
        )
        self.norm = tf.math.sqrt(tf.cast(input_shape[-1], dtype=tf.float32))

    def calc_entropy(self, W):
        norm = tf.shape(W, out_type=tf.float32)[0]
        return -tf.reduce_sum(W * tf.math.log(W + self.epsilon)) / norm * self.alpha

    def hard_relu(self, W):
        # unstable gradients because of division
        norm = tf.math.abs(W - self.sparsity_treshold) + self.epsilon
        W = (tf.nn.relu(W - self.sparsity_treshold) * (W)) / norm
        # this still zeroes out and remains similar size
        # W = tf.nn.relu(W - self.sparsity_treshold) * (W + self.sparsity_treshold)
        # clip nans
        # value_not_nan = tf.dtypes.cast(
        #    tf.math.logical_not(tf.math.is_nan(W)), dtype=tf.float32
        # )
        # W = tf.math.multiply_no_nan(W, value_not_nan)
        # clip to 0 ... 1
        # W = tf.clip_by_value(W, 0.0, 1.0)
        return W

    def call(self, z, training=False):  # Defines the computation from inputs to outputs
        # (B, N) where N is number of memory entries
        # M is (N, C), z is (B, C)
        W = tf.matmul(z, self.M, transpose_b=True)
        # Ws is (B, N)
        # self-attention style norm
        W = W / self.norm
        # stable softmax
        # softmax over memory only
        W_max = tf.reduce_max(W, axis=1, keepdims=True)
        Ws = tf.nn.softmax(W - W_max, axis=1)

        # induce sparsity
        if self.sparsity_treshold > 0.0:
            Ws = self.hard_relu(Ws)
        if training:
            # minimize entropy
            # max entropy -> dense
            # min entropy -> sparse
            self.add_loss(self.calc_entropy(Ws))

        # normalize to keep the scale normal (prob. sums to 1)
        Ws = Ws / tf.norm(Ws, ord=1, axis=1, keepdims=True)

        # output (B, C)
        # (B, N) times (N, C)
        out = tf.matmul(Ws, self.M) + self.bias
        return out


class Target_network(tf.keras.Model):
    def __init__(
        self,
        dense_units=[64],
        dense_activ="relu",
        dense_layernorm=False,
        dense_batchnorm=False,
        input_batchnorm=False,
        dense_out=1,
        dense_out_activ="sigmoid",
        layernorm_out=False,
        batchnorm_out=False,
        initializer="he_uniform",
        spectral=False,
        orthonormal=False,
        l2_reg=0.0000,
        l1_reg=0.0000,
        l1_reg_out=0.0000,
        l2_reg_out=0.0000,
        l2_reg_activ=0.0000,
        l1_reg_activ=0.0000,
        unit_constraint=False,
        cnn=False,
        kernel_size=[3],
        strides=[2],
        padding="valid",
        noise_output=False,
        dcnn=False,
        lcnn=False,
        lcnn_first_layers=False,
        skip_after_cnn=False,
        split_slow=False,
        gaussian_dropout=0.0,
    ):
        super().__init__(self)

        if input_batchnorm:
            self.input_batchnorm_layer = tf.keras.layers.BatchNormalization()
        else:
            self.input_batchnorm_layer = tf.keras.layers.Activation("linear")
        self.split_slow = split_slow
        if skip_after_cnn:
            self.skip_con = tf.keras.layers.Dense(dense_out)
        else:
            self.skip_con = None
        if not spectral:
            self.initializer = initializer
        else:
            self.initializer = "orthogonal"
        self.dense_layers = []
        self.layernorm_layers = []
        self.activation_layers = []
        self.dropout_layers = []
        # dense layers
        if cnn:
            self.reshape_layer = tf.keras.layers.Reshape((-1, 1))
            self.gmp_layer = tf.keras.layers.GlobalMaxPool1D()
        else:
            self.reshape_layer = tf.keras.layers.Activation("linear")
            self.gmp_layer = tf.keras.layers.Activation("linear")

        for i, units_layer in enumerate(dense_units):
            if dense_activ == "prelu":
                dense_activ_lay = tf.keras.layers.PReLU()
            elif dense_activ == "lerelu":
                dense_activ_lay = tf.keras.layers.LeakyReLU(alpha=0.01)
            elif dense_activ == "snake":
                dense_activ_lay = tfa.activations.snake
            elif dense_activ == "psnake":
                dense_activ_lay = Psnake()
            elif dense_activ == "uaf":
                dense_activ_lay = UAF()
            elif dense_activ == "gelu":
                dense_activ_lay = tfa.activations.gelu
            elif dense_activ == "lisht":
                dense_activ_lay = tfa.activations.lisht
            elif dense_activ == "mish":
                dense_activ_lay = tfa.activations.mish
            elif dense_activ == "ftswishplus":
                dense_activ_lay = ftswishplus

            else:
                dense_activ_lay = tf.keras.layers.Activation(dense_activ)

            if unit_constraint:
                constraint = tf.keras.constraints.UnitNorm()
            else:
                constraint = None
            if not cnn:
                self.dense_layers.append(
                    tf.keras.layers.Dense(
                        units_layer,
                        kernel_initializer=self.initializer,
                        kernel_regularizer=tf.keras.regularizers.L1L2(
                            l1=l1_reg, l2=l2_reg
                        ),
                        activity_regularizer=tf.keras.regularizers.L1L2(
                            l1=l1_reg_activ, l2=l2_reg_activ
                        ),
                        kernel_constraint=constraint,
                    )
                )
            else:
                if dcnn:
                    self.dense_layers.append(
                        tf.keras.layers.SeparableConv1D(
                            units_layer,
                            kernel_size[i],
                            strides=strides[i],
                            depthwise_initializer=self.initializer,
                            depthwise_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg, l2=l2_reg
                            ),
                            activity_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg_activ, l2=l2_reg_activ
                            ),
                            depthwise_constraint=constraint,
                            padding=padding,
                        )
                    )
                elif lcnn:
                    self.dense_layers.append(
                        tf.keras.layers.LocallyConnected1D(
                            units_layer,
                            kernel_size[i],
                            strides=strides[i],
                            kernel_initializer=self.initializer,
                            kernel_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg, l2=l2_reg
                            ),
                            activity_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg_activ, l2=l2_reg_activ
                            ),
                            kernel_constraint=constraint,
                            padding=padding,
                        )
                    )
                    if lcnn_first_layers:
                        lcnn = False

                else:
                    self.dense_layers.append(
                        tf.keras.layers.Conv1D(
                            units_layer,
                            kernel_size[i],
                            strides=strides[i],
                            kernel_initializer=self.initializer,
                            kernel_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg, l2=l2_reg
                            ),
                            activity_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg_activ, l2=l2_reg_activ
                            ),
                            kernel_constraint=constraint,
                            padding=padding,
                        )
                    )

            self.activation_layers.append(dense_activ_lay)

            if dense_layernorm:
                self.layernorm_layers.append(tf.keras.layers.LayerNormalization())
            elif dense_batchnorm:
                self.layernorm_layers.append(tf.keras.layers.BatchNormalization())
            else:
                self.layernorm_layers.append(tf.keras.layers.Activation("linear"))
            self.dropout_layers.append(
                tf.keras.layers.GaussianDropout(gaussian_dropout)
            )

        # Add additional dense at the end when architecture is CNN
        self.dense_preout = []
        self.activation_preout = []
        self.layernorm_preout = []
        self.dropout_preout = []
        if cnn:
            for i in range(3):
                if dense_activ == "prelu":
                    dense_activ_lay = tf.keras.layers.PReLU()
                elif dense_activ == "lerelu":
                    dense_activ_lay = tf.keras.layers.LeakyReLU(alpha=0.01)
                elif dense_activ == "snake":
                    dense_activ_lay = tfa.activations.snake
                elif dense_activ == "psnake":
                    dense_activ_lay = Psnake()
                elif dense_activ == "uaf":
                    dense_activ_lay = UAF()
                elif dense_activ == "gelu":
                    dense_activ_lay = tfa.activations.gelu
                elif dense_activ == "lisht":
                    dense_activ_lay = tfa.activations.lisht
                elif dense_activ == "mish":
                    dense_activ_lay = tfa.activations.mish
                elif dense_activ == "ftswishplus":
                    dense_activ_lay = ftswishplus
                else:
                    dense_activ_lay = tf.keras.layers.Activation(dense_activ)

                self.activation_preout.append(dense_activ_lay)

                if dense_layernorm:
                    self.layernorm_preout.append(tf.keras.layers.LayerNormalization())
                elif dense_batchnorm:
                    self.layernorm_preout.append(tf.keras.layers.BatchNormalization())
                else:
                    self.layernorm_preout.append(tf.keras.layers.Activation("linear"))

                neurons = int(dense_out * 2**4 // (2 ** (i + 1)))

                self.dense_preout.append(
                    tf.keras.layers.Dense(
                        neurons,
                        kernel_initializer=self.initializer,
                        kernel_regularizer=tf.keras.regularizers.L1L2(
                            l1=l1_reg, l2=l2_reg
                        ),
                        activity_regularizer=tf.keras.regularizers.L1L2(
                            l1=l1_reg_activ, l2=l2_reg_activ
                        ),
                    )
                )

        if orthonormal:
            self.dense_out = tf.keras.layers.Dense(
                dense_out,
                activation=dense_out_activ,
                kernel_initializer="orthogonal",
                kernel_constraint=Orthonormal(beta=0.1),
                use_bias=True,
            )
        else:
            if unit_constraint:
                constraint = tf.keras.constraints.UnitNorm()
            else:
                constraint = None
            if self.split_slow:
                activ_out = tfa.activations.mish
            else:
                activ_out = dense_out_activ
            self.dense_out = tf.keras.layers.Dense(
                dense_out,
                activation=activ_out,
                kernel_initializer=self.initializer,
                kernel_regularizer=tf.keras.regularizers.L1L2(
                    l1=l1_reg_out, l2=l2_reg_out
                ),
                activity_regularizer=tf.keras.regularizers.L1L2(
                    l1=l1_reg_activ, l2=l2_reg_activ
                ),
                use_bias=True,
                kernel_constraint=constraint,
            )
        if self.split_slow:
            self.dense_out_t0 = tf.keras.layers.Dense(
                dense_out,
                activation=dense_out_activ,
                kernel_initializer=self.initializer,
                kernel_regularizer=tf.keras.regularizers.L1L2(
                    l1=l1_reg_out, l2=l2_reg_out
                ),
                activity_regularizer=tf.keras.regularizers.L1L2(
                    l1=l1_reg_activ, l2=l2_reg_activ
                ),
                use_bias=True,
                kernel_constraint=constraint,
            )

            self.dense_out_tt = tf.keras.layers.Dense(
                dense_out,
                activation=dense_out_activ,
                kernel_initializer=self.initializer,
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                activity_regularizer=tf.keras.regularizers.L1L2(
                    l1=l1_reg_activ, l2=l2_reg_activ
                ),
                use_bias=True,
                kernel_constraint=constraint,
            )

        if noise_output:
            self.noise_out = tf.keras.layers.GaussianNoise(1)
        else:
            self.noise_out = tf.keras.layers.Activation("linear")

        if layernorm_out:
            self.layernorm_out = tf.keras.layers.LayerNormalization()
        if batchnorm_out:
            self.layernorm_out = tf.keras.layers.BatchNormalization()
        else:
            self.layernorm_out = tf.keras.layers.Activation("linear")
        self.dropout_preout.append(tf.keras.layers.GaussianDropout(gaussian_dropout))
        assert len(self.dense_layers) != 0
        assert len(self.layernorm_layers) != 0
        assert len(self.activation_layers) != 0
        assert len(self.activation_layers) == len(self.dense_layers)
        assert len(self.layernorm_layers) == len(self.dense_layers)

    def call(self, inputs, training=False):
        x = inputs
        x = self.reshape_layer(x)
        x = self.input_batchnorm_layer(x, training=training)

        for dense_layer, layernorm_layer, activation_layer, drop_layer in zip(
            self.dense_layers,
            self.layernorm_layers,
            self.activation_layers,
            self.dropout_layers,
        ):
            x = dense_layer(x, training=training)
            x = activation_layer(x)
            x = layernorm_layer(x, training=training)
            x = drop_layer(x, training=training)

        # replaced from flatten, better performance, less cost
        x = self.gmp_layer(x)

        if self.skip_con is not None:
            x_skip = self.skip_con(x)

        for dense_layer, layernorm, activation_layer, drop_layer in zip(
            self.dense_preout,
            self.layernorm_preout,
            self.activation_preout,
            self.dropout_preout,
        ):
            x = dense_layer(x, training=training)
            x = activation_layer(x)
            x = layernorm(x, training=training)
            x = drop_layer(x, training=training)
        x = self.dense_out(x, training=training)

        if self.skip_con is not None:
            x = x + x_skip

        x = self.noise_out(x, training=training)
        x = self.layernorm_out(x, training=training)
        if self.split_slow:
            x_t0 = self.dense_out_t0(x)
            x_tt = self.dense_out_tt(x)
        else:
            x_t0 = x
            x_tt = x
        return x_t0, x_tt

    def build(self, x, new_reshape_size=None):
        if new_reshape_size is not None:
            self.reshape_layer = tf.keras.layers.Reshape((-1, new_reshape_size))
            _ = self.reshape_layer(x)


class Predictor_network(tf.keras.Model):
    def __init__(
        self,
        dense_units=[64],
        dense_activ="relu",
        dense_layernorm=False,
        dense_batchnorm=False,
        dense_out=1,
        dense_out_activ="sigmoid",
        layernorm_out=False,
        batchnorm_out=False,
        initializer="he_uniform",
        spectral=False,
        orthonormal=False,
        l1_reg=0.0000,
        l2_reg=0.0000,
        l1_reg_out=0.0000,
        l2_reg_out=0.0000,
        l1_reg_activ=0.0000,
        l2_reg_activ=0.0000,
        unit_constraint=False,
        cnn=False,
        kernel_size=[3],
        strides=[2],
        padding="valid",
        noise_output=False,
        dcnn=False,
        lcnn=False,
        lcnn_first_layers=False,
        skip_after_cnn=False,
        input_batchnorm=False,
        memory_units=False,
    ):
        super().__init__(self)

        if input_batchnorm:
            self.input_batchnorm_layer = tf.keras.layers.BatchNormalization()
        else:
            self.input_batchnorm_layer = tf.keras.layers.Activation("linear")

        if skip_after_cnn:
            self.skip_con_mean = tf.keras.layers.Dense(dense_out)
            self.skip_con_scale = tf.keras.layers.Dense(dense_out)
        else:
            self.skip_con_mean = None

        if not spectral:
            self.initializer = initializer
        else:
            self.initializer = "orthogonal"
        self.dense_layers = []
        self.layernorm_layers = []
        self.activation_layers = []
        # dense layers
        if cnn:
            self.reshape_layer = tf.keras.layers.Reshape((-1, 1))
            self.flatten_layer = tf.keras.layers.Flatten()
        else:
            self.reshape_layer = tf.keras.layers.Activation("linear")
            self.flatten_layer = tf.keras.layers.Activation("linear")

        for i, units_layer in enumerate(dense_units):
            if dense_activ == "prelu":
                dense_activ_lay = tf.keras.layers.PReLU()
            elif dense_activ == "lerelu":
                dense_activ_lay = tf.keras.layers.LeakyReLU(alpha=0.01)
            elif dense_activ == "snake":
                dense_activ_lay = tfa.activations.snake
            elif dense_activ == "psnake":
                dense_activ_lay = Psnake()
            elif dense_activ == "uaf":
                dense_activ_lay = UAF()
            elif dense_activ == "gelu":
                dense_activ_lay = tfa.activations.gelu
            elif dense_activ == "lisht":
                dense_activ_lay = tfa.activations.lisht
            elif dense_activ == "mish":
                dense_activ_lay = tfa.activations.mish
            elif dense_activ == "ftswishplus":
                dense_activ_lay = ftswishplus

            else:
                dense_activ_lay = tf.keras.layers.Activation(dense_activ)

            if unit_constraint:
                constraint = tf.keras.constraints.UnitNorm()
            else:
                constraint = None
            if not cnn:
                self.dense_layers.append(
                    tf.keras.layers.Dense(
                        units_layer,
                        kernel_initializer=self.initializer,
                        kernel_regularizer=tf.keras.regularizers.L1L2(
                            l1=l1_reg, l2=l2_reg
                        ),
                        activity_regularizer=tf.keras.regularizers.L1L2(
                            l1=l1_reg_activ, l2=l2_reg_activ
                        ),
                        kernel_constraint=constraint,
                    )
                )
            else:
                if dcnn:
                    self.dense_layers.append(
                        tf.keras.layers.SeparableConv1D(
                            units_layer,
                            kernel_size[i],
                            strides=strides[i],
                            depthwise_initializer=self.initializer,
                            depthwise_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg, l2=l2_reg
                            ),
                            activity_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg_activ, l2=l2_reg_activ
                            ),
                            depthwise_constraint=constraint,
                            padding=padding,
                        )
                    )
                elif lcnn:
                    self.dense_layers.append(
                        tf.keras.layers.LocallyConnected1D(
                            units_layer,
                            kernel_size[i],
                            strides=strides[i],
                            kernel_initializer=self.initializer,
                            kernel_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg, l2=l2_reg
                            ),
                            activity_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg_activ, l2=l2_reg_activ
                            ),
                            kernel_constraint=constraint,
                            padding=padding,
                        )
                    )
                    if lcnn_first_layers:
                        lcnn = False

                else:
                    self.dense_layers.append(
                        tf.keras.layers.Conv1D(
                            units_layer,
                            kernel_size[i],
                            strides=strides[i],
                            kernel_initializer=self.initializer,
                            kernel_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg, l2=l2_reg
                            ),
                            activity_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg_activ, l2=l2_reg_activ
                            ),
                            kernel_constraint=constraint,
                            padding=padding,
                        )
                    )

            self.activation_layers.append(dense_activ_lay)
            if dense_layernorm:
                self.layernorm_layers.append(tf.keras.layers.LayerNormalization())
            elif dense_batchnorm:
                self.layernorm_layers.append(tf.keras.layers.BatchNormalization())
            else:
                self.layernorm_layers.append(tf.keras.layers.Activation("linear"))

        # Add additional dense at the end when architecture is CNN
        self.dense_preout = []
        self.activation_preout = []
        self.layernorm_preout = []
        if cnn:
            if memory_units:
                iters = 2
            else:
                iters = 3
            for i in range(iters):
                if dense_activ == "prelu":
                    dense_activ_lay = tf.keras.layers.PReLU()
                elif dense_activ == "lerelu":
                    dense_activ_lay = tf.keras.layers.LeakyReLU(alpha=0.01)
                elif dense_activ == "snake":
                    dense_activ_lay = tfa.activations.snake
                elif dense_activ == "psnake":
                    dense_activ_lay = Psnake()
                elif dense_activ == "uaf":
                    dense_activ_lay = UAF()
                elif dense_activ == "gelu":
                    dense_activ_lay = tfa.activations.gelu
                elif dense_activ == "lisht":
                    dense_activ_lay = tfa.activations.lisht
                elif dense_activ == "mish":
                    dense_activ_lay = tfa.activations.mish
                elif dense_activ == "ftswishplus":
                    dense_activ_lay = ftswishplus
                else:
                    dense_activ_lay = tf.keras.layers.Activation(dense_activ)

                self.activation_preout.append(dense_activ_lay)

                if dense_layernorm:
                    self.layernorm_preout.append(tf.keras.layers.LayerNormalization())
                elif dense_batchnorm:
                    self.layernorm_preout.append(tf.keras.layers.BatchNormalization())
                else:
                    self.layernorm_preout.append(tf.keras.layers.Activation("linear"))

                if memory_units:
                    neurons = dense_out
                elif dense_out * 100 < 512:
                    neurons = 512
                else:
                    neurons = dense_out * 100
                if (not memory_units) or i < 1:
                    self.dense_preout.append(
                        tf.keras.layers.Dense(
                            neurons,
                            kernel_initializer=self.initializer,
                            kernel_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg, l2=l2_reg
                            ),
                            activity_regularizer=tf.keras.regularizers.L1L2(
                                l1=l1_reg_activ, l2=l2_reg_activ
                            ),
                        )
                    )
                else:
                    neurons = 2000
                    self.dense_preout.append(
                        MemoryUnit(neurons, kernel_initializer=self.initializer),
                    )
                    self.activation_preout[-1] = tf.keras.layers.Activation("linear")
                    self.layernorm_preout[-1] = tf.keras.layers.Activation("linear")

        if memory_units:
            self.dense_out = tf.keras.layers.Activation("linear")
        elif orthonormal:
            self.dense_out = tf.keras.layers.Dense(
                dense_out,
                activation=dense_out_activ,
                kernel_initializer="orthogonal",
                kernel_constraint=Orthonormal(beta=0.1),
                use_bias=True,
            )
        else:
            if unit_constraint:
                constraint = tf.keras.constraints.UnitNorm()
            else:
                constraint = None

            self.dense_out = tf.keras.layers.Dense(
                dense_out,
                activation=dense_out_activ,
                kernel_initializer=self.initializer,
                kernel_regularizer=tf.keras.regularizers.L1L2(
                    l1=l1_reg_out, l2=l2_reg_out
                ),
                activity_regularizer=tf.keras.regularizers.L1L2(
                    l1=l1_reg_activ, l2=l2_reg_activ
                ),
                use_bias=True,
                kernel_constraint=constraint,
            )

        if noise_output:
            self.noise_out = tf.keras.layers.GaussianNoise(1)
        else:
            self.noise_out = tf.keras.layers.Activation("linear")

        if layernorm_out:
            self.layernorm_out = tf.keras.layers.LayerNormalization()
        if batchnorm_out:
            self.layernorm_out = tf.keras.layers.BatchNormalization()
        else:
            self.layernorm_out = tf.keras.layers.Activation("linear")
        assert len(self.dense_layers) != 0
        assert len(self.layernorm_layers) != 0
        assert len(self.activation_layers) != 0
        assert len(self.activation_layers) == len(self.dense_layers)
        assert len(self.layernorm_layers) == len(self.dense_layers)

    def call(self, inputs, training=False):
        x = inputs
        x = self.reshape_layer(x)
        x = self.input_batchnorm_layer(x)
        for dense_layer, layernorm_layer, activation_layer in zip(
            self.dense_layers, self.layernorm_layers, self.activation_layers
        ):
            x = dense_layer(x, training=training)
            x = activation_layer(x)
            x = layernorm_layer(x, training=training)

        # gmp makes training far worse
        x = self.flatten_layer(x)
        if self.skip_con_mean is not None:
            x_skip_mean = self.skip_con_mean(x)
            x_skip_scale = self.skip_con_scale(x)

        for dense_layer, layernorm, activation_layer in zip(
            self.dense_preout, self.layernorm_preout, self.activation_preout
        ):
            x = dense_layer(x, training=training)
            x = activation_layer(x)
            x = layernorm(x, training=training)

        x = self.dense_out(x, training=training)
        if self.skip_con_mean is not None:
            x = (x + x_skip_mean) * x_skip_scale
        x = self.noise_out(x, training=training)
        x = self.layernorm_out(x, training=training)
        return x

    def build(self, x, new_reshape_size=None):
        if new_reshape_size is not None:
            self.reshape_layer = tf.keras.layers.Reshape((-1, new_reshape_size))
            _ = self.reshape_layer(x)
