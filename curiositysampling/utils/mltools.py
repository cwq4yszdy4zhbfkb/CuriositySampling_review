import math

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from diskcache import Cache, Deque
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend, constraints, initializers, regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow_addons.activations import snake


def ftswishplus(x, T=-0.25, mean_shift=-0.1):
    # https://arxiv.org/abs/1812.06247
    x = tf.keras.activations.relu(x) * tf.sigmoid(x) + T
    x -= mean_shift
    return x


class Psnake(Layer):
    """Parametric Rectified Linear Unit.
    It follows:
    ```
      f(x) = alpha * x for x < 0
      f(x) = x for x >= 0
    ```
    where `alpha` is a learned array with the same shape as x.
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as the input.
    Args:
      alpha_initializer: Initializer function for the weights.
      alpha_regularizer: Regularizer for the weights.
      alpha_constraint: Constraint for the weights.
      shared_axes: The axes along which to share learnable
        parameters for the activation function.
        For example, if the incoming feature maps
        are from a 2D convolution
        with output shape `(batch, height, width, channels)`,
        and you wish to share parameters across space
        so that each filter only has one set of parameters,
        set `shared_axes=[1, 2]`.
    """

    def __init__(
        self,
        alpha_initializer="ones",
        alpha_regularizer=None,
        alpha_constraint=None,
        shared_axes=None,
        **kwargs
    ):
        super(Psnake, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        self.alpha = self.add_weight(
            shape=param_shape,
            name="alpha",
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
        )
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        up = inputs + (1 - tf.math.cos(2 * self.alpha * inputs))
        down = 2 * self.alpha
        return up / down

    def get_config(self):
        config = {
            "alpha_initializer": initializers.serialize(self.alpha_initializer),
            "alpha_regularizer": regularizers.serialize(self.alpha_regularizer),
            "alpha_constraint": constraints.serialize(self.alpha_constraint),
            "shared_axes": self.shared_axes,
        }
        base_config = super(Psnake, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


def reinitialize(model, dont_reset_cnn=False):
    for l in model.layers:
        # if we dont reset all kinds of CNNs, look for strides
        if hasattr(l, "strides") and dont_reset_cnn:
            continue
        if hasattr(l, "kernel_initializer") and hasattr(l, "kernel"):
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l, "depthwise_initializer") and hasattr(l, "depthwise_kernel"):
            l.depthwise_kernel.assign(
                l.depthwise_initializer(tf.shape(l.depthwise_kernel))
            )

        if hasattr(l, "kernel_initializer") and hasattr(l, "pointwise_kernel"):
            l.depthwise_kernel.assign(
                l.depthwise_initializer(tf.shape(l.depthwise_kernel))
            )

        if hasattr(l, "bias_initializer"):
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        if hasattr(l, "recurrent_initializer"):
            l.recurrent_kernel.assign(
                l.recurrent_initializer(tf.shape(l.recurrent_kernel))
            )


class UAF(Layer):
    """Parametric Rectified Linear Unit.
    It follows:
    ```
      f(x) = alpha * x for x < 0
      f(x) = x for x >= 0
    ```
    where `alpha` is a learned array with the same shape as x.
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as the input.
    Args:
      alpha_initializer: Initializer function for the weights.
      alpha_regularizer: Regularizer for the weights.
      alpha_constraint: Constraint for the weights.
      shared_axes: The axes along which to share learnable
        parameters for the activation function.
        For example, if the incoming feature maps
        are from a 2D convolution
        with output shape `(batch, height, width, channels)`,
        and you wish to share parameters across space
        so that each filter only has one set of parameters,
        set `shared_axes=[1, 2]`.
    """

    def __init__(
        self,
        alpha_initializer="ones",
        alpha_regularizer=None,
        alpha_constraint=None,
        shared_axes=None,
        **kwargs
    ):
        super(UAF, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        self.a = self.add_weight(
            shape=param_shape,
            name="a",
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
        )

        self.b = self.add_weight(
            shape=param_shape,
            name="b",
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
        )

        self.c = self.add_weight(
            shape=param_shape,
            name="c",
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
        )

        self.d = self.add_weight(
            shape=param_shape,
            name="d",
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
        )

        self.e = self.add_weight(
            shape=param_shape,
            name="e",
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
        )

        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        x = inputs
        left = tf.math.log(1 + tf.exp(self.a * (x + self.b)) + self.c * x**2)
        right = tf.math.log(1 + tf.exp(self.d * (x - self.b))) + self.e
        return left - right

    def get_config(self):
        config = {
            "alpha_initializer": initializers.serialize(self.alpha_initializer),
            "alpha_regularizer": regularizers.serialize(self.alpha_regularizer),
            "alpha_constraint": constraints.serialize(self.alpha_constraint),
            "shared_axes": self.shared_axes,
        }
        base_config = super(UAF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class AutoClipper:
    """
    Source: https://github.com/pseeth/autoclip/blob/master/autoclip_tf.py
    """

    def __init__(self, clip_percentile, history_size=10000):
        self.clip_percentile = clip_percentile
        self.grad_history = tf.Variable(tf.zeros(history_size), trainable=False)
        self.i = tf.Variable(0, trainable=False)
        self.history_size = history_size

    def __call__(self, grads):
        grad_norms = [self._get_grad_norm(g) for g in grads]
        total_norm = tf.norm(grad_norms)
        assign_idx = tf.math.mod(self.i, self.history_size)
        self.grad_history = self.grad_history[assign_idx].assign(total_norm)
        self.i = self.i.assign_add(1)
        clip_value = tfp.stats.percentile(
            self.grad_history[: self.i], q=self.clip_percentile
        )
        return [tf.clip_by_norm(g, clip_value) for g in grads]

    def _get_grad_norm(self, t, axes=None, name=None):
        values = tf.convert_to_tensor(
            t.values if isinstance(t, tf.IndexedSlices) else t, name="t"
        )

        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        l2sum = tf.math.reduce_sum(values * values, axes, keepdims=True)
        pred = l2sum > 0
        # Two-tap tf.where trick to bypass NaN gradients
        l2sum_safe = tf.where(pred, l2sum, tf.ones_like(l2sum))
        return tf.squeeze(tf.where(pred, tf.math.sqrt(l2sum_safe), l2sum))


class Dequemax(Deque):
    def __init__(self, iterable=(), directory=None, maxlen=None, pickle_protocol=0):
        """Initialize deque instance.
        If directory is None then temporary directory created. The directory
        :param iterable: iterable of items to append to deque
        :param directory: deque directory (default None)
        """
        self._cache = Cache(
            directory, eviction_policy="none", pickle_protocol=pickle_protocol
        )
        self._maxlen = float("inf") if maxlen is None else maxlen
        self.extend(iterable)

    def append(self, value):
        """Add `value` to back of deque.
        >>> deque = Deque()
        >>> deque.append('a')
        >>> deque.append('b')
        >>> deque.append('c')
        >>> list(deque)
        ['a', 'b', 'c']
        :param value: value to add to back of deque
        """
        with self._cache.transact(retry=True):
            self._cache.push(value, retry=True)
            if len(self._cache) > self._maxlen:
                self.popleft()

        def appendleft(self, value):
            """Add `value` to front of deque.
            :param value: value to add to front of deque
            :param value: value to add to front of deque

            """
            self._cache.push(value, side="front", retry=True)
            with self._cache.transact(retry=True):
                self._cache.push(value, side="front", retry=True)
                if len(self._cache) > self._maxlen:
                    self.pop()

    @classmethod
    def fromcache(cls, cache, iterable=(), maxlen=None):
        """Initialize deque using `cache`.
        :param Cache cache: cache to use
        :param iterable: iterable of items
        :return: initialized Deque
        """
        # pylint: disable=no-member,protected-access
        self = cls.__new__(cls)
        self._cache = cache
        self._maxlen = float("inf") if maxlen is None else maxlen
        self.extend(iterable)
        return self


def m_inv(M, return_sqrt=False, eps=1e-6, tf_based=True, regularize=True):
    if regularize:
        I = tf.eye(tf.shape(M)[0], tf.shape(M)[1], dtype=M.dtype)
        M = M + I * eps

    if not tf_based:
        # regularize

        # expand to batch
        M = tf.expand_dims(M, axis=0)
        # eigval, eigvec = tf.linalg.eigh(M)
        S, U, V = tf.linalg.svd(M)
        # collapse batch dim
        S = tf.squeeze(S, axis=0)
        U = tf.squeeze(U, axis=0)
        V = tf.squeeze(V, axis=0)
        # transpose as in deeptime _vampnet / doesnt work
        # eigvec = tf.transpose(eigvec, perm=[0, 1])
        if return_sqrt:
            Si = tf.math.sqrt(1 / S)
        else:
            Si = 1 / S

        Mi = tf.matmul(V * Si, U, transpose_b=True)
    else:
        Mi = tf.linalg.inv(M)
        if return_sqrt:
            Mi = tf.linalg.sqrtm(Mi)

    return Mi


@tf.function
def loss_func_vamp(
    shift,
    back,
    reversible=False,
    nonrev_srv=False,
    k=2,
    vampe_score=True,
    eps=1e-6,
):
    """Calculates the VAMP-2 score with respect to the network lobes.

    Based on:
        https://github.com/markovmodel/deeptime/blob/master/vampnet/vampnet/vampnet.py
        https://github.com/hsidky/srv/blob/master/hde/hde.py
    Arguments:
        shift: tensorflow tensor, shifted by tau and truncated in the
                batch dimension.

        back: tensorlfow tensor, not shifted by tau in the batch dimension and
                truncated so by size of tau.
    Returns:
        loss_score: tensorflow tensor with shape (1, ).
    """
    N = tf.shape(shift)[0]
    # Remove the mean from the data
    ztt, zt0 = (shift, back)
    # shape (batch_size, output)
    zt0_mean = tf.reduce_mean(zt0, axis=0, keepdims=True)
    ztt_mean = tf.reduce_mean(ztt, axis=0, keepdims=True)
    # Try to keep the mean about 0
    zt0 = zt0 - zt0_mean
    # shape (batch_size, output)
    ztt = ztt - ztt_mean
    # Calculate the covariance matrices
    # shape (output, output)
    # we can't use rblw and shrinkage for non-symmetric matrix
    cov_01 = calc_cov(zt0, ztt, rblw=False, use_shrinkage=False, double=True)
    cov_00 = calc_cov(zt0, zt0, double=True)
    cov_11 = calc_cov(ztt, ztt, double=True)

    if vampe_score and not reversible:
        cov_00_inv = m_inv(cov_00, return_sqrt=True)
        cov_11_inv = m_inv(cov_11, return_sqrt=True)
        vamp_matrix = tf.matmul(tf.matmul(cov_00_inv, cov_01), cov_11_inv)

        S, Up, Vp = tf.linalg.svd(vamp_matrix)

        U = tf.matmul(cov_00_inv, Up)
        UT = tf.transpose(U)
        V = tf.matmul(cov_11_inv, Vp)
        VT = tf.transpose(V)

        S = tf.linalg.diag(S)
        S = tf.clip_by_value(S, 0.0, 1.0)
        sut = tf.matmul(S, UT)
        left_part = 2.0 * tf.matmul(tf.matmul(sut, cov_01), V)
        right_part = tf.matmul(
            tf.matmul(
                tf.matmul(tf.matmul(tf.matmul(tf.matmul(sut, cov_00), U), S), VT),
                cov_11,
            ),
            V,
        )
        return -(
            1.0 + tf.cast(tf.linalg.trace(left_part - right_part), dtype=tf.float32)
        )
    elif not reversible:
        # Calculate the inverse of the self-covariance matrices
        cov_00_inv = m_inv(cov_00, return_sqrt=True)
        cov_11_inv = m_inv(cov_11, return_sqrt=True)

        vamp_matrix = tf.matmul(tf.matmul(cov_00_inv, cov_01), cov_11_inv)
        vamp_score = tf.norm(vamp_matrix, ord=k)
        vamp_score = tf.cast(vamp_score, tf.float32)
        return -1 - tf.square(vamp_score)

    else:
        if not nonrev_srv:
            cov_10 = calc_cov(ztt, zt0, rblw=False, use_shrinkage=False, double=True)
            cov_0 = 0.5 * (cov_00 + cov_11)
            cov_1 = 0.5 * (cov_01 + cov_10)
        else:
            cov_0 = cov_00
            cov_1 = cov_01

        L = tf.linalg.cholesky(cov_0)
        Linv = m_inv(L)
        A = tf.matmul(tf.matmul(Linv, cov_1), Linv, transpose_b=True)
        if not nonrev_srv:
            lambdas, eig_v = tf.linalg.eigh(A)
        else:
            lambdas, eig_v = tf.linalg.eig(A)
            lambdas, eig_v = (tf.math.real(lambdas), tf.math.real(eig_v))

        lambdas = tf.cast(lambdas, dtype=tf.float32)
        eig_v = tf.cast(eig_v, dtype=tf.float32)

        loss = -1 - tf.reduce_sum(tf.math.abs(lambdas) ** k)

        return loss


def metric_VAMP2(shift, back):
    """Returns the sum of the squared top k eigenvalues of the vamp matrix,
    with k determined by the wrapper parameter k_eig, and the vamp matrix
    defined as:
        V = cov_00 ^ -1/2 * cov_01 * cov_11 ^ -1/2
    Can be used as a metric function in model.fit()

    Arguments:
        shift: shifted data by tau

        back: not shifted data

    Returns:
        eig_sum_sq: tensorflow float
        sum of the squared k highest eigenvalues in the vamp matrix
    """
    N = tf.shape(shift)[0]
    zt0 = back
    ztt = shift
    # shape (batch_size, output)
    zt0 = zt0 - tf.reduce_mean(zt0, axis=0, keepdims=True)
    # shape (batch_size, output)
    ztt = ztt - tf.reduce_mean(ztt, axis=0, keepdims=True)
    # Calculate the covariance matrices
    # shape (output, output)
    # we can't use rblw and shrinkage for non-symmetric matrix
    cov_01 = calc_cov(zt0, ztt, rblw=False, use_shrinkage=False, double=True)
    cov_00 = calc_cov(zt0, zt0, double=True)
    cov_11 = calc_cov(ztt, ztt, double=True)
    cov_00_inv = m_inv(cov_00, return_sqrt=True)
    cov_11_inv = m_inv(cov_11, return_sqrt=True)
    vamp_matrix = tf.matmul(tf.matmul(cov_00_inv, cov_01), cov_11_inv)

    return 1 + tf.linalg.norm(vamp_matrix)


def loss_func(
    true,
    pred,
    mask=True,
    regul_loss=None,
    cos_sim=False,
    cos_rew=False,
):
    """Loss function to calculate loss, the returned loss is of
    batch dimension shape.
    Arguments:
        true: groundtruth tensor
        pred: predicted tensor
        mask: if to mask values, that are not finite, to prevent nan losses
        regul_loss: pass your regularization loss here
    """
    loss_nm = (true - pred) ** 2
    if not cos_sim:
        loss_f = tf.reduce_mean(loss_nm)
    else:
        # see Yan, S., Shao, H., Xiao, Y., Liu, B. and Wan, J., 2023. Hybrid robust convolutional autoencoder for unsupervised anomaly detection of machine tools under noises. Robotics and Computer-Integrated Manufacturing, 79, p.102441.
        lamb = 1
        cos = (1 + tf.keras.losses.cosine_similarity(true, pred)) / 2
        loss_f = tf.reduce_mean(loss_nm) + lamb * tf.reduce_mean(cos)
        if cos_rew:
            loss_nm = loss_nm + tf.reshape(cos, shape=(-1, 1))

    if mask:
        loss_f = tf.clip_by_value(loss_f, 0, 1e4)
        loss_nm = tf.clip_by_value(loss_nm, 0, 1e16)

    if regul_loss is not None:
        loss_f += regul_loss

    return loss_f, loss_nm


def rao_blackwell_ledoit_wolf(cov, N):
    """Rao-Blackwellized Ledoit-Wolf shrinkaged estimator of the covariance
    matrix.
    Arguments:
    ----------
    S : array, shape=(n, n)
    Sample covariance matrix (e.g. estimated with np.cov(X.T))
    n : int
    Number of data points.
    Returns
    .. [1] Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. "Shrinkage
    estimation of high dimensional covariance matrices" ICASSP (2009)
    Based on: https://github.com/msmbuilder/msmbuilder/blob/master/msmbuilder/decomposition/tica.py
    """
    p = tf.cast(tf.shape(cov)[0], dtype=cov.dtype)

    alpha = (N - 2) / (N * (N + 2))
    beta = ((p + 1) * N - 2) / (N * (N + 2))

    trace_cov2 = tf.reduce_sum(cov * cov)
    U = (p * trace_cov2 / tf.linalg.trace(cov) ** 2) - 1
    rho = tf.minimum(alpha + beta / U, 1)

    F = (tf.linalg.trace(cov) / p) * tf.eye(p, dtype=cov.dtype)
    return (1 - rho) * cov + rho * F, rho


def calc_cov(
    x,
    y,
    rblw=True,
    use_shrinkage=True,
    no_normalize=False,
    double=False,
    shrinkage=0.0,
):
    if double:
        x = tf.cast(x, tf.float64)
        y = tf.cast(y, tf.float64)
        N = tf.cast(tf.shape(x)[0], tf.float64)
        feat = tf.cast(tf.shape(x)[1], tf.float64)
    else:
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        N = tf.cast(tf.shape(x)[0], tf.float32)
        feat = tf.cast(tf.shape(x)[1], tf.float32)

    if not no_normalize:
        cov = 1 / (N - 1) * tf.matmul(x, y, transpose_a=True)
    else:
        cov = tf.matmul(x, y, transpose_a=True)
    if rblw and use_shrinkage and shrinkage <= 0:
        cov_shrink, shrinkage = rao_blackwell_ledoit_wolf(cov, N)
        return cov_shrink
    elif use_shrinkage:
        shrinkage = shrinkage
        ident = tf.eye(feat, dtype=cov.dtype)
        mu = tf.linalg.trace(cov) / feat
        cov_shrink = (1 - shrinkage) * cov + shrinkage * mu * ident
        return cov_shrink
    else:
        return cov


def whiten_data(X, W=None, mean=None, eps=1e-8, pca=False):
    """Whiten the data before passing further
    arguments:
        X: input data
        W: P_matrix/W matrix
        mean: mean used for whitening
    returns:
        X_w: Whitened input data
    """
    if W is not None and mean is not None:
        X -= mean
        W = tf.convert_to_tensor(W)
    else:
        mean = tf.reduce_mean(X, axis=0)
        X -= mean
        cov = calc_cov(X, X)
        D, L = tf.linalg.eigh(cov)
    if pca:
        # save it to constnt vector
        if W is None:
            trunc_v = tf.math.reduce_min(D) * 10
            mask = tf.math.less(D, trunc_v)
            W = tf.boolean_mask(L, mask, axis=1)

        return (
            tf.matmul(X, W),
            W,
            mean,
        )
    else:
        if W is None:
            D_sqrt = tf.linalg.diag(1 / tf.math.sqrt(D + eps))
            # save it to constant vector
            W = tf.matmul(tf.matmul(L, D_sqrt), L, transpose_b=True)

        return tf.transpose(tf.matmul(W, X, transpose_b=True)), W, mean


def return_most_freq(X, bins=200):
    """Clips to the most frequent value"""
    X = np.array(X)
    hist = [np.histogram(X[:, i], bins=bins) for i in range(X.shape[-1])]
    freq, val = (np.array([o[0] for o in hist]).T, np.array([o[1] for o in hist]).T)
    most_freq = np.argmax(freq, axis=0)
    most_freq_values = np.diagonal(val[most_freq])

    return most_freq_values


def clip_to_value(X, vmin, vmax, value):
    mask_max = tf.math.less(X, vmax)
    mask_min = tf.math.greater(X, vmin)
    mask = tf.math.logical_and(mask_min, mask_max)
    value_mat = tf.zeros_like(X) + tf.reshape(value, (1, tf.shape(X)[-1]))
    X_up = tf.where(mask, X, value_mat)

    return X_up
