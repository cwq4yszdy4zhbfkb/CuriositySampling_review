import gc
import inspect
import os
import random
from collections import deque
from copy import deepcopy
from datetime import datetime
from time import time

import numpy as np
import openmm as omm
import ray
import scipy
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from curiositysampling.utils.checkpointutils import (
    append_json_object,
    save_json_object,
    save_pickle_object,
)
from scipy import interpolate
from sklearn.preprocessing import PowerTransformer

# relative imports to avoid TF import
from ..models.rnd import Predictor_network, Target_network
from ..utils.mltools import (
    AutoClipper,
    Dequemax,
    calc_cov,
    clip_to_value,
    loss_func,
    loss_func_vamp,
    m_inv,
    metric_VAMP2,
    reinitialize,
    return_most_freq,
    whiten_data,
)
from .base_loger import logger


class RunningMeanStd:
    """
    The class allows for calculating mean and average with Welford's online algorithm.
    The mean and variance is calculated across axis=0, so the resulting mean and variance
    can also be a tensor.
    source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    source: https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/utils.py
    adapted to work with TF2

    Arguments:
        epsilon: a float number to make mean and variance calculations more stable.
        shape: shape of the tensor, whose mean and variance is calculated.
    """

    def __init__(self, epsilon=1e-4, shape=None):
        self.mean = tf.Variable(
            tf.zeros(shape=shape), name="mean", dtype=tf.float32, trainable=False
        )
        self.var = tf.Variable(
            tf.ones(shape=shape), name="var", dtype=tf.float32, trainable=False
        )

        self.current_mean = tf.Variable(
            tf.zeros(shape=shape),
            name="current_mean",
            dtype=tf.float32,
            trainable=False,
        )
        self.current_var = tf.Variable(
            tf.ones(shape=shape), name="current_var", dtype=tf.float32, trainable=False
        )

        self.max_var = tf.Variable(
            tf.ones(shape=shape), name="max_var", dtype=tf.float32, trainable=False
        )

        self.min_var = tf.Variable(
            tf.ones(shape=shape), name="min_var", dtype=tf.float32, trainable=False
        )

        self.count = tf.Variable(
            epsilon, name="count", dtype=tf.float32, trainable=False
        )

    def current_mean_var(self):
        """Returns current mean and variance for the given tensor across axis=0
        Arguments:
            x: batch tensor of shape defined in the init
        Returns
            mean, variance across axis 0
        """

        return self.current_mean.read_value(), self.current_var.read_value()

    def current_min_max(self):
        """Returns current min and max for the given tensor across axis=0
        Arguments:
            x: batch tensor of shape defined in the init
        Returns
            min, max across axis 0
        """

        return self.min_var.read_value(), self.max_var.read_value()

    def update(self, x):
        """Updates mean and variance with batch tensor x
        Arguments:
            x: batch tensor of shape defined in the init
        """
        batch_mean = tf.reduce_mean(x, axis=0, keepdims=True)
        batch_var = tf.math.reduce_variance(x, axis=0, keepdims=True)
        self.current_mean.assign(batch_mean)
        self.current_var.assign(batch_var)
        self.min_var.assign(tf.math.reduce_min(x, axis=0, keepdims=True))
        self.max_var.assign(tf.math.reduce_max(x, axis=0, keepdims=True))
        batch_count = tf.cast(tf.shape(x)[0], dtype=tf.float32)
        self.update_from_moments(batch_mean, batch_var, batch_count)

        return self.mean.read_value(), self.var.read_value()

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean.read_value()
        tot_count = self.count.read_value() + batch_count
        new_mean = self.mean.read_value() + delta * batch_count / tot_count
        m_a = self.var.read_value() * (self.count.read_value())
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + tf.square(delta)
            * self.count.read_value()
            * batch_count
            / (self.count.read_value() + batch_count)
        )
        new_var = M2 / (self.count.read_value() + batch_count)

        new_count = batch_count + self.count.read_value()

        self.mean.assign(new_mean, read_value=False)
        self.var.assign(new_var, read_value=False)
        self.count.assign(new_count, read_value=False)

    def get_mean_and_var(self):
        """Returns means and variance tensors"""
        return self.mean.read_value(), self.var.read_value()


class RndTrain:
    """The class's object allows for training a RND network with batches of
    molecular dynamics simulation's features (distance matrix, cos of dihedral
    angles). Model's parameters (number of layers, activations, normalization
    layers) are passed through `config` dictionary.
    Arguments:
        config: dictionary with fallowing parameters:
            model: defines model used in the RND, it has fallowing parameters:
                dense_units: a list with number of neurons for Curiosity part (list of uints)
                dense_units_ae_enc: a list with number of neurons for Autoencoder encoder part (list of uints)
                dense_activ: a string with Keras's activation function or deel-lip activation
                             (if spectral=True)
                dense_layernorm: whether to use layer normalization or not (bool)
                dense_batch: whether to use batch renormalization or not (bool)
                dense_out: number of outputs (uint)
                dense_out_activ: a string with Keras's activation function (str)
                curiosity_activ: activation functions for curiosity part of the
                                 algorithm (str)
                spectral: Use 1-Lipschitz comp. layers (True or False)
                ae_spectral_only: Use 1-Lipschitz comp. layers only for AE (bool)
                orthonormal: Whetever to constraint output's transformation (W of last layer) of the dim. red. to be orthogonal (bool)
                l1_reg: whether to L1 regularize encoder/node of AE/mTAE/VAMP/SRV (ufloat)
                l2_reg: whether to L2 regularize encoder/node of AE/mTAE/VAMP/SRB (ufloat)
            autoencoder: if to use autoencoder (bool).
            autoencoder_lagtime: lagtime for time-lagged autoencoder (uint,
                                 max is number of steps/stride)
            vampnet: whetever to use Vampnet of Noe for the dim. red. part (bool)
            reversible_vampnet: whetever to use SRV of Ferguson to perform dim. red part (bool)
            num_of_train_updates: number of optimization iterations (uint)
            num_of_ae_train_updates: number of optimization iterations
                                     for autoencoder (uint)
            learning_rate_cur: Learning rate for the curiosity part (ufloat)
            learning_rate_ae: learning rate for autoencoder part (ufloat)
            obs_stand: if standarize observation by mean and variance (bool).
                         Sometimes it gives better performance.
            reward_stand: If standarize reward with its (bool)
                          standard deviation. Sometimes it's
                          easier to analyze reward with it
            optimizer: Optimizer (curiosity) to choose. You can choose between
                       nsgd (Nesterov SGD), sgd (SGD), msgd (mSGD),
                       amsgrad (AMSgrad), rmsprop (RMSprop)
                       adadelta (adadelta).
            optimizer_ae: Optimizer (AE) to choose. You can choose between
                       nsgd (Nesterov SGD), sgd (SGD), msgd (mSGD),
                       amsgrad (AMSgrad), rmsprop (RMSprop)
                       adadelta (adadelta).

            train_buffer_size: Size of buffer for training example
                               for curiosity part (uint).
            target_network_update_freq: Every how many cycles, the autoencoder
                                        part should be updated and used
                                        to judge curiosity. If it's 1 it's updated
                                        everytime but curiosity part of the algorithm
                                        may not catch up and output trash (uint).
            vamp2_metric: If to calculate VAMP-2 score for outputs. It may take
                          a quite portion of computational time.
        An example of such config is below:
         ```config_rnd = {'model': 'dense_units': [4, 4],
                                   'dense_units_ae_enc': [4, 4],
                                   'dense_units_ae_dec': [4, 4],
                                   'dense_activ': 'fullsort',
                                   'dense_layernorm': False,
                                   'dense_batchnorm': False,
                                   'dense_out': 1,
                                   'dense_out_activ': 'linear',
                                   'curiosity_activ': 'tanh',
                                   'initializer': 'glorot_uniform',
                                   'spectral': False,
                                   'ae_spectral_only': False,
                                   'orthonormal': False,
                                   'l1_reg': 0.0,
                                   'l2_reg': 0.0001,
                                   'unit_constraint': False},
                          'autoencoder': True,
                          'autoencoder_lagtime': 450,
                          'vampnet': False,
                          'reversible_vampnet': False,
                          'num_of_train_updates': 1,
                          'num_of_ae_train_updates': 2,
                          'learning_rate_cur': 0.0001,
                          'learning_rate_ae': 0.0001,
                          'obs_stand': False,
                          'reward_stand': False,
                          'train_buffer_size': 50000,
                          'optimizer_ae': 'nsgd',
                          'optimizer': 'nsgd',
                          'target_network_update_freq': 20,
                          'vamp2_metric': True,
                          'classic_loss': False
                          }```
    """

    def __init__(
        self,
        diskcache=True,
        data_path=None,
        fname_checkpoint_filenames=None,
        checkpoint_save_frequency=None,
        config=None,
        oneframe_stepsize=None,
        if_use_positions=None,
        if_use_dihedrals=None,
        if_use_distances=None,
    ):
        # use only one GPU
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                # use last GPU on the visible NODE!
                tf.config.set_visible_devices(gpus[-1], "GPU")
                logical_gpus = tf.config.list_logical_devices("GPU")
                logger.debug(
                    "Physical GPUs {}, Logical GPU {}".format(gpus, logical_gpus)
                )
            except RuntimeError as e:
                logger.error(
                    "Visible devices must be set before GPUs have been initialized"
                )
                logger.error("RuntimeError: {}".format(e))
        else:
            logger.error("There is no GPU in the machine or they aren't initialized")
        # setup memory growth
        try:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            logger.warning("Memory growth failed, because {}".format(e))

        # Test if tensorflow works
        # Also it will block changing GPU visibility
        try:
            x = tf.Variable([[4, 4, 4]])
            tf.matmul(x, x, transpose_b=True)
            tf.Variable(
                tf.ones(shape=(400, 400)),
                name="test",
                dtype=tf.float32,
                trainable=False,
            )
            logger.debug("Tensorflow works")
        except Exception as e:
            logger.error("Tensorflow failed to work")
            logger.error(e)
            sys.exit("Exiting program, tensorflow doesn't work")

        # checkpoint:
        self.checkpoint_save_frequency = checkpoint_save_frequency
        if fname_checkpoint_filenames is None:
            raise ValueError("fname_checkpoint_filenames has to be a string")
        self.fname_checkpoint_filenames = fname_checkpoint_filenames
        # Setting controling perfromance
        self.diskcache = diskcache
        # directory for diskcache
        self.data_path = data_path
        self.decay_num_of_opt_iterations = True
        self.tb_writer = tf.summary.create_file_writer(
            self.data_path
            + "/tensorboard/rndstats/"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + "/train"
        )
        self.rohith = np.array([1])
        # Variables used for later
        self.oneframe_stepsize = oneframe_stepsize
        self.if_use_positions = if_use_positions
        self.if_use_dihedrals = if_use_dihedrals
        self.if_use_distances = if_use_distances
        # Hyperparameters
        ## Optimizer
        self.momentum_curiosity = 0.9
        self.momentum_ae = 0.99
        clipping_percentage = 10
        self.epsilon = tf.convert_to_tensor(1e-8)
        self.remove_last_cycle = False  # setting it to true, makes is to fallow RND paper (training after choosing actions)
        logger.warning("Remove last cycle is {}".format(self.remove_last_cycle))

        # lagtime
        self.flex_tau = False

        ## Reward
        ### if to use DSSP in the reward
        self.dssp_reward = False
        ### if to use k-tournament selection style of dssp reward selection
        self.sorted_dssp = False

        # Variables used to store temporary state
        self.one_simulation_size = None
        self.cycle = 0
        self.target_model_count = 0
        self.logconst = None

        ## statistics objects
        self.total_steps_ae = 0
        self.total_steps = 0
        self.W_whiten = None
        self.mean_whiten = None
        self.obs_stat = None
        self.reward_stat = None
        self.reward_robscaler = PowerTransformer(method="box-cox")
        self.energy_robscaler = PowerTransformer()
        self.dssp_robscaler = PowerTransformer(method="box-cox")
        self.max_reward = None
        self.max_reward_prev = None
        ## Vampnet
        self.means_ = None
        self.means2_ = None
        self.cov_0 = None
        self.cov_1 = None
        self.beta = 0.99
        self.norms_ = None
        self.eigenvectors_copy = None
        self.eigenvalues_copy = None
        self.prev_implied_time = None
        self.norms_copy = None
        self.means_copy = None
        self.srv_lower_copy = None
        self.srv_upper_copy = None
        self.means2_copy = None
        self.implied_time_array = []

        ## Batch statistics
        self.n_batch_stored = 0
        self.n_train_stored = 0
        self.batch_store_buffer = []
        self.n_batch_ae_stored = 0
        self.batch_ae_store_buffer = []

        ## Loss statistics
        self.last_curiosity_loss = 1e10
        self.last_ae_loss = 1e10
        self.loss_count = 0

        # Confirguration of ANN training
        self.config = config
        self.minibatch_size_cur = config["minibatch_size_cur"]
        self.minibatch_size_ae = config["minibatch_size_ae"]
        self.nonrev_srv = config["nonrev_srv"]
        self.num_of_train_updates = config["num_of_train_updates"]
        self.num_of_train_updates_current = config["num_of_train_updates"]
        self.num_of_ae_train_updates = config["num_of_ae_train_updates"]
        self.num_of_ae_train_updates_current = config["num_of_ae_train_updates"]
        self.learning_rate_cur = float(config["learning_rate_cur"])
        self.learning_rate_ae = float(config["learning_rate_ae"])
        self.ae_lagtime = config["autoencoder_lagtime"]
        self.train_buffer_size = config["train_buffer_size"]
        self.obs_stand = config["obs_stand"]
        self.reward_stand = config["reward_stand"]
        self.chosen_optimizer = config["optimizer"]
        self.chosen_ae_optimizer = config["optimizer_ae"]
        self.target_network_update_freq = config["target_network_update_freq"]
        self.timescale_mode = config["timescale_mode_target_network_update"]
        self.timescale_mode_lock = False
        self.implied_time_last_update = None
        self.vamp2_metric = config["vamp2_metric"]
        try:
            self.reinitialize_predictor_network = config[
                "reinitialize_predictor_network"
            ]
        except:
            self.reinitialize_predictor_network = True
        self.slowp_vector = tf.reshape(
            tf.convert_to_tensor(config["slowp_vector"], dtype=tf.float32), (1, -1)
        )
        self.slowp_kinetic_like_scaling = config["slowp_kinetic_like_scaling"]
        logger.info(
            "slowp_kinetic_like_scaling is {}".format(self.slowp_kinetic_like_scaling)
        )
        self.classic_loss = config["classic_loss"]
        self.shrinkage = config["shrinkage"]
        self.whiten = config["whiten"]
        self.protein_cnn = config["protein_cnn"]
        self.logtrans = config["logtrans"]

        # Initialisation of RND model
        self.autoencoder = config["autoencoder"]
        self.vampnet = config["vampnet"]
        self.reversible_vampnet = config["reversible_vampnet"]
        self.spectral = config["model"]["target"]["spectral"]
        self.dense_out = config["model"]["target"]["dense_out"]
        self.cnn = config["model"]["target"]["cnn"]
        self.cnn_cur = config["model"]["predictor"]["cnn"]

        org_conf = deepcopy(config)
        conf_copy = deepcopy(config["model"])

        # Create predictor model for RND
        conf_copy_keys = list(conf_copy["predictor"].keys())
        # remove not used parameters in predictor network
        aval_args_target = inspect.getfullargspec(Predictor_network.__init__)[0]
        for key in conf_copy_keys:
            if key not in aval_args_target:
                del conf_copy["predictor"][key]
        self.predictor_model = Predictor_network(**conf_copy["predictor"])

        conf_copy = deepcopy(org_conf["model"])

        # makes training more stable for VAMPNET
        if self.dense_out < 6:
            conf_copy["target"]["dense_out"] = 6

        # Create target model for RND
        conf_copy_keys = list(conf_copy["target"].keys())
        # remove not used parameters
        aval_args_target = inspect.getfullargspec(Target_network.__init__)[0]
        for key in conf_copy_keys:
            if key not in aval_args_target:
                del conf_copy["target"][key]
        self.target_model = Target_network(**conf_copy["target"])
        if self.autoencoder:
            self.target_model_copy = Target_network(**conf_copy["target"])

        # Here we set optimizer
        if self.chosen_optimizer == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate_cur,
                centered=True,
            )
        elif self.chosen_optimizer == "nsgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_cur,
                nesterov=True,
                momentum=self.momentum_curiosity,
            )
        elif self.chosen_optimizer == "msgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_cur,
                nesterov=False,
                momentum=self.momentum_curiosity,
            )
        if self.chosen_optimizer == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_cur,
            )
        if self.chosen_optimizer == "amsgrad":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate_cur,
                amsgrad=True,
            )
        if self.chosen_optimizer == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate_cur,
                amsgrad=False,
            )

        if self.chosen_optimizer == "adab":
            self.optimizer = tfa.optimizers.AdaBelief(
                learning_rate=self.learning_rate_cur,
                amsgrad=False,
                rectify=True,
            )

        if self.chosen_optimizer == "adabLA":
            self.optimizer = tfa.optimizers.Lookahead(
                tfa.optimizers.AdaBelief(
                    learning_rate=self.learning_rate_cur,
                    amsgrad=False,
                ),
                sync_period=6,
                slow_step_size=0.5,
            )

        if self.chosen_optimizer == "nadam":
            self.optimizer = tf.keras.optimizers.Nadam(
                learning_rate=self.learning_rate_cur,
            )

        if self.chosen_optimizer == "adadelta":
            self.optimizer = tf.keras.optimizers.Adadelta(
                learning_rate=self.learning_rate_cur,
            )
        if self.chosen_optimizer == "adamax":
            self.optimizer = tf.keras.optimizers.Adamax(
                learning_rate=self.learning_rate_cur,
            )

        if self.chosen_optimizer == "adagrad":
            self.optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=self.learning_rate_cur,
            )

        if self.chosen_optimizer == "adamw":
            self.optimizer = tfa.optimizers.AdamW(
                learning_rate=self.learning_rate_cur,
                weight_decay=self.learning_rate_cur / 1e4,
            )

        if self.chosen_optimizer == "novograd":
            self.optimizer = tfa.optimizers.NovoGrad(
                learning_rate=self.learning_rate_cur,
            )

        if self.chosen_optimizer == "swa":
            self.optimizer = tfa.optimizers.SWA(
                tf.keras.optimizers.SGD(
                    learning_rate=self.learning_rate_cur,
                ),
                0,
                5,
            )

        if self.chosen_optimizer == "swarmsprop":
            self.optimizer = tfa.optimizers.SWA(
                tf.keras.optimizers.RMSprop(
                    learning_rate=self.learning_rate_cur,
                ),
                0,
                5,
            )

        if self.chosen_optimizer == "movingaverage":
            self.optimizer = tfa.optimizers.MovingAverage(
                tf.keras.optimizers.SGD(
                    learning_rate=self.learning_rate_cur,
                )
            )

        logger.info("Optimzer is {}".format(self.chosen_optimizer))

        # Here we set optimizer for AE
        if self.chosen_ae_optimizer == "rmsprop":
            self.optimizer_ae = tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate_ae,
                centered=True,
            )
        elif self.chosen_ae_optimizer == "nsgd":
            self.optimizer_ae = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_ae,
                nesterov=True,
                momentum=self.momentum_ae,
            )
        elif self.chosen_ae_optimizer == "msgd":
            self.optimizer_ae = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_ae,
                nesterov=False,
                momentum=self.momentum_ae,
            )
        if self.chosen_ae_optimizer == "sgd":
            self.optimizer_ae = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate_ae,
            )
        if self.chosen_ae_optimizer == "amsgrad":
            self.optimizer_ae = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate_ae,
                amsgrad=True,
                beta_1=0.99,
            )
        if self.chosen_ae_optimizer == "adam":
            self.optimizer_ae = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate_ae,
                amsgrad=False,
                beta_1=0.99,
            )

        if self.chosen_ae_optimizer == "adab":
            self.optimizer_ae = tfa.optimizers.AdaBelief(
                learning_rate=self.learning_rate_ae,
                amsgrad=False,
                rectify=True,
            )

        if self.chosen_ae_optimizer == "adabLA":
            self.optimizer_ae = tfa.optimizers.Lookahead(
                tfa.optimizers.AdaBelief(
                    learning_rate=self.learning_rate_ae,
                    amsgrad=False,
                ),
                sync_period=6,
                slow_step_size=0.5,
            )

        if self.chosen_ae_optimizer == "nadam":
            self.optimizer_ae = tf.keras.optimizers.Nadam(
                learning_rate=self.learning_rate_ae,
                beta_1=0.99,
            )

        if self.chosen_ae_optimizer == "adadelta":
            self.optimizer_ae = tf.keras.optimizers.Adadelta(
                learning_rate=self.learning_rate_ae,
            )
        if self.chosen_ae_optimizer == "adamax":
            self.optimizer_ae = tf.keras.optimizers.Adamax(
                learning_rate=self.learning_rate_ae,
                beta_1=0.99,
            )
        if self.chosen_ae_optimizer == "adagrad":
            self.optimizer_ae = tf.keras.optimizers.Adagrad(
                learning_rate=self.learning_rate_ae,
            )

        if self.chosen_ae_optimizer == "adamw":
            self.optimizer_ae = tfa.optimizers.AdamW(
                learning_rate=self.learning_rate_ae,
                weight_decay=self.learning_rate_ae / 1e4,
            )

        if self.chosen_ae_optimizer == "novograd":
            self.optimizer_ae = tfa.optimizers.NovoGrad(
                learning_rate=self.learning_rate_ae,
            )

        if self.chosen_ae_optimizer == "swa":
            self.optimizerae_ = tfa.optimizers.SWA(
                tf.keras.optimizers.SGD(
                    learning_rate=self.learning_rate_ae,
                ),
                0,
                10,
            )

        if self.chosen_ae_optimizer == "swarmsprop":
            self.optimizer_ae = tfa.optimizers.SWA(
                tf.keras.optimizers.RMSprop(
                    learning_rate=self.learning_rate_ae,
                ),
                0,
                10,
            )

        if self.chosen_ae_optimizer == "movingaverage":
            self.optimizer_ae = tfa.optimizers.MovingAverage(
                tf.keras.optimizers.SGD(
                    learning_rate=self.learning_rate_ae,
                )
            )

        self.autoclip = AutoClipper(clipping_percentage)
        self.autoclip_ae = AutoClipper(clipping_percentage)
        logger.info("AE Optimzer is {}".format(self.chosen_ae_optimizer))
        if "tensorboard_log_nnweights" in config:
            self.tensorboard_log_nnweights = config["tensorboard_log_nnweights"]
        else:
            self.tensorboard_log_nnweights = False
        logger.warning(
            "Logging neural network weights is set to {}".format(
                self.tensorboard_log_nnweights
            )
        )
        self.energy_mode = config["energy_mode"]
        if self.energy_mode == "sort":
            self.sorted_energy = True
            self.include_energy = False
        elif self.energy_mode == "continuous":
            self.sorted_energy = False
            self.include_energy = True
        else:
            self.sorted_energy = False
            self.include_energy = False
        if self.include_energy:
            try:
                self.energy_continuous_constant = float(
                    config["energy_continuous_constant"]
                )

            except:
                logger.info("energy_continuous_constant not set. Setting to 5.0")
                self.energy_continuous_constant = 5

            else:
                logger.info(
                    "energy_continuous_constant is {}".format(
                        self.energy_continuous_constant
                    )
                )

        if self.sorted_energy:
            try:
                self.sorted_energy_constant = int(config["sorted_energy_constant"])

            except:
                logger.info("sorted_energy_constant not set. Setting to 20")
                self.sorted_energy_constant = 20

            else:
                logger.info(
                    "sorted_energy_constant is {}".format(self.sorted_energy_constant)
                )

        # check if vamp or other is true if autoencoder is true
        if not self.autoencoder:
            if any((self.vampnet, self.reversible_vampnet)):
                raise ValueError(
                    "To any of the two to work (vampnet, reversible_vampnet), autoencoder have to be set True"
                )
        # check if vamp is true, when reversible_vamp is true
        if self.reversible_vampnet:
            if not self.vampnet:
                raise ValueError(
                    "vampnet have to be true, when reversible_vampnet is true"
                )

    def norm_inputs(self, inp, mean, var, tanh=False, logtrans=False, rnd_clip=False):
        if logtrans:
            if self.logconst is None:
                self.logconst = tf.reduce_max(tf.math.abs(inp)) * 100
            inp = tf.math.log(inp + self.logconst) - tf.math.log(self.logconst)
        if not self.obs_stand:
            return inp

        z_norm = (inp - mean) / tf.sqrt(var + self.epsilon)

        if rnd_clip:
            z_norm = tf.clip_by_value(z_norm, -5, 5)

        if tanh:
            return 0.5 * (tf.math.tanh(0.01 * z_norm) + 1)
        else:
            return z_norm

    def initialise(self, initial_observation):
        """Initializes RND network, means and variances
        with some initial observation.
        """
        # if vampnet is turned on, then batch_size have to be aligned with sampled
        # data, shifted by lagtime.
        if self.vampnet:
            self.init_size = initial_observation["dist_matrix"].shape[0]
            logger.info(
                "Mini-batch size is {} for AE/VAMPnet".format(self.minibatch_size_ae)
            )
            logger.info(
                "Mini-batch size is {} for Curiosity".format(self.minibatch_size_cur)
            )

        if self.train_buffer_size > 0:
            if self.diskcache:
                self.train_buffer = Dequemax(
                    maxlen=self.train_buffer_size // self.init_size
                    + bool(self.train_buffer_size % self.init_size),
                    directory=self.data_path + "/" + "diskcachedeque_trainbuff",
                    pickle_protocol=5,
                )
            else:
                self.train_buffer = deque(
                    maxlen=self.train_buffer_size // self.init_size
                    + bool(self.train_buffer_size % self.init_size)
                )
        else:
            self.train_buffer = []

        self.add_to_buffer(initial_observation, shuffle=False)
        self._first_run()

        assert self.norms_copy is not None
        assert self.means_copy is not None
        assert self.srv_upper_copy is not None
        assert self.srv_lower_copy is not None
        assert self.means2_copy is not None
        assert self.eigenvectors_copy is not None
        assert self.eigenvalues_copy is not None

        if any(
            (
                tf.math.reduce_any(tf.math.is_nan(self.eigenvectors_copy)),
                tf.math.reduce_any(tf.math.is_nan(self.eigenvalues_copy)),
                tf.math.reduce_any(tf.math.is_nan(self.means_copy)),
                tf.math.reduce_any(tf.math.is_nan(self.srv_upper_copy)),
                tf.math.reduce_any(tf.math.is_nan(self.srv_lower_copy)),
                tf.math.reduce_any(tf.math.is_nan(self.means2_copy)),
                tf.math.reduce_any(tf.math.is_nan(self.norms_copy)),
            )
        ):
            raise StandardError(
                "Eigenvector, Means or Norms for reversible vampnet are nan"
            )

    def _first_run(self):
        first_time = True

        if not self.obs_stand:
            obs_mean, obs_var = (0.0, 1.0)

        dist_matrix_p = tf.convert_to_tensor(self.train_buffer[0])
        for example in self.train_buffer:
            dist_matrix = tf.convert_to_tensor(example)
            # standarization of state target
            if first_time:
                conf_copy = deepcopy(self.config["model"])
                self.obs_stat = RunningMeanStd(shape=(1, *dist_matrix.shape[1:]))
                self.reward_stat = RunningMeanStd(shape=(1, self.dense_out))

                first_time = False

            obs_mean, obs_var = self.obs_stat.update(dist_matrix)

        dist_matrix_standarized = self.norm_inputs(
            dist_matrix_p, obs_mean, obs_var, logtrans=self.logtrans
        )

        self.obs_mean_copy = obs_mean
        self.obs_var_copy = obs_var

        input_shape = list(dist_matrix_standarized.shape)
        input_shape[0] = None
        if self.cnn and self.protein_cnn:
            if self.if_use_distances and not (
                self.if_use_positions and self.if_use_dihedrals
            ):
                # solution for number of combinations of distances
                # assuming we use only distances to describe protein
                c = int((1 + (1 + 4 * 2 * input_shape[-1]) ** 0.5) // 2)
                # mdtraj works differently for even and odd # of AA
                if c % 2 == 0:
                    c = c - 1
                else:
                    c = c
                new_reshape_size = input_shape[-1] // c

            elif self.if_use_dihedrals and not (
                self.if_use_distances and self.if_use_positions
            ):
                # psi (cos and sin) and phi (cos and sin)
                new_reshape_size = 4
            elif self.if_use_positions:
                # x y z as features
                new_reshape_size = 3
            else:
                raise ValueError(
                    "You can only chose between distances or dihedrals, not together"
                )
        else:
            new_reshape_size = None
        if new_reshape_size is not None:
            l = tf.keras.layers.Reshape((-1, new_reshape_size))
        else:
            l = tf.keras.layers.Reshape((-1, 1))
        new_shape = tf.shape(l(dist_matrix_standarized))
        logger.warning(
            "Input feature size is (batch, seq, features) {}".format(new_shape)
        )
        self.target_model.build(
            x=dist_matrix_standarized, new_reshape_size=new_reshape_size
        )
        self.target_model_copy.build(
            x=dist_matrix_standarized, new_reshape_size=new_reshape_size
        )
        _, state_target = self.target_model(dist_matrix_standarized, training=False)
        _, state_target_copy = self.target_model_copy(
            dist_matrix_standarized, training=False
        )

        self._calc_basis(obs_mean, obs_var, self.ae_lagtime)
        if self.flex_tau:
            self.new_tau = self.ae_lagtime
        if self.eigenvectors_copy is None:
            # Initialize values
            self.eigenvectors_copy = tf.Variable(self.eigenvectors_)
            self.eigenvalues_copy = tf.Variable(self.eigenvalues_)
            self.means_copy = tf.Variable(self.means_)
            self.srv_upper_copy = tf.Variable(self.srv_upper_)
            self.srv_lower_copy = tf.Variable(self.srv_lower_)
            self.means2_copy = tf.Variable(self.means2_)
            self.norms_copy = tf.Variable(self.norms_)
        else:
            self.eigenvectors_copy.assign(self.eigenvectors_)
            self.eigenvalues_copy.assign(self.eigenvalues_)
            self.means_copy.assign(self.means_)
            self.srv_lower_copy.assign(self.srv_lower_)
            self.srv_upper_copy.assign(self.srv_upper_)
            self.means2_copy.assign(self.means2_)
            self.norms_copy.assign(self.norms_)

        if self.ae_lagtime > 0 and self.autoencoder:
            state_target = self.project_state(state_target)

        if self.cnn_cur and self.protein_cnn:
            # from cnn
            new_reshape_size = new_reshape_size
        else:
            new_reshape_size = None

        self.predictor_model.build(
            x=dist_matrix_standarized, new_reshape_size=new_reshape_size
        )
        state_predict = self.predictor_model(dist_matrix_standarized, training=False)

        regul_loss = tf.reduce_sum(self.predictor_model.losses)
        loss, loss_nm = loss_func(state_target, state_predict, regul_loss=regul_loss)
        # loss standarized
        loss_mean, loss_var = self.reward_stat.update(loss_nm)
        loss_mean, loss_var = self.reward_stat.current_mean_var()
        # also with mean, not like in the papaer
        loss_standarized = (loss_nm - loss_mean) / tf.sqrt(loss_var + self.epsilon)

        # we add back after to the buffer in the curioussampling.py
        self.train_buffer.clear()
        # initialize tensorboard

    def train(self, total_num_of_obs, shuffle_inside_curiosity=True):
        """Performs training for the current training buffer (samples observed during current MD simulation)
        with predefined number of optimization steps in the `config`. The buffer is emptied at the end
        of training.
        Arguments:
        """

        if self.cycle % self.checkpoint_save_frequency == 0:
            ### Checkpoint saving
            internal_data = {
                "one_simulation_size": self.one_simulation_size,
                "cycle": self.cycle,
                "target_model_count": self.target_model_count,
                "logconst": self.logconst,
            }
            full_fname = save_pickle_object(
                internal_data, self.data_path, fname="int_data"
            )
            append_json_object(
                {"rnd_filename_int": full_fname},
                path=self.data_path,
                fname=self.fname_checkpoint_filenames,
            )

            stats = {
                "W_whiten": self.W_whiten,
                "mean_whiten": self.mean_whiten,
                "obs_stat": self.obs_stat,
                "reward_stat": self.reward_stat,
                "reward_robscaler": self.reward_robscaler,
                "dssp_robscaler": self.dssp_robscaler,
                "energy_robscaler": self.energy_robscaler,
                "max_reward": self.max_reward,
                "max_reward_prev": self.max_reward_prev,
                "implied_time_last_update": self.implied_time_last_update,
                "timescale_mode_lock": self.timescale_mode_lock,
            }
            full_fname = save_pickle_object(stats, self.data_path, fname="stats")
            append_json_object(
                {"rnd_filename_stats": full_fname},
                path=self.data_path,
                fname=self.fname_checkpoint_filenames,
            )

            vamp_stats = {
                "means_": self.means_,
                "means2_": self.means2_,
                "cov_0": self.cov_0,
                "cov_1": self.cov_1,
                "beta": self.beta,
                "norms": self.norms_,
            }
            full_fname = save_pickle_object(
                vamp_stats, self.data_path, fname="vamp_stats"
            )
            append_json_object(
                {"rnd_filename_vamp": full_fname},
                path=self.data_path,
                fname=self.fname_checkpoint_filenames,
            )

            target_weights = self.target_model.get_weights()
            full_fname = save_pickle_object(
                target_weights, self.data_path, fname="target_weights"
            )
            append_json_object(
                {"rnd_filename_target_weights": full_fname},
                path=self.data_path,
                fname=self.fname_checkpoint_filenames,
            )

            predictor_weights = self.predictor_model.get_weights()
            full_fname = save_pickle_object(
                predictor_weights, self.data_path, fname="predictor_weights"
            )
            append_json_object(
                {"rnd_filename_predictor_weights": full_fname},
                path=self.data_path,
                fname=self.fname_checkpoint_filenames,
            )

        # dist_matrix is of dimension (batch_size, distm_dimension, distm_dimension)
        loss_standarized_list = []
        loss_ae_list = []
        vamp2_score_list = []
        # update obs statistics
        if self.obs_stand:
            for k, batch_example in enumerate(self.train_buffer):
                batch_dist_matrix = tf.convert_to_tensor(
                    batch_example, dtype=tf.float32
                )
                obs_mean, obs_var = self.obs_stat.update(batch_example)
        # SRV/VAMPNET part
        if self.ae_lagtime > 0 and self.autoencoder:
            len_of_buffer = len(self.train_buffer)
            one_example_size = self.train_buffer[0].shape[0]
            num_of_envs = total_num_of_obs // self.train_buffer[0].shape[0]
            train_buff_ind = list(range(len(self.train_buffer)))
            for i in range(self.num_of_ae_train_updates_current):
                n_cached = min(
                    10
                    * num_of_envs
                    * max(1, self.minibatch_size_ae // one_example_size)
                    + random.randint(0, 5),
                    len_of_buffer,
                )
                last_batch = False
                # indirect shuffle
                random.shuffle(train_buff_ind)

                for j, k in enumerate(train_buff_ind):
                    if j == len_of_buffer - 1:
                        last_batch = True

                    batch = tf.convert_to_tensor(self.train_buffer[k], dtype=tf.float32)

                    local_training_data = self.batch_ae_shuffle_aftern(
                        batch,
                        shuffle=shuffle_inside_curiosity,
                        n=n_cached,
                        last_batch=last_batch,
                        lagtime=self.ae_lagtime,
                    )
                    if local_training_data is None:
                        continue

                    for batch_shift, batch_back in local_training_data:
                        if self.vamp2_metric:
                            shift_standarized = self.norm_inputs(
                                batch_shift,
                                self.obs_mean_copy,
                                self.obs_var_copy,
                                logtrans=self.logtrans,
                            )
                            back_standarized = self.norm_inputs(
                                batch_back,
                                self.obs_mean_copy,
                                self.obs_var_copy,
                                logtrans=self.logtrans,
                            )
                            _, ztt = self.target_model_copy(shift_standarized)
                            zt0, _ = self.target_model_copy(back_standarized)
                            vamp2_metric = metric_VAMP2(ztt, zt0)
                            with self.tb_writer.as_default(step=self.total_steps_ae):
                                tf.summary.scalar(
                                    "train/Target raw batch VAMP2 score", vamp2_metric
                                )

                            vamp2_score_list.append(vamp2_metric)

                        assert tf.rank(batch_back) == 2
                        batch_ae_loss, g_norm_ae = self._train_step_vamp(
                            batch_back,
                            batch_shift,
                            self.obs_mean_copy,
                            self.obs_var_copy,
                        )
                        with self.tb_writer.as_default(step=self.total_steps_ae):
                            tf.summary.scalar(
                                "train/Target raw batch loss", batch_ae_loss
                            )
                            tf.summary.scalar(
                                "train/Target batch gradient norm", g_norm_ae
                            )
                        loss_ae_list.append(batch_ae_loss.numpy())
                        self.total_steps_ae += 1
                avg_loss = tf.reduce_mean(loss_ae_list)
                if_stop = self.stop_loss(
                    epoch_loss=avg_loss,
                    prev_loss=self.last_ae_loss,
                    tries=10,
                    treshhold=0.01,
                )
                self.last_ae_loss = avg_loss
                if if_stop:
                    logger.debug(
                        "Used projection stopped training after {} iterations".format(
                            i + 1
                        )
                    )
                    break
        if self.flex_tau:
            self.new_tau = self._est_lagtime(
                self.obs_mean_copy,
                self.obs_var_copy,
                initial_tau=self.new_tau,
                max_tau=batch.shape[0] // 2,
            )

            self._calc_basis(self.obs_mean_copy, self.obs_var_copy, self.new_tau)
            self.ae_lagtime = self.new_tau
        else:
            self._calc_basis(self.obs_mean_copy, self.obs_var_copy, self.ae_lagtime)

        if self.ae_lagtime > 0 and self.autoencoder:
            # size of buffer
            true_train_buff_size = sum(
                [self.train_buffer[i].shape[0] for i in train_buff_ind]
            )
            logger.info(
                "Lagged AE training buffer current size is: {0}".format(
                    true_train_buff_size
                )
            )
            # number of training iterations, decay with number of training examples
            # if self.decay_num_of_opt_iterations:
            #    frac_of_buff = max(0.0, true_train_buff_size / self.train_buffer_size)
            #    frac_of_updates = 1 - ((frac_of_buff * 100) / 100) ** 0.5
            #    self.num_of_ae_train_updates_current = max(
            #        1, int(self.num_of_ae_train_updates * frac_of_updates)
            #    )

        if (
            (
                self.target_model_count >= self.target_network_update_freq
                and self.autoencoder
            )
            and not self.timescale_mode
        ) or (self.timescale_mode_lock and self.timescale_mode):
            self.target_model_copy.set_weights(self.target_model.get_weights())
            self.norms_copy.assign(self.norms_)
            self.means_copy.assign(self.means_)
            self.srv_upper_copy.assign(self.srv_upper_)
            self.srv_lower_copy.assign(self.srv_lower_)
            self.means2_copy.assign(self.means2_)
            self.eigenvectors_copy.assign(self.eigenvectors_)
            self.eigenvalues_copy.assign(self.eigenvalues_)
            logger.debug("Eigenvec: {}".format(self.eigenvectors_copy))
            logger.debug("Eigenval: {}".format(self.eigenvalues_copy))
            logger.debug("Norms: {}".format(self.norms_copy.read_value()))
            logger.debug("Means: {}".format(self.means_copy.read_value()))

            if self.obs_stand:
                obs_mean, obs_var = self.obs_stat.get_mean_and_var()
                self.obs_mean_copy = obs_mean
                self.obs_var_copy = obs_var

            # TODO: Set properly memory units (so that they are output)
            # TODO: Reset only memory units to save computational time
            if self.reinitialize_predictor_network:
                reinitialize(self.predictor_model, dont_reset_cnn=True)
                for var in self.optimizer.variables():
                    var.assign(tf.zeros_like(var))

                logger.info("Predictor Network reiinitialized")
            logger.info("Target model updated!!!")
            self.target_model_count = 0
            self.timescale_mode_lock = False

        # VAMP statistics START
        self.target_model_count += 1

        first_time_reversible = True
        assert self.norms_copy is not None
        assert self.means_copy is not None
        assert self.srv_upper_copy is not None
        assert self.srv_lower_copy is not None
        assert self.means2_copy is not None
        assert self.eigenvectors_copy is not None
        assert self.eigenvalues_copy is not None
        # calc implied time scale
        if self.oneframe_stepsize is not None:
            if self.flex_tau:
                tau = self.new_tau
            else:
                tau = self.ae_lagtime
            implied_time = (
                tau * (-1 / np.log(self.eigenvalues_)) * self.oneframe_stepsize
            )
            logger.info("Timescales for this epoch: {} ".format(implied_time))
        else:
            implied_time = tau * (-1 / np.log(self.eigenvalues_))
            logger.info(
                "Timescales for this epoch: {} of sampled frames".format(implied_time)
            )
        self.implied_time_array.append(implied_time)
        # print difference in eigenvalues
        if self.prev_implied_time is not None:
            imp_error = implied_time - self.prev_implied_time
            logger.info("Diff. between epoch eigenvalues: {}".format(imp_error))
            if self.timescale_mode and not np.isnan(implied_time).any():
                if self.implied_time_last_update is None:
                    self.implied_time_last_update = implied_time
                rel_error = (implied_time - self.implied_time_last_update) / (
                    implied_time + self.implied_time_last_update
                )
                # we only care about positive change (longer timescales)
                rel_error_mask = rel_error >= 0.05
                if rel_error_mask.any():
                    self.timescale_mode_lock = True
                    self.implied_time_last_update = implied_time
                logger.info(
                    "Relative timescale error from last update: {}".format(rel_error)
                )
        with self.tb_writer.as_default(self.cycle):
            if self.flex_tau:
                if self.oneframe_stepsize is not None:
                    tf.summary.scalar(
                        "implied/lagtime tau (ps) in each episode",
                        self.new_tau * self.oneframe_stepsize._value,
                    )

                else:
                    tf.summary.scalar(
                        "implied/lagtime tau in each episode", self.new_tau
                    )

            for i, t in enumerate(implied_time):
                tf.summary.scalar(
                    "implied/Implied timescale {} in each episode".format(i), t
                )

            for i, t in enumerate(self.eigenvalues_):
                tf.summary.scalar("implied/Eigenvalues {} in each episode".format(i), t)

        self.prev_implied_time = implied_time

        # VAMP statistics END

        # CURIOSITY LOOP PART
        # switch for first time in the batch loop
        first_time = True
        train_buff_ind = list(range(len(self.train_buffer)))
        # Remove last n examples
        if self.remove_last_cycle:
            for i in range(num_of_envs):
                del train_buff_ind[-1]
        len_of_buffer = len(train_buff_ind)
        one_example_size = self.train_buffer[0].shape[0]
        for i in range(self.num_of_train_updates_current):
            # max 10 per walker
            # use randint to mix between different batches in every iter
            n_cached = min(
                10 * num_of_envs * max(1, self.minibatch_size_cur // one_example_size)
                + random.randint(0, 5),
                len_of_buffer,
            )

            # shuffle train_buffer
            random.shuffle(train_buff_ind)
            for j, k in enumerate(train_buff_ind):
                last_batch = False
                batch_example_j = tf.convert_to_tensor(
                    self.train_buffer[k], dtype=tf.float32
                )

                if j == len_of_buffer - 1:
                    last_batch = True
                # accumulate number of examples
                train_buffer_local_temp = self.batch_shuffle_aftern(
                    batch_example_j,
                    shuffle=shuffle_inside_curiosity,
                    n=n_cached,
                    last_batch=last_batch,
                )
                if train_buffer_local_temp is None:
                    continue
                for batch_example in train_buffer_local_temp:
                    batch_dist_matrix = tf.convert_to_tensor(
                        batch_example, dtype=tf.float32
                    )

                    # perform one train step
                    assert tf.rank(batch_dist_matrix) == 2

                    batch_loss, batch_loss_nm, g_norm = self._train_step(
                        batch_dist_matrix, self.obs_mean_copy, self.obs_var_copy
                    )
                    with self.tb_writer.as_default(step=self.total_steps):
                        tf.summary.scalar(
                            "train/Predictor raw batch loss",
                            batch_loss,
                        )
                        tf.summary.scalar("train/Predictor batch gradient norm", g_norm)
                    loss_standarized_list.append(tf.reduce_mean(batch_loss))
                    self.total_steps += 1
            avg_loss = tf.reduce_mean(loss_standarized_list)
            if_stop = self.stop_loss(
                epoch_loss=avg_loss,
                prev_loss=self.last_curiosity_loss,
                tries=10,
                treshhold=0.01,
            )
            self.last_curiosity_loss = avg_loss
            if if_stop:
                logger.debug(
                    "Curiosity stopped training after {} iterations".format(i + 1)
                )
                break
        true_train_buff_size = sum(
            [self.train_buffer[i].shape[0] for i in train_buff_ind]
        )
        # Decay number of curiosity opt iterations with number of examples
        # if self.decay_num_of_opt_iterations:
        #    frac_of_buff = max(0.0, true_train_buff_size / self.train_buffer_size)
        #    frac_of_updates = 1 - ((frac_of_buff * 100) / 100) ** 0.5
        #    self.num_of_train_updates_current = max(
        #        2, int(self.num_of_train_updates * frac_of_updates)
        #    )

        # log weights
        # TODO: Causes OMM when input is quite big
        # TODO: Thus either remove it (through some if option)
        # TODO: Or don't save all weights
        if self.tensorboard_log_nnweights:
            with self.tb_writer.as_default(self.cycle):
                for w in self.predictor_model.weights:
                    tf.summary.histogram(
                        "pweights/Predictor network {}".format(w.name), w, buckets=100
                    )
                for w in self.target_model.weights:
                    tf.summary.histogram(
                        "tweights/Target network {}".format(w.name), w, buckets=100
                    )

        logger.info(
            "Main training buffer current size is: {}".format(true_train_buff_size)
        )

        return loss_standarized_list, loss_ae_list, vamp2_score_list

    @tf.function
    def _calc_basis_estimate(self, obs_mean, obs_var, batch_shift, batch_back):
        zt0, _ = self.target_model(
            self.norm_inputs(batch_back, obs_mean, obs_var, logtrans=self.logtrans)
        )
        _, ztt = self.target_model(
            self.norm_inputs(batch_shift, obs_mean, obs_var, logtrans=self.logtrans)
        )

        return zt0, ztt

    def _est_lagtime(
        self,
        obs_mean,
        obs_var,
        lags=10,
        samples=10,
        initial_tau=None,
        max_tau=None,
        tresh=0.01,
        f=0.1,
        deterministic_search=True,
    ):
        if initial_tau >= max_tau:
            logger.warning(
                "Warning: current lagtime can't increase, because trajectory is too short"
            )
        # ensure int type
        initial_tau = int(initial_tau)
        if not deterministic_search:
            while True:
                prop_taus = tf.random.truncated_normal(
                    (lags,), mean=float(initial_tau), stddev=max_tau
                )
                prop_taus = tf.math.round(prop_taus)
                # filter anyway for too big m
                # if true, redraw
                if_unique = len(prop_taus) > len(set(prop_taus.numpy()))
                m = (
                    tf.reduce_all(prop_taus < max_tau)
                    and tf.reduce_all(prop_taus > 0)
                    and if_unique
                )
                if m:
                    break
        else:
            # TODO: If length is lower (e.g. too small lag time)
            # extend it to set length

            # them by 0.05 of initial tau
            # because for large tau it is meaningless to go by one lag
            step = tf.math.maximum(1, int(tf.math.round(initial_tau * f)))
            left = tf.range(initial_tau - lags * step, initial_tau, delta=step)
            right = tf.range(initial_tau, initial_tau + lags * step, delta=step)
            # if there's no initial tau, add it from the right to left
            # so that it is in the middle
            if (initial_tau not in left) and (initial_tau not in right):
                left = tf.concat([left, [initial_tau]], axis=0)
            prop_taus = tf.concat(
                [left, right],
                axis=0,
            )

            m = prop_taus > 0
            prop_taus = prop_taus[m]
        prop_taus = list(tf.cast(prop_taus, dtype=tf.int64).numpy())
        tlags = []
        for tau in prop_taus:
            tlags.append(None)
        for i in range(samples):
            for j, tau in enumerate(prop_taus):
                tries = 5
                while tries > 0:
                    tries -= 1
                    try:
                        ts = self._calc_basis(
                            obs_mean,
                            obs_var,
                            tau=tau,
                            return_timescales=True,
                            subsampling=0.9,
                            estim_mode=True,
                        ).numpy()
                        break
                    except Exception:
                        if tries > 0:
                            pass
                        else:
                            raise Exception("Lag inv. didn't work 5 times in row")

                if tlags[j] is None:
                    tlags[j] = np.zeros_like(ts)
                tlags[j] += ts / samples
        # filter those that have entries lower than timelag
        ind_to_remove = []
        for i, (timescales, tau) in enumerate(zip(tlags, prop_taus)):
            if np.any(timescales < tau):
                ind_to_remove.append(i)
        for i in sorted(ind_to_remove, reverse=True):
            del tlags[i]
            del prop_taus[i]

        # if all were wrong, chose previous lag
        k = 3
        if not (len(tlags) > k):
            return initial_tau
        # interpolate slowest first process
        x = np.array(prop_taus)
        y = np.array(tlags)[:, 0]
        s = len(prop_taus) - np.sqrt(2 * len(prop_taus))
        f1 = interpolate.splrep(x, y, k=k, s=s)  # interpolate
        xnew = np.linspace(np.min(x), np.max(x), 400, endpoint=True)
        # calculate derivate)
        y_der = interpolate.splev(xnew, f1, der=1)
        if np.any(y_der <= tresh):
            # first element which is true
            current_tau = int(np.round(xnew[np.argmax(y_der <= tresh)]))
        else:
            # else chose maximum
            current_tau = int(prop_taus[-1])

        # prevent oscilations around the same tau
        if np.abs(current_tau - initial_tau) <= int(
            np.round(np.max([1.0, (f * 1.1) * initial_tau]))
        ):
            current_tau = initial_tau
        return current_tau

    # TODO: Automatic optimal lagtime, by adjusting linear VAC
    # Eg. by doing 5 steps back and forward and calculating convergence
    # check if timescales are larger than timelag
    # move timelag as an argument to function, to not change that one for ANN
    # or something like:https://pubs.acs.org/doi/full/10.1021/acs.jpcb.2c03711
    def _calc_basis(
        self,
        obs_mean,
        obs_var,
        tau=None,
        return_timescales=False,
        subsampling=None,
        estim_mode=False,
    ):
        """Calculates basis vectors for the SVR/VAMPNet method, based on the previous data
        approximated covariance matrices.
        Arguments:
             train_buffer: train buffer with the examples
             obs_mean: observation mean to normalize input
             obs_var: observation mean to normalize input
             tau: lagtime
             subsampling: how fraction of all samples
             return_timescales: if to return timescales of every slow dynamics process
         Returns:
             timescales if specified
        """
        M = self.dense_out
        zt0_buffer = []
        ztt_buffer = []
        obs = 0
        for batch in self.train_buffer:
            batch_shift = batch[tau:]
            batch_back = batch[:-tau]
            if subsampling is not None:
                val_len = int(batch_back.shape[0] * subsampling)
                ival = tf.range(val_len)
                ival = tf.random.shuffle(ival)
                batch_shift = tf.gather(batch_shift, ival, axis=0)
                batch_back = tf.gather(batch_back, ival, axis=0)
            assert tf.rank(batch_back) == 2
            assert tf.rank(batch_shift) == 2
            zt0_nom, ztt_nom = self._calc_basis_estimate(
                obs_mean, obs_var, batch_shift, batch_back
            )
            zt0_buffer.append(tf.cast(zt0_nom, tf.float32))
            ztt_buffer.append(tf.cast(ztt_nom, tf.float32))
        ztt_concat = tf.concat(ztt_buffer, axis=0)
        zt0_concat = tf.concat(zt0_buffer, axis=0)
        x_concat = tf.concat([ztt_concat, zt0_concat], axis=0)
        self.means_ = tf.reduce_mean(x_concat, axis=0)
        zt0_concat -= tf.reduce_mean(zt0_concat, axis=0)
        ztt_concat -= tf.reduce_mean(ztt_concat, axis=0)
        zt0_concat = tf.cast(zt0_concat, tf.float64)
        ztt_concat = tf.cast(ztt_concat, tf.float64)
        # calc covariances
        cov_01 = calc_cov(
            zt0_concat,
            ztt_concat,
            rblw=False,
            use_shrinkage=False,
            double=True,
            shrinkage=self.shrinkage,
        )
        cov_10 = calc_cov(
            ztt_concat,
            zt0_concat,
            rblw=False,
            use_shrinkage=False,
            double=True,
            shrinkage=self.shrinkage,
        )
        cov_00 = calc_cov(zt0_concat, zt0_concat, double=True, shrinkage=self.shrinkage)
        cov_11 = calc_cov(ztt_concat, ztt_concat, double=True, shrinkage=self.shrinkage)
        if not self.nonrev_srv:
            self.cov_0 = 0.5 * (cov_00 + cov_11)
            self.cov_1 = 0.5 * (cov_01 + cov_10)
        else:
            self.cov_0 = cov_00
            self.cov_1 = cov_01
        assert self.cov_0.shape[0] == zt0_nom.shape[1]
        cov_1_numpy = self.cov_1.numpy()
        cov_0_numpy = self.cov_0.numpy()
        assert cov_1_numpy.dtype == np.float64
        assert cov_0_numpy.dtype == np.float64
        if not self.reversible_vampnet:
            cov_00_inv = m_inv(cov_00, tf_based=True, return_sqrt=True)
            cov_11_inv = m_inv(cov_11, tf_based=True, return_sqrt=True)
            vamp_matrix = tf.matmul(tf.matmul(cov_00_inv, cov_01), cov_11_inv)
            ## transpose as in deeptime implementation
            # vamp_matrix = tf.transpose(vamp_matrix)
            # Select the all singular values of the VAMP matrix
            # SVD in Tensorflow gives you S from largest to smallest
            S, Up, Vp = tf.linalg.svd(vamp_matrix, compute_uv=True)
            U = tf.matmul(cov_00_inv, Up)
            V = tf.matmul(cov_11_inv, Vp)
            S = tf.clip_by_value(S, 0.0, 1.0)
            # we take only left singular vectors
            eigvals, eigvecs = S.numpy(), U.numpy()
        elif self.reversible_vampnet and not self.nonrev_srv:
            try:
                eigvals, eigvecs = scipy.linalg.eigh(cov_1_numpy, b=cov_0_numpy)
            except:
                eigvecs = np.ones(cov_1_numpy.shape) * 0.001
                eigvals = np.ones(cov_1_numpy.shape[0]) * 0.1
        elif self.reversible_vampnet:
            try:
                eigvals, eigvecs = scipy.linalg.eig(cov_1_numpy, b=cov_0_numpy)
                eigvals, eigvecs = (np.real(eigvals), np.real(eigvecs))
            except:
                eigvecs = np.ones(cov_1_numpy.shape) * 0.001
                eigvals = np.ones(cov_1_numpy.shape[0]) * 0.1

        else:
            raise ValueError(
                "Reversible vampnet, nonrev_svr or vampnet should be chosen"
            )

        if not estim_mode:
            # sorts descending
            # filter number of eigenvectors up to dense_out
            idx = np.argsort(np.abs(eigvals))[::-1][0 : self.dense_out]
            # remove the slowest process
            self.eigenvectors_ = tf.convert_to_tensor(eigvecs[:, idx], dtype=tf.float32)
            self.eigenvalues_ = tf.convert_to_tensor(eigvals[idx], dtype=tf.float32)
            # transform features trough eigenvectors/basis vectors
            self.means_ = tf.cast(self.means_, tf.float32)
            x_concat = tf.cast(x_concat, tf.float32)
            z = tf.matmul((x_concat - self.means_), self.eigenvectors_)

            self.means2_ = tf.reduce_mean(z, axis=0)
            self.norms_ = tf.math.sqrt(tf.reduce_mean(z * z, axis=0))
            if self.reversible_vampnet:
                out = z / self.norms_
            else:
                out = z
        else:
            idx = np.argsort(np.abs(eigvals))[::-1]
            # remove the slowest process
            eigenvectors_ = tf.convert_to_tensor(eigvecs[:, idx], dtype=tf.float32)
            eigenvalues_ = tf.convert_to_tensor(eigvals[idx], dtype=tf.float32)

        if not estim_mode:
            with self.tb_writer.as_default(step=self.cycle):
                for i in range(self.dense_out):
                    tf.summary.histogram(
                        "SP/Slow process projected data, process {}".format(i),
                        out[:, i],
                        buckets=200,
                    )
                    tf.summary.histogram(
                        "SP/Slow process ANN output data, dim {}".format(i),
                        x_concat[:, i],
                        buckets=200,
                    )
            percentile = 99.7
            self.srv_upper_ = tfp.stats.percentile(out, percentile, axis=0)
            self.srv_lower_ = tfp.stats.percentile(out, 100 - percentile, axis=0)
            self.most_freq = tf.convert_to_tensor(return_most_freq(out, bins=200))
            self.cycle += 1

        if return_timescales:
            timescales = -tau / tf.math.log(eigenvalues_)
            return timescales

    def project_state(
        self, state, use_projection=True, clip_projection=False, clip_by_value=False
    ):
        if use_projection and (self.reversible_vampnet or self.vampnet):
            # U.T @ X0 is equvilant to X0 @ U, when first dim of data is batch
            p_state = tf.matmul(
                state - self.means_copy.read_value(),
                self.eigenvectors_copy.read_value(),
            )
            if self.reversible_vampnet:
                p_state = p_state / self.norms_copy.read_value()
        else:
            ind = tf.range(0, self.dense_out)
            p_state = tf.gather(state, ind, axis=1)
        # usually tails have a lot of structures that are ANN noise
        # we clip above 99.7% and below 0.3%
        if clip_projection:
            if clip_by_value:
                p_state = tf.clip_by_value(
                    p_state,
                    self.srv_lower_copy.read_value(),
                    self.srv_upper_copy.read_value(),
                )
            else:
                p_state = clip_to_value(
                    p_state,
                    self.srv_lower_copy.read_value(),
                    self.srv_upper_copy.read_value(),
                    self.most_freq,
                )
        return p_state

    @tf.function
    def _train_step(self, inp_tup, obs_mean, obs_var):
        """One train step as the TF 2 graph.
        Arguments:
            inp_tup: Feature matrix which is an tensor
            obs_mean: observation mean, used for standarization
            obs_var: observation var, used for standarization
        Returns:
            Tuple with loss and global norm
        """

        with tf.GradientTape() as tape:
            # standarization of obs
            inp_tup_standarized = self.norm_inputs(
                inp_tup, obs_mean, obs_var, logtrans=self.logtrans, rnd_clip=False
            )
            if not self.autoencoder:
                state_target = self.target_model(inp_tup_standarized, training=False)
            else:
                _, state_target = self.target_model_copy(
                    inp_tup_standarized, training=False
                )
            if self.ae_lagtime > 0:
                state_target = self.project_state(state_target)
            state_predict = self.predictor_model(inp_tup_standarized, training=True)
            regul_loss = tf.reduce_sum(self.predictor_model.losses)
            loss, loss_nm = loss_func(
                state_target, state_predict, regul_loss=regul_loss
            )
            trainable_var = self.predictor_model.trainable_variables
        grads = tape.gradient((loss), trainable_var)
        global_norm = tf.linalg.global_norm(grads)

        grads = self.autoclip(grads)
        self.optimizer.apply_gradients(zip(grads, trainable_var))

        return loss, loss_nm, global_norm

    # TODO: work with (batch, sample_size, features)
    # instead of (sample_size, features)
    # so that eigendecomposition batch dimension > 1
    @tf.function
    def _train_step_vamp(self, shift, back, obs_mean, obs_var):
        """One train step as the TF 2 graph for autoencoder.
        Arguments:
            shift: shifted data
            back: not shifted data,
            obs_mean: observation mean, used for standarization
            obs_var: observation var, used for standarization
        Returns:
            Tuple with loss and global norm
        """
        with tf.GradientTape() as tape:
            # standarization of obs
            shift_standarized = self.norm_inputs(
                shift, obs_mean, obs_var, logtrans=self.logtrans
            )
            back_standarized = self.norm_inputs(
                back, obs_mean, obs_var, logtrans=self.logtrans
            )
            _, state_shift = self.target_model(shift_standarized, training=True)
            state_back, _ = self.target_model(back_standarized, training=True)
            regul_loss = tf.reduce_sum(self.target_model.losses)
            loss = loss_func_vamp(
                state_shift,
                state_back,
                reversible=self.reversible_vampnet,
                nonrev_srv=self.nonrev_srv,
            )
            loss += regul_loss
            trainable_var = self.target_model.trainable_variables
        grads = tape.gradient((loss), trainable_var)
        global_norm = tf.linalg.global_norm(grads)

        grads = self.autoclip_ae(grads)
        self.optimizer_ae.apply_gradients(zip(grads, trainable_var))

        return loss, global_norm

    def add_to_buffer(self, obs, shuffle=False):
        """Adds feature tensors from the observation, to the training buffer.
        Arguments:
            obs: observation sampled from MD simulation, that contains
            `dist_matrix` and `trajectory` keys.
            shuffle: shuffles input with respect to first dimension. It is turned off
                     if lag time is greater than 0.
        Return:
            None
        """
        if not self.train_buffer_size > 0:
            self.train_buffer.clear()
        dist_matrix_org = obs["dist_matrix"]
        # whiten
        # TODO: rewhiten data on a new cov00
        # by unwhitening them, calculating cov00
        # then whitening once again
        # you can just create a correction matrix WW^-1
        # where you accumulate cov00 somewhere in a variable on the unbiased data
        if self.whiten and self.W_whiten is None:
            logger.warning(
                "Data whitening is turned on, it will be based on the first data batch to decorrelate local noise"
            )
            dist_matrix_org, cov, mean = whiten_data(dist_matrix_org)
            self.W_whiten, self.mean_whiten = cov, mean
            dist_matrix_org = dist_matrix_org.numpy()
        elif self.whiten:
            dist_matrix_org, _, _ = whiten_data(
                dist_matrix_org, self.W_whiten, self.mean_whiten
            )
            dist_matrix_org = dist_matrix_org.numpy()
        self.one_simulation_size = dist_matrix_org.shape[0]

        if shuffle and ae_lagtime == 0:
            np.random.shuffle(perm)
            dist_matrix = dist_matrix_org[perm]
        else:
            dist_matrix = dist_matrix_org

        # add training examples to buffer

        self.train_buffer.append(dist_matrix)

    def predict_action(
        self, obs, n=1, random_actions=False, update_stats=False, instance=None
    ):
        """Predicts `n` structures that are least sampled during all the cycles.
        Arguments:
            obs: observation sampled from MD simulation, that contains
            `dist_matrix` and `trajectory` keys.
            n: number of highest rewards observations to return
            random_actions: Instead of maximum reward, use structures with indices picked from a uniform dist.
        Returns:
            A tuple of four variables is returned, first index are `n` molecular structures sorted from the highest
            reward to lowest. The structures correspond to the `trajectory` key in the observation dict. The second
            are feature tensors, sorted the same way as previous, and also correspond to the `dist_matrix` in the
            obs dict. The third are rewards corresponding to the two previous. The last is array of all rewards,
            unsorted.
        """

        # TODO: don't take whole observation, only the features
        dist_matrix = obs["dist_matrix"]
        batch_dist_matrix = tf.convert_to_tensor(dist_matrix)

        batch_reward = self._calc_reward(
            batch_dist_matrix,
            obs["energy"],
            obs["dssp"],
            update_stats=update_stats,
            instance=instance,
        )
        if not batch_reward.shape[0] > 0:
            Exception(
                "Batch reward shape first shape is zero, batch reward: {}".format(
                    batch_reward
                )
            )
        batch_reward_array = batch_reward.numpy()
        # both have to be equal after spliting
        assert batch_reward_array.shape[0] == dist_matrix.shape[0]
        # get p % of top reward
        if self.sorted_energy:
            # topk of reward
            k = self.sorted_energy_constant
            # sort by reward
            indices_reward = np.argpartition(batch_reward_array, -k)[-k:]
            # sort by energy
            energy = -obs["energy"][indices_reward]
            indices_energy = np.argpartition(energy, -n)[-n:]
            indices = indices_reward[indices_energy]
        elif self.sorted_dssp:
            # topk of reward
            percentage = 0.05
            k = max(20, int(batch_reward_array.shape[0] * percentage))
            # sort by reward
            indices_reward = np.argpartition(batch_reward_array, -k)[-k:]
            # sort by dssp
            dssp = obs["dssp"]
            dssp_frac = np.sum(dssp == "C", axis=1) / dssp.shape[-1]
            logger.info(
                "Current average DSSP frac. of unstructured {}".format(
                    np.mean(dssp_frac)
                )
            )
            dssp_frac = -dssp_frac[indices_reward]
            indices_dssp = np.argpartition(dssp_frac, -n)[-n:]
            indices = indices_reward[indices_dssp]

        else:
            # get indices of top n
            # see for more here https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
            indices = np.argpartition(batch_reward_array, -n)[-n:]
        # reverse to have descending order
        # TODO: return index, not whole object
        if not random_actions:
            indices_sorted = indices[np.argsort(batch_reward_array[indices])][::-1]
        else:
            indices_sorted = np.random.randint(0, len(obs["trajectory"]), n)
        # TODO: remove trajectory from the function, too much memory
        actions = [
            obs["trajectory"][ind].reshape((1, *obs["trajectory"][ind].shape))
            for ind in indices_sorted
        ]
        dist_matrixs = [
            obs["dist_matrix"][ind].reshape((1, *obs["dist_matrix"][ind].shape))
            for ind in indices_sorted
        ]

        energy_matrix = [
            obs["energy"][ind].reshape((1, *obs["energy"][ind].shape))
            for ind in indices_sorted
        ]

        box_list = [
            obs["box"][ind].reshape((1, *obs["box"][ind].shape))
            for ind in indices_sorted
        ]

        if obs["dssp"] is not None:
            dssp_matrix = [
                obs["dssp"][ind].reshape((1, *obs["dssp"][ind].shape))
                for ind in indices_sorted
            ]
        else:
            dssp_matrix = None

        return (
            actions,
            dist_matrixs,
            batch_reward_array[indices_sorted],
            batch_reward_array,
            energy_matrix,
            dssp_matrix,
            box_list,
        )

    def _calc_reward(self, inp_tup, energy, dssp, update_stats=False, instance=None):
        """Calculates reward based on the feature tensor provided in the input.
        Arguments:
            inp_tup: Feature tensor
            energy: array with potential energy
            dssp: array with dssp decomposition of the protein
        Returns:
            a tensor of minibatch size.
        """
        # data here come from different sourec than training buffer
        # they need to undergo all the preprocessing as to the training buffer
        if self.whiten:
            inp_tup, _, _ = whiten_data(inp_tup, self.W_whiten, self.mean_whiten)
        obs_mean = self.obs_mean_copy
        obs_var = self.obs_var_copy
        inp_tup_standarized = self.norm_inputs(
            inp_tup, obs_mean, obs_var, logtrans=self.logtrans, rnd_clip=False
        )
        if not self.autoencoder:
            state_target = self.target_model(inp_tup_standarized, training=False)
        else:
            _, state_target = self.target_model_copy(
                inp_tup_standarized, training=False
            )
            if self.ae_lagtime > 0:
                state_target = self.project_state(state_target)
        assert tf.rank(inp_tup_standarized) == 2
        state_predict = self.predictor_model(inp_tup_standarized, training=False)
        tf.debugging.check_numerics(
            state_target, "Results are NaN for state target", name=None
        )
        tf.debugging.check_numerics(
            state_predict, "Results are NaN for state predict", name=None
        )

        loss, loss_nm = loss_func(state_target, state_predict)
        if instance is not None:
            with self.tb_writer.as_default(self.cycle):
                for i in range(loss_nm.shape[-1]):
                    tf.summary.histogram(
                        "rewardrnd/Agent {0} reward before standarization, for each error component {1}".format(
                            instance, i
                        ),
                        loss_nm[:, i],
                        buckets=200,
                    )
                for i in range(loss_nm.shape[-1]):
                    tf.summary.histogram(
                        "rewardrnd/Agent {0} variance standarized reward before standarization, for each error component {1}".format(
                            instance, i
                        ),
                        loss_nm[:, i] / np.var(loss_nm[:, i]),
                        buckets=200,
                    )

        # loss standarized
        # we substract mean, not like in the paper
        if not self.reward_stand:
            loss_var = 1.0
            loss_standarized = loss_nm.numpy()
        else:
            if update_stats:
                self.reward_robscaler.fit(loss_nm.numpy())
                if self.include_energy:
                    self.energy_robscaler.fit(energy)

            loss_standarized = self.reward_robscaler.transform(loss_nm.numpy())
        # weight the rewards with respect to the slow process weights
        if loss_standarized.shape[-1] != self.slowp_vector.shape[-1]:
            raise ValueError(
                "slowp_vector dimension has to be the same as number of slow process outputs"
            )

        if instance is not None:
            with self.tb_writer.as_default(self.cycle):
                for i in range(loss_standarized.shape[-1]):
                    tf.summary.histogram(
                        "rewardrnd/Agent {0} reward before scaling, for each error component {1}".format(
                            instance, i
                        ),
                        loss_standarized[:, i],
                        buckets=200,
                    )
                for i in range(loss_standarized.shape[-1]):
                    tf.summary.histogram(
                        "rewardrnd/Agent {0} variance standarized reward before scaling, for each error component {1}".format(
                            instance, i
                        ),
                        loss_standarized[:, i] / np.var(loss_standarized[:, i]),
                        buckets=200,
                    )
        if not self.slowp_kinetic_like_scaling:
            loss_standarized = loss_standarized * self.slowp_vector
        else:
            loss_standarized = loss_standarized * self.eigenvalues_copy

        if instance is not None:
            with self.tb_writer.as_default(self.cycle):
                for i in range(loss_standarized.shape[-1]):
                    tf.summary.histogram(
                        "rewardrnd/Agent {0} reward after scaling for each error component {1}".format(
                            instance, i
                        ),
                        loss_standarized[:, i],
                        buckets=200,
                    )

                    tf.summary.histogram(
                        "rewardrnd/Agent {0} variance standarized reward after scaling for each error component {1}".format(
                            instance, i
                        ),
                        loss_standarized[:, i] / np.var(loss_standarized[:, i]),
                        buckets=200,
                    )

        mean_loss_standarized = np.mean(loss_standarized, axis=-1)
        if update_stats:
            reward_copy = self.max_reward
            self.max_reward = np.max(mean_loss_standarized)
            if self.max_reward_prev is None:
                self.max_reward_prev = self.max_reward
            else:
                self.max_reward_prev = reward_copy
        if self.include_energy:
            energy = energy.reshape(-1, 1)
            # make lowest energy positive
            energy = -energy[0]

            energy_reward_standarized = self.energy_robscaler.transform(energy)
            # mean of unstandarized loss
            min_energy = -np.min(energy_reward_standarized)
            c = self.energy_continuous_constant
            if self.max_reward > 0:
                coef = self.max_reward / (self.max_reward + c * min_energy)
            else:
                coef = 0
            logger.debug(
                "Current curiosity_reward-energy coefficient is: {}".format(coef)
            )
            energy_reward_standarized = coef * energy_reward_standarized.squeeze()
            slowp_vector_coef = np.sum(self.slowp_vector) / self.slowp_vector.shape[0]
            mean_loss_standarized = 0.5 * (
                mean_loss_standarized + slowp_vector_coef * energy_reward_standarized
            )
        if self.dssp_reward and dssp is not None:
            # count only unstructured
            dssp_frac = np.sum(dssp == "C", axis=1) / dssp.shape[-1]
            logger.debug(
                "Current average DSSP frac. of unstructured {}".format(
                    np.mean(dssp_frac)
                )
            )
            # panelize unstructured
            dssp_frac = -dssp_frac.reshape(-1, 1)
            if update_stats:
                self.dssp_robscaler.fit(dssp_frac)

            c = 0.5
            if self.max_reward - self.max_reward_prev > 0:
                coef = 2 * (
                    1 / (np.exp(-c * (self.max_reward - self.max_reward_prev)) + 1)
                    - 0.5
                )
            else:
                coef = 0
            logger.debug(
                "Current curiosity_reward-dssp coefficient is: {}".format(coef)
            )
            dssp_frac_standarized = self.dssp_robscaler.transform(dssp_frac).squeeze()
            mean_loss_standarized = 0.5 * (
                mean_loss_standarized + coef * dssp_frac_standarized
            )

        if instance is not None:
            with self.tb_writer.as_default(self.cycle):
                tf.summary.histogram(
                    "rewardrnd/Agent {0} reward after scaling, averaged over all components".format(
                        instance
                    ),
                    np.mean(loss_standarized, axis=-1),
                    buckets=200,
                )

                tf.summary.histogram(
                    "rewardrnd/Agent {0} variance standarized reward after scaling and averaging over all components".format(
                        instance
                    ),
                    np.mean(
                        loss_standarized / np.var(loss_standarized, axis=0), axis=-1
                    ),
                    buckets=200,
                )

                for i in range(state_target.shape[-1]):
                    tf.summary.histogram(
                        "actiontargetoutput/Agent {0} state target for component {1}".format(
                            instance, i
                        ),
                        state_target[:, i],
                        buckets=200,
                    )
                    tf.summary.histogram(
                        "actiontargetoutput/Agent {0} state target for component {1} for the highest reward".format(
                            instance, i
                        ),
                        state_target[np.argmax(mean_loss_standarized), i],
                        buckets=1,
                    )

        return tf.convert_to_tensor(mean_loss_standarized)

    def get_reward_mean_variance(self):
        """Returns online variance of reward, that is used to
        standarize reward.

        """
        mean, var = self.reward_stat.current_mean_var()
        return mean.numpy(), var.numpy()

    def get_state_mean_variance(self):
        """Return mean and variance, that is used to normalize
        state/observation.
        """
        mean, var = self.obs_stat.get_mean_and_var()
        return mean.numpy(), var.numpy()

    def get_implied_time_scales(self):
        """Returns implied time scales if available"""
        return self.implied_time_array

    def calc_input_latent(self, inp_tup):
        obs_mean = self.obs_mean_copy
        obs_var = self.obs_var_copy
        inp_tup_standarized = self.norm_inputs(
            inp_tup, obs_mean, obs_var, logtrans=self.logtrans
        )
        if not self.autoencoder:
            state_target = self.target_model(inp_tup_standarized, training=False)
        else:
            _, state_target = self.target_model_copy(
                inp_tup_standarized, training=False
            )
            if self.ae_lagtime > 0:
                state_target = self.project_state(state_target)
            else:
                state_target = state_target
        return state_target

    def train_last_n_batch(self, n=None, iter=0):
        """
        Trains last n example from training buffer.
        Arguments:
            n: should be negative and indicate position in the
               buffer, e.g. -1, -2 ... -N.
        """
        if n >= 0:
            raise ValueError
        batch_example = self.train_buffer[n]
        batch_dist_matrix = tf.convert_to_tensor(batch_example, dtype=tf.float32)
        assert tf.rank(batch_dist_matrix) == 2
        for i in range(iter):
            batch_loss, batch_loss_nm, g_norm = self._train_step(
                batch_dist_matrix, self.obs_mean_copy, self.obs_var_copy
            )

    def batch_shuffle_aftern(
        self, current_batch, shuffle=True, n=3, last_batch=False, remove_last_batch=True
    ):
        """Shuffles and divides into batches n stored batches.
        Works by calling the function n times, when number of calls
        reaches n, then shuffled/divided data are returned
        """
        if self.n_batch_stored == 0 and not last_batch:
            self.batch_store_buffer = []
            self.batch_store_buffer.append(current_batch)
            self.n_batch_stored += 1
            return None
        # skip one, because the last on batch should return batches
        elif self.n_batch_stored + 1 < n and not last_batch:
            self.batch_store_buffer.append(current_batch)
            self.n_batch_stored += 1
            return None
        else:
            self.n_batch_stored = 0
            if last_batch and len(self.batch_store_buffer) < 1:
                self.batch_store_buffer = []
                self.batch_store_buffer.append(current_batch)
            else:
                # the last one iteration accepts and returns batches
                self.batch_store_buffer.append(current_batch)

            if shuffle:
                self.batch_store_buffer = np.concatenate(self.batch_store_buffer)
                np.random.shuffle(self.batch_store_buffer)
                self.batch_store_buffer = np.split(
                    self.batch_store_buffer,
                    range(
                        self.minibatch_size_cur,
                        self.batch_store_buffer.shape[0],
                        self.minibatch_size_cur,
                    ),
                )
            # Change batch size to the one chosen by user
            else:
                self.batch_store_buffer = np.concatenate(self.batch_store_buffer)
                self.batch_store_buffer = np.split(
                    self.batch_store_buffer,
                    range(
                        self.minibatch_size_cur,
                        self.batch_store_buffer.shape[0],
                        self.minibatch_size_cur,
                    ),
                )
            buff_copy = deepcopy(self.batch_store_buffer)
            if (
                remove_last_batch and buff_copy[-1].shape[0] < self.minibatch_size_cur
            ) and buff_copy[0].shape[0] >= self.minibatch_size_cur:
                del buff_copy[-1]
            self.batch_store_buffer = []
            return buff_copy

    def batch_ae_shuffle_aftern(
        self, current_batch, shuffle=True, n=3, last_batch=False, lagtime=None
    ):
        """Shuffles and divides into batches n stored batches.
        Works by calling the function n times, when number of calls
        reaches n, then shuffled/divided data are returned
        """
        if self.n_batch_ae_stored == 0 and not last_batch:
            self.batch_ae_store_buffer = []
            self.batch_ae_store_buffer.append(current_batch)
            self.n_batch_ae_stored += 1
            return None
        # skip one, because the last on batch should return batches
        elif self.n_batch_ae_stored + 1 < n and not last_batch:
            self.batch_ae_store_buffer.append(current_batch)
            self.n_batch_ae_stored += 1
            return None
        else:
            self.n_batch_ae_stored = 0
            if last_batch and len(self.batch_ae_store_buffer) < 1:
                self.batch_ae_store_buffer = []
                self.batch_ae_store_buffer.append(current_batch)
            else:
                # the last one iteration accepts and returns batches
                self.batch_ae_store_buffer.append(current_batch)

            # split onto shift, back
            # every batch (That is alligned with simulation time)
            # is shifted by tau bach and forth (shift and back var.)
            shift = [batch[lagtime:] for batch in self.batch_ae_store_buffer]
            back = [batch[:-lagtime] for batch in self.batch_ae_store_buffer]
            self.batch_ae_store_buffer.clear()
            batch_size = self.minibatch_size_ae
            shift = np.concatenate(shift, axis=0)
            back = np.concatenate(back, axis=0)
            # generate indices
            indices = np.arange(shift.shape[0])
            if shuffle:
                np.random.shuffle(indices)
            # shuffle pairs of zt0 ztt
            shift = shift[indices]
            back = back[indices]
            shift = np.split(
                shift,
                range(
                    batch_size,
                    shift.shape[0],
                    batch_size,
                ),
            )
            back = np.split(
                back,
                range(
                    batch_size,
                    back.shape[0],
                    batch_size,
                ),
            )

            # remove samples smaller than batch size
            if shift[-1].shape[0] != batch_size:
                del shift[-1]
                del back[-1]

            if len(shift) == 0:
                return None

            return zip(shift, back)

    def stop_loss(self, epoch_loss=None, prev_loss=None, tries=5, treshhold=0.05):
        if epoch_loss is None or prev_loss is None:
            return None
        # relative decrease, negative when decrease, positive when increase
        rel = abs((epoch_loss - prev_loss) / (epoch_loss + prev_loss)) * np.sign(
            epoch_loss - prev_loss
        )
        # when decrease is lower than threshold, add +1
        if abs(rel) <= treshhold and rel < 0:
            self.loss_count += 1
        # When increase is lower than threshold, do add +1
        elif abs(rel) <= treshhold and rel > 0:
            self.loss_count += 1
        # when it increases or increases too much, reset
        else:
            self.loss_count = 0

        if self.loss_count >= tries:
            status = True
        else:
            status = False

        return status
