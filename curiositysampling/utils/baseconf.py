DEFAULT_OMM_SETTINGS = {
    "positions": None,
    "system": None,
    "topology": None,
    "integrator": None,
    "temperature": 300,
    "steps": 25000,
    "reporter_stride": 100,
    "file_per_cycle": True,
    "warmup_cycles": 0,
    "warmup_steps": None,
    "warmup_reporter_stride": 0,
    "regular_md": False,
    "saving_frequency": 100,
    "cuda": False,
    "hip": False,
    "use_positions": True,
    "use_distances": False,
    "use_dihedrals": False,
    "selection": "protein and type N C O S",
    "boosting": True,
    "boosting_temp": 200,
    "position_list": None,
    "box_list": None,
    "boosting_steps": 5000,
    "gpus_per_agent": None,
    "cpus_per_agent": None,
}

DEFAULT_RND_SETTINGS = {
    "model": {
        "target": {
            "dense_units": [16, 32, 64],
            "dense_activ": "mish",
            "lcnn": False,
            "dense_layernorm": False,
            "dense_batchnorm": False,
            "dense_layernorm": False,
            "input_batchnorm": False,
            "dense_out": 2,
            "gaussian_dropout": 0.0,
            "dense_out_activ": "linear",
            "layernorm_out": False,
            "initializer": "lecun_normal",
            "spectral": False,
            "orthonormal": False,
            "l1_reg": 0.0001,
            "l1_reg_activ": 0.0000,
            "l2_reg": 0.0001,
            "l2_reg_activ": 0.0000,
            "unit_constraint": False,
            "cnn": True,
            "strides": [1, 1, 1],
            "kernel_size": [3, 3, 1],
            "padding": "same",
            "skip_after_cnn": True,
            "split_slow": False,
            "anglecomparison": False,
        },
        "predictor": {
            "dense_units": [16, 32, 64],
            "dense_activ": "mish",
            "dcnn": False,
            "dense_batchnorm": False,
            "dense_layernorm": False,
            "input_batchnorm": False,
            "dense_out": 2,
            "dense_out_activ": "linear",
            "layernorm_out": False,
            "initializer": "lecun_normal",
            "spectral": False,
            "orthonormal": False,
            "l1_reg": 0.0001,
            "l1_reg_activ": 0.0000,
            "l2_reg": 0.0001,
            "l2_reg_activ": 0.0000,
            "unit_constraint": False,
            "cnn": True,
            "strides": [1, 1, 1],
            "kernel_size": [3, 3, 1],
            "padding": "same",
            "skip_after_cnn": True,
            "memory_units": False,
        },
    },
    "vampnet": True,
    "nonrev_srv": False,
    "reversible_vampnet": False,
    "autoencoder": True,
    "mae": False,
    "autoencoder_lagtime": 50,
    "minibatch_size_cur": 200,
    "minibatch_size_ae": 2000,
    "reinitialize_predictor_network": True,
    "clip_by_global_norm": False,
    "num_of_train_updates": 500,
    "num_of_ae_train_updates": 500,
    "learning_rate_cur": 0.01,
    "learning_rate_ae": 0.001,
    "clr": False,
    "obs_stand": False,
    "reward_stand": False,
    "train_buffer_size": 1000000,
    "optimizer": "adab",
    "optimizer_ae": "adab",
    "target_network_update_freq": 1,
    "hard_momentum": True,
    "vamp2_metric": True,
    "slowp_vector": [1.0, 1.0],
    "classic_loss": False,
    "whiten": False,
    "logtrans": False,
    "shrinkage": 0.0,
    "energy_mode": None,
    "energy_continuous_constant": 25,
    "protein_cnn": True,
    "slowp_kinetic_like_scaling": False,
    "timescale_mode_target_network_update": False,
}
