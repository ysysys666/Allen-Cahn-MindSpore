import time

import numpy as np

import mindspore
from mindspore import context, nn, ops, Tensor, jit, set_seed
from mindspore import dtype as mstype
from mindspore import load_checkpoint, load_param_into_net
from mindflow.pde import PDEWithLoss
from sympy import diff, symbols, Function

from mindflow.loss import get_loss_metric

from mindflow.pde import sympy_to_mindspore
from src import MultiScaleFCSequentialOutputTransform
from mindflow.utils import load_yaml_config

from src import create_training_dataset, create_test_dataset, visual, calculate_l2_error

set_seed(123456)
np.random.seed(123456)

# set context for training: using graph mode for high performance training with GPU acceleration
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)
use_ascend = context.get_context(attr_key='device_target') == "Ascend"

# load configuration
config = load_yaml_config('./configs/allen_cahn_cfg.yaml')

# create training dataset
ac_train_dataset = create_training_dataset(config)
train_dataset = ac_train_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                shuffle=True,
                                                prebatched_data=True,
                                                drop_remainder=True)
# create test dataset
inputs, label = create_test_dataset(config["test_dataset_path"])

# define models and optimizers
model = MultiScaleFCSequentialOutputTransform(in_channels=config["model"]["in_channels"],
                                              out_channels=config["model"]["out_channels"],
                                              layers=config["model"]["layers"],
                                              neurons=config["model"]["neurons"],
                                              residual=config["model"]["residual"],
                                              act=config["model"]["activation"],
                                              num_scales=1)
if config["load_ckpt"]:
    param_dict = load_checkpoint(config["load_ckpt_path"])
    load_param_into_net(model, param_dict)
    
epochs = config["train_epochs"]
visual(model, epochs=epochs, resolution=config["visual_resolution"])