#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

from parlai.agents.hy_lib.common_utils import override_opt
from parlai.scripts.train_model import setup_args, TrainLoop
from projects.contrastive_learning.param import (
    DEFAULT_OVERRIDE,
    DEFAULT_PARAMS,
    add_cl_cmdline_args
)

PARLAI_HOME = os.getenv('PARLAI_HOME')

OVERRIDE = {
    "batchsize": 30,
    "validation_every_n_secs": 600,
    "validation_every_n_epochs": -1,
    "batch_sort_field": 'label',
    # CL Training
    "ref_model_update_freq": 30,
    "pretrain_steps": 30,
    "compute_tokenized_bleu": True,
    # "ref_model_file": os.path.join(
    #     PARLAI_HOME,
    #     'models/contrastive_learning/hran/host-172-20-189-131_GPU3/main_exp_v1/personachat_extend_lr_0.001'
    # )
}

if __name__ == '__main__':
    parser = setup_args()
    parser = add_cl_cmdline_args(parser)

    parser.set_defaults(**DEFAULT_PARAMS)
    parser.set_defaults(**DEFAULT_OVERRIDE)
    parser.set_defaults(**OVERRIDE)

    parser.set_defaults(
        task='personachat_extend',
        model='parlai.agents.contrastive_learning.dialog_wae:CLHredAgent',
        model_file=os.path.join(PARLAI_HOME,
                                'models/contrastive_learning/debug_cross_ref/personachat_extend'),
        rnn_class='lstm',
        hiddensize=256,
        attention='none',
        numlayers=2,
        dict_lower=True,
        dict_tokenizer='split',
        embedding_type=os.path.join(PARLAI_HOME,
                                    'data/PersonaChatExtend/personachat_extend.embed.vec'),
        eval_embedding_type=os.path.join(PARLAI_HOME,
                                         'data/PersonaChatExtend/personachat_extend.embed.vec'),
        skip_generation=False,
        split_lines=True,
        person_tokens=True,
        delimiter='__EOT__',
        hred=True,
        vhred=False,
        optimizer='adam',
        learningrate=0.001,
        momentum=0.95,
        nesterov=True,
        gradient_clip=1.0,
        lr_scheduler="reduceonplateau",
        lr_scheduler_decay=0.8,
        lr_scheduler_patience=3,
        tensorboard_log=True,
    )

    opt = parser.parse_args()

    opt = override_opt(opt, DEFAULT_OVERRIDE)
    opt = override_opt(opt, OVERRIDE)

    TrainLoop(opt).train()
