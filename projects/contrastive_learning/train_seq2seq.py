#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train model for ppl metric with pre-selected parameters.
These parameters have some variance in their final perplexity, but they were
used to achieve the pre-trained model.
"""
import os
import random

from parlai.agents.hy_lib.common_utils import override_opt
from parlai.scripts.multiprocessing_train import setup_args, launch_and_train
from parlai.scripts.train_model import TrainLoop
from projects.contrastive_learning.param import (
    DEFAULT_OVERRIDE,
    DEFAULT_PARAMS,
    add_cl_cmdline_args
)

PARLAI_HOME = os.getenv('PARLAI_HOME')

OVERRIDE = {
    "batchsize": 128,
    "eval_batchsize": 128,
    "validation_every_n_secs": 120,
    "validation_every_n_epochs": -1,
    "learningrate": 0.001,
    "compute_tokenized_bleu": True,
    # CL Training (For debugging)
    "ref_model_update_freq": 30,
    "pretrain_steps": 30,
    "ref_model_file": os.path.join(
        PARLAI_HOME,
        'models/contrastive_learning/seq2seq/baseline_seq2seq/gpu-154-36-v100_GPU0/personachat_extend'
    )
}

if __name__ == '__main__':
    parser = setup_args()
    parser = add_cl_cmdline_args(parser)

    parser.set_defaults(**DEFAULT_PARAMS)
    parser.set_defaults(**DEFAULT_OVERRIDE)
    parser.set_defaults(**OVERRIDE)

    parser.set_defaults(
        task='personachat_extend',
        model='parlai.agents.contrastive_learning.seq2seq:CLSeq2seqAgent',
        model_file=os.path.join(
            PARLAI_HOME, 'models/contrastive_learning/tmp/personachat_extend'
        ),
        hiddensize=256,
        attention='general',
        attention_time='post',
        numlayers=2,
        rnn_class='lstm',
        lookuptable='enc_dec',
        optimizer='adam',
        weight_decay=0,
        embedding_type='glove',
        momentum=0.95,
        bidirectional=True,
        numsoftmax=1,
        lr_scheduler='reduceonplateau',
        lr_scheduler_patience=3,
        lr_scheduler_decay=0.8,
        warmup_updates=-1,
    )

    opt = parser.parse_args()

    opt = override_opt(opt, DEFAULT_OVERRIDE)
    opt = override_opt(opt, OVERRIDE)

    if opt.get('multigpu', False):
        port = random.randint(32000, 48000)
        launch_and_train(opt, port)
    else:
        TrainLoop(opt).train()
