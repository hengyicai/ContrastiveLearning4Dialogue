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
from projects.contrastive_learning.param import DEFAULT_OVERRIDE, DEFAULT_PARAMS, add_cl_cmdline_args

PARLAI_HOME = os.getenv('PARLAI_HOME')

OVERRIDE = {
    "batchsize": 64,
    "eval_batchsize": 64,
    "validation_every_n_secs": 1200,
    "validation_every_n_epochs": -1,
    "learningrate": 5e-4,
    # CL Training
    "ref_model_update_freq": 100,
    "pretrain_steps": 100,
    "compute_tokenized_bleu": True,
    # Prevent NaN issues
    # "gradient_clip": 10,
    # "weight_decay": 1e-6,
    # "ref_model_file": os.path.join(PARLAI_HOME, "models/contrastive_learning/transformer/gpu-154-36-v100_GPU1/douban_ref_update_12000:pretrain_12000:num_epochs_50:sample_k_5:contrast_by_context:periodical_replacement_False:naive_neg_sampling_True"),
}

if __name__ == '__main__':
    parser = setup_args()
    parser = add_cl_cmdline_args(parser)

    parser.set_defaults(**DEFAULT_PARAMS)
    parser.set_defaults(**DEFAULT_OVERRIDE)
    parser.set_defaults(**OVERRIDE)

    parser.set_defaults(
        task='douban',
        model='parlai.agents.contrastive_learning.transformer:CLTransformerAgent',
        model_file=os.path.join(PARLAI_HOME,
                                'models/contrastive_learning/debug_cl_transformer/douban'),
        n_layers=6,
        n_heads=8,
        ffn_size=2048,
        embedding_size=512,
        n_positions=128,
        optimizer='adam',
        clip_norm=0.1,
        betas="0.9,0.98",
        warmup_updates=8000,
        clip=0.1,
        lr_scheduler='invsqrt',
        embedding_type='glove',
        attention_dropout=0.1,
        relu_dropout=0.1,
        learn_positional_embeddings=True,
        variant='xlm',
        activation='gelu',
    )

    opt = parser.parse_args()

    opt = override_opt(opt, DEFAULT_OVERRIDE)
    opt = override_opt(opt, OVERRIDE)

    if opt.get('multigpu', False):
        port = random.randint(32000, 48000)
        launch_and_train(opt, port)
    else:
        TrainLoop(opt).train()
