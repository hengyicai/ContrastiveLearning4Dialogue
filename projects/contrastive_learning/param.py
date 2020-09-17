DEFAULT_OVERRIDE = {
    "datatype": 'train',
    "max_train_time": -1,
    "learningrate": 5e-4,
    "dropout": 0.2,
    "gradient_clip": 1,
    "batch_sort": True,
    "validation_every_n_secs": 1200,
    "validation_every_n_epochs": -1,
    "validation_metric": 'ppl',
    "validation_metric_mode": 'min',
    "validation_patience": 5,
    "log_every_n_secs": 1,
    "shuffle": True,
    "numworkers": 40,
    "multigpu": False,
    "num_epochs": 50,
    "display_examples": False,
    "history_size": -1,
    "text_truncate": 120,
    "label_truncate": 40,
    "beam_size": 1,
    "inference": 'greedy',  # greedy, beam, topk, nucleus
    "topp": 0.3,
}

DEFAULT_PARAMS = {
    "dict_lower": True,
    "dict_minfreq": -1,
    "embeddingsize": 300,
    "no_cuda": False,
    "dict_maxtokens": 40000,
    "split_lines": False,
    "delimiter": '__EOT__',
    "tensorboard_log": True,
    "save_after_valid": False,
}


def add_cl_cmdline_args(argparse):
    argparse.add_argument(
        '--ref_model_update_freq',
        type=int,
        default=3000,
        help='training steps to update the reference model',
    )
    argparse.add_argument(
        '--pretrain_steps',
        type=int,
        default=3000,
    )
    argparse.add_argument(
        '--contrast_by',
        choices=['context', 'response', 'both'],
        default='both',
    )
    argparse.add_argument(
        '--sample_k',
        type=int,
        default=5,
    )
    argparse.add_argument(
        '--periodical_replacement',
        type='bool',
        default=True
    )
    argparse.add_argument(
        '--naive_neg_sampling',
        type='bool',
        default=False,
    )
    argparse.add_argument(
        '--ref_model_file',
        type=str,
        default=None,
    )
    argparse.add_argument(
        '--cl_threshold',
        type=float,
        default=0.5
    )
    argparse.add_argument(
        '--soft_normalize_score',
        type='bool',
        default=True,
    )
    argparse.add_argument(
        '--filter_normalize_score',
        type='bool',
        default=False
    )
    argparse.add_argument(
        '--neg_threshold',
        type=float,
        default=0.5,
    )
    argparse.add_argument(
        '--pos_threshold',
        type=float,
        default=0.5,
    )
    argparse.add_argument(
        '--cl_anneal',
        type='bool',
        default=True
    )
    argparse.add_argument(
        '--anneal_speed',
        type=float,
        default=1.0
    )
    argparse.add_argument(
        '--nll_w',
        type=float,
        default=0.5,
    )
    argparse.add_argument(
        '--use_eval_ref_agent',
        type='bool',
        default=True,
    )
    argparse.add_argument(
        '--cl_loss_per_token',
        type='bool',
        default=False
    )
    argparse.add_argument(
        '--only_pos',
        type='bool',
        default=False
    )
    argparse.add_argument(
        '--only_neg',
        type='bool',
        default=False,
    )
    argparse.add_argument(
        '--eval_naive_neg_sampling',
        type='bool',
        default=False
    )
    argparse.add_argument(
        '--eval_naive_neg_sampling_k',
        type=int,
        default=1,
    )

    return argparse
