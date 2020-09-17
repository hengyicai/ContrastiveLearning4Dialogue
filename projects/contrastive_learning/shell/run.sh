#!/usr/bin/env bash

set -e
set -x

FLAG=main_exp_v1

declare -A model_arr=(
  ["cl_seq2seq"]="parlai.agents.contrastive_learning.seq2seq:CLSeq2seqAgent"
  ["seq2seq"]="parlai.agents.contrastive_learning.seq2seq:OrigSeq2seqAgent"
  ["cl_transformer"]="parlai.agents.contrastive_learning.transformer:CLTransformerAgent"
  ["transformer"]="parlai.agents.contrastive_learning.transformer:OrigTransformerAgent"
  ["hred"]="parlai.agents.contrastive_learning.dialog_wae:OrigDialogWaeAgent"
  ["cl_hred"]="parlai.agents.contrastive_learning.dialog_wae:CLHredAgent"
  ["hran"]="parlai.agents.contrastive_learning.dialog_wae:OrigDialogWaeAgent"
  ["cl_hran"]="parlai.agents.contrastive_learning.dialog_wae:CLHredAgent"
)

declare -A ref_model_files=(
  ["none"]=None
  ["personachat_extend_seq2seq"]="${PARLAI_HOME}/models/contrastive_learning/seq2seq/baseline"
)

declare -A init_model_files=(
  ["none"]=None
)

declare -A bszs=(
  ["cl_seq2seq"]=128
  ["seq2seq"]=128
  ["cl_transformer"]=128
  ["transformer"]=128
  ["hred"]=128
  ["cl_hred"]=128
  ["hran"]=128
  ["cl_hran"]=128
)

declare -A topps=(
  ["cl_seq2seq"]=0.1
  ["seq2seq"]=0.1
  ["cl_transformer"]=0.1
  ["transformer"]=0.1
  ["hred"]=0.1
  ["cl_hred"]=0.1
  ["hran"]=0.1
  ["cl_hran"]=0.1
)

declare -A lrs=(
  ["transformer"]=1e-3 # adam
  ["seq2seq"]=1e-3 # adam
  ["hred"]=1e-3 # adam
  ["hran"]=1e-3 # adam
)

declare -A optims=(
  ["transformer"]=adam
  ["seq2seq"]=adam
  ["hred"]=adam
  ["hran"]=adam
)

declare -A task_dirs=(
  ["douban"]=${PARLAI_HOME}/data/DoubanConversaionCorpus
  ["personachat_extend"]=${PARLAI_HOME}/data/PersonaChatExtend
  ["opensub_extend"]=${PARLAI_HOME}/data/OpenSubExtend
)

# CL training args
ref_model=personachat_extend_seq2seq
init_model=none
ref_model_update_freq=6000
periodical_replacement=False

# Transformer args
n_layers=6
n_heads=8
learn_positional_embeddings=True
warmup_updates=8000

# ParlAI args
num_epochs=50
patience=5
valid_every_epochs=0.5

function common_args() {
  echo "--validation_metric_mode min " \
    "--validation_patience ${patience} " \
    "--validation_every_n_secs -1 " \
    "--validation_every_n_epochs ${valid_every_epochs} " \
    "--num_epochs ${num_epochs} " \
    "--tensorboard_log True " \
    "--ref_model_update_freq ${ref_model_update_freq} " \
    "--periodical_replacement ${periodical_replacement} " \
    "--init_model ${init_model_files[$init_model]} " \
    "--ref_model_file ${ref_model_files[$ref_model]} ""$1"
}

function train_model() {
  local model_=$1
  local to_minimize=$2
  local task=$3
  local pretrain_steps=$4
  local sample_k=$5
  local contrast_by=$6
  local naive_neg_sampling=$7
  local cl_threshold=$8
  local cl_anneal=$9
  local anneal_speed=${10}

  local model_name=${model_arr[$model_]}
  local model_dir

  model_dir=${PARLAI_HOME}/models/contrastive_learning/${model_}/$(hostname)_GPU${CUDA_VISIBLE_DEVICES}/${FLAG}

  if [[ ! -d "$model_dir" ]]; then
    mkdir -p "${model_dir}"
  fi

  local cl_tag
  local emb=${task_dirs[$task]}/${task}.embed.vec
  local train_args
  local train_script
  local lr
  local optim

  train_args=$(common_args " --task ${task} --model ${model_name} ")
  train_args+=" --batchsize ${bszs[$model_]} --eval_batchsize ${bszs[$model_]} "
  train_args+=" --topp ${topps[$model_]} "
  train_args+=" --validation_metric ${to_minimize} "
  train_args+=" --eval_embedding_type ${emb} --embedding_type ${emb} "

  if [[ "${model_}" == *"seq2seq" ]]; then
    train_script=train_seq2seq.py
    lr=${lrs["seq2seq"]}
    optim=${optims["seq2seq"]}
  elif [[ "${model_}" == *"hred" ]]; then
    train_script=train_hred.py
    lr=${lrs["hred"]}
    optim=${optims["hred"]}
    train_args+=" --hred True "
    train_args+=" --vhred False "
  elif [[ "${model_}" == *"hran" ]]; then
    train_script=train_hred.py
    lr=${lrs["hran"]}
    optim=${optims["hran"]}
    train_args+=" --hred True "
    train_args+=" --vhred False "
    train_args+=" --attention general "
  elif [[ "${model_}" == *"transformer" ]]; then
    train_script=train_transformer.py
    lr=${lrs["transformer"]}
    optim=${optims["transformer"]}
    train_args+=" --n_layers ${n_layers} "
    train_args+=" --n_heads ${n_heads} "
    train_args+=" --learn_positional_embeddings ${learn_positional_embeddings} "
    train_args+=" --warmup_updates ${warmup_updates} "
  fi

  train_args+=" --learningrate ${lr} "
  train_args+=" --optimizer ${optim} "
  train_args+=" --max_train_time -1 "

  local model_file=${model_dir}/${task}

  cl_tag=ref_update_${ref_model_update_freq}
  cl_tag+=:pretrain_${pretrain_steps}
  cl_tag+=:sample_k_${sample_k}
  cl_tag+=:contrast_by_${contrast_by}
  cl_tag+=:periodical_replacement_${periodical_replacement}
  cl_tag+=:naive_neg_sampling_${naive_neg_sampling}
  cl_tag+=:ref_model_${ref_model}
  cl_tag+=:cl_threshold_${cl_threshold}
  cl_tag+=:cl_anneal_${cl_anneal}
  cl_tag+=:anneal_speed_${anneal_speed}

  #if [[ "${model_}" == "cl_"* ]]; then
  #  model_file=${model_file}_${cl_tag}
  #fi
  # Disable this to prevent too long filename issue in Linux

  train_args+=" --model_file ${model_file}"
  train_args+=" --pretrain_steps ${pretrain_steps}"
  train_args+=" --sample_k ${sample_k}"
  train_args+=" --contrast_by ${contrast_by}"
  train_args+=" --naive_neg_sampling ${naive_neg_sampling}"
  train_args+=" --cl_threshold ${cl_threshold}"
  train_args+=" --cl_anneal ${cl_anneal}"
  train_args+=" --anneal_speed ${anneal_speed}"

  cd "${PARLAI_HOME}"

  nohup python ./projects/contrastive_learning/${train_script} ${train_args} >>${model_file}.log 2>&1 &

  cd -
}

# MODEL_NAME TO_MINIMIZE TASK PRETRAIN_STEPS SAMPLE_K CONTRAST_BY NAIVE_NEG_SAMPLING CL_THRESHOLD CL_ANNEAL ANNEAL_SPEED
export CUDA_VISIBLE_DEVICES=0; train_model cl_seq2seq to_minimize personachat_extend 5000 6 both False 0.5 True 1.0
