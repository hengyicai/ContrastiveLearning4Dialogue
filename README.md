# Group-wise Contrastive Learning for Neural Dialogue Generation

This repo contains preliminary code of the EMNLP2020 paper named "[Group-wise Contrastive Learning for Neural Dialogue Generation](https://arxiv.org/abs/2009.07543)".

This codebase is built upon the [ParlAI](https://parl.ai/) project. 
Check `parlai/agents/contrastive_learning` for framework implementations.
Running scripts can be found in `projects/contrastive_learning`.


## Requirements
- Python3
- Pytorch 1.2 or newer

Dependencies of the core modules are listed in requirement.txt.

## Dataset
TBA

## Installing
```
git clone git@github.com:hengyicai/ContrastiveLearning4Dialogue.git ~/ContrastiveLearning4Dialogue
cd ~/ContrastiveLearning4Dialogue; python setup.py develop
echo "export PARLAI_HOME=~/ContrastiveLearning4Dialogue" >> ~/.bashrc; source ~/.bashrc
```

## Running

```
cd ~/ContrastiveLearning4Dialogue
bash projects/contrastive_learning/shell/run.sh
```

The last line of `projects/contrastive_learning/shell/run.sh` specifies preliminary arguments for the training:
```

# MODEL_NAME TO_MINIMIZE TASK PRETRAIN_STEPS SAMPLE_K CONTRAST_BY NAIVE_NEG_SAMPLING CL_THRESHOLD CL_ANNEAL ANNEAL_SPEED
export CUDA_VISIBLE_DEVICES=0; train_model cl_seq2seq to_minimize personachat_extend 5000 6 both False 0.5 True 1.0
```

See `projects/adaptive_learning/shell/run.sh` for details.

### Running Details

TBA
