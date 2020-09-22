import copy
import random

import math
import nltk
import numpy as np
import torch
from nltk.util import bigrams, trigrams

from parlai.agents.dialog_evaluator.auto_evaluator import CorpusSavedDictionaryAgent
from parlai.agents.dialog_wae.dialog_wae import make_floor
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric
from parlai.core.opt import Opt
from parlai.core.torch_generator_agent import Batch
from parlai.utils.torch import padded_3d

RANDOM_SAMPLING = -1
NO_SAMPLING = -2


class SampleExtendBatch(Batch):
    def __init__(self,
                 samp_cs_vecs=None,
                 samp_cs=None,
                 samp_rs_vecs=None,
                 samp_rs=None,
                 c_vs_samp_r_scores=None,
                 samp_c_vs_r_scores=None,
                 **kwargs):
        super().__init__(
            samp_cs_vecs=samp_cs_vecs,
            samp_cs=samp_cs,
            samp_rs_vecs=samp_rs_vecs,
            samp_rs=samp_rs,
            c_vs_samp_r_scores=c_vs_samp_r_scores,
            samp_c_vs_r_scores=samp_c_vs_r_scores,
            **kwargs
        )


class SampleExtendDictionaryAgent(CorpusSavedDictionaryAgent):
    def act(self):
        """
        Add words in the last observation to the dictionary.

        This checks any fields in the message present in the --dict-textfields
        argument (e.g. "text,labels").
        """
        for textfield in self.textfields:
            source = self.observation.get(textfield)
            if source is None:
                continue
            # fields may be singleton strings or lists of strings.
            # wrap the singleton strings in a list to iterate over them
            if type(source) is str:
                source = [source]
            for text in source:
                if text:
                    items = text.split('__SAMP__')
                    text = items[0].strip()
                    tokens = self.tokenize(text)
                    self.add_to_dict(tokens)
                    unigram_ = nltk.ngrams(tokens, 1)
                    bigrams_ = bigrams(tokens)
                    trigrams_ = trigrams(tokens)
                    self.unigram_freq.update(unigram_)
                    self.bigram_freq.update(bigrams_)
                    self.trigram_freq.update(trigrams_)
        return {'id': 'Dictionary'}


def cl_init(self, shared=None):
    assert not self.opt['multigpu'], "CL training now does not support multigpu training!" \
                                     "Set --multigpu False."
    self.ref_update = False
    self.cl_training_steps = 0

    if shared:
        self.ref_agent = shared['ref_agent']
        self.eval_ref_agent = shared['eval_ref_agent']
        if 'neg_samples' in shared:
            self.neg_samples = shared['neg_samples']
    else:
        if self.opt['naive_neg_sampling']:
            self.neg_samples = set()
        cl_build_ref_agent(self)

        # loading the saved ref_agent
        init_model, _ = self._get_init_model(self.opt, shared)
        if init_model is not None:
            import parlai.utils.pickle
            states = torch.load(
                init_model, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle
            )
            load_ref_agent(self, states)

    if self.use_cuda:
        self.ref_agent.model.cuda()
        self.eval_ref_agent.model.cuda()


def cl_share(self, shared):
    shared['ref_agent'] = self.ref_agent
    shared['eval_ref_agent'] = self.eval_ref_agent
    if self.opt['naive_neg_sampling']:
        shared['neg_samples'] = self.neg_samples
    return shared


def observe_samp_expanded_observation(observation, multi_turn=False):
    """
    Process incoming message in preparation for producing a response.

    This includes remembering the past history of the conversation.
    """
    # TODO: Migration plan: TorchAgent currently supports being passed
    # observations as vanilla dicts for legacy interop; eventually we
    # want to remove this behavior and demand that teachers return Messages
    observation = Message(observation)

    if 'text' in observation:
        # ---> refactor the observation
        orig_text: str = observation['text']
        items = orig_text.split('__SAMP__')
        real_text = items[0].strip()

        samp_cs, samp_rs, c_vs_samp_r_scores, samp_c_vs_r_scores = None, None, None, None
        if len(items) > 1:
            samples = [d.strip() for d in items[1].split('__EOD__')]
            samp_cs = [d.split('__EOC__')[0].strip() for d in samples]
            samp_rs = [d.split('__EOC__')[1].strip() for d in samples]
            if multi_turn:
                samp_cs = [[utt.strip() for utt in samp_c.split('__EOT__')] for samp_c in samp_cs]

        if len(items) > 2:
            c_vs_samp_r_scores = [float(score) for score in items[2].split()]
        if len(items) > 3:
            samp_c_vs_r_scores = [float(score) for score in items[3].split()]

        observation.force_set('text', real_text)
        observation['samp_cs'] = samp_cs
        observation['samp_rs'] = samp_rs
        observation['c_vs_samp_r_scores'] = c_vs_samp_r_scores
        observation['samp_c_vs_r_scores'] = samp_c_vs_r_scores
        # <--- refactor the observation

    return observation


EMPTY = torch.zeros(0, dtype=torch.long)


def cl_batchify(self, batch):
    if len(batch) == 0:
        return batch

    exs = batch.observations

    # SAMP_CS
    samp_cs, samp_cs_vecs = None, None
    if any('samp_cs_vecs' in ex for ex in exs):
        samp_cs = [ex.get('samp_cs', [""]) for ex in exs]
        samp_cs_vecs = [ex.get('samp_cs_vecs', [EMPTY]) for ex in exs]

    # SAMP_RS
    samp_rs, samp_rs_vecs = None, None
    if any('samp_rs_vecs' in ex for ex in exs):
        samp_rs = [ex.get('samp_rs', None) for ex in exs]
        samp_rs_vecs = [ex.get('samp_rs_vecs', None) for ex in exs]

    # SCORES
    c_vs_samp_r_scores, samp_c_vs_r_scores = None, None
    if any(ex.get('c_vs_samp_r_scores') is not None for ex in exs):
        # noinspection PyArgumentList
        c_vs_samp_r_scores = torch.FloatTensor([ex.get('c_vs_samp_r_scores') for ex in exs])
        if self.use_cuda:
            c_vs_samp_r_scores = c_vs_samp_r_scores.cuda()
    if any(ex.get('samp_c_vs_r_scores') is not None for ex in exs):
        # noinspection PyArgumentList
        samp_c_vs_r_scores = torch.FloatTensor([ex.get('samp_c_vs_r_scores') for ex in exs])
        if self.use_cuda:
            samp_c_vs_r_scores = samp_c_vs_r_scores.cuda()

    extend_batch = SampleExtendBatch(**{k: v for k, v in batch.items()})
    extend_batch.samp_cs_vecs = samp_cs_vecs
    extend_batch.samp_cs = samp_cs
    extend_batch.samp_rs_vecs = samp_rs_vecs
    extend_batch.samp_rs = samp_rs

    extend_batch.c_vs_samp_r_scores = c_vs_samp_r_scores
    extend_batch.samp_c_vs_r_scores = samp_c_vs_r_scores

    return extend_batch


def _set_samp_label_vec(self, obs, add_start, add_end, truncate):
    if 'samp_rs' not in obs or obs['samp_rs'] is None:
        return

    elif 'samp_rs_vecs' in obs:
        # check truncation of pre-computed vector
        samp_rs_list_of_vec = []
        for samp_r_vec in obs['samp_rs_vecs']:
            truncated_vec = self._check_truncate(samp_r_vec, truncate)
            # noinspection PyArgumentList
            samp_rs_list_of_vec.append(torch.LongTensor(truncated_vec))
        obs.force_set('samp_rs_vecs', samp_rs_list_of_vec)
    else:
        samp_rs_list_of_vec = []
        for samp_r in obs['samp_rs']:
            truncated_vec = self._vectorize_text(samp_r, add_start, add_end, truncate, False)
            samp_rs_list_of_vec.append(truncated_vec)
        obs['samp_rs_vecs'] = samp_rs_list_of_vec
    return obs


def _set_samp_text_vec(self, obs, truncate):
    if 'samp_cs' not in obs or obs['samp_cs'] is None:
        return

    if 'samp_cs_vecs' in obs:
        if truncate is not None:
            # check truncation of pre-computed vectors
            vecs = obs['samp_cs_vecs']
            for i, samp_c in enumerate(vecs):
                vecs[i] = self._check_truncate(samp_c, truncate)
    elif obs.get('samp_cs'):
        obs['samp_cs_vecs'] = [
            self._vectorize_text(samp_c, truncate=truncate)
            for samp_c in obs['samp_cs']
        ]
    return obs


def _set_samp_multi_turn_text_vec(self, obs, truncate):
    if 'samp_cs' not in obs or obs['samp_cs'] is None:
        return

    if 'samp_cs_vecs' in obs:
        if truncate is not None:
            # check truncation of pre-computed vectors
            vecs = obs['samp_cs_vecs']
            for i, samp_c in enumerate(vecs):
                vecs[i] = [self._check_truncate(c_utt, truncate) for c_utt in samp_c]
    elif obs.get('samp_cs'):
        obs['samp_cs_vecs'] = [
            [self._vectorize_text(c_utt, truncate=truncate) for c_utt in samp_c]
            for samp_c in obs['samp_cs']
        ]
    return obs


def _log_p(self, scores, ys):
    score_view = scores.view(-1, scores.size(-1))
    loss = self.criterion(score_view, ys.reshape(-1))
    loss = loss.view(scores.shape[:-1]).sum(dim=1)  # bsz
    return -loss


def _compute_sample_cl_loss(self, target_scores, target_ys, ref_scores, ref_ys, matching_scores):
    log_p_m = _log_p(self, target_scores, target_ys)
    log_p_n = _log_p(self, ref_scores, ref_ys)

    """
    target_log_p = F.log_softmax(target_scores, -1).gather(-1, target_ys.unsqueeze(-1)).squeeze(-1)  # bsz, seq_len
    ref_log_p = F.log_softmax(ref_scores, -1).gather(-1, ref_ys.unsqueeze(-1)).squeeze(-1)

    target_notnull = target_ys.ne(self.NULL_IDX)
    ref_notnull = ref_ys.ne(self.ref_agent.NULL_IDX)
    # target_tokens = target_notnull.float().sum()

    target_log_p = target_log_p.masked_fill(~target_notnull, 0.)
    ref_log_p = ref_log_p.masked_fill(~ref_notnull, 0.)
    log_p_m = torch.sum(target_log_p, dim=-1)
    log_p_n = torch.sum(ref_log_p, dim=-1)
    """

    # matching_scores = matching_scores.unsqueeze(-1).expand(ys.size(0), ys.size(1)).reshape(-1)[notnull]
    # log_p_m = target_log_p.view(-1)[notnull]
    # log_p_n = ref_log_p.view(-1)[notnull]

    g = torch.sub(log_p_m, log_p_n)
    h = torch.sigmoid(g)

    if self.opt.get('only_pos', False) and self.model.training:
        matching_scores[matching_scores < 0] = 0.0
    if self.opt.get('only_neg', False) and self.model.training:
        matching_scores[matching_scores > 0] = 0.0

    # noinspection PyTypeChecker
    batch_cl_loss = -torch.log(torch.clamp(-matching_scores * (0.5 - h) + 0.5, 1e-20, 1e20))

    if self.opt['cl_loss_per_token']:
        target_notnull = target_ys.ne(self.NULL_IDX)
        target_tokens = target_notnull.float().sum()
        crt_cl_loss = torch.sum(
            batch_cl_loss
        ) / target_tokens.sum()  # average loss per token
    else:
        crt_cl_loss = torch.sum(
            batch_cl_loss
        ) / target_ys.size(0)  # average loss per sample

    # batch_cl_loss = -matching_scores * (log_p_m - torch.log(torch.exp(log_p_m) + torch.exp(log_p_n)))

    return crt_cl_loss, batch_cl_loss


def cl_build_ref_agent(self):
    ref_model_file = self.opt['ref_model_file']
    if ref_model_file is None or ref_model_file.lower() == "none":
        raise RuntimeError("CL training requires reference model!")
    else:
        from parlai.core.agents import create_agent_from_opt_file
        ref_agent = create_agent_from_opt_file(Opt({'model_file': ref_model_file}))
        eval_ref_agent = create_agent_from_opt_file(Opt({'model_file': ref_model_file}))
        if ref_agent is None:
            raise RuntimeError("Build reference model failed! check your `ref_model_file`:{}!".format(ref_model_file))
        if self.id == ref_agent.id and dict_same(self, ref_agent):
            self.use_external_ref_model = False
        else:
            self.use_external_ref_model = True
        # No need to do this
        # # check dict
        # if self.dict.tok2ind != ref_agent.dict.tok2ind or self.dict.ind2tok != ref_agent.dict.ind2tok:
        #     raise RuntimeError("Reference model is using different dict!")

    self.eval_ref_agent = eval_ref_agent
    self.ref_agent = ref_agent


def cl_state_dict(self, states):
    if hasattr(self, 'ref_agent'):
        states['ref_agent'] = self.ref_agent.model.state_dict()
    if hasattr(self, 'eval_ref_agent'):
        states['eval_ref_agent'] = self.eval_ref_agent.model.state_dict()
    if hasattr(self, 'neg_samples'):
        states['neg_samples'] = self.neg_samples
    return states


def load_ref_agent(self, states):
    if 'ref_agent' in states:
        try:
            self.ref_agent.model.load_state_dict(states['ref_agent'])
        except RuntimeError:
            raise
    if 'eval_ref_agent' in states:
        try:
            self.eval_ref_agent.model.load_state_dict(states['eval_ref_agent'])
        except RuntimeError:
            raise
    if 'neg_samples' in states:
        self.neg_samples = states['neg_samples']


def _soft_normalize(scores, threshold=0.5):
    return torch.clamp(2 * (scores + 0.5 - threshold) - 1, -1, 1)


def _hard_normalize(scores, neg_threshold, pos_threshold):
    assert 0 < neg_threshold <= pos_threshold < 1
    scores[scores >= pos_threshold] = 1.0
    scores[scores < neg_threshold] = 0.0
    return _soft_normalize(scores)


def _filter_normalize(scores, neg_threshold, pos_threshold):
    assert 0 <= neg_threshold <= pos_threshold <= 1
    mask_gt_neg = scores > neg_threshold
    mask_lt_pos = scores < pos_threshold
    scores[mask_gt_neg & mask_lt_pos] = 0.5
    return _soft_normalize(scores)


def normalize_score(scores, self_opt):
    # scores: ~(0, 1)
    # return: ~(-1, 1)
    # ret_scores = torch.tanh(-3.0 + 6.0 * scores)  # ~(-1, 1)

    assert (self_opt.get('soft_normalize_score', True) and self_opt.get('filter_normalize_score', False)) is not True, \
        "Conflict options with both soft_normalize_score==True and filter_normalize_score==True!"

    if self_opt.get('soft_normalize_score', True):
        return _soft_normalize(scores, self_opt.get('cl_threshold', 0.5))
    elif self_opt.get('filter_normalize_score', False):
        return _filter_normalize(scores, self_opt.get('neg_threshold', 0.5), self_opt.get('pos_threshold', 0.5))
    else:
        return _hard_normalize(scores, self_opt.get('neg_threshold', 0.5), self_opt.get('pos_threshold', 0.5))


def __sample_batchify_pos(model_agent, batch, ref_agent_share_same_dict_with_target, multi_turn):
    if ref_agent_share_same_dict_with_target:
        return {}, {}
    else:
        if not ref_agent_share_same_dict_with_target and not multi_turn:
            # ref_agent is another type of model, like GPT2, target_agent is seq2seq or transformer
            texts = [obs['full_text'] for obs in batch.observations]
            text_vec = [model_agent._vectorize_text(utt, truncate=model_agent.opt['text_truncate']) for utt in texts]
            to_replace = __text_batchify(model_agent, text_vec)
        else:
            multi_turn_text = [[utt.strip() for utt in obs['full_text'].split(model_agent.history.delimiter)]
                               for obs in batch.observations]
            multi_turn_text_vec = [
                [model_agent._vectorize_text(
                    c_utt,
                    truncate=model_agent.opt['text_truncate']
                ) for c_utt in samp_c] for samp_c in multi_turn_text
            ]
            to_replace = __text_batchify_multi_turn(model_agent, multi_turn_text_vec)

        label_vec = [
            model_agent._vectorize_text(
                label, True, True, model_agent.opt['label_truncate']
            ) for label in batch.labels
        ]
        to_replace.update(__label_batchify(model_agent, label_vec))

        to_restore = dict((k, batch.get(k, None)) for k in to_replace.keys())
        return to_replace, to_restore


def __sample_batchify(model_agent, samp_idx, batch, contrast_by,
                      ref_agent_share_same_dict_with_target=True,
                      multi_turn=False):
    # Assume:
    #   ref_agent_share_same_dict_with_target: True,  multi_turn: False --> both ref and target are seq2seq or transformer
    #   ref_agent_share_same_dict_with_target: True,  multi_turn: True  --> both ref and target are hred or hran
    #   ref_agent_share_same_dict_with_target: False, multi_turn: False --> ref_agent is another type of model, like GPT2,
    #                                                                       target_agent is seq2seq or transformer
    #   ref_agent_share_same_dict_with_target: False, multi_turn: True  --> ref_agent is in the type of DialogWAE
    #                                                                       target_agent is seq2seq or transformer
    if samp_idx == NO_SAMPLING:
        return __sample_batchify_pos(model_agent, batch, ref_agent_share_same_dict_with_target, multi_turn)
    if ref_agent_share_same_dict_with_target:
        samp_vecs_name = 'samp_cs_vecs' if contrast_by == 'context' else 'samp_rs_vecs'
        batch_samp = [samp[samp_idx] for samp in batch[samp_vecs_name]]
    else:
        samp_cs = [obs['samp_cs'][samp_idx] for obs in batch.observations]
        samp_rs = [obs['samp_rs'][samp_idx] for obs in batch.observations]

        if multi_turn:
            samp_cs = [[utt.strip() for utt in samp_c.split('__EOT__')] for samp_c in samp_cs]
            samp_cs_vecs = [
                [model_agent._vectorize_text(
                    c_utt,
                    truncate=model_agent.opt['text_truncate']
                ) for c_utt in samp_c]
                for samp_c in samp_cs
            ]
        else:
            samp_cs_vecs = [
                model_agent._vectorize_text(
                    samp_c,
                    truncate=model_agent.opt['text_truncate']
                ) for samp_c in samp_cs
            ]
        samp_rs_vecs = [
            model_agent._vectorize_text(
                r,
                add_start=True if multi_turn else False,
                add_end=True,
                truncate=model_agent.opt['label_truncate'],
                truncate_left=False
            ) for r in samp_rs
        ]
        batch_samp = samp_cs_vecs if contrast_by == 'context' else samp_rs_vecs

    if contrast_by == 'response':
        to_replace = __label_batchify(model_agent, batch_samp)
        if not ref_agent_share_same_dict_with_target:
            if multi_turn:
                to_replace.update(__text_batchify_multi_turn(model_agent, samp_cs_vecs))
            else:
                to_replace.update(__text_batchify(model_agent, samp_cs_vecs))
    else:
        if multi_turn:
            to_replace = __text_batchify_multi_turn(model_agent, batch_samp)
        else:
            to_replace = __text_batchify(model_agent, batch_samp)
        if not ref_agent_share_same_dict_with_target:
            to_replace.update(__label_batchify(model_agent, samp_rs_vecs))

    to_restore = dict((k, batch.get(k, None)) for k in to_replace.keys())

    return to_replace, to_restore


def sample_batchify(model_agent, samp_idx, batch, contrast_by,
                    ref_agent_share_same_dict_with_target=True):
    return __sample_batchify(model_agent, samp_idx, batch, contrast_by,
                             ref_agent_share_same_dict_with_target)


def sample_batchify_multi_turn(model_agent, samp_idx, batch, contrast_by,
                               ref_agent_share_same_dict_with_target=True):
    return __sample_batchify(model_agent, samp_idx, batch, contrast_by,
                             ref_agent_share_same_dict_with_target, multi_turn=True)


def __text_batchify(model_agent, batch_samp):
    batch_samp, x_lens = model_agent._pad_tensor(batch_samp)  # TODO: left_padded=True?
    return {'text_vec': batch_samp, 'text_lengths': x_lens}


def __text_batchify_multi_turn(model_agent, batch_samp):
    samp_x = padded_3d(batch_samp, model_agent.NULL_IDX, model_agent.use_cuda)
    samp_x_lens = (samp_x != model_agent.NULL_IDX).sum(dim=-1)
    samp_context_lens = (samp_x_lens != 0).sum(dim=-1)
    samp_floors, _ = model_agent._pad_tensor([make_floor(c_len.item()) for c_len in samp_context_lens])
    return {'text_vec': samp_x, 'text_lengths': samp_x_lens,
            'context_lens': samp_context_lens, 'floors': samp_floors}


def __label_batchify(model_agent, batch_samp):
    samp_y, samp_y_lens = model_agent._pad_tensor(batch_samp)
    samp_y_lens = torch.LongTensor(samp_y_lens)
    if model_agent.use_cuda:
        samp_y_lens = samp_y_lens.cuda()
    return {'label_vec': samp_y, 'label_lengths': samp_y_lens}


def _create_sample_batch(
        model_agent,
        batch,
        self_opt,
        samp_idx=-1,
        contrast_by='context',
        ref_agent_share_same_dict_with_target=True,
        # cl_threshold=0.5,
):
    if contrast_by == 'context':
        samp_vecs_name = 'samp_cs_vecs'
        scores_name = 'samp_c_vs_r_scores'
    elif contrast_by == 'response':
        samp_vecs_name = 'samp_rs_vecs'
        scores_name = 'c_vs_samp_r_scores'
    else:
        raise ValueError("Unrecognized value for `contrast_by`!")

    if samp_vecs_name not in batch or batch[samp_vecs_name] is None:
        raise ValueError('Cannot compute cl loss without samples.')

    if samp_idx == RANDOM_SAMPLING:
        samp_idx = random.choice(range(len(batch[samp_vecs_name][0])))
    if samp_idx == NO_SAMPLING:
        crt_matching_scores = None
    else:
        crt_matching_scores = normalize_score(batch[scores_name][:, samp_idx], self_opt)  # ~(-1, 1)

    to_replace, to_restore = model_agent.sample_batchify_func(
        model_agent,
        samp_idx,
        batch,
        contrast_by,
        ref_agent_share_same_dict_with_target=ref_agent_share_same_dict_with_target
    )

    for k, v in to_replace.items():
        batch[k] = v

    return {'sample_batch': batch, 'matching_scores': crt_matching_scores, 'to_restore': to_restore}


def dict_same(agent1, agent2):
    if not agent1 or not agent2:
        return False
    return agent1.dict.ind2tok == agent2.dict.ind2tok and agent1.dict.tok2ind == agent2.dict.tok2ind


def compute_cl_loss(
        self,
        batch,
        naive_neg_sampling=False,
        samp_idx=-1,
        contrast_by='context'
):
    if naive_neg_sampling:
        return compute_cl_loss_random_neg(self, batch, contrast_by)

    # replace some fields in batch with sampled content
    sample_constrcuted = _create_sample_batch(
        self,
        batch,
        self.opt,
        samp_idx,
        contrast_by,
        ref_agent_share_same_dict_with_target=True,
        # cl_threshold=self.opt['cl_threshold']
    )
    target_model_output, target_ys = self.compute_loss(
        sample_constrcuted['sample_batch'],
        return_output_only=True
    )
    # restore the original batch
    for k, v in sample_constrcuted['to_restore'].items():
        batch[k] = v

    with torch.no_grad():
        ref_sample_constrcuted = _create_sample_batch(
            self.ref_agent,
            batch,
            self.opt,
            samp_idx,
            contrast_by,
            ref_agent_share_same_dict_with_target=dict_same(self, self.ref_agent),
            # cl_threshold=self.opt['cl_threshold']
        )
        ref_model_output, ref_ys = self.ref_agent.compute_loss(
            ref_sample_constrcuted['sample_batch'],
            return_output_only=True
        )
        self.ref_agent._local_metrics.clear()
        # restore the original batch
        for k, v in ref_sample_constrcuted['to_restore'].items():
            batch[k] = v

    target_scores, *_ = target_model_output
    ref_scores, *_ = ref_model_output
    # do actual contrastive loss computation
    samp_cl_loss, batch_samp_cl_loss = _compute_sample_cl_loss(
        self, target_scores, target_ys, ref_scores, ref_ys,
        sample_constrcuted['matching_scores']
    )

    return samp_cl_loss, batch_samp_cl_loss


def compute_cl_loss_random_neg(self, batch, contrast_by):
    # TODO: ref_agent and target_agent are not applicable to HRED or HRAN, require more fields
    bsz = batch.text_vec.size(0)
    neg_scores = -torch.ones(bsz)
    if self.use_cuda:
        neg_scores = neg_scores.cuda()

    neg_sample = random.sample(self.neg_samples, 1)[0]
    orig = {'text_vec': batch.text_vec, 'label_vec': batch.label_vec}
    replace_field = 'text_vec' if contrast_by == 'context' else 'label_vec'
    neg_vec = neg_sample[0] if contrast_by == 'context' else neg_sample[1]
    if neg_vec.size(0) > bsz:
        neg_vec = neg_vec[:bsz, :]

    if self.use_cuda:
        neg_vec = neg_vec.cuda()
    batch[replace_field] = neg_vec

    target_model_output, target_ys = self.compute_loss(batch, return_output_only=True)

    with torch.no_grad():
        ref_model_output, ref_ys = self.ref_agent.compute_loss(batch, return_output_only=True)
        self.ref_agent._local_metrics.clear()

    target_scores, *_ = target_model_output
    ref_scores, *_ = ref_model_output

    to_return = _compute_sample_cl_loss(self, target_scores, target_ys, ref_scores, ref_ys, neg_scores)
    batch[replace_field] = orig[replace_field]
    return to_return


def compute_cl_loss_pos(self, batch):
    bsz = batch.text_vec.size(0)
    ones_score = torch.ones(bsz)
    if self.use_cuda:
        ones_score = ones_score.cuda()
    target_model_output, target_ys = self.compute_loss(batch, return_output_only=True)

    with torch.no_grad():
        ref_sample_constrcuted = _create_sample_batch(
            self.ref_agent,
            batch,
            self.opt,
            NO_SAMPLING,
            ref_agent_share_same_dict_with_target=dict_same(self, self.ref_agent),
            # cl_threshold=self.opt['cl_threshold']
        )
        ref_model_output, ref_ys = self.ref_agent.compute_loss(
            ref_sample_constrcuted['sample_batch'],
            return_output_only=True
        )
        self.ref_agent._local_metrics.clear()
        for k, v in ref_sample_constrcuted['to_restore'].items():
            batch[k] = v

    target_scores, *_ = target_model_output
    ref_scores, *_ = ref_model_output
    return _compute_sample_cl_loss(self, target_scores, target_ys, ref_scores, ref_ys, ones_score)


def anneal_weight(step, anneal_speed=1):
    return np.clip((math.tanh((step * anneal_speed - 3500) / 1000) + 1) / 2, 0., 1.)


def cl_train_step(self, batch):
    self._init_cuda_buffer(self.opt['batchsize'], self.label_truncate or 256)
    self.model.train()
    self.zero_grad()

    assert self.opt.get('update_freq', 1) == 1

    try:
        if self.opt['cl_anneal']:
            cl_anneal = anneal_weight(self.cl_training_steps, anneal_speed=self.opt.get('anneal_speed', 1.0))
            nll_loss = self.compute_loss(batch)
            nll_loss *= (1 - cl_anneal)
            self.backward(nll_loss)
        else:
            cl_anneal = 1.0

        batch_loss = 0.0
        contrast_by = self.opt['contrast_by']
        if contrast_by == 'context':
            coin_face = 0.
        elif contrast_by == 'response':
            coin_face = 1.
        else:
            coin_face = .5

        for _ in range(self.opt['sample_k']):
            sample_cl_loss, batch_sample_cl_loss = compute_cl_loss(
                self,
                batch,
                naive_neg_sampling=self.opt['naive_neg_sampling'],
                contrast_by='context' if random.random() >= coin_face else 'response'
            )

            batch_loss += batch_sample_cl_loss.data
            sample_cl_loss /= (self.opt['sample_k'] + 1)
            sample_cl_loss *= cl_anneal
            self.backward(sample_cl_loss)

        pos_cl_loss, batch_pos_cl_loss = compute_cl_loss_pos(self, batch)
        batch_loss += batch_pos_cl_loss.data

        batch_loss /= (self.opt['sample_k'] + 1)
        pos_cl_loss /= (self.opt['sample_k'] + 1)
        pos_cl_loss *= cl_anneal
        # record metrics
        # noinspection PyTypeChecker
        self.record_local_metric('cl_loss', AverageMetric.many(batch_loss))
        self.backward(pos_cl_loss)

        self.update_params()
    except RuntimeError as e:
        # catch out of memory exceptions during fwd/bck (skip batch)
        if 'out of memory' in str(e):
            print(
                '| WARNING: ran out of memory, skipping batch. '
                'if this happens frequently, decrease batchsize or '
                'truncate the inputs to the model.'
            )
            # self.global_metrics.update('skipped_batches', SumMetric(1))  # TODO: bug here
            # gradients are synced on backward, now this model is going to be
            # out of sync! catch up with the other workers
            self._init_cuda_buffer(8, 8, True)
        else:
            raise e


def cl_eval_step(super_, self, batch):
    output = super_.eval_step(batch)

    if batch.label_vec is not None:
        if self._number_training_updates >= self.opt['pretrain_steps']:
            # >>> CL eval >>>
            real_ref_agent = self.ref_agent
            if self.opt['use_eval_ref_agent']:
                # eval_ref_agent never conduct periodical replacement
                self.ref_agent = self.eval_ref_agent

            with torch.no_grad():
                _, batch_cl_loss = compute_cl_loss_pos(self, batch)

            if not self.opt['naive_neg_sampling']:
                contrast_by = self.opt['contrast_by']
                if contrast_by != 'both':
                    # only evaluating for either `cl_context` or `cl_response`
                    for i in range(len(batch.samp_cs_vecs[0])):
                        with torch.no_grad():
                            _, batch_samp_cl_loss = compute_cl_loss(
                                self,
                                batch,
                                naive_neg_sampling=False,
                                samp_idx=i,
                                contrast_by=contrast_by)
                        batch_cl_loss += batch_samp_cl_loss
                    batch_cl_loss /= (len(batch.samp_cs_vecs[0]) + 1)
                else:
                    # evaluating both `cl_context` and `cl_response`
                    for i in range(len(batch.samp_cs_vecs[0])):
                        with torch.no_grad():
                            _, batch_samp_c_cl_loss = compute_cl_loss(
                                self,
                                batch,
                                naive_neg_sampling=False,
                                samp_idx=i,
                                contrast_by='context')
                            _, batch_samp_r_cl_loss = compute_cl_loss(
                                self,
                                batch,
                                naive_neg_sampling=False,
                                samp_idx=i,
                                contrast_by='response')
                        batch_cl_loss += batch_samp_c_cl_loss
                        batch_cl_loss += batch_samp_r_cl_loss
                    batch_cl_loss /= (len(batch.samp_cs_vecs[0]) * 2 + 1)
            else:
                if self.opt['eval_naive_neg_sampling']:
                    # Also evaluate neg samples for `naive_neg_sampling'
                    for i in range(self.opt['eval_naive_neg_sampling_k']):
                        sample_cl_loss, batch_sample_cl_loss = compute_cl_loss(
                            self,
                            batch,
                            naive_neg_sampling=self.opt['naive_neg_sampling'],
                        )
                        batch_cl_loss += batch_sample_cl_loss

            self.record_local_metric('cl_loss', AverageMetric.many(batch_cl_loss))
            self.record_local_metric(
                'to_minimize',
                [cl.value() + nll.value() * self.opt['nll_w'] for cl, nll in zip(self._local_metrics['cl_loss'], self._local_metrics['loss'])])

            if self.opt['use_eval_ref_agent']:
                self.ref_agent = real_ref_agent
            # <<< CL eval <<<
        else:
            self.record_local_metric('to_minimize', self._local_metrics['ppl'])

    return output


def cl_train(super_, self, batch):
    if not self.use_external_ref_model and self._number_training_updates >= self.opt['pretrain_steps']:
        if self.opt['ref_model_update_freq'] > 0 and self.cl_training_steps % self.opt['ref_model_update_freq'] == 1:
            if self.opt['periodical_replacement'] and self.cl_training_steps > self.opt['ref_model_update_freq']:
                self.ref_agent.model = copy.deepcopy(self.model)

    if self._number_training_updates < self.opt['pretrain_steps']:
        super_.train_step(batch)
    else:
        # CL training
        # with torch.autograd.detect_anomaly():
        cl_train_step(self, batch)
        self.cl_training_steps += 1

    if self.opt['naive_neg_sampling']:
        # TODO: neg_sampling may not applicable to HRED, since its model inputs
        #   require more fields.
        self.neg_samples.add((batch.text_vec.cpu(), batch.label_vec.cpu()))
