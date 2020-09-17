import random

import torch
import torch.optim as optim

from parlai.agents.dialog_evaluator.auto_evaluator import (
    TorchGeneratorWithDialogEvalAgent,
    CorpusSavedDictionaryAgent
)
from parlai.core.metrics import AverageMetric
from parlai.core.torch_agent import History
from parlai.core.torch_generator_agent import Output
from parlai.core.torch_generator_agent import PPLMetric
from parlai.utils.misc import round_sigfigs, warn_once, AttrDict
from parlai.utils.torch import padded_tensor, padded_3d
from .modules import DialogWAE_GMP, DialogWAE


def make_floor(n):
    floor = [0 for _ in range(n)]
    for i in range(0, n, 2):
        floor[i] = 1
    return floor


class Batch(AttrDict):
    def __init__(self, text_vec=None, text_lengths=None, context_lens=None,
                 floors=None, label_vec=None, label_lengths=None, labels=None,
                 valid_indices=None, candidates=None, candidate_vecs=None,
                 image=None, observations=None, **kwargs):
        super().__init__(
            text_vec=text_vec, text_lengths=text_lengths, context_lens=context_lens,
            floors=floors, label_vec=label_vec, label_lengths=label_lengths, labels=labels,
            valid_indices=valid_indices,
            candidates=candidates, candidate_vecs=candidate_vecs,
            image=image, observations=observations,
            **kwargs)


class PersonDictionaryAgent(CorpusSavedDictionaryAgent):
    def __init__(self, opt, shared=None):
        """Initialize DictionaryAgent."""
        super().__init__(opt, shared)
        if not shared:
            delimiter = opt.get('delimiter', '\n')
            self.add_token(delimiter)
            self.freq[delimiter] = 999999999

            if DialogWaeAgent.P1_TOKEN:
                self.add_token(DialogWaeAgent.P1_TOKEN)

            if DialogWaeAgent.P2_TOKEN:
                self.add_token(DialogWaeAgent.P2_TOKEN)

            if DialogWaeAgent.P1_TOKEN:
                self.freq[DialogWaeAgent.P1_TOKEN] = 999999998

            if DialogWaeAgent.P2_TOKEN:
                self.freq[DialogWaeAgent.P2_TOKEN] = 999999997


class MultiTurnOnOneRowHistory(History):
    def update_history(self, obs, add_next=None):
        """
        Update the history with the given observation.

        :param add_next:
            string to append to history prior to updating it with the
            observation
        """
        coin_flip = 0

        if self.field in obs and obs[self.field] is not None:
            if self.split_on_newln:
                next_texts = obs[self.field].split(self.delimiter)
            else:
                next_texts = [obs[self.field]]
            for text in next_texts:
                self._update_raw_strings(text)
                if self.add_person_tokens:
                    text = self._add_person_tokens(
                        text, self.p1_token if coin_flip % 2 == 0 else self.p2_token)
                    coin_flip += 1
                # update history string
                self._update_strings(text)
                # update history vecs
                self._update_vecs(text)

    def get_history_vec(self):
        """Returns a vectorized version of the history."""
        if len(self.history_vecs) == 0:
            return None

        # if self.vec_type == 'deque':
        #     history = deque(maxlen=self.max_len)
        #     for vec in self.history_vecs[:-1]:
        #         history.extend(vec)
        #         history.extend(self.delimiter_tok)
        #     history.extend(self.history_vecs[-1])
        # else:
        #     # vec type is a list
        #     history = []
        #     for vec in self.history_vecs[:-1]:
        #         history += vec
        #         history += self.delimiter_tok
        #     history += self.history_vecs[-1]
        history = self.history_vecs
        return history


class DialogWaeAgent(TorchGeneratorWithDialogEvalAgent):

    @classmethod
    def history_class(cls):
        """
        Return the history class that this agent expects to use.

        Can be overriden if a more complex history is required.
        """
        return MultiTurnOnOneRowHistory

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return PersonDictionaryAgent

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('DialogWAE Arguments')
        # Model Arguments
        agent.add_argument('--rnn_class', type=str, default='gru', choices=['gru', 'lstm'])
        agent.add_argument('-esz', '--embeddingsize', type=int, default=300,
                           help='Size of all embedding layers')
        agent.add_argument('--maxlen', type=int, default=60,
                           help='maximum utterance length')
        agent.add_argument('--hiddensize', type=int, default=512,
                           help='number of hidden units per layer')
        agent.add_argument('--numlayers', type=int, default=2,
                           help='number of layers')
        agent.add_argument('--noise_radius', type=float, default=0.2,
                           help='stdev of noise for autoencoder (regularizer)')
        agent.add_argument('--z_size', type=int, default=200,
                           help='dimension of z (300 performs worse)')
        agent.add_argument('--lambda_gp', type=int, default=10,
                           help='Gradient penalty lambda hyperparameter.')
        agent.add_argument('--temp', type=float, default=1.0,
                           help='softmax temperature (lower --> more discrete)')
        agent.add_argument('--input_dropout', type=float, default=0.0)
        agent.add_argument('--dropout', type=float, default=0.2)
        agent.add_argument('--gmp', type='bool', default=False)
        # -- with the following two arguments, we have model ``DialogWAE_GMP''
        agent.add_argument('--n_prior_components', type=int, default=3)
        agent.add_argument('--gumbel_temp', type=float, default=0.1)
        # -- if hred or vhred to be true, then this model degenerate into the vanilla HRED or VHRED
        agent.add_argument('--hred', type='bool', default=False)
        agent.add_argument('--vhred', type='bool', default=False)
        agent.add_argument('--bow_w', type='bool', default=0.01)
        # -- for HRAN
        agent.add_argument('-attl', '--attention-length', default=48, type=int,
                           help='Length of local attention.')
        agent.add_argument('-att', '--attention', default='none',
                           choices=['none', 'concat', 'general', 'dot', 'local'],
                           help='Choices: none, concat, general, local. '
                                'If set local, also set attention-length. '
                                '(see arxiv.org/abs/1508.04025)')

        # Training Arguments
        agent.add_argument('--n_iters_d', type=int, default=5,
                           help='number of discriminator iterations in training')
        agent.add_argument('--lr_gan_g', type=float, default=5e-05,
                           help='model learning rate')
        agent.add_argument('--lr_gan_d', type=float, default=1e-05,
                           help='critic/discriminator learning rate')
        agent.add_argument('--gan_clamp', type=float, default=0.01,
                           help='WGAN clamp (Do not use clamp when you apply gradient penelty')
        agent.add_argument('--norm_z', type='bool', default=False)

        cls.dictionary_class().add_cmdline_args(argparser)

        super(DialogWaeAgent, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        self.id = 'DialogWAE'
        if opt.get('hred', False) and opt.get('vhred', False):
            raise RuntimeError('The flags hred and vhred can not set to be True simultaneously!')
        if not opt.get('split_lines', False):
            raise RuntimeError('"split_lines" must be True for DialogWAE!')

        if not shared:
            self.add_metric('loss_G', 0.0)
            self.add_metric('loss_G_cnt', 0)
            self.add_metric('loss_D', 0.0)
            self.add_metric('loss_D_cnt', 0)
            self.add_metric('kl_loss', 0.0)
            self.add_metric('kl_loss_cnt', 0)
            self.add_metric('bow_loss', 0.0)
            self.add_metric('bow_loss_cnt', 0)
            self.add_metric('to_minimize', 0.0)

        if (
                # only build an optimizer if we're training
                'train' in opt.get('datatype', '') and
                # and this is the main model, or on every fork if doing hogwild
                (shared is None or self.opt.get('numthreads', 1) > 1)
        ):
            self.optimizer_G = optim.RMSprop(list(self.model.post_net.parameters())
                                             + list(self.model.post_generator.parameters())
                                             + list(self.model.prior_net.parameters())
                                             + list(self.model.prior_generator.parameters()), lr=opt['lr_gan_g'])
            self.optimizer_D = optim.RMSprop(self.model.discriminator.parameters(), lr=opt['lr_gan_d'])

    def build_model(self, states=None):
        special_tokens = [self.START_IDX,
                          self.END_IDX,
                          self.NULL_IDX,
                          self.dict[self.dict.unk_token]]

        if self.opt.get('gmp', False) and not self.opt['hred']:
            model = DialogWAE_GMP(self.opt, len(self.dict),
                                  PAD_token=self.NULL_IDX,
                                  unknown_idx=self.dict[self.dict.unk_token],
                                  use_cuda=self.use_cuda,
                                  special_tokens=special_tokens)
        else:
            model = DialogWAE(self.opt, len(self.dict),
                              PAD_token=self.NULL_IDX,
                              unknown_idx=self.dict[self.dict.unk_token],
                              use_cuda=self.use_cuda,
                              special_tokens=special_tokens)

        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.embedder.weight, self.opt['embedding_type'])
        return model

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        self.model.load_state_dict(state_dict, strict=False)

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['loss_G_cnt'] = 0
        self.metrics['loss_G'] = 0.0
        self.metrics['loss_D_cnt'] = 0
        self.metrics['loss_D'] = 0.0
        self.metrics['kl_loss_cnt'] = 0
        self.metrics['kl_loss'] = 0.0
        self.metrics['bow_loss_cnt'] = 0
        self.metrics['bow_loss'] = 0.0
        self.metrics['to_minimize'] = 0.0

    def report(self):
        base = super().report()
        m = dict()

        if self.metrics['loss_G_cnt'] > 0:
            m['loss_G'] = self.metrics['loss_G'] / self.metrics['loss_G_cnt']
        if self.metrics['loss_D_cnt'] > 0:
            m['loss_D'] = self.metrics['loss_D'] / self.metrics['loss_D_cnt']

        if self.metrics['kl_loss_cnt'] > 0:
            m['kl_loss'] = self.metrics['kl_loss'] / self.metrics['kl_loss_cnt']

        if self.metrics['bow_loss_cnt'] > 0:
            m['bow_loss'] = self.metrics['bow_loss'] / self.metrics['bow_loss_cnt']

        if 'loss_G' in m and 'loss_D' in m:
            m['to_minimize'] = m['loss_G'] + m['loss_D']

        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def _set_text_vec(self, obs, history, truncate):
        """
        Sets the 'text_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            # text vec is not precomputed, so we set it using the history
            history_string = history.get_history_str()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if history_string is None:
                return obs

            obs['full_text'] = history_string
            if history_string:
                obs['text_vec'] = history.get_history_vec()

        # check truncation
        if 'text_vec' in obs:
            for idx, vec in enumerate(obs['text_vec']):
                truncated_vec = self._check_truncate(vec, truncate, True)
                obs['text_vec'][idx] = torch.LongTensor(truncated_vec)

        return obs

    def batchify(self, obs_batch, sort=False):
        """
        Create a batch of valid observations from an unchecked batch.

        A valid observation is one that passes the lambda provided to the
        function, which defaults to checking if the preprocessed 'text_vec'
        field is present which would have been set by this agent's 'vectorize'
        function.

        Returns a namedtuple Batch. See original definition above for in-depth
        explanation of each field.

        If you want to include additonal fields in the batch, you can subclass
        this function and return your own "Batch" namedtuple: copy the Batch
        namedtuple at the top of this class, and then add whatever additional
        fields that you want to be able to access. You can then call
        super().batchify(...) to set up the original fields and then set up the
        additional fields in your subclass and return that batch instead.

        :param obs_batch:
            List of vectorized observations

        :param sort:
            Default False, orders the observations by length of vectors. Set to
            true when using torch.nn.utils.rnn.pack_padded_sequence.  Uses the text
            vectors if available, otherwise uses the label vectors if available.
        """
        if len(obs_batch) == 0:
            return Batch()

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if
                     self.is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch()

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs, x_lens, context_lens, floors = None, None, None, None
        if any('text_vec' in ex for ex in exs):
            _xs = [ex.get('text_vec', [self.EMPTY]) for ex in exs]
            xs = padded_3d(
                _xs, self.NULL_IDX, self.use_cuda, fp16friendly=self.opt.get('fp16'),
            )
            x_lens = (xs != self.NULL_IDX).sum(dim=-1)  # bsz, context_len
            context_lens = (x_lens != 0).sum(dim=-1)  # bsz
            floors, _ = padded_tensor([make_floor(c_len.item()) for c_len in context_lens],
                                      use_cuda=self.use_cuda)
            # We do not sort on the xs which in the shape of [bsz, context_len, utt_len] is this agent
            # if sort:
            #     sort = False  # now we won't sort on labels
            #     xs, x_lens, valid_inds, exs = argsort(
            #         x_lens, xs, x_lens, valid_inds, exs, descending=True
            #     )

        # LABELS
        labels_avail = any('labels_vec' in ex for ex in exs)
        some_labels_avail = (labels_avail or
                             any('eval_labels_vec' in ex for ex in exs))

        ys, y_lens, labels = None, None, None
        if some_labels_avail:
            field = 'labels' if labels_avail else 'eval_labels'

            label_vecs = [ex.get(field + '_vec', self.EMPTY) for ex in exs]
            labels = [ex.get(field + '_choice') for ex in exs]
            y_lens = [y.shape[0] for y in label_vecs]

            ys, y_lens = padded_tensor(
                label_vecs, self.NULL_IDX, self.use_cuda,
                fp16friendly=self.opt.get('fp16')
            )
            y_lens = torch.LongTensor(y_lens)
            if self.use_cuda:
                y_lens = y_lens.cuda()
            # We do not sort examples in batch for this agent
            # if sort and xs is None:
            #     ys, valid_inds, label_vecs, labels, y_lens = argsort(
            #         y_lens, ys, valid_inds, label_vecs, labels, y_lens,
            #         descending=True
            #     )

        # LABEL_CANDIDATES
        cands, cand_vecs = None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        return Batch(text_vec=xs, text_lengths=x_lens, context_lens=context_lens,
                     floors=floors, label_vec=ys, label_lengths=y_lens,
                     labels=labels, valid_indices=valid_inds, candidates=cands,
                     candidate_vecs=cand_vecs, image=imgs, observations=exs)

    def vectorize(self, obs, history, add_start=True, add_end=True,
                  text_truncate=None, label_truncate=None):
        """
        Make vectors out of observation fields and store in the observation.

        In particular, the 'text' and 'labels'/'eval_labels' fields are
        processed and a new field is added to the observation with the suffix
        '_vec'.

        If you want to use additional fields on your subclass, you can override
        this function, call super().vectorize(...) to process the text and
        labels, and then process the other fields in your subclass.

        Additionally, if you want to override some of these default parameters,
        then we recommend using a pattern like:

        .. code-block:: python

          def vectorize(self, *args, **kwargs):
              kwargs['add_start'] = False
              return super().vectorize(*args, **kwargs)


        :param obs:
            Single observation from observe function.

        :param add_start:
            default True, adds the start token to each label.

        :param add_end:
            default True, adds the end token to each label.

        :param text_truncate:
            default None, if set truncates text vectors to the specified
            length.

        :param label_truncate:
            default None, if set truncates label vectors to the specified
            length.

        :return:
            the input observation, with 'text_vec', 'label_vec', and
            'cands_vec' fields added.
        """
        self._set_text_vec(obs, history, text_truncate)
        self._set_label_vec(obs, True, True, label_truncate)
        self._set_label_cands_vec(obs, add_start, add_end, label_truncate)
        return obs

    def _model_input(self, batch):
        return (batch.text_vec,
                batch.context_lens,
                batch.text_lengths,
                batch.floors,)

    def compute_loss(self, batch, return_output=False):
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch),
                                  ys=batch.label_vec,
                                  res_lens=batch.label_lengths)
        scores, preds, vhred_kl_loss, bow_loss, *_ = model_output
        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(
            score_view / self.opt['temp'],
            batch.label_vec[:, 1:].contiguous().view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)
        # save loss to metrics
        notnull = batch.label_vec[:, :-1].ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec[:, :-1] == preds) * notnull).sum(dim=-1)

        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )
        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token

        # for vhred
        if vhred_kl_loss != -1 and bow_loss != -1:
            loss += (vhred_kl_loss * self.model.anneal_weight(self._number_training_updates)
                     + self.opt['bow_w'] * bow_loss)
            self.metrics['kl_loss_cnt'] += 1
            self.metrics['kl_loss'] += vhred_kl_loss.item()
            self.metrics['bow_loss_cnt'] += 1
            self.metrics['bow_loss'] += bow_loss.item()

        if return_output:
            return (loss, model_output)
        else:
            return loss

    def _dummy_batch(self, batchsize, maxlen):
        context_lens = torch.LongTensor([3] * batchsize)

        return Batch(
            text_vec=torch.ones(batchsize, 3, maxlen).long().cuda(),
            text_lengths=(torch.ones(batchsize, 3) * maxlen).long().cuda(),
            context_lens=context_lens.cuda(),
            floors=padded_tensor([make_floor(c_len.item()) for c_len in context_lens],
                                 use_cuda=self.use_cuda)[0],
            label_vec=torch.ones(batchsize, 2).long().cuda(),
            label_lengths=torch.LongTensor([2] * batchsize).cuda()
        )

    def wae_gan_train_step(self, batch):
        loss_G = self.model.train_G(batch.text_vec, batch.context_lens, batch.text_lengths,
                                    batch.floors, batch.label_vec, batch.label_lengths,
                                    self.optimizer_G)
        self.metrics['loss_G_cnt'] += 1
        self.metrics['loss_G'] += loss_G['train_loss_G']

        for i in range(self.opt['n_iters_d']):
            loss_D = self.model.train_D(batch.text_vec, batch.context_lens, batch.text_lengths,
                                        batch.floors, batch.label_vec, batch.label_lengths,
                                        self.optimizer_D)
            if i == 0:
                self.metrics['loss_D_cnt'] += 1
                self.metrics['loss_D'] += loss_D['train_loss_D']

    def train_step(self, batch):
        super(DialogWaeAgent, self).train_step(batch)
        if not (self.opt.get('hred', False) or self.opt.get('vhred', False)):
            self.wae_gan_train_step(batch)

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        self.model.eval()

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            _, model_output = self.compute_loss(batch, return_output=True)

            if not (self.opt.get('hred', False) or self.opt.get('vhred', False)):
                *_, x, c = model_output
                costG, costD = self.model.wae_gan_valid(x, c)
                self.metrics['loss_G_cnt'] += 1
                self.metrics['loss_G'] += costG
                self.metrics['loss_D_cnt'] += 1
                self.metrics['loss_D'] += costD

        preds = None
        if self.skip_generation:
            # noinspection PyTypeChecker
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning
            )
        else:
            sample_words, sample_lens = self.model.sample(
                batch.text_vec, batch.context_lens, batch.text_lengths,
                batch.floors, self.START_IDX, self.END_IDX
            )
            preds = torch.from_numpy(sample_words)

        text = [self._v2t(p) for p in preds] if preds is not None else None
        output = Output(text)

        label_text = batch.labels
        context = [obs['text'] for obs in batch.observations]
        if label_text is not None:
            self._eval_embedding_metrics(output, label_text, context)
            self._eval_distinct_metrics(output, label_text)
            self._eval_entropy_metrics(output, label_text)

            # sampling predictions for printing
            if output.text is not None:
                for i in range(len(output.text)):
                    if random.random() > (1 - self.opt['report_freq']):
                        context_text = batch.observations[i]['text']
                        target_text = self._v2t(batch.label_vec[i])
                        print('TEXT: ', context_text.replace(self.dict[self.NULL_IDX], ''))
                        print('TARGET: ', target_text)
                        print('PREDICTION: ', output.text[i], '\n~')

            if text and self.compute_tokenized_bleu:
                # compute additional bleu scores
                self._compute_fairseq_bleu(batch, preds)
                self._compute_nltk_bleu(batch, text)

        return output
