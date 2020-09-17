import os
import pickle
import random
import sys
from collections import Counter

import nltk
import numpy as np
from nltk.util import bigrams, trigrams

from parlai.agents.hy_lib.embedding_metrics import sentence_average_score, \
    sentence_greedy_score, sentence_extrema_score
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.misc import round_sigfigs


class CorpusSavedDictionaryAgent(DictionaryAgent):
    """This DictionaryAgent save the training corpus for building the language model
        TODO: Add warnings once some subclass does not inherent this DictionaryAgent
    """

    def __init__(self, opt, shared=None):
        if shared:
            self.unigram_freq = shared.get('unigram_freq', Counter())
            self.bigram_freq = shared.get('bigram_freq', Counter())
            self.trigram_freq = shared.get('trigram_freq', Counter())
        else:
            self.unigram_freq = Counter()
            self.bigram_freq = Counter()
            self.trigram_freq = Counter()
        super().__init__(opt, shared=shared)

    def share(self):
        shared = super().share()
        shared['unigram_freq'] = self.unigram_freq
        shared['bigram_freq'] = self.bigram_freq
        shared['trigram_freq'] = self.trigram_freq
        return shared

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
                    tokens = self.tokenize(text)
                    self.add_to_dict(tokens)
                    unigram_ = nltk.ngrams(tokens, 1)
                    bigrams_ = bigrams(tokens)
                    trigrams_ = trigrams(tokens)
                    self.unigram_freq.update(unigram_)
                    self.bigram_freq.update(bigrams_)
                    self.trigram_freq.update(trigrams_)
        return {'id': 'Dictionary'}

    def save(self, filename=None, append=False, sort=True):
        super().save(filename, append, sort)
        filename = self.opt['dict_file'] if filename is None else filename
        with open(filename + '.unigram_freq.pkl', 'wb') as f:
            pickle.dump(self.unigram_freq, f)
        with open(filename + '.bigram_freq.pkl', 'wb') as f:
            pickle.dump(self.bigram_freq, f)
        with open(filename + '.trigram_freq.pkl', 'wb') as f:
            pickle.dump(self.trigram_freq, f)

    def load(self, filename):
        super().load(filename)
        with open(filename + '.unigram_freq.pkl', 'rb') as f:
            self.unigram_freq = pickle.load(f)
        with open(filename + '.bigram_freq.pkl', 'rb') as f:
            self.bigram_freq = pickle.load(f)
        with open(filename + '.trigram_freq.pkl', 'rb') as f:
            self.trigram_freq = pickle.load(f)


def clip_value(val, threshold):
    if val < threshold:
        return -1
    else:
        return val


class TorchGeneratorWithDialogEvalAgent(TorchGeneratorAgent):
    """Agent with metrics for evaluating dialogue agent."""

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overridden if a more complex dictionary is required.
        """
        return CorpusSavedDictionaryAgent

    def build_model(self):
        """
        Construct the model.

        The model should be set to self.model, and support
        the TorchGeneratorModel interface.
        """
        raise NotImplementedError(
            "AbstractClass: build_model must be implemented by the user."
        )

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('Dialog Evaluation Arguments')
        agent.add_argument('--eval_embedding_type', type=str, default='glove',
                           help='Embedding type (or embedding file) for response evaluation.')
        agent.add_argument('--arora_weighting', type='bool', default=True)
        agent.add_argument('--metrics_exclude_from_total_metric', type=str, default='')
        # ---------------------- For logging ----------------------------------#
        agent.add_argument('--report_freq', type=float, default=0.05)

        super(TorchGeneratorWithDialogEvalAgent, cls).add_cmdline_args(argparser)
        return agent

    def _init_eval_embedding(self, embedding_type=None):
        if embedding_type is None:
            embedding_type = 'glove'
        print('[ Init {} embeddings for evaluation ]'.format(embedding_type))
        embs, _ = self._get_embtype(embedding_type)
        self.eval_embs = embs

    def _get_embtype(self, emb_type):
        # set up preinitialized embeddings
        try:
            import torchtext.vocab as vocab
        except ImportError as ex:
            print('Please install torch text with `pip install torchtext`')
            raise ex
        pretrained_dim = 300
        if emb_type.startswith('glove'):
            if 'twitter' in emb_type:
                init = 'glove-twitter'
                name = 'twitter.27B'
                pretrained_dim = 200
            else:
                init = 'glove'
                name = '840B'
            embs = vocab.GloVe(
                name=name, dim=pretrained_dim,
                cache=modelzoo_path(self.opt.get('datapath'),
                                    'models:glove_vectors'))
        elif emb_type.startswith('fasttext_cc'):
            init = 'fasttext_cc'
            embs = vocab.FastText(
                language='en',
                cache=modelzoo_path(self.opt.get('datapath'),
                                    'models:fasttext_cc_vectors'))
        elif emb_type.startswith('fasttext'):
            init = 'fasttext'
            embs = vocab.FastText(
                language='en',
                cache=modelzoo_path(self.opt.get('datapath'),
                                    'models:fasttext_vectors'))
        else:
            # emb_type does not matching any type embeddings list above,
            # so we think it is a file_path to the embedding file,
            # if not, raise error
            assert os.path.isfile(emb_type), \
                'emb_type: {} does not matching any type embeddings list above, '.format(emb_type) + \
                'so we think it is a file_path to the embedding file!'
            init = os.path.basename(emb_type)
            cache = '.vector_cache'
            if not os.path.exists(cache):
                os.makedirs(cache)
            embs = vocab.Vectors(emb_type, cache=cache)
        return embs, init

    def __init__(self, opt: Opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        self.id = 'TorchGeneratorWithDialogEval'
        if shared:
            self.eval_embs = shared['eval_embs']
            self.total_unigrams = shared['total_unigrams']
            self.total_bigrams = shared['total_bigrams']
            self.total_trigrams = shared['total_trigrams']
        else:
            # Add the metrics for distinct evaluations
            self.add_metric('total_unigram_cnt', 0)
            self.add_metric('total_bigram_cnt', 0)
            self.add_metric('total_trigram_cnt', 0)
            self.add_metric('dist_unigram_tokens', set())
            self.add_metric('dist_bigram_tokens', set())
            self.add_metric('dist_trigram_tokens', set())

            # Intra_distinct
            self.add_metric('intra_unigram_cnt', 0)
            self.add_metric('intra_unigram', 0.0)
            self.add_metric('intra_bigram_cnt', 0)
            self.add_metric('intra_bigram', 0.0)
            self.add_metric('intra_trigram_cnt', 0)
            self.add_metric('intra_trigram', 0.0)

            # Response length
            self.add_metric('response_length_cnt', 0)
            self.add_metric('response_length', 0.0)

            # BOW Embedding
            self.add_metric('embed_avg_cnt', 0)
            self.add_metric('embed_avg', 0.0)
            self.add_metric('embed_greedy_cnt', 0)
            self.add_metric('embed_greedy', 0.0)
            self.add_metric('embed_extrema_cnt', 0)
            self.add_metric('embed_extrema', 0.0)

            self.add_metric('embed_coh_cnt', 0)
            self.add_metric('embed_coh', 0.0)

            # Entropy
            self.add_metric('sent_entropy_uni_cnt', 0)
            self.add_metric('sent_entropy_uni', 0.0)
            self.add_metric('sent_entropy_bi_cnt', 0)
            self.add_metric('sent_entropy_bi', 0.0)
            self.add_metric('sent_entropy_tri_cnt', 0)
            self.add_metric('sent_entropy_tri', 0.0)
            self.add_metric('word_entropy_uni_cnt', 0)
            self.add_metric('word_entropy_uni', 0.0)
            self.add_metric('word_entropy_bi_cnt', 0)
            self.add_metric('word_entropy_bi', 0.0)
            self.add_metric('word_entropy_tri_cnt', 0)
            self.add_metric('word_entropy_tri', 0.0)

            # Ground-truth metrics
            self.add_metric('human_total_unigram_cnt', 0)
            self.add_metric('human_total_bigram_cnt', 0)
            self.add_metric('human_total_trigram_cnt', 0)
            self.add_metric('human_dist_unigram_tokens', set())
            self.add_metric('human_dist_bigram_tokens', set())
            self.add_metric('human_dist_trigram_tokens', set())

            self.add_metric('human_intra_unigram_cnt', 0)
            self.add_metric('human_intra_unigram', 0.0)
            self.add_metric('human_intra_bigram_cnt', 0)
            self.add_metric('human_intra_bigram', 0.0)
            self.add_metric('human_intra_trigram_cnt', 0)
            self.add_metric('human_intra_trigram', 0.0)

            self.add_metric('human_response_length_cnt', 0)
            self.add_metric('human_response_length', 0.0)
            self.add_metric('human_embed_coh_cnt', 0)
            self.add_metric('human_embed_coh', 0.0)

            self.add_metric('human_sent_entropy_uni_cnt', 0)
            self.add_metric('human_sent_entropy_uni', 0.0)
            self.add_metric('human_sent_entropy_bi_cnt', 0)
            self.add_metric('human_sent_entropy_bi', 0.0)
            self.add_metric('human_sent_entropy_tri_cnt', 0)
            self.add_metric('human_sent_entropy_tri', 0.0)
            self.add_metric('human_word_entropy_uni_cnt', 0)
            self.add_metric('human_word_entropy_uni', 0.0)
            self.add_metric('human_word_entropy_bi_cnt', 0)
            self.add_metric('human_word_entropy_bi', 0.0)
            self.add_metric('human_word_entropy_tri_cnt', 0)
            self.add_metric('human_word_entropy_tri', 0.0)

            self._init_eval_embedding(embedding_type=opt.get('eval_embedding_type'))

            self.total_unigrams = sum(self.dict.unigram_freq.values())
            self.total_bigrams = sum(self.dict.bigram_freq.values())
            self.total_trigrams = sum(self.dict.trigram_freq.values())

        self.max_response_len = self.opt.get('label_truncate', -1)
        self.metrics_exclude_from_total_metric = [item.strip() for item in
                                                  self.opt.get('metrics_exclude_from_total_metric', '').split(',')]
        if self.max_response_len < 1:
            self.max_response_len = 100

    def report(self):
        base = super().report()
        m = dict()

        if self.metrics['total_unigram_cnt'] > 0:
            m['dist_1_cnt'] = len(self.metrics['dist_unigram_tokens'])
            m['dist_1_ratio'] = m['dist_1_cnt'] / self.metrics['total_unigram_cnt']

        if self.metrics['total_bigram_cnt'] > 0:
            m['dist_2_cnt'] = len(self.metrics['dist_bigram_tokens'])
            m['dist_2_ratio'] = m['dist_2_cnt'] / self.metrics['total_bigram_cnt']

        if self.metrics['total_trigram_cnt'] > 0:
            m['dist_3_cnt'] = len(self.metrics['dist_trigram_tokens'])
            m['dist_3_ratio'] = m['dist_3_cnt'] / self.metrics['total_trigram_cnt']

        if self.metrics['intra_unigram_cnt'] > 0:
            m['intra_dist_1'] = self.metrics['intra_unigram'] / self.metrics['intra_unigram_cnt']

        if self.metrics['intra_bigram_cnt'] > 0:
            m['intra_dist_2'] = self.metrics['intra_bigram'] / self.metrics['intra_bigram_cnt']

        if self.metrics['intra_trigram_cnt'] > 0:
            m['intra_dist_3'] = self.metrics['intra_trigram'] / self.metrics['intra_trigram_cnt']

        if self.metrics['response_length_cnt'] > 0:
            m['response_length'] = self.metrics['response_length'] / self.metrics['response_length_cnt']

        if self.metrics['embed_avg_cnt'] > 0:
            m['embed_avg'] = self.metrics['embed_avg'] / self.metrics['embed_avg_cnt']
        if self.metrics['embed_extrema_cnt'] > 0:
            m['embed_extrema'] = self.metrics['embed_extrema'] / self.metrics['embed_extrema_cnt']
        if self.metrics['embed_greedy_cnt'] > 0:
            m['embed_greedy'] = self.metrics['embed_greedy'] / self.metrics['embed_greedy_cnt']
        if self.metrics['embed_coh_cnt'] > 0:
            m['embed_coh'] = self.metrics['embed_coh'] / self.metrics['embed_coh_cnt']

        # Entropy
        if self.metrics['sent_entropy_uni_cnt'] > 0:
            m['sent_entropy_uni'] = self.metrics['sent_entropy_uni'] / self.metrics['sent_entropy_uni_cnt']
        if self.metrics['sent_entropy_bi_cnt'] > 0:
            m['sent_entropy_bi'] = self.metrics['sent_entropy_bi'] / self.metrics['sent_entropy_bi_cnt']
        if self.metrics['sent_entropy_tri_cnt'] > 0:
            m['sent_entropy_tri'] = self.metrics['sent_entropy_tri'] / self.metrics['sent_entropy_tri_cnt']
        if self.metrics['word_entropy_uni_cnt'] > 0:
            m['word_entropy_uni'] = self.metrics['word_entropy_uni'] / self.metrics['word_entropy_uni_cnt']
        if self.metrics['word_entropy_bi_cnt'] > 0:
            m['word_entropy_bi'] = self.metrics['word_entropy_bi'] / self.metrics['word_entropy_bi_cnt']
        if self.metrics['word_entropy_tri_cnt'] > 0:
            m['word_entropy_tri'] = self.metrics['word_entropy_tri'] / self.metrics['word_entropy_tri_cnt']

        # -- Ground-truth metrics
        if self.metrics['human_total_unigram_cnt'] > 0:
            m['human_dist_1_cnt'] = len(self.metrics['human_dist_unigram_tokens'])
            m['human_dist_1_ratio'] = m['human_dist_1_cnt'] / self.metrics['human_total_unigram_cnt']

        if self.metrics['human_total_bigram_cnt'] > 0:
            m['human_dist_2_cnt'] = len(self.metrics['human_dist_bigram_tokens'])
            m['human_dist_2_ratio'] = m['human_dist_2_cnt'] / self.metrics['human_total_bigram_cnt']

        if self.metrics['human_total_trigram_cnt'] > 0:
            m['human_dist_3_cnt'] = len(self.metrics['human_dist_trigram_tokens'])
            m['human_dist_3_ratio'] = m['human_dist_3_cnt'] / self.metrics['human_total_trigram_cnt']

        if self.metrics['human_intra_unigram_cnt'] > 0:
            m['human_intra_dist_1'] = self.metrics['human_intra_unigram'] / self.metrics['human_intra_unigram_cnt']

        if self.metrics['human_intra_bigram_cnt'] > 0:
            m['human_intra_dist_2'] = self.metrics['human_intra_bigram'] / self.metrics['human_intra_bigram_cnt']

        if self.metrics['human_intra_trigram_cnt'] > 0:
            m['human_intra_dist_3'] = self.metrics['human_intra_trigram'] / self.metrics['human_intra_trigram_cnt']

        if self.metrics['human_response_length_cnt'] > 0:
            m['human_response_length'] = self.metrics['human_response_length'] / self.metrics[
                'human_response_length_cnt']

        if self.metrics['human_embed_coh_cnt'] > 0:
            m['human_embed_coh'] = self.metrics['human_embed_coh'] / self.metrics['human_embed_coh_cnt']

        if self.metrics['human_sent_entropy_uni_cnt'] > 0:
            m['human_sent_entropy_uni'] = self.metrics['human_sent_entropy_uni'] / self.metrics[
                'human_sent_entropy_uni_cnt']
        if self.metrics['human_sent_entropy_bi_cnt'] > 0:
            m['human_sent_entropy_bi'] = self.metrics['human_sent_entropy_bi'] / self.metrics[
                'human_sent_entropy_bi_cnt']
        if self.metrics['human_sent_entropy_tri_cnt'] > 0:
            m['human_sent_entropy_tri'] = self.metrics['human_sent_entropy_tri'] / self.metrics[
                'human_sent_entropy_tri_cnt']
        if self.metrics['human_word_entropy_uni_cnt'] > 0:
            m['human_word_entropy_uni'] = self.metrics['human_word_entropy_uni'] / self.metrics[
                'human_word_entropy_uni_cnt']
        if self.metrics['human_word_entropy_bi_cnt'] > 0:
            m['human_word_entropy_bi'] = self.metrics['human_word_entropy_bi'] / self.metrics[
                'human_word_entropy_bi_cnt']
        if self.metrics['human_word_entropy_tri_cnt'] > 0:
            m['human_word_entropy_tri'] = self.metrics['human_word_entropy_tri'] / self.metrics[
                'human_word_entropy_tri_cnt']

        if not self.model.training:
            # TODO: add other metrics and balance these metrics
            m['total_metric'] = \
                (-base.get('ppl' if 'ppl' not in self.metrics_exclude_from_total_metric else 'NON_EXIST', 0) * 0.25) / 100 + \
                (m.get('dist_1_ratio', 0) + m.get('dist_2_ratio', 0) + clip_value(m.get('dist_3_ratio', 0), 0.001)) + \
                (m.get('embed_avg', 0) + m.get('embed_greedy', 0) + m.get('embed_extrema', 0) + m.get('embed_coh', 0)) + \
                (m.get('intra_dist_1', 0) + m.get('intra_dist_2', 0) + m.get('intra_dist_3', 0)) / 10 + \
                (m.get('word_entropy_uni', 0) + m.get('word_entropy_bi', 0) + m.get('word_entropy_tri', 0)) / 50 + \
                (m.get('response_length' if 'response_length' not in self.metrics_exclude_from_total_metric else 'NON_EXIST',
                       0)) / self.max_response_len

            # sent_entropy is strong correlated with word_entropy, so we only compute one of them
            # m.get('sent_entropy_uni', 0) + m.get('sent_entropy_bi', 0) + m.get('sent_entropy_tri', 0)

        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 5)
        return base

    def share(self):
        shared = super().share()
        shared['eval_embs'] = self.eval_embs
        shared['total_unigrams'] = self.total_unigrams
        shared['total_bigrams'] = self.total_bigrams
        shared['total_trigrams'] = self.total_trigrams
        return shared

    def add_metric(self, metric_name: str, default_value):
        assert self.metrics is not None, 'The metrics is not initialized!'
        assert type(metric_name) == str, 'metric_name should be a string!'
        self.metrics[metric_name] = default_value

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['total_unigram_cnt'] = 0
        self.metrics['total_bigram_cnt'] = 0
        self.metrics['total_trigram_cnt'] = 0
        self.metrics['dist_unigram_tokens'] = set()
        self.metrics['dist_bigram_tokens'] = set()
        self.metrics['dist_trigram_tokens'] = set()
        self.metrics['intra_unigram_cnt'] = 0
        self.metrics['intra_unigram'] = 0.0
        self.metrics['intra_bigram_cnt'] = 0
        self.metrics['intra_bigram'] = 0.0
        self.metrics['intra_trigram_cnt'] = 0
        self.metrics['intra_trigram'] = 0.0

        self.metrics['response_length_cnt'] = 0
        self.metrics['response_length'] = 0.0

        self.metrics['embed_avg_cnt'] = 0
        self.metrics['embed_avg'] = 0.0
        self.metrics['embed_greedy_cnt'] = 0
        self.metrics['embed_greedy'] = 0.0
        self.metrics['embed_extrema_cnt'] = 0
        self.metrics['embed_extrema'] = 0.0
        self.metrics['embed_coh_cnt'] = 0
        self.metrics['embed_coh'] = 0.0

        self.metrics['sent_entropy_uni_cnt'] = 0.0
        self.metrics['sent_entropy_uni'] = 0
        self.metrics['sent_entropy_bi_cnt'] = 0.0
        self.metrics['sent_entropy_bi'] = 0
        self.metrics['sent_entropy_tri_cnt'] = 0.0
        self.metrics['sent_entropy_tri'] = 0

        self.metrics['word_entropy_uni_cnt'] = 0
        self.metrics['word_entropy_uni'] = 0.0
        self.metrics['word_entropy_bi_cnt'] = 0
        self.metrics['word_entropy_bi'] = 0.0
        self.metrics['word_entropy_tri_cnt'] = 0
        self.metrics['word_entropy_tri'] = 0.0

        self.metrics['human_total_unigram_cnt'] = 0
        self.metrics['human_total_bigram_cnt'] = 0
        self.metrics['human_total_trigram_cnt'] = 0
        self.metrics['human_dist_unigram_tokens'] = set()
        self.metrics['human_dist_bigram_tokens'] = set()
        self.metrics['human_dist_trigram_tokens'] = set()
        self.metrics['human_intra_unigram_cnt'] = 0
        self.metrics['human_intra_unigram'] = 0.0
        self.metrics['human_intra_bigram_cnt'] = 0
        self.metrics['human_intra_bigram'] = 0.0
        self.metrics['human_intra_trigram_cnt'] = 0
        self.metrics['human_intra_trigram'] = 0.0

        self.metrics['human_response_length_cnt'] = 0
        self.metrics['human_response_length'] = 0.0
        self.metrics['human_embed_coh_cnt'] = 0
        self.metrics['human_embed_coh'] = 0.0

        self.metrics['human_sent_entropy_uni_cnt'] = 0.0
        self.metrics['human_sent_entropy_uni'] = 0
        self.metrics['human_sent_entropy_bi_cnt'] = 0.0
        self.metrics['human_sent_entropy_bi'] = 0
        self.metrics['human_sent_entropy_tri_cnt'] = 0.0
        self.metrics['human_sent_entropy_tri'] = 0

        self.metrics['human_word_entropy_uni_cnt'] = 0
        self.metrics['human_word_entropy_uni'] = 0.0
        self.metrics['human_word_entropy_bi_cnt'] = 0
        self.metrics['human_word_entropy_bi'] = 0.0
        self.metrics['human_word_entropy_tri_cnt'] = 0
        self.metrics['human_word_entropy_tri'] = 0.0

    def _arora_weight(self, tokens):
        if self.opt.get('arora_weighting', True):
            a = 0.001
            weights = [a / (a + self.dict.unigram_freq[(tok,)] / self.total_unigrams) for tok in tokens]
        else:
            weights = [1.0 for _ in tokens]
        return weights

    def _eval_embedding_metrics(self, output, label_text, context):
        # Evaluation of embedding distance between predictions and labels

        text = output.text
        for i in range(len(text)):
            pred_sent = text[i].split()
            target_sent = label_text[i].split()
            target_sent_w, pred_sent_w = self._arora_weight(target_sent), self._arora_weight(pred_sent)
            emb_avg = sentence_average_score(
                target_sent, pred_sent, self.eval_embs,
                target_sent_w, pred_sent_w
            )  # maybe None

            emb_greedy1 = sentence_greedy_score(
                target_sent, pred_sent, self.eval_embs,
                # target_sent_w, pred_sent_w
            )
            emb_greedy2 = sentence_greedy_score(
                pred_sent, target_sent, self.eval_embs,
                # pred_sent_w, target_sent_w
            )
            emb_greedy = (emb_greedy1 + emb_greedy2) / 2.0

            emb_extrema = sentence_extrema_score(
                target_sent, pred_sent, self.eval_embs,
                # target_sent_w, pred_sent_w
            )  # maybe None
            if emb_avg is not None:
                self.metrics['embed_avg_cnt'] += 1
                self.metrics['embed_avg'] += emb_avg

            self.metrics['embed_greedy_cnt'] += 1
            self.metrics['embed_greedy'] += emb_greedy

            if emb_extrema is not None:
                self.metrics['embed_extrema_cnt'] += 1
                self.metrics['embed_extrema'] += emb_extrema

            current_context = context[i]
            lastutt = list(filter(lambda x: not (x == self.P1_TOKEN or x == self.P2_TOKEN),
                                  current_context.split(self.opt.get('delimiter', '\n'))[-1].split()))
            lastutt_w = self._arora_weight(lastutt)
            emb_coh = sentence_average_score(
                lastutt, pred_sent, self.eval_embs,
                lastutt_w, pred_sent_w
            )
            if emb_coh is not None:
                self.metrics['embed_coh_cnt'] += 1
                self.metrics['embed_coh'] += emb_coh

            human_emb_coh = sentence_average_score(
                lastutt, target_sent, self.eval_embs,
                lastutt_w, target_sent_w
            )
            if human_emb_coh is not None:
                self.metrics['human_embed_coh_cnt'] += 1
                self.metrics['human_embed_coh'] += human_emb_coh

    def _eval_distinct_metrics(self, output, labels):
        text = output.text

        for i in range(len(text)):
            pred_sent = text[i]
            unigram_tokens = pred_sent.split()
            bigram_tokens = list(bigrams(unigram_tokens))
            trigram_tokens = list(trigrams(unigram_tokens))

            self.metrics['total_unigram_cnt'] += len(unigram_tokens)
            self.metrics['total_bigram_cnt'] += len(bigram_tokens)
            self.metrics['total_trigram_cnt'] += len(trigram_tokens)
            self.metrics['dist_unigram_tokens'] = set.union(
                self.metrics['dist_unigram_tokens'], set(unigram_tokens)
            )
            self.metrics['dist_bigram_tokens'] = set.union(
                self.metrics['dist_bigram_tokens'], set(bigram_tokens)
            )
            self.metrics['dist_trigram_tokens'] = set.union(
                self.metrics['dist_trigram_tokens'], set(trigram_tokens)
            )

            self.metrics['intra_unigram_cnt'] += 1
            self.metrics['intra_unigram'] += ((len(set(unigram_tokens)) + sys.float_info.epsilon) / (
                    len(unigram_tokens) + sys.float_info.epsilon))
            self.metrics['intra_bigram_cnt'] += 1
            self.metrics['intra_bigram'] += ((len(set(bigram_tokens)) + sys.float_info.epsilon) / (
                    len(bigram_tokens) + sys.float_info.epsilon))
            self.metrics['intra_trigram_cnt'] += 1
            self.metrics['intra_trigram'] += ((len(set(trigram_tokens)) + sys.float_info.epsilon) / (
                    len(trigram_tokens) + sys.float_info.epsilon))

            self.metrics['response_length_cnt'] += 1
            self.metrics['response_length'] += len(unigram_tokens)

            # Record ground-truth metrics
            label_sent = labels[i]
            human_unigram_tokens = label_sent.split()
            human_bigram_tokens = list(bigrams(human_unigram_tokens))
            human_trigram_tokens = list(trigrams(human_unigram_tokens))

            self.metrics['human_total_unigram_cnt'] += len(human_unigram_tokens)
            self.metrics['human_total_bigram_cnt'] += len(human_bigram_tokens)
            self.metrics['human_total_trigram_cnt'] += len(human_trigram_tokens)
            self.metrics['human_dist_unigram_tokens'] = set.union(
                self.metrics['human_dist_unigram_tokens'], set(human_unigram_tokens)
            )
            self.metrics['human_dist_bigram_tokens'] = set.union(
                self.metrics['human_dist_bigram_tokens'], set(human_bigram_tokens)
            )
            self.metrics['human_dist_trigram_tokens'] = set.union(
                self.metrics['human_dist_trigram_tokens'], set(human_trigram_tokens)
            )

            self.metrics['human_intra_unigram_cnt'] += 1
            self.metrics['human_intra_unigram'] += ((len(set(human_unigram_tokens)) + sys.float_info.epsilon) / (
                    len(human_unigram_tokens) + sys.float_info.epsilon))
            self.metrics['human_intra_bigram_cnt'] += 1
            self.metrics['human_intra_bigram'] += ((len(set(human_bigram_tokens)) + sys.float_info.epsilon) / (
                    len(human_bigram_tokens) + sys.float_info.epsilon))
            self.metrics['human_intra_trigram_cnt'] += 1
            self.metrics['human_intra_trigram'] += ((len(set(human_trigram_tokens)) + sys.float_info.epsilon) / (
                    len(human_trigram_tokens) + sys.float_info.epsilon))

            self.metrics['human_response_length_cnt'] += 1
            self.metrics['human_response_length'] += len(human_unigram_tokens)

    def _eval_sent_entropy(self, sents, human=''):
        for i in range(len(sents)):
            sent = sents[i]
            sent_tokens = sent.split()
            unigrams_ = nltk.ngrams(sent_tokens, 1)
            bigrams_ = bigrams(sent_tokens)
            trigrams_ = trigrams(sent_tokens)
            prob_unigrams = [self.dict.unigram_freq[uni_tok] / self.total_unigrams for uni_tok in unigrams_]
            prob_bigrams = [self.dict.bigram_freq[bi_tok] / self.total_bigrams for bi_tok in bigrams_]
            prob_trigrams = [self.dict.trigram_freq[tri_tok] / self.total_trigrams for tri_tok in trigrams_]

            # smoothing zero values
            prob_unigrams = np.asarray([p if p > 0 else 1 for p in prob_unigrams])
            prob_bigrams = np.asarray([p if p > 0 else 1 for p in prob_bigrams])
            prob_trigrams = np.asarray([p if p > 0 else 1 for p in prob_trigrams])

            sent_entropy_uni = -np.sum(np.log2(prob_unigrams))
            sent_entropy_bi = -np.sum(np.log2(prob_bigrams))
            sent_entropy_tri = -np.sum(np.log2(prob_trigrams))

            word_entropy_uni = sent_entropy_uni / (len(prob_unigrams) + sys.float_info.epsilon)
            word_entropy_bi = sent_entropy_bi / (len(prob_bigrams) + sys.float_info.epsilon)
            word_entropy_tri = sent_entropy_tri / (len(prob_trigrams) + sys.float_info.epsilon)

            self.metrics[human + 'sent_entropy_uni_cnt'] += 1
            self.metrics[human + 'sent_entropy_uni'] += sent_entropy_uni
            self.metrics[human + 'sent_entropy_bi_cnt'] += 1
            self.metrics[human + 'sent_entropy_bi'] += sent_entropy_bi
            self.metrics[human + 'sent_entropy_tri_cnt'] += 1
            self.metrics[human + 'sent_entropy_tri'] += sent_entropy_tri

            self.metrics[human + 'word_entropy_uni_cnt'] += 1
            self.metrics[human + 'word_entropy_uni'] += word_entropy_uni
            self.metrics[human + 'word_entropy_bi_cnt'] += 1
            self.metrics[human + 'word_entropy_bi'] += word_entropy_bi
            self.metrics[human + 'word_entropy_tri_cnt'] += 1
            self.metrics[human + 'word_entropy_tri'] += word_entropy_tri

    def _eval_entropy_metrics(self, output, labels):
        text = output.text
        self._eval_sent_entropy(text)
        self._eval_sent_entropy(labels, human='human_')

    def inference(self, input_sents):
        assert type(input_sents) is list and len(input_sents) > 0
        if self.opt['dict_lower']:
            batch_obs = [{'text': text.lower(), 'episode_done': True} for text in input_sents]
        else:
            batch_obs = [{'text': text, 'episode_done': True} for text in input_sents]

        # batch_observed_obs = [self.observe(obs) for obs in batch_obs]
        batch_observed_obs = []
        for idx, obs in enumerate(batch_obs):
            obs = self.observe(obs)
            batch_observed_obs.append(obs)
            self.self_observe(obs)

        batch = self.batchify(batch_observed_obs)
        eval_output = self.eval_step(batch)

        # self.history.reset()
        # self.observation = None
        # self.__expecting_clear_history = False

        output = eval_output
        return output

    def eval_step(self, batch):
        # There is no need for the subclasses to evaluate these metrics again
        output = super().eval_step(batch)
        label_text = batch.labels
        context = [obs['text'] for obs in batch.observations]
        if label_text and output:
            self._eval_embedding_metrics(output, label_text, context)
            self._eval_distinct_metrics(output, label_text)
            self._eval_entropy_metrics(output, label_text)

        # sampling predictions for printing
        if output.text is not None and batch.label_vec is not None:
            preds = output.text
            for i in range(len(preds)):
                if random.random() > (1 - self.opt['report_freq']):
                    target_text = self._v2t(batch.label_vec[i])
                    print('TEXT: ', context[i])
                    print('TARGET: ', target_text)
                    print('PREDICTION: ', preds[i], '\n~')

        return output
