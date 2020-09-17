"""
Copyright 2018 NAVER Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)

from parlai.agents.dialog_wae.helper import gVar, gData
from parlai.agents.hy_lib.common_utils import texts_to_bow
from parlai.agents.seq2seq.modules import OutputLayer
from parlai.agents.seq2seq.modules import UnknownDropout
from parlai.agents.seq2seq.modules import (
    _transpose_hidden_state,
    AttentionLayer,
)


class Variation(nn.Module):
    def __init__(self, input_size, z_size, use_cuda=False):
        super(Variation, self).__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.use_cua = use_cuda
        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
        )
        self.context_to_mu = nn.Linear(z_size, z_size)
        self.context_to_logsigma = nn.Linear(z_size, z_size)

        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mu)
        self.init_weights(self.context_to_logsigma)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def forward(self, context):
        batch_size, _ = context.size()
        context = self.fc(context)
        mu = self.context_to_mu(context)
        logsigma = self.context_to_logsigma(context)

        # mu = torch.clamp(mu, -30, 30)
        logsigma = torch.clamp(logsigma, -20, 20)
        std = torch.exp(0.5 * logsigma)
        epsilon = gVar(torch.randn([batch_size, self.z_size]), use_cuda=self.use_cua)
        z = epsilon * std + mu
        return z, mu, logsigma


class MixVariation(nn.Module):
    def __init__(self, input_size, z_size, n_components,
                 gumbel_temp=0.1,
                 use_cuda=False):
        super(MixVariation, self).__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.n_components = n_components
        self.gumbel_temp = gumbel_temp
        self.use_cua = use_cuda

        self.pi_net = nn.Sequential(
            nn.Linear(z_size, z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, n_components),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_size, z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
        )
        self.context_to_mu = nn.Linear(z_size, n_components * z_size)
        self.context_to_logsigma = nn.Linear(z_size, n_components * z_size)
        self.pi_net.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mu)
        self.init_weights(self.context_to_logsigma)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def forward(self, context):
        batch_size, _ = context.size()
        context = self.fc(context)

        pi = self.pi_net(context)
        pi = F.gumbel_softmax(pi, tau=self.gumbel_temp, hard=True, eps=1e-10)
        pi = pi.unsqueeze(1)

        mus = self.context_to_mu(context)
        logsigmas = self.context_to_logsigma(context)

        # mus = torch.clamp(mus, -30, 30)
        logsigmas = torch.clamp(logsigmas, -20, 20)

        stds = torch.exp(0.5 * logsigmas)

        epsilons = gVar(torch.randn([batch_size, self.n_components * self.z_size]), self.use_cua)

        zi = (epsilons * stds + mus).view(batch_size, self.n_components, self.z_size)
        z = torch.bmm(pi, zi).squeeze(1)  # [batch_sz x z_sz]
        mu = torch.bmm(pi, mus.view(batch_size, self.n_components, self.z_size))
        logsigma = torch.bmm(pi, logsigmas.view(batch_size, self.n_components, self.z_size))
        return z, mu, logsigma


class Encoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, n_layers,
                 noise_radius=0.2, input_dropout=0., unknown_idx=None, dropout=0.2,
                 rnn_class='gru', use_cuda=False):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.noise_radius = noise_radius
        self.n_layers = n_layers
        self.bidirectional = True
        self.dirs = 2
        assert type(self.bidirectional) == bool

        self.dropout = nn.Dropout(p=dropout)
        if input_dropout > 0 and unknown_idx is None:
            raise RuntimeError('input_dropout > 0 but unknown_idx not set')
        self.input_dropout = UnknownDropout(unknown_idx, input_dropout)
        self.embedding = embedder
        self.rnn_class = rnn_class
        if rnn_class == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers,
                              dropout=dropout if n_layers > 1 else 0,
                              batch_first=True,
                              bidirectional=self.bidirectional)
        elif rnn_class == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers,
                               dropout=dropout if n_layers > 1 else 0,
                               batch_first=True,
                               bidirectional=self.bidirectional)
        else:
            raise RuntimeError('RNN class {} is not supported yet!'.format(rnn_class))
        self.use_cuda = use_cuda
        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, inputs, input_lens=None, noise=False):
        inputs = self.input_dropout(inputs)
        attn_mask = inputs.ne(0)
        if self.embedding is not None:
            inputs = self.embedding(inputs)

        batch_size, seq_len, emb_size = inputs.size()
        inputs = self.dropout(inputs)

        self.rnn.flatten_parameters()
        encoder_output, hidden = self.rnn(inputs)

        h_n = hidden[0] if self.rnn_class == 'lstm' else hidden
        h_n = h_n.view(self.n_layers, self.dirs, batch_size, self.hidden_size)
        enc = h_n[-1].transpose(1, 0).contiguous().view(batch_size, -1)  # bsz, num_dirs*hidden_size

        if isinstance(self.rnn, nn.LSTM):
            hidden = (
                hidden[0].view(-1, self.dirs, batch_size, self.hidden_size).sum(1),
                hidden[1].view(-1, self.dirs, batch_size, self.hidden_size).sum(1),
            )
        else:
            hidden = hidden.view(-1, self.dirs, batch_size, self.hidden_size).sum(1)

        hidden = _transpose_hidden_state(hidden)

        if noise and self.noise_radius > 0:
            gauss_noise = gVar(torch.normal(means=torch.zeros(enc.size()), std=self.noise_radius),
                               self.use_cuda)
            enc = enc + gauss_noise

        utt_encoder_states = (encoder_output, hidden, attn_mask)
        return enc, utt_encoder_states


class ContextEncoder(nn.Module):
    def __init__(
            self,
            utt_encoder,
            input_size,
            hidden_size,
            n_layers=1,
            noise_radius=0.2,
            dropout=0.1,
            rnn_class='gru',
            use_cuda=False,
            attn_type='none',
            attn_length=-1,
    ):
        super(ContextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.noise_radius = noise_radius
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

        self.utt_encoder = utt_encoder
        self.rnn_class = rnn_class
        if self.rnn_class == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,
                              num_layers=n_layers,
                              dropout=dropout if n_layers > 1 else 0,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size,
                               num_layers=n_layers,
                               dropout=dropout if n_layers > 1 else 0,
                               batch_first=True)
        self.use_cuda = use_cuda
        self.word_attention = AttentionLayer(
            attn_type=attn_type,
            hiddensize=hidden_size,
            embeddingsize=input_size,
            bidirectional=True,  # UtteranceEncoder is always bidirectional
            attn_length=attn_length,
            attn_time='post',
        )
        self.attn_type = attn_type
        self.init_weights()

    def init_weights(self):
        # initialize the gate weights with orthogonal
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, context, context_lens, utt_lens, floors, noise=False):
        batch_size, max_context_len, max_utt_len = context.size()
        utts = context.view(-1, max_utt_len)
        batch_max_lens = torch.arange(max_context_len).expand(batch_size, max_context_len)
        if self.use_cuda:
            batch_max_lens = batch_max_lens.cuda()
        context_mask = batch_max_lens < context_lens.unsqueeze(1)
        utt_lens = utt_lens.view(-1)
        utt_encs, utt_encoder_states = self.utt_encoder(utts, utt_lens)
        utt_encs = utt_encs.view(batch_size, max_context_len, -1)
        utt_encoder_output, utt_hidden, utt_attn_mask = utt_encoder_states
        utt_encoder_output = utt_encoder_output.view(
            batch_size,
            max_context_len,
            max_utt_len,
            self.utt_encoder.dirs * self.utt_encoder.hidden_size)
        utt_hidden = _transpose_hidden_state(utt_hidden)
        if isinstance(utt_hidden, tuple):
            utt_hidden = tuple(
                x.view(
                    self.utt_encoder.n_layers,
                    batch_size,
                    max_context_len,
                    self.utt_encoder.hidden_size
                ).contiguous() for x in utt_hidden)
        else:
            utt_hidden = utt_hidden.view(
                self.utt_encoder.n_layers,
                batch_size,
                max_context_len,
                self.utt_encoder.hidden_size
            ).contiguous()
        utt_attn_mask = utt_attn_mask.view(batch_size, max_context_len, max_utt_len)

        floor_one_hot = gVar(torch.zeros(floors.numel(), 2), self.use_cuda)
        floor_one_hot.data.scatter_(1, floors.view(-1, 1), 1)
        floor_one_hot = floor_one_hot.view(-1, max_context_len, 2)
        utt_floor_encs = torch.cat([utt_encs, floor_one_hot], 2)

        utt_floor_encs = self.dropout(utt_floor_encs)
        self.rnn.flatten_parameters()

        if self.rnn_class == 'lstm':
            new_hidden = tuple(x[:, :, -1, :].contiguous() for x in utt_hidden)
        else:
            new_hidden = utt_hidden[:, :, -1, :].contiguous()

        if self.attn_type != 'none':
            output = []
            for i in range(max_context_len):
                o, new_hidden = self.rnn(utt_floor_encs[:, i, :].unsqueeze(1), new_hidden)
                o, _ = self.word_attention(
                    o,
                    new_hidden,
                    (utt_encoder_output[:, i, :, :], utt_attn_mask[:, i, :])
                )
                output.append(o)

            context_encoder_output = torch.cat(output, dim=1).to(utt_floor_encs.device)
        else:
            utt_floor_encs = pack_padded_sequence(utt_floor_encs, context_lens,
                                                  batch_first=True,
                                                  enforce_sorted=False)

            context_encoder_output, new_hidden = self.rnn(utt_floor_encs, new_hidden)
            context_encoder_output, _ = pad_packed_sequence(
                context_encoder_output, batch_first=True, total_length=max_context_len
            )

        new_hidden = _transpose_hidden_state(new_hidden)
        if self.rnn_class == 'lstm':
            enc = new_hidden[0]
        else:
            enc = new_hidden
        enc = enc.contiguous().view(batch_size, -1)

        if noise and self.noise_radius > 0:
            gauss_noise = gVar(torch.normal(means=torch.zeros(enc.size()), std=self.noise_radius),
                               self.use_cuda)
            enc = enc + gauss_noise
        return enc, (context_encoder_output, new_hidden, context_mask)


class Decoder(nn.Module):
    def __init__(
            self,
            embedder,
            input_size,
            hidden_size,
            vocab_size,
            n_layers=1,
            dropout=0.2,
            rnn_class='gru',
            use_cuda=False,
            padding_idx=0,
            topp=0.9,
            attn_type='none',
            attn_length=-1,
    ):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=dropout)
        self.topp = topp
        self.embedding = embedder
        self.rnn_class = rnn_class
        if self.rnn_class == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers,
                              dropout=dropout if n_layers > 1 else 0,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers,
                               dropout=dropout if n_layers > 1 else 0,
                               batch_first=True)

        self.use_cuda = use_cuda
        self.out = OutputLayer(
            vocab_size,
            input_size,
            hidden_size,
            dropout=dropout,
            padding_idx=padding_idx
        )
        self.context_attention = AttentionLayer(
            attn_type=attn_type,
            hiddensize=hidden_size,
            embeddingsize=input_size,
            bidirectional=False,  # ContextRNN is unidirectional
            attn_length=attn_length,
            attn_time='post',
        )
        self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def forward(self, init_hidden, context=None, inputs=None, lens=None, context_encoder_states=None):
        batch_size, maxlen = inputs.size()
        if self.embedding is not None:
            inputs = self.embedding(inputs)

        if context is not None:
            repeated_context = context.unsqueeze(1).repeat(1, maxlen, 1)
            inputs = torch.cat([inputs, repeated_context], 2)

        inputs = self.dropout(inputs)
        self.rnn.flatten_parameters()
        if context_encoder_states is not None:
            # attention on the context encoder outputs
            context_enc_state, context_enc_hidden, context_attn_mask = context_encoder_states
            context_attn_params = (context_enc_state, context_attn_mask)
            context_hidden = _transpose_hidden_state(context_enc_hidden)
            if isinstance(context_hidden, tuple):
                context_hidden = tuple(x.contiguous() for x in context_hidden)
            else:
                context_hidden = context_hidden.contiguous()
            new_hidden = context_hidden
            output = []

            for i in range(maxlen):
                o, new_hidden = self.rnn(inputs[:, i, :].unsqueeze(1), new_hidden)
                o, _ = self.context_attention(o, new_hidden, context_attn_params)
                output.append(o)
            output = torch.cat(output, dim=1).to(inputs.device)
        else:
            init_hidden = init_hidden.view(batch_size, self.n_layers, self.hidden_size)
            init_hidden = init_hidden.transpose(0, 1).contiguous()
            if self.rnn_class == 'lstm':
                init_hidden = (init_hidden, init_hidden)
            output, _ = self.rnn(inputs, init_hidden)

        decoded = self.out(output)
        decoded = decoded.view(batch_size, maxlen, self.vocab_size)
        return decoded

    def sampling(self, init_hidden, context, maxlen, SOS_tok, EOS_tok,
                 mode='greedy', context_encoder_states=None):
        batch_size = init_hidden.size(0)
        decoded_words = np.zeros((batch_size, maxlen), dtype=np.int)
        sample_lens = np.zeros(batch_size, dtype=np.int)

        # noinspection PyArgumentList
        decoder_input = gVar(torch.LongTensor([[SOS_tok] * batch_size]).view(batch_size, 1), self.use_cuda)
        decoder_input = self.embedding(decoder_input) if self.embedding is not None else decoder_input
        decoder_input = torch.cat([decoder_input, context.unsqueeze(1)], 2) if context is not None else decoder_input

        if context_encoder_states is not None:
            context_enc_state, context_enc_hidden, context_attn_mask = context_encoder_states
            context_attn_params = (context_enc_state, context_attn_mask)
            context_hidden = _transpose_hidden_state(context_enc_hidden)
            if isinstance(context_hidden, tuple):
                context_hidden = tuple(x.contiguous() for x in context_hidden)
            else:
                context_hidden = context_hidden.contiguous()
            decoder_hidden = context_hidden
        else:
            decoder_hidden = init_hidden.view(batch_size, self.n_layers, self.hidden_size)
            decoder_hidden = decoder_hidden.transpose(0, 1).contiguous()
            if self.rnn_class == 'lstm':
                decoder_hidden = (decoder_hidden, decoder_hidden)

        for di in range(maxlen):
            decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
            if context_encoder_states is not None:
                # apply attention
                decoder_output, _ = self.context_attention(decoder_output, decoder_hidden, context_attn_params)

            decoder_output = self.out(decoder_output)

            if mode == 'greedy':
                topi = decoder_output[:, -1].max(1, keepdim=True)[1]
            elif mode == 'nucleus':
                # Nucelus, aka top-p sampling (Holtzman et al., 2019).
                logprobs = decoder_output[:, -1]
                probs = torch.softmax(logprobs, dim=-1)
                sprobs, sinds = probs.sort(dim=-1, descending=True)
                mask = (sprobs.cumsum(dim=-1) - sprobs[:, :1]) >= self.topp
                sprobs[mask] = 0
                sprobs.div_(sprobs.sum(dim=-1).unsqueeze(1))
                choices = torch.multinomial(sprobs, 1)[:, 0]
                hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
                topi = sinds[hyp_ids, choices].unsqueeze(dim=1)
            else:
                raise RuntimeError('inference method: {} not supported yet!')

            decoder_input = self.embedding(topi) if self.embedding is not None else topi
            decoder_input = torch.cat([decoder_input, context.unsqueeze(1)], 2) if context is not None else decoder_input
            ni = topi.squeeze().data.cpu().numpy()
            decoded_words[:, di] = ni

        for i in range(batch_size):
            for word in decoded_words[i]:
                if word == EOS_tok:
                    break
                sample_lens[i] += 1
        return decoded_words, sample_lens


class DialogWAE(nn.Module):
    def __init__(
            self,
            config,
            vocab_size,
            PAD_token=0,
            unknown_idx=None,
            use_cuda=True,
            special_tokens=None,
    ):
        super(DialogWAE, self).__init__()
        if not config['hred']:
            # attention is only applicable for HRED, HRED+Attention-->HRAN
            config['attention'] = 'none'

        self.vocab_size = vocab_size
        self.maxlen = config['maxlen']
        self.lambda_gp = config['lambda_gp']
        self.PAD_token = PAD_token
        self.special_tokens = special_tokens
        self.use_cuda = use_cuda
        self.embedder = nn.Embedding(vocab_size, config['embeddingsize'], padding_idx=PAD_token)
        self.utt_encoder = Encoder(self.embedder,
                                   input_size=config['embeddingsize'],
                                   hidden_size=config['hiddensize'],
                                   n_layers=config['numlayers'],
                                   noise_radius=config['noise_radius'],
                                   input_dropout=config['input_dropout'],
                                   unknown_idx=unknown_idx,
                                   dropout=config['dropout'],
                                   rnn_class=config.get('rnn_class', 'gru'),
                                   use_cuda=self.use_cuda)

        self.context_encoder = ContextEncoder(utt_encoder=self.utt_encoder,
                                              input_size=config['hiddensize'] * 2 + 2,
                                              hidden_size=config['hiddensize'],
                                              n_layers=config['numlayers'],
                                              dropout=config['dropout'],
                                              noise_radius=config['noise_radius'],
                                              rnn_class=config.get('rnn_class', 'gru'),
                                              use_cuda=self.use_cuda,
                                              attn_type=config['attention'],
                                              attn_length=config['attention_length'])

        self.prior_net = Variation(config['hiddensize'] * config['numlayers'], config['z_size'], use_cuda=self.use_cuda)  # p(e|c)
        self.post_net = Variation(config['hiddensize'] * (2 + config['numlayers']), config['z_size'], use_cuda=self.use_cuda)  # q(e|c,x)

        self.post_generator = nn.Sequential(
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size'] * config['numlayers'])
        )
        self.post_generator.apply(self.init_weights)

        self.prior_generator = nn.Sequential(
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size'] * config['numlayers'])
        )

        self.prior_generator.apply(self.init_weights)
        if config.get('hred', False):
            decoder_hidden_size = config['hiddensize']
        else:
            decoder_hidden_size = config['hiddensize'] + config['z_size']

        if config.get('vhred', False):
            self.vhred_priori = Variation(
                config['hiddensize'] * config['numlayers'],
                config['z_size'] * config['numlayers'],
                use_cuda=self.use_cuda
            )
            self.vhred_bow_project = nn.Sequential(
                nn.Linear((config['hiddensize'] + config['z_size']) * config['numlayers'], config['hiddensize'] * 2),
                nn.BatchNorm1d(config['hiddensize'] * 2, eps=1e-05, momentum=0.1),
                nn.Tanh(),
                nn.Linear(config['hiddensize'] * 2, config['hiddensize'] * 2),
                nn.BatchNorm1d(config['hiddensize'] * 2, eps=1e-05, momentum=0.1),
                nn.Tanh(),
                nn.Linear(config['hiddensize'] * 2, vocab_size)
            )

            self.vhred_posterior = Variation(
                config['hiddensize'] * (2 + config['numlayers']),
                config['z_size'] * config['numlayers'],
                use_cuda=self.use_cuda
            )

        self.decoder = Decoder(self.embedder,
                               input_size=config['embeddingsize'],
                               hidden_size=decoder_hidden_size,
                               vocab_size=vocab_size,
                               n_layers=config['numlayers'],
                               dropout=config['dropout'],
                               rnn_class=config.get('rnn_class', 'gru'),
                               use_cuda=self.use_cuda,
                               padding_idx=PAD_token,
                               topp=config['topp'],
                               attn_type=config['attention'],
                               attn_length=config['attention_length'])

        self.discriminator = nn.Sequential(
            nn.Linear((config['hiddensize'] + config['z_size']) * config['numlayers'], config['hiddensize'] * 2),
            nn.BatchNorm1d(config['hiddensize'] * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config['hiddensize'] * 2, config['hiddensize'] * 2),
            nn.BatchNorm1d(config['hiddensize'] * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config['hiddensize'] * 2, 1),
        )
        self.discriminator.apply(self.init_weights)
        self.config = config

        self.one = gData(torch.FloatTensor([1]), self.use_cuda)
        self.minus_one = self.one * -1

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def sample_code_post(self, x, c):
        e, _, _ = self.post_net(torch.cat((x, c), 1))
        z = self.post_generator(e)
        return z

    def sample_code_prior(self, c):
        e, _, _ = self.prior_net(c)
        z = self.prior_generator(e)
        return z

    def normal_kl_div(self, mean1, logvar1, mean2=None, logvar2=None):
        if mean2 is None:
            mean2 = Variable(torch.FloatTensor([0.0])).unsqueeze(dim=1).expand(mean1.size(0), mean1.size(1))
            if self.use_cuda:
                mean2 = mean2.cuda()
        if logvar2 is None:
            logvar2 = Variable(torch.FloatTensor([0.0])).unsqueeze(dim=1).expand(logvar1.size(0), logvar1.size(1))
            if self.use_cuda:
                logvar2 = logvar2.cuda()
        kl_div = 0.5 * torch.sum(
            logvar2 - logvar1 + (torch.exp(logvar1) + (mean1 - mean2).pow(2)) / torch.exp(logvar2) - 1.0,
            dim=1).mean().squeeze()
        return kl_div

    @staticmethod
    def anneal_weight(step):
        return (math.tanh((step - 3500) / 1000) + 1) / 2

    def _compute_bow_loss(self, bow_logits, response):
        target_bow = texts_to_bow(response, self.vocab_size, self.special_tokens)
        if self.use_cuda:
            target_bow = target_bow.cuda()
        bow_loss = -F.log_softmax(bow_logits, dim=1) * target_bow
        # Compute per token loss
        # bow_loss = torch.sum(bow_loss) / torch.sum(target_bow)
        bow_loss = torch.sum(bow_loss) / response.size(0)
        return bow_loss

    def train_G(self, context, context_lens, utt_lens, floors, response, res_lens, optimizer_G):
        # self.context_encoder.eval()
        optimizer_G.zero_grad()

        for p in self.discriminator.parameters():
            p.requires_grad = False

        with torch.no_grad():
            context_hidden, *_ = self.context_encoder(context, context_lens, utt_lens, floors)
            # context_hidden: (bsz, num_context_rnn_layers * hiddensize)
            x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
            # x: (bsz, num_dirs*hiddensize)
        # -----------------posterior samples ---------------------------
        z_post = self.sample_code_post(x.detach(), context_hidden.detach())

        errG_post = torch.mean(self.discriminator(torch.cat((z_post, context_hidden.detach()), 1)))
        errG_post.backward(self.minus_one)

        # ----------------- prior samples ---------------------------
        prior_z = self.sample_code_prior(context_hidden.detach())
        errG_prior = torch.mean(self.discriminator(torch.cat((prior_z, context_hidden.detach()), 1)))
        errG_prior.backward(self.one)

        optimizer_G.step()

        for p in self.discriminator.parameters():
            p.requires_grad = True

        costG = errG_prior - errG_post
        return {'train_loss_G': costG.item()}

    def train_D(self, context, context_lens, utt_lens, floors, response, res_lens, optimizer_D):
        # self.context_encoder.eval()
        self.discriminator.train()

        optimizer_D.zero_grad()

        batch_size = context.size(0)

        with torch.no_grad():
            context_hidden, *_ = self.context_encoder(context, context_lens, utt_lens, floors)
            # context_hidden: (bsz, num_context_rnn_layers * hiddensize)
            x, _ = self.utt_encoder(response[:, 1:], res_lens - 1)
            # x: (bsz, num_dirs*hiddensize)

        post_z = self.sample_code_post(x, context_hidden)
        errD_post = torch.mean(self.discriminator(torch.cat((post_z.detach(), context_hidden.detach()), 1)))
        errD_post.backward(self.one)

        prior_z = self.sample_code_prior(context_hidden)
        errD_prior = torch.mean(self.discriminator(torch.cat((prior_z.detach(), context_hidden.detach()), 1)))
        errD_prior.backward(self.minus_one)

        alpha = gData(torch.rand(batch_size, 1), self.use_cuda)
        alpha = alpha.expand(prior_z.size())
        interpolates = alpha * prior_z.data + ((1 - alpha) * post_z.data)
        interpolates = Variable(interpolates, requires_grad=True)
        d_input = torch.cat((interpolates, context_hidden.detach()), 1)
        disc_interpolates = torch.mean(self.discriminator(d_input))
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=gData(torch.ones(disc_interpolates.size()), self.use_cuda),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.contiguous().view(
            gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        gradient_penalty.backward()

        optimizer_D.step()
        costD = -(errD_prior - errD_post) + gradient_penalty
        return {'train_loss_D': costD.item()}

    def wae_gan_valid(self, x, c):
        post_z = self.sample_code_post(x, c)
        prior_z = self.sample_code_prior(c)
        errD_post = torch.mean(self.discriminator(torch.cat((post_z, c), 1)))
        errD_prior = torch.mean(self.discriminator(torch.cat((prior_z, c), 1)))
        costD = -(errD_prior - errD_post)
        costG = -costD

        return costG.item(), costD.item()

    def sample(self, context, context_lens, utt_lens, floors, SOS_tok, EOS_tok):
        context_hidden, context_encoder_states = self.context_encoder(context, context_lens, utt_lens, floors)
        # context_hidden: (bsz, num_context_rnn_layers * hiddensize)
        if self.config.get('hred', False):
            dec_input = context_hidden
        elif self.config.get('vhred', False):
            prior_z, _, _ = self.vhred_priori(context_hidden)
            dec_input = torch.cat((prior_z, context_hidden), 1)
            context_encoder_states = None
        else:
            prior_z = self.sample_code_prior(context_hidden)
            dec_input = torch.cat((prior_z, context_hidden), 1)
            if self.config.get('norm_z', False):
                dec_input = F.layer_norm(dec_input, [dec_input.size(-1), ])
            context_encoder_states = None

        sample_words, sample_lens = self.decoder.sampling(
            dec_input, None, self.maxlen, SOS_tok, EOS_tok,
            self.config['inference'], context_encoder_states)
        return sample_words, sample_lens

    def forward(self, *xs, ys, res_lens):
        context, context_lens, utt_lens, floors = xs
        context_hidden, context_encoder_states = self.context_encoder(context, context_lens, utt_lens, floors)
        # context_hidden: (bsz, num_context_rnn_layers * hiddensize)
        vhred_kl_loss = -1
        bow_loss = -1

        if self.config.get('hred', False):
            output = self.decoder(context_hidden, None, ys[:, :-1], (res_lens - 1),
                                  context_encoder_states=context_encoder_states)
            x = None
        elif self.config.get('vhred', False):
            # x: (bsz, num_dirs*hiddensize)
            x, _ = self.utt_encoder(ys[:, 1:], res_lens - 1)
            vhred_post_z, vhred_post_mu, vhred_post_logsigma = self.vhred_posterior(torch.cat((x, context_hidden), 1))
            vhred_priori_z, vhred_priori_mu, vhred_priori_logsigma = self.vhred_priori(context_hidden)

            vhred_z = vhred_post_z

            output = self.decoder(torch.cat((vhred_z, context_hidden), 1), None, ys[:, :-1], (res_lens - 1))
            vhred_kl_loss = self.normal_kl_div(vhred_post_mu, vhred_post_logsigma, vhred_priori_mu, vhred_priori_logsigma)

            bow_logits = self.vhred_bow_project(torch.cat([context_hidden, vhred_priori_z], dim=1))
            bow_loss = self._compute_bow_loss(bow_logits, ys)
        else:
            x, _ = self.utt_encoder(ys[:, 1:], res_lens - 1)
            z = self.sample_code_post(x, context_hidden)

            c_z = torch.cat((z, context_hidden), 1)
            if self.config.get('norm_z', False):
                c_z = F.layer_norm(c_z, [c_z.size(-1), ])
            output = self.decoder(c_z, None, ys[:, :-1], (res_lens - 1))

        _, preds = output.max(dim=2)
        return output, preds, vhred_kl_loss, bow_loss, x, context_hidden


class DialogWAE_GMP(DialogWAE):
    def __init__(self, config, vocab_size, PAD_token=0,
                 unknown_idx=None, use_cuda=True, special_tokens=None):
        super(DialogWAE_GMP, self).__init__(config, vocab_size,
                                            PAD_token, unknown_idx,
                                            use_cuda, special_tokens)
        self.n_components = config['n_prior_components']
        self.gumbel_temp = config['gumbel_temp']

        self.prior_net = MixVariation(config['hiddensize'] * config['numlayers'],
                                      config['z_size'],
                                      self.n_components,
                                      self.gumbel_temp,
                                      self.use_cuda)  # p(e|c)
