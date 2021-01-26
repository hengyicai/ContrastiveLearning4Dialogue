import argparse
import pickle
from typing import List, Sized

import torch

from parlai.utils.torch import padded_tensor
from projects.contrastive_learning.retrieval_models import MSN


def _check_truncate(vec, truncate, truncate_left=False):
    """
    Check that vector is truncated correctly.
    """
    if truncate is None:
        return vec
    if len(vec) <= truncate:
        return vec
    if truncate_left:
        return vec[-truncate:]
    else:
        return vec[:truncate]


def padded_3d(
    tensors: List[torch.LongTensor],
    pad_idx: int = 0,
    use_cuda: bool = False,
    left_padded: bool = False,
    max_len=None,
    dtype=torch.long,
):
    a = len(tensors)
    b = max(len(row) for row in tensors)  # type: ignore
    c = max(len(item) for row in tensors for item in row) if max_len is None else max_len  # type: ignore

    c = max(c, 1)

    output = torch.full((a, b, c), pad_idx, dtype=dtype)

    for i, row in enumerate(tensors):
        item: Sized
        idx = [i for i in range(row.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        for j, item in enumerate(row.index_select(0, idx)):  # type: ignore
            if len(item) == 0:
                continue
            if not isinstance(item, torch.Tensor):
                item = torch.Tensor(item, dtype=dtype)
            if left_padded:
                output[i, -j - 1, c - len(item):] = item
            else:
                output[i, -j - 1, : len(item)] = item

    if use_cuda:
        output = output.cuda()

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_embed_path', type=str, required=True)
    parser.add_argument('--training_data', type=str, required=True)
    parser.add_argument('--msn_model', type=str, required=True)
    parser.add_argument('--saved_path', type=str, required=True)

    # MSN default parameters
    parser.add_argument("--gru_hidden", default=300, type=int,
                        help="The hidden size of GRU in layer 1")
    parser.add_argument('--score_file_path', default='/tmp/null')

    args = parser.parse_args()

    vocab, word_embeddings = pickle.load(file=open(args.vocab_embed_path, 'rb'))
    model = MSN(word_embeddings, args=args)
    model.load_model(args.msn_model)
    max_len = 50

    with open(args.training_data) as f, open(args.saved_path, 'w') as saved_f:
        for line in f.readlines():
            line = line.strip()
            id_context_samples = line.split('\t')[0]
            response = line.split('\t')[1].strip()
            context = [item.strip() for item in
                       ' '.join(id_context_samples.split()[1:]).split('__SAMP__')[0].strip().split('__EOT__')]
            if len(context) == 1:
                context += context

            samples_arr = []
            samples = ' '.join(id_context_samples.split()[1:]).split('__SAMP__')[1].strip()
            samples = samples.split('__EOD__')
            for samp in samples:
                samp_context = [item.strip() for item in samp.strip().split('__EOC__')[0].strip().split('__EOT__')]
                samp_response = samp.strip().split('__EOC__')[1].strip()
                samples_arr.append((samp_context, samp_response))

            # tokens --> ids
            context_ids = [_check_truncate([vocab.get(token.strip(), vocab['<UNK>']) for token in utterance.split()],
                                           max_len, truncate_left=True) for utterance in context]
            response_ids = _check_truncate([vocab.get(token.strip(), vocab['<UNK>']) for token in response.split()],
                                           max_len, truncate_left=False)

            samples_context_ids_arr = []
            samples_response_ids_arr = []
            for samp_context, samp_response in samples_arr:
                samp_context_ids = [
                    _check_truncate([vocab.get(token.strip(), vocab['<UNK>']) for token in utterance.split()],
                                    max_len, truncate_left=True) for utterance in samp_context]
                samp_response_ids = _check_truncate(
                    [vocab.get(token.strip(), vocab['<UNK>']) for token in samp_response.split()],
                    max_len, truncate_left=False)
                samples_context_ids_arr.append(samp_context_ids)
                samples_response_ids_arr.append(samp_response_ids)

            # padding
            # max_len = max(max([len(u) for u in context_ids]), len(response_ids))
            # max_len = max(max_len, max([len(u) for c in samples_context_ids_arr for u in c]))
            # max_len = max(max_len, max([len(r) for r in samples_response_ids_arr]))

            context_ids_duplicated = [padded_tensor(context_ids, max_len=max_len, left_padded=True)[0]
                                      for _ in samples_context_ids_arr]
            response_ids_duplicated = [response_ids for _ in samples_response_ids_arr]

            samples_context_ids_arr = [padded_tensor(samp_context_ids, max_len=max_len, left_padded=True)[0]
                                       for samp_context_ids in samples_context_ids_arr]

            context_ids_duplicated_padded = padded_3d(context_ids_duplicated, max_len=max_len, left_padded=True)
            response_ids_duplicated_padded = padded_tensor(
                response_ids_duplicated, max_len=max_len, left_padded=True)[0]
            samples_context_ids_arr_padded = padded_3d(samples_context_ids_arr, max_len=max_len, left_padded=True)
            samples_response_ids_arr_padded = padded_tensor(
                samples_response_ids_arr, max_len=max_len, left_padded=True)[0]

            with torch.no_grad():
                c_vs_sample_r_scores = model.inference(context_ids_duplicated_padded, samples_response_ids_arr_padded)
                sample_c_vs_r_scores = model.inference(samples_context_ids_arr_padded, response_ids_duplicated_padded)

            c_vs_sample_r_scores = [str(item) for item in c_vs_sample_r_scores]
            sample_c_vs_r_scores = [str(item) for item in sample_c_vs_r_scores]

            saved_f.write(
                "{} __SAMP__ {} __SAMP__ {}\t {}\n".format(
                    id_context_samples,
                    ' '.join(c_vs_sample_r_scores),
                    ' '.join(sample_c_vs_r_scores),
                    response
                )
            )
