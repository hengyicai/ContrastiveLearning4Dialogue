import argparse
import locale
import multiprocessing as mp
import os
import pickle

from projects.contrastive_learning.BM25.bm25 import WrappedBM25

locale.setlocale(locale.LC_ALL, 'en_US')


class DumpTuple(object):
    def __init__(self, bm25, context2response):
        self.bm25 = bm25
        self.context2response = context2response


def build_data_worker(arg):
    bm25, topk, tailk, id, context, response, context2response, cached_exists = arg
    query = context
    topk_res = bm25.find_topk_doc(query, topk=topk, rm_first=not cached_exists)
    topk_context_response_res = ' __EOD__ '.join(['{} __EOC__ {}'.format(c, context2response[c]) for c in topk_res])
    tailk_res = bm25.find_tailk_doc(query, tailk=tailk)
    tailk_context_response_res = ' __EOD__ '.join(['{} __EOC__ {}'.format(c, context2response[c]) for c in tailk_res])

    enhanced_context = context + ' __SAMP__ ' + topk_context_response_res + ' __EOD__ ' + tailk_context_response_res

    ret_line = "{} {}\t{}".format(id, enhanced_context, response)
    return ret_line
    # sys.stdout.flush()


def build_data(fb_dialog_file, cached_model_path, topk=10, tailk=10, cores=-1):
    with open(fb_dialog_file, encoding='utf-8') as f:
        dialogs = [
            (
                int(item.strip().split('\t')[0].split()[0]),
                ' '.join(item.strip().split('\t')[0].split()[1:]),
                item.strip().split('\t')[1]
            )
            if len(item.strip().split('\t')) == 2 and len(item.strip().split('\t')[0].split()) > 1
            else None
            for item in f.readlines()
        ]
        dialogs = filter(lambda x: x is not None, dialogs)

    cached_exists = os.path.exists(cached_model_path)
    if not cached_exists:
        dialogs = rm_duplicated_context(dialogs)

    if not cached_exists:
        context2response = {}
        for id, context, response in dialogs:
            context2response[context] = response
        # build the BM25 model for the training file
        contexts = [item[1] for item in dialogs]
        bm25 = WrappedBM25(contexts, tokenizer='split')
        # cache the model
        pickle.dump(DumpTuple(bm25, context2response), open(cached_model_path, 'wb'))
    else:
        # load the cached BM25 model for the valid or test file
        dumped_obj = pickle.load(open(cached_model_path, 'rb'))
        bm25 = dumped_obj.bm25
        context2response = dumped_obj.context2response

    pool = mp.Pool(cores)
    list_of_res_lines = pool.map(
        build_data_worker,
        ((bm25, topk, tailk, id, context, response, context2response, cached_exists) for id, context, response in
         dialogs))

    pool.close()
    pool.join()

    for line in list_of_res_lines:
        print(line)


def rm_duplicated_context(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x[1] in seen or seen_add(x[1]))]


def build_data_from_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fb_dialog_file",
                        required=True,
                        type=str,
                        help="The training set used for building the data")
    parser.add_argument("--cache_path",
                        type=str,
                        required=True)
    parser.add_argument("--topk",
                        required=False,
                        type=int,
                        default=10)
    parser.add_argument("--tailk",
                        required=False,
                        type=int,
                        default=10)
    parser.add_argument("--cores",
                        default=-1,
                        type=int)

    args = parser.parse_args()
    fb_dialog_file = args.fb_dialog_file
    cache_path = args.cache_path
    topk = args.topk
    tailk = args.tailk

    if args.cores == -1:
        args.cores = mp.cpu_count()
    build_data(fb_dialog_file, cache_path, topk, tailk, args.cores)


if __name__ == '__main__':
    build_data_from_cmd_line()
