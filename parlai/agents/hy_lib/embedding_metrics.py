"""
Word embedding based evaluation metrics for dialogue.

This method implements three evaluation metrics based on Word2Vec word embeddings, which compare a target utterance with a model utterance:
1) Computing cosine-similarity between the mean word embeddings of the target utterance and of the model utterance
2) Computing greedy meatching between word embeddings of target utterance and model utterance (Rus et al., 2012)
3) Computing word embedding extrema scores (Forgues et al., 2014)

We believe that these metrics are suitable for evaluating dialogue systems.

Example run:

    python embedding_metrics.py path_to_ground_truth.txt path_to_predictions.txt path_to_embeddings.bin

The script assumes one example per line (e.g. one dialogue or one sentence per line), where line n in 'path_to_ground_truth.txt' matches that of line n in 'path_to_predictions.txt'.

NOTE: The metrics are not symmetric w.r.t. the input sequences. 
      Therefore, DO NOT swap the ground truths with the predicted responses.

References:

A Comparison of Greedy and Optimal Assessment of Natural Language Student Input Word Similarity Metrics Using Word to Word Similarity Metrics. Vasile Rus, Mihai Lintean. 2012. Proceedings of the Seventh Workshop on Building Educational Applications Using NLP, NAACL 2012.

Bootstrapping Dialog Systems with Word Embeddings. G. Forgues, J. Pineau, J. Larcheveque, R. Tremblay. 2014. Workshop on Modern Machine Learning and Natural Language Processing, NIPS 2014.


"""
__docformat__ = 'restructedtext en'
__authors__ = ("Chia-Wei Liu", "Iulian Vlad Serban")

import numpy as np


def greedy_match(fileone, filetwo, w2v):
    res1 = greedy_score(fileone, filetwo, w2v)
    res2 = greedy_score(filetwo, fileone, w2v)
    res_sum = (res1 + res2) / 2.0
    for s in res_sum:
        print(s)
    return np.mean(res_sum), 1.96 * np.std(res_sum) / float(len(res_sum)), np.std(res_sum)


def sentence_greedy_score(tokens1, tokens2, w2v, tokens1_w=None, tokens2_w=None):
    if tokens1_w and tokens2_w:
        assert len(tokens1_w) == len(tokens1)
        assert len(tokens2_w) == len(tokens2)
    else:
        tokens1_w = [1 for _ in tokens1]
        tokens2_w = [1 for _ in tokens2]

    dim = w2v.vectors.shape[1]
    X = np.zeros((dim,))
    y_count = 0
    x_count = 0
    o = 0.0
    Y = np.zeros((dim, 1))
    for tok, w in zip(tokens2, tokens2_w):
        if tok in w2v.stoi:
            Y = np.hstack((Y, ((w * w2v[tok].numpy()).reshape((dim, 1)))))
            y_count += 1

    if y_count > 0:
        Y = np.delete(Y, 0, axis=1)
        Y_norm = np.linalg.norm(Y, axis=0)
        for tok, w in zip(tokens1, tokens1_w):
            if tok in w2v.stoi:
                W = (w2v[tok].numpy() * w).reshape((1, dim))
                tmp = W.dot(Y) / np.linalg.norm(W) / Y_norm
                o += np.max(tmp)
                x_count += 1

    # if none of the words in response or ground truth have embeddings, count result as zero
    if x_count < 1 or y_count < 1:
        return 0

    o /= float(x_count)
    return o


def greedy_score(fileone, filetwo, w2v):
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
    r1 = f1.readlines()
    r2 = f2.readlines()

    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        o = sentence_greedy_score(tokens1, tokens2, w2v)
        scores.append(o)

    return np.asarray(scores)


def sentence_extrema_score(tokens1, tokens2, w2v, tokens1_w=None, tokens2_w=None):
    if tokens1_w and tokens2_w:
        assert len(tokens1_w) == len(tokens1)
        assert len(tokens2_w) == len(tokens2)
    else:
        tokens1_w = [1 for _ in tokens1]
        tokens2_w = [1 for _ in tokens2]

    X = []
    for tok, w in zip(tokens1, tokens1_w):
        if tok in w2v.stoi:
            X.append(w2v[tok].numpy() * w)
    Y = []
    for tok, w in zip(tokens2, tokens2_w):
        if tok in w2v.stoi:
            Y.append(w2v[tok].numpy() * w)

    # if none of the words have embeddings in ground truth, skip
    if np.linalg.norm(X) < 0.00000000001:
        return None

    # if none of the words have embeddings in response, count result as zero
    if np.linalg.norm(Y) < 0.00000000001:
        return 0

    xmax = np.max(X, 0)  # get positive max
    xmin = np.min(X, 0)  # get abs of min
    xtrema = []
    for i in range(len(xmax)):
        if np.abs(xmin[i]) > xmax[i]:
            xtrema.append(xmin[i])
        else:
            xtrema.append(xmax[i])
    X = np.array(xtrema)  # get extrema

    ymax = np.max(Y, 0)
    ymin = np.min(Y, 0)
    ytrema = []
    for i in range(len(ymax)):
        if np.abs(ymin[i]) > ymax[i]:
            ytrema.append(ymin[i])
        else:
            ytrema.append(ymax[i])
    Y = np.array(ytrema)

    o = np.dot(X, Y.T) / np.linalg.norm(X) / np.linalg.norm(Y)

    return o


def extrema_score(fileone, filetwo, w2v):
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
    r1 = f1.readlines()
    r2 = f2.readlines()

    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")

        o = sentence_extrema_score(tokens1, tokens2, w2v)

        if o is not None:
            scores.append(o)
    for s in scores:
        print(s)
    scores = np.asarray(scores)
    return np.mean(scores), 1.96 * np.std(scores) / float(len(scores)), np.std(scores)


def sentence_average_score(tokens1, tokens2, w2v, tokens1_w=None, tokens2_w=None):
    dim = w2v.vectors.shape[1]
    X = np.zeros((dim,))
    if tokens1_w and tokens2_w:
        assert len(tokens1_w) == len(tokens1)
        assert len(tokens2_w) == len(tokens2)
    else:
        tokens1_w = [1 for _ in tokens1]
        tokens2_w = [1 for _ in tokens2]

    for tok, w in zip(tokens1, tokens1_w):
        if tok in w2v.stoi:
            X += (w2v[tok].numpy() * w)
    Y = np.zeros((dim,))
    for tok, w in zip(tokens2, tokens2_w):
        if tok in w2v.stoi:
            Y += (w2v[tok].numpy() * w)

    # if none of the words in ground truth have embeddings, skip
    if np.linalg.norm(X) < 0.00000000001:
        return None

    # if none of the words have embeddings in response, count result as zero
    if np.linalg.norm(Y) < 0.00000000001:
        return 0

    X = np.array(X) / np.linalg.norm(X)
    Y = np.array(Y) / np.linalg.norm(Y)
    o = np.dot(X, Y.T) / np.linalg.norm(X) / np.linalg.norm(Y)
    return o


def average(fileone, filetwo, w2v):
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
    r1 = f1.readlines()
    r2 = f2.readlines()

    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")

        o = sentence_average_score(tokens1, tokens2, w2v)
        if o is not None:
            scores.append(o)
    for s in scores:
        print(s)
    scores = np.asarray(scores)
    return np.mean(scores), 1.96 * np.std(scores) / float(len(scores)), np.std(scores)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('ground_truth', help="ground truth text file, one example per line")
#     parser.add_argument('predicted', help="predicted text file, one example per line")
#     parser.add_argument('embeddings', help="embeddings bin file")
#     args = parser.parse_args()
#
#     print("loading embeddings file {} ...".format(args.embeddings))
#     w2v = KeyedVectors.load_word2vec_format(args.embeddings, binary=False)
#
#     r = average(args.ground_truth, args.predicted, w2v)
#     print("Embedding Average Score: {:f} +/- {:f} ( {:f} )".format(r[0], r[1], r[2]))
#
#     r = greedy_match(args.ground_truth, args.predicted, w2v)
#     print("Greedy Matching Score: {:f} +/- {:f} ( {:f} )".format(r[0], r[1], r[2]))
#
#     r = extrema_score(args.ground_truth, args.predicted, w2v)
#     print("Extrema Score: {:f} +/- {:f} ( {:f} )".format(r[0], r[1], r[2]))
