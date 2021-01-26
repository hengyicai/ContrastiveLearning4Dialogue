import numpy
from gensim.summarization.bm25 import BM25


class WrappedBM25(BM25):
    def __init__(self, docs, tokenizer='spacy'):
        self.docs = docs
        if tokenizer == 'spacy':
            try:
                import spacy
            except ImportError:
                raise ImportError('Please install spacy and spacy "en" model: '
                                  '`pip install -U spacy && '
                                  'python -m spacy download en` '
                                  'or find alternative installation options '
                                  'at spacy.io')
            self._spacy = spacy.load('en')
            self.tokenizer = self.spacy_tokenize
        else:
            self.tokenizer = self.split_tokenize

        corpus = []
        for doc in self.docs:
            corpus.append(self.tokenizer(doc))
        super().__init__(corpus)
        self.average_idf = sum(map(lambda k: float(self.idf[k]), self.idf.keys())) / len(self.idf.keys())

    def find_topk_doc(self, doc, topk=10, rm_first=True):
        scores = self.get_scores(self.tokenizer(doc))
        arg_idx = numpy.argsort(scores)
        arg_idx = arg_idx[::-1]
        result = []
        for i in range(topk):
            result.append(self.docs[arg_idx[i]])
        if rm_first:
            del result[0]  # remove self
        return result

    def find_tailk_doc(self, doc, tailk=10):
        scores = self.get_scores(self.tokenizer(doc))
        arg_idx = numpy.argsort(scores)
        result = []
        for i in range(tailk):
            result.append(self.docs[arg_idx[i]])
        return result

    def spacy_tokenize(self, text):
        tokens = self._spacy.tokenizer(text)
        return [t.text for t in tokens]

    def split_tokenize(self, text):
        return text.strip().split(' ')
