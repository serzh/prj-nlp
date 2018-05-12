import spacy
import spacy.tokens
import en_core_web_lg
from sklearn.externals import joblib
import pandas as pd
from arg_extractor import extract_args

def extract(tokens):
    N = len(tokens)
    all_features = []
    for i in range(N):
        features = {
            'text': tokens[i].text,
            'lemma': tokens[i].lemma_,
            'pos': tokens[i].tag_,
            'text-1': tokens[i-1].text if i-1 >= 0 else 'NONE',
            'lemma-1': tokens[i-1].lemma_ if i-1 >= 0 else 'NONE',
            'pos-1': tokens[i-1].pos_ if i-1 >= 0 else 'NONE',
            'text-2': tokens[i-2].text if i-2 >= 0 else 'NONE',
            'lemma-2': tokens[i-2].lemma_ if i-2 >= 0 else 'NONE',
            'pos-2': tokens[i-2].pos_ if i-2 >= 0 else 'NONE',
            'text+1': tokens[i+1].text if i+1 < N else 'NONE',
            'lemma+1': tokens[i+1].lemma_ if i+1 < N else 'NONE',
            'pos+1': tokens[i+1].pos_ if i+1 < N else 'NONE',
            'text+2': tokens[i+2].text if i+2 < N else 'NONE',
            'lemma+2': tokens[i+2].lemma_ if i+2 < N else 'NONE',
            'pos+2': tokens[i+2].pos_ if i+2 < N else 'NONE'
        }
        all_features.append(features)
    return all_features

def sent2features(nlp, sent):
    tokens = spacy.tokens.Doc(nlp.vocab, words=[pair[0] for pair in sent])
    nlp.tagger(tokens)
    nlp.parser(tokens)
    return extract(tokens)

CONJUNCTIONS = {'and', 'or', 'but', 'except'}

def split(nlp, splitter, sent, verbose=False):
    tokens = nlp(sent)
    features = extract(tokens)
    stops = [1 if stop == 'E' else 0 for stop in splitter.predict([features])[0]]
    if verbose:
        print(stops)
    parts = []
    part = (0,0)
    conj = (0,0)
    prev = 0
    for token,stop in zip(tokens, stops):
        if stop == 0:
            part = (part[0], part[1]+len(token.text)+1)
        else:
            parts.append({'type': 'phrase',
                          'phrase': sent[part[0]:part[1]]})
            if token.text in CONJUNCTIONS:
                parts.append({'type': 'conj',
                              'conj': token.text})
                part = (part[1]+len(token.text)+1, part[1]+len(token.text)+1)

            else:
                part = (part[1], part[1]+len(token.text)+1)
    parts.append({'type': 'phrase',
                          'phrase': sent[part[0]:].strip()})
    return parts

def classify(filter_cls, part):
    part['class'] = filter_cls.predict([part['phrase']])[0]
    return part


def organize_in_tree_(parts, tree):
    if parts:
        part = parts[0]
        rest = parts[1:]
        (l, n, r) = tree
        if part['type'] == 'phrase':
            if not l:
                return organize_in_tree_(rest, (part, n, r))
            elif not r:
                return organize_in_tree_(rest, (l, n, part))
            else:
                raise Exception("node in invalid position")
        else:
            if not n:
                return organize_in_tree_(rest, (l, part, r))
            else:
                return organize_in_tree_(rest, (tree, part, None))
    else:
        return tree


def organize_in_tree(parts):
    return organize_in_tree_(parts[:], (None, None, None))

def spacy_tokenizer(nlp):
    return lambda x: [t.lemma_ for t in nlp(x)]

class Transformer():
    def __init__(self, splitter='../crf_splitter.pkl', filter_cls='../filter_classifier.pkl'):
        self.nlp = en_core_web_lg.load()
        self.splitter = joblib.load(splitter)
        self.filter_cls = joblib.load(filter_cls)

    def transform(self, sent, verbose=False):
        parts = split(self.nlp, self.splitter, sent, verbose)
        if verbose:
            print(parts)
        parts = parts[1:]
        parts = parts[1:] if parts[0]['type'] == 'conj' else parts
        parsed = [extract_args(self.nlp,
                               classify(self.filter_cls, part),
                               verbose)
                  if part['type'] != 'conj' else part
                  for part in parts]
        print(parsed)
        return organize_in_tree(parsed)
