import os
import sys

import numpy as np
import pandas as pd
import jieba.analyse
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torchtext import data

sys.path.append('../')
from util.zhihu_util import mkdirs_if_not_exist

TFIDF_FEATURE_NUM = 200
W2V_DIMENSION = 300
D2V_DIMENSION = 100
BATCH_SIZE = 16
EPOCH = 20
COMMENTS_DIR = '../spider/ZhihuLiveComments'

STOP_DENOTATION = [u'《', u'》', u'%', u'\n', u'、', u'=', u' ', u'+', u'-', u'~', u'', u'#', u'＜', u'＞']


def read_corpus():
    """
    read corpus from Excel file and cut words
    :return:
    """
    documents = []
    rates = []

    print('loading corpus...')
    for xlsx in os.listdir(COMMENTS_DIR):
        print('reading Excel: %s ...' % xlsx)
        df = pd.read_excel(os.path.join(COMMENTS_DIR, xlsx), encoding='GBK', index_col=None)
        df = df.dropna(how='any')

        documents += df['content'].tolist()
        rates += df['score'].tolist()

    print('tokenizer starts working...')

    texts = []
    import jieba.analyse

    jieba.load_userdict('./userdict.txt')
    jieba.analyse.set_stop_words('./stopwords.txt')
    stopwords = [_.replace('\n', '') for _ in open('./stopwords.txt', encoding='utf-8').readlines()]

    for doc in documents:
        words_in_doc = list(jieba.cut(str(doc).strip()))
        words_in_doc = list(filter(lambda w: w not in STOP_DENOTATION + stopwords, words_in_doc))
        texts.append(words_in_doc)

    return texts, rates


def load_torchtext():
    texts = data.Field(sequential=True, lower=False, batch_first=True, fix_length=30)
    labels = data.LabelField(sequential=False)


def corpus_to_tfidf_vector(texts, rate_label):
    """
    convert segmented corpus in a list into TF-IDF array
    :param texts:
    :return:
    """
    vectorizer = CountVectorizer(min_df=1, max_features=TFIDF_FEATURE_NUM)
    transformer = TfidfTransformer(smooth_idf=True)
    corpus_list = [''.join(text) for text in texts]
    X = vectorizer.fit_transform(corpus_list)
    tfidf = transformer.fit_transform(X)

    return tfidf.toarray(), rate_label


def get_w2v(texts, rate_label, train=True):
    """
    get word2vec representation
    :return:
    """
    print(texts)
    if train:
        print('training word2vec...')
        model = Word2Vec(texts, size=W2V_DIMENSION, window=5, min_count=1, workers=4, iter=20)
        if not os.path.isdir('./model') or not os.path.exists('./model'):
            os.makedirs('./model')
        print('word2vec training successfully...')
        model.save('./model/zhihulive_comment_w2v.model')

    else:
        print('loading pretrained word2vec model...')
        model = Word2Vec.load('./model/zhihulive_comment_w2v.model')

    # print(model.wv['表情'])
    # similarity = model.wv.similarity('算法', '机器学习')
    # print(similarity)
    features = list()
    labels = list()

    for i in range(len(texts)):
        w2v = []
        for tx in texts[i]:
            try:
                w2v.append(model.wv[tx])
            except:
                pass

        # print(np.array(w2v).shape)
        f = np.array(w2v).mean(axis=0).flatten().tolist()
        if len(f) == W2V_DIMENSION:
            features.append(f)
            labels.append(rate_label[i])

    return np.array(features), np.array(labels)


def get_d2v(words_list, labels, train=True):
    """
    get doc2vec representation
    :param words_list:
    :param labels:
    :param train:
    :return:
    """
    if train:
        print('training doc2vec...')
        documents = []
        for i in range(len(words_list)):
            documents.append(TaggedDocument(words_list[i], [labels[i]]))

        model = Doc2Vec(size=D2V_DIMENSION, min_count=1, workers=4)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=20)
        if not os.path.isdir('./model') or not os.path.exists('./model'):
            os.makedirs('./model')
        model.save('./model/zhihulive_comment_d2v.model')
    else:
        print('loading pretrained doc2vec model...')
        model = Doc2Vec.load('./model/zhihulive_comment_d2v.model')

    features = list()

    for words in words_list:
        f = model.infer_vector(words)
        if len(f) == D2V_DIMENSION:
            features.append(f)

    return np.array(features), np.array(labels)


def cal_hot_words():
    """
    calculate TOP-K hot words by TextRank
    :return:
    """
    documents = []

    print('loading corpus...')
    for xlsx in os.listdir(COMMENTS_DIR):
        print('reading Excel: %s ...' % xlsx)
        df = pd.read_excel(os.path.join(COMMENTS_DIR, xlsx), encoding='GBK', index_col=None)
        df = df.dropna(how='any')

        documents += [str(_) for _ in df['content'].tolist()]

    words_and_weights = jieba.analyse.textrank(' '.join(documents), topK=50, withWeight=True, allowPOS=('ns', 'n',
                                                                                                        'vn', 'v'))
    words, weights = [], []
    for _ in words_and_weights:
        words.append(_[0])
        weights.append(_[1])

    return words, weights


if __name__ == '__main__':
    # words, weights = cal_hot_words()
    # print(words)
    # print(weights)

    texts, rates = read_corpus()
    print("There are {0} records in total...".format(len(rates)))
    X, y = get_w2v(texts, rates, train=False)

    print('start training sentiment classifier...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    mkdirs_if_not_exist('./model')
    joblib.dump(clf, './model/rfc.pkl')

    cm = confusion_matrix(y_test, clf.predict(X_test))
    print(cm)
    print('finish training sentiment classifier...')
