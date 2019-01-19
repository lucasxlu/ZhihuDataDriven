import os
import sys

import numpy as np
import pandas as pd
import jieba.analyse
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import fasttext
import jieba.analyse

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

    jieba.load_userdict('./userdict.txt')
    jieba.analyse.set_stop_words('./stopwords.txt')
    stopwords = [_.replace('\n', '') for _ in open('./stopwords.txt', encoding='utf-8').readlines()]

    for doc in documents:
        words_in_doc = list(jieba.cut(str(doc).strip()))
        words_in_doc = list(filter(lambda w: w not in STOP_DENOTATION + stopwords, words_in_doc))
        texts.append(words_in_doc)

    return texts, rates


def get_comments_and_live_score():
    """
    get comments and ZhiHuLive scores
    :return:
    """
    documents = []
    rates = []

    live = pd.read_excel("../spider/ZhihuLiveDB.xlsx")
    live = live[live['review_count'] >= 11]

    mp = {}
    for i in range(len(live)):
        mp[live['id'].tolist()[i]] = live['review_score'].tolist()[i]

    print('loading corpus...')
    for xlsx, score in mp.items():
        print('reading Excel: %s ...' % xlsx)
        df = pd.read_excel(os.path.join(COMMENTS_DIR, '{0}.xlsx'.format(xlsx)), encoding='GBK', index_col=None)
        df = df.dropna(how='any')

        documents += df['content'].tolist()
        rates.append(score)

    print('tokenizer starts working...')

    texts = []

    jieba.load_userdict('./userdict.txt')
    jieba.analyse.set_stop_words('./stopwords.txt')
    stopwords = [_.replace('\n', '') for _ in open('./stopwords.txt', encoding='utf-8').readlines()]

    for doc in documents:
        words_in_doc = list(jieba.cut(str(doc).strip()))
        words_in_doc = list(filter(lambda w: w not in STOP_DENOTATION + stopwords, words_in_doc))
        texts.append(words_in_doc)

    return texts, rates


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

        f = np.array(w2v).mean(axis=0).flatten().tolist()
        if len(f) == W2V_DIMENSION:
            features.append(f)
            labels.append(rate_label[i])

    return np.array(features), np.array(labels)


def get_fast_text_repr(fastTextRepr, texts, rate_label):
    """
    get FastText representation
    :return:
    """
    features = list()
    labels = list()

    for i in range(len(texts)):
        w2v = []
        for tx in texts[i]:
            try:
                w2v.append(fastTextRepr[tx])
            except:
                pass

        f = np.array(w2v).mean(axis=0).flatten().tolist()
        print(f)
        features.append(f)
        labels.append(rate_label[i])

    return np.array(features), np.array(labels).ravel()


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


class FastTextClassifier:
    def __init__(self):
        self.excel_dir = COMMENTS_DIR

    def prepare_fast_text_data(self):
        documents = []
        rates = []

        for xlsx in os.listdir(self.excel_dir):
            print('reading Excel: %s ...' % xlsx)
            df = pd.read_excel(os.path.join(COMMENTS_DIR, xlsx), encoding='GBK', index_col=None)
            df = df.dropna(how='any')

            documents += df['content'].tolist()
            rates += df['score'].tolist()

        X_train, X_test, y_train, y_test = train_test_split(documents, rates, test_size=0.2, random_state=42,
                                                            stratify=rates)
        train_lines = []
        for i in range(len(X_train)):
            line = '{0}\t__label__{1}\n'.format(" ".join(jieba.cut(str(X_train[i]).strip())), int(y_train[i]))
            train_lines.append(line)

        with open('./train.txt', mode='wt', encoding='utf-8') as f:
            f.write("".join(train_lines))

        test_lines = []
        for i in range(len(X_test)):
            line = '{0}\t__label__{1}\n'.format(" ".join(jieba.cut(str(X_test[i]).strip())), int(y_test[i]))
            test_lines.append(line)

        with open('./test.txt', mode='wt', encoding='utf-8') as f:
            f.write("".join(test_lines))

    def train_and_eval(self):
        classifier = fasttext.supervised('./train.txt', 'fastTextClassifier', label_prefix="__label__", thread=4,
                                         epoch=100)
        result = classifier.test('./test.txt')
        print('P@1:', result.precision)
        print('R@1:', result.recall)

    def train_word_repr(self):
        documents = []

        for xlsx in os.listdir(self.excel_dir):
            print('reading Excel: %s ...' % xlsx)
            df = pd.read_excel(os.path.join(COMMENTS_DIR, xlsx), encoding='GBK', index_col=None)
            df = df.dropna(how='any')

            documents += df['content'].tolist()

        lines = []
        for i in range(len(documents)):
            line = " ".join(jieba.cut(str(documents[i]).strip()))
            lines.append(line)

        with open('./comments.txt', mode='wt', encoding='utf-8') as f:
            f.write("".join(lines))

        fasttext.skipgram('comments.txt', 'fastTextRepr')

    def get_word_repr(self, text):
        model = fasttext.load_model('fastTextRepr.bin')

        return model[text]


if __name__ == '__main__':
    # fast_text_classifier = FastTextClassifier()
    # fast_text_classifier.prepare_fast_text_data()
    # fast_text_classifier.train_and_eval()

    # fast_text_classifier.train_word_repr()
    # print(fast_text_classifier.get_word_repr("知乎"))

    texts, rates = get_comments_and_live_score()

    print("There are {0} records in total...".format(len(rates)))
    X, y = get_fast_text_repr(fasttext.load_model('fastTextRepr.bin'), texts, rates)

    print(y)

    print('start training regressor...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.shape)
    print(y_train.shape)

    X_train, X_test = X_train[0:100], X_test[0:100]
    y_train, y_test = y_train[0:100], y_test[0:100]

    clr = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0)
    clr.fit(X_train, y_train)
    mkdirs_if_not_exist('./model')
    joblib.dump(clr, './model/rfc.pkl')

    y_pred = clr.predict(X_test)
    mae_lr = round(mean_absolute_error(y_test, y_pred), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(y_test, y_pred)), 4)
    print('===============The Mean Absolute Error is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error is {0}===================='.format(rmse_lr))
    print('finish training regressor...')
