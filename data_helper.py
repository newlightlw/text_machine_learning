import re
import os
import sys
import csv
import time
import json
import collections
import numpy as np
import pandas as pd
from tensorflow.contrib import learn
from imblearn.over_sampling import RandomOverSampler, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import EasyEnsemble

csv.field_size_limit(100000000)


class ImbalancedSample(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def randomOverSampling(self, random_state=42):
        # ros = RandomOverSampler(random_state = 3, ratio = {1:5229, 0:52290}) # 按比例抽取样本
        ros = RandomOverSampler(ratio='minority', random_state=random_state)
        x_res, y_res = ros.fit_sample(self.x, self.y)
        return x_res, y_res

    def adasyn(self, random_state=42):
        ada = ADASYN(ratio='minority', random_state=random_state)
        x_res, y_res = ada.fit_sample(self.x, self.y)
        return x_res, y_res

    def smoteEnn(self, random_state=42):
        sme = SMOTEENN(random_state=random_state)
        x_res, y_res = sme.fit_sample(self.x, self.y)
        return x_res, y_res

    def smoteTomek(self, random_state=42):
        smt = SMOTETomek(random_state=random_state)
        x_res, y_res = smt.fit_sample(self.x, self.y)
        return x_res, y_res

    def easyEnsemble(self, random_state=42):
        ee = EasyEnsemble(random_state=random_state)
        x_res, y_res = ee.fit_sample(self.x, self.y)
        return x_res, y_res


def load_data(file_path, sw_path=None, min_frequency=0, max_length=0, language='ch', re_sampling=True,
              vocab_processor=None, shuffle=True):
    """
    Build dataset for mini-batch iterator
    :param file_path: Data file path
    :param sw_path: Stop word file path
    :param language: 'ch' for Chinese and 'en' for English
    :param min_frequency: the minimal frequency of words to keep
    :param max_length: the max document length
    :param vocab_processor: the predefined vocabulary processor
    :param shuffle: whether to shuffle the data
    :return data, labels, lengths, vocabulary processor
    """
    print("Building dataset ...")
    start = time.time()
    df = pd.read_excel(file_path, encoding='utf-8')
    selected = ['cluster_name', 'text']
    non_selected = list(set(df.columns) - set(selected))
    df = df.drop(non_selected, axis=1)  # 只选取selected的列
    df = df.dropna(axis=0, how='any', subset=selected)  # Drop null rows/删除空行
    df = df.reindex(np.random.permutation(df.index))  # Shuffle the dataframe/重生成Index
    # Map the actual labels to one hot label
    labels = sorted(list(set(df[selected[0]].tolist())))    # 选取label
    one_hot = np.zeros((len(labels), len(labels)), int)     # 用one-hot表示label
    np.fill_diagonal(one_hot, 1)    # 生成对角阵
    label_dict = dict(zip(labels, one_hot))     # 将labels与one-hot表示映射
    if sw_path is not None:
        sw = _stop_words(sw_path)
    else:
        sw = None
    if language == 'ch':
        df[selected[1]].apply(lambda x: _clean_data(_tradition_2_simple(x), sw=sw, language='ch'))
        x_raw = df[selected[1]].apply(lambda x: _word_segmentation(x)).tolist()
        y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    elif language == 'en':
        df[selected[1]].apply(lambda x: _clean_data(x.lower(), sw=sw, language='en'))
        x_raw = df[selected[1]].apply(lambda x: _word_segmentation(x)).tolist()
        y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    else:
        raise ValueError('language should be one of [ch.cn].')
    labels = np.array(y_raw)
    lengths = np.array(list(map(len, [sent.strip().split(' ') for sent in x_raw])))
    if max_length == 0:
        max_length = max(lengths)
    if vocab_processor is None:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=min_frequency)
        data = np.array(list(vocab_processor.fit_transform(x_raw)))
    else:
        data = np.array(list(vocab_processor.transform(x_raw)))
    if re_sampling:
        data_size_before = len(data)
        print("采样前维度：", data.shape)
        data, labels = ImbalancedSample(data, labels).randomOverSampling(random_state=42)
        print("采样后维度：", data.shape, labels.shape)
        data_size = len(data)
        lengths = np.asarray(lengths.tolist() + [max_length] * (data_size - data_size_before))
    else:
        data_size = len(data)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]
        lengths = lengths[shuffle_indices]
    end = time.time()
    print('Dataset has been built successfully.')
    print('Run time: {}'.format(end - start))
    print('Number of sentences: {}'.format(len(data)))
    print('Vocabulary size: {}'.format(len(vocab_processor.vocabulary_._mapping)))
    print('Max document length: {}\n'.format(vocab_processor.max_document_length))
    return data, labels, lengths, label_dict, vocab_processor


def batch_iter(data, labels, lengths, batch_size, num_epochs):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param data: a list of sentences. each sentence is a vector of integers
    :param labels: a list of labels
    :param batch_size: the size of mini-batch
    :param num_epochs: number of epochs
    :return: a mini-batch iterator
    """
    assert len(data) == len(labels) == len(lengths)

    data_size = len(data)
    epoch_length = data_size // batch_size

    for _ in range(num_epochs):
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size

            xdata = data[start_index: end_index]
            ydata = labels[start_index: end_index]
            sequence_length = lengths[start_index: end_index]

            yield xdata, ydata, sequence_length


def _tradition_2_simple(sent):
    """ Convert Traditional Chinese to Simplified Chinese """
    # Please download langconv.py and zh_wiki.py first
    # langconv.py and zh_wiki.py are used for converting between languages
    try:
        from zhtools import langconv
    except ImportError as e:
        error = "Please download langconv.py and zh_wiki.py at "
        error += "https://github.com/skydark/nstools/tree/master/zhtools."
        print(str(e) + ': ' + error)
        sys.exit()

    return langconv.Converter('zh-hans').convert(sent)


def _word_segmentation(sent):
    """ Tokenizer for Chinese """
    import jieba
    sent = ' '.join(list(jieba.cut(sent, cut_all=False, HMM=True)))
    return re.sub(r'\s+', ' ', sent)


def _stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = list()
        for line in f:
            sw.append(line.strip())

    return set(sw)


def _clean_data(sent, sw, language='ch'):
    """ Remove special characters and stop words """
    if language == 'ch':
        sent = re.sub(r"[^\u4e00-\u9fa5A-z0-9！？，。]", " ", sent)
        sent = re.sub('！{2,}', '！', sent)
        sent = re.sub('？{2,}', '！', sent)
        sent = re.sub('。{2,}', '。', sent)
        sent = re.sub('，{2,}', '，', sent)
        sent = re.sub('\s{2,}', ' ', sent)
    if language == 'en':
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\'s", " \'s", sent)
        sent = re.sub(r"\'ve", " \'ve", sent)
        sent = re.sub(r"n\'t", " n\'t", sent)
        sent = re.sub(r"\'re", " \'re", sent)
        sent = re.sub(r"\'d", " \'d", sent)
        sent = re.sub(r"\'ll", " \'ll", sent)
        sent = re.sub(r",", " , ", sent)
        sent = re.sub(r"!", " ! ", sent)
        sent = re.sub(r"\(", " \( ", sent)
        sent = re.sub(r"\)", " \) ", sent)
        sent = re.sub(r"\?", " \? ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
    if sw is not None:
        sent = "".join([word for word in sent if word not in sw])

    return sent
