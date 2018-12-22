import sys
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import requests
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_regression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

sys.path.append('../')
from util.cfg import cfg
from util.zhihu_util import mkdirs_if_not_exist
from analysis.models import MLP, MTNet, ZhihuLiveDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def split_train_test(excel_path, dl=True):
    # df = pd.read_excel(excel_path, sheet_name="ZhihuLive").fillna(value=0)
    df = pd.read_excel(excel_path, sheet_name="ZhihuLive")
    df = df[df['review_count'] >= 11]
    print("*" * 100)
    if not dl:
        dataset = df.loc[:, ['duration', 'reply_message_count', 'source', 'purchasable', 'is_refundable',
                             'has_authenticated', 'user_type', 'gender', 'badge', 'tag_id',
                             'speaker_audio_message_count',
                             'attachment_count', 'liked_num', 'is_commercial', 'audition_message_count',
                             'is_audition_open',
                             'seats_taken', 'seats_max', 'speaker_message_count', 'amount', 'original_price',
                             'has_audition', 'has_feedback', 'review_count']]
        dataset['tag_id'] = dataset['tag_id'].fillna(value=0)
        print(dataset.describe())

        imp = Imputer(missing_values='NaN', strategy='median', axis=0)
        imp.fit(dataset)

        source_le = LabelEncoder()
        source_labels = source_le.fit_transform(dataset['source'])
        dataset['source'] = source_labels
        # source_mappings = {index: label for index, label in enumerate(source_le.classes_)}

        enc = preprocessing.OneHotEncoder()
        enc.fit_transform(
            dataset[['source', 'purchasable', 'is_refundable', 'user_type', 'is_commercial', 'is_audition_open',
                     'has_audition', 'has_feedback']])

        tag_id_le = LabelEncoder()
        tag_id_labels = tag_id_le.fit_transform(dataset['tag_id'])
        dataset['tag_id'] = tag_id_labels

        min_max_scaler = preprocessing.MinMaxScaler()
        dataset = min_max_scaler.fit_transform(dataset)

        labels = df.loc[:, ['review_score']]

        dataset, labels = feature_selection(dataset, labels, k=15)

        dataset = pd.DataFrame(dataset)
        labels = pd.DataFrame(labels)

    else:
        dataset = df.loc[:, ['duration', 'reply_message_count', 'source', 'purchasable', 'is_refundable',
                             'has_authenticated', 'user_type', 'gender', 'badge', 'speaker_audio_message_count',
                             'attachment_count', 'liked_num', 'is_commercial', 'audition_message_count',
                             'is_audition_open', 'seats_taken', 'seats_max', 'speaker_message_count', 'amount',
                             'original_price', 'has_audition', 'has_feedback', 'review_count']]
        print(dataset.describe())
        min_max_scaler = preprocessing.MinMaxScaler()
        dataset = min_max_scaler.fit_transform(dataset)
        dataset = pd.DataFrame(dataset)
        labels = df.loc[:, ['review_score']]

    print("*" * 10)
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, random_state=42)

    return X_train, X_test, y_train, y_test


def train_and_test_model(train, test, train_Y, test_Y):
    # model = Pipeline([('poly', PolynomialFeatures(degree=3)),
    #                   ('linear', LinearRegression(fit_intercept=False))])

    # model = LassoCV(alphas=[_ * 0.1 for _ in range(1, 1000, 1)])
    # model = RidgeCV(alphas=[_ * 0.1 for _ in range(1, 1000, 1)])
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # model = SVR(kernel='linear', C=1e3)
    # model = SVR(kernel='poly', C=1e3, degree=2)
    # model = KNeighborsRegressor(n_neighbors=10, n_jobs=4)

    # model = MLPRegressor(hidden_layer_sizes=(16, 8, 8, 4), early_stopping=True, alpha=1e-4,
    #                      batch_size=16, learning_rate='adaptive')
    model.fit(train, train_Y.values.ravel())
    mkdirs_if_not_exist('./model')
    joblib.dump(model, './model/svr.pkl')
    predicted_score = model.predict(test)
    mae_lr = round(mean_absolute_error(test_Y, predicted_score), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(test_Y, predicted_score)), 4)
    # pc = round(np.corrcoef(test_Y.values.ravel(), predicted_score)[0, 1], 4)
    print('===============The Mean Absolute Error is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error is {0}===================='.format(rmse_lr))
    # print('===============The Pearson Correlation of Model is {0}===================='.format(pc))

    from util.zhihu_util import out_result
    out_result(predicted_score, test_Y)


def feature_selection(X, y, k=15):
    """
    feature selection
    :param X:
    :param y:
    :return:
    """
    print(X.shape)
    X = SelectKBest(f_regression, k=k).fit_transform(X, y)
    print(X.shape)

    return X, y


def train_and_test_mtnet(train, test, train_Y, test_Y, epoch=100):
    """
    train and test with MTNet
    :param train:
    :param test:
    :param train_Y:
    :param test_Y:
    :return:
    """
    trainloader = torch.utils.data.DataLoader(ZhihuLiveDataset(train, train_Y), batch_size=cfg['batch_size'],
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(ZhihuLiveDataset(test, test_Y), batch_size=cfg['batch_size'],
                                             shuffle=False, num_workers=4)

    mtbdnn = MTNet(K=3)
    print(mtbdnn)
    # mlp = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mtbdnn.parameters(), lr=0.001, weight_decay=1e-3)
    # learning_rate_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    for epoch in range(epoch):

        running_loss = 0.0
        for i, data_batch in enumerate(trainloader, 0):
            # learning_rate_scheduler.step()
            inputs, labels = data_batch['data'], data_batch['label']

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            mtbdnn = mtbdnn.to(DEVICE)
            # mlp = mlp.to(DEVICE)

            optimizer.zero_grad()

            # outputs = mlp.forward(inputs)
            outputs = mtbdnn.forward(inputs)
            loss = criterion(outputs, labels)
            # loss.requires_grad = True  # explicitly declare require gradient

            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 10 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training\n')
    print('save trained model...')
    model_path_dir = './model'
    if not os.path.isdir(model_path_dir) or not os.path.exists(model_path_dir):
        os.makedirs(model_path_dir)
    torch.save(mtbdnn.state_dict(), os.path.join(model_path_dir, 'zhihu_live_mtbdnn.pth'))
    # torch.save(mlp.state_dict(), os.path.join(model_path_dir, 'zhihu_live_mlp.pth'))

    predicted_labels = []
    gt_labels = []
    for data_batch in testloader:
        inputs, labels = data_batch['data'], data_batch['label']
        inputs = inputs.to(DEVICE)
        mtbdnn = mtbdnn.to(DEVICE)
        # mlp = mlp.to(DEVICE)

        # outputs = mlp.forward(Variable(inputs))
        outputs = mtbdnn.forward(inputs)
        predicted_labels += outputs.to("cpu").data.numpy().tolist()
        gt_labels += labels.numpy().tolist()

    mae_lr = round(mean_absolute_error(np.array(gt_labels), np.array(predicted_labels)), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(gt_labels), np.array(predicted_labels))), 4)
    print('===============The Mean Absolute Error of MTB-DNN is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of MTB-DNN is {0}===================='.format(rmse_lr))


if __name__ == '__main__':
    train_set, test_set, train_label, test_label = split_train_test("../spider/ZhihuLiveDB.xlsx", False)
    # train_and_test_model(train_set, test_set, train_label, test_label)
    train_and_test_mtnet(train_set, test_set, train_label, test_label, 30)
