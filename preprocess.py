import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.utils import shuffle
import re

# ok
def _preprocess(data_set, random_state=None, frac=0.5):
    with open('data/uci/' + data_set + '.data', 'r') as _file:
        data_array = []
        for line in _file:
            tmp = line.rstrip().split(',')
            data_array.append(tmp)
        data_array = np.array(data_array)
        features = data_array[:, :-1].astype(np.float)
        labels = data_array[:, -1]
        classes = list(set(labels))
        if data_set != 'kin8nm':
            for i in range(labels.shape[0]):
                if labels[i] == classes[0]:
                    labels[i] = 1
                else:
                    labels[i] = -1
        labels = labels.astype(np.float)
        features, labels = shuffle(features, labels, random_state=random_state)
        frac = int( features.shape[0] * frac )
        xTrain = features[:frac, :]
        xTest = features[frac:, :]
        yTrain = labels[:frac]
        yTest = labels[frac:]
        scaler = preprocessing.MinMaxScaler(feature_range=(0., 1.))
        xTrain = scaler.fit_transform(xTrain)
        xTest = scaler.transform(xTest)
        mTrain = np.mean(xTrain,axis=0)
        xTrain -= mTrain
        xTest -= mTrain
        _list = [xTrain, yTrain, xTest, yTest]
        with open('data_python/' + data_set, 'wb') as _file:
            pickle.dump(_list, _file)

def preprocessor_libsvm_data(filename, format_label_func=lambda _: _):
    with open('data/uci/' + filename + '.data', 'r') as inputfile:
        features = []
        labels = []
        for line in inputfile:
            container = line.rstrip().split()
            label = float(container[0])
            label = int(format_label_func(label))
            del container[0]
            pattern = re.compile(r"[-+]?\d+:([-+]?\d*\.\d+|[-+]?\d+)")
            feature = []
            for phrase in container:
                # print(phrase)
                target = re.findall(pattern, phrase)
                # print(target)
                feature.append(float(target[0]))
            features.append(feature)
            labels.append(label)
        classes = list(set(labels))
        for i in range(len(labels)):
            if labels[i] == classes[0]:
                labels[i] = 1
            else:
                labels[i] = -1
        features = np.array(features)
        labels = np.array(labels).reshape((-1,1))
        labels = labels.astype(np.float)
        data = np.concatenate((features,labels),axis=1)
        #np.random.shuffle(data)
        x = data[:,:-1]
        y = data[:,-1]
        frac = int( features.shape[0] * 0.8 )
        xTrain = x[:frac, :]
        xTest = x[frac:, :]
        yTrain = y[:frac]
        yTest = y[frac:]
        scaler = preprocessing.MinMaxScaler(feature_range=(0., 1.))
        xTrain = scaler.fit_transform(xTrain)
        xTest = scaler.transform(xTest)
        _mean = np.mean(xTrain,axis=0)
        xTrain = xTrain - _mean
        xTest = xTest - _mean
        _list = [xTrain, yTrain, xTest, yTest]
        with open('data_python/' + filename, 'wb') as _file:
            pickle.dump(_list, _file)

def _load_and_save(dataset):
    xTrain = np.load('data/data_chris/' + dataset + '_features_train.npy')
    yTrain = np.load('data/data_chris/' + dataset + '_labels_train.npy')
    xTest = np.load('data/data_chris/' + dataset + '_features_test.npy')
    yTest = np.load('data/data_chris/' + dataset + '_labels_test.npy')
    x = np.concatenate((xTrain,xTest),axis=0)
    y = np.concatenate((yTrain,yTest),axis=0).reshape((-1,1))
    data = np.concatenate((x,y),axis=1)
    data_array = np.array(data)
    features = data_array[:, :-1].astype(np.float)
    labels = data_array[:, -1]
    classes = list(set(labels))
    if dataset != 'kin8nm':
        for i in range(labels.shape[0]):
            if labels[i] == classes[0]:
                labels[i] = 1
            else:
                labels[i] = -1
    labels = labels.astype(np.float)
    features, labels = shuffle(features, labels)
    frac = .5
    frac = int( features.shape[0] * frac )
    xTrain = features[:frac, :]
    xTest = features[frac:, :]
    yTrain = labels[:frac]
    yTest = labels[frac:]
    scaler = preprocessing.MinMaxScaler(feature_range=(0., 1.))
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)
    mTrain = np.mean(xTrain,axis=0)
    xTrain -= mTrain
    xTest -= mTrain
    _list = [xTrain, yTrain, xTest, yTest]
    with open('data_python/' + dataset, 'wb') as _file:
        pickle.dump(_list, _file)

if __name__ == '__main__':
    data_sets = ['kin8nm', 'ionosphere', 'sonar']
    for dataset in data_sets:
        _preprocess(dataset)
    data_sets_chris = ['breast-cancer', 'diabetes', 'fourclass', 'german',
        'heart', 'madelon']
    for dataset in data_sets_chris:
        preprocessor_libsvm_data(dataset)
    data_sets_load = ['supernova']
    for dataset in data_sets_load:
        _load_and_save(dataset)
