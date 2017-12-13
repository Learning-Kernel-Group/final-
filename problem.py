import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pgd_l2
import pgd_new
import pgd_new_gd
import pgd
import kernel
import preprocess

class Problem():

    def __init__(self, dataset=None, alg=None, degree=1, lam_range=[1.],
        eta=1., L_range=[1.], mu0=1., mu_init=1., eps=1e-3, subsampling=1):
        
        self.dataset = dataset
        self.find_kernel = eval(alg).find_kernel
        self.sum_weight_kernels = eval(alg).sum_weight_kernels
        self.degree = degree
        self.lam_range = lam_range
        self.eta = eta
        self.L_range = L_range
        with open('data_python/' + self.dataset, 'rb') as _file:
            [self.xTrain, self.yTrain, self.xTest, self.yTest] = pickle.load(_file)
        self.x = np.concatenate((self.xTrain, self.xTest), axis=0)
        self.y = np.concatenate((self.yTrain, self.yTest))
        self.n_features = self.xTrain.shape[1]
        self.mu0 = mu0 * np.ones(self.n_features)
        self.mu_init = mu_init * np.ones(self.n_features)
        self.eps = eps
        self.subsampling = subsampling
        self.make_test_kernels = kernel.make_test_kernels
        self.cv_error = {}
        
    def get_classifier(self, c=1.):
        return KernelRidge(alpha=c, kernel='precomputed')
    
    def get_kernel(self, x, y, lam=1., L=1.):
        return self.find_kernel(x, y, degree=self.degree, lam=lam, eta=self.eta,
            L=L, mu0=self.mu0, mu_init=self.mu_init, eps=self.eps, subsampling=self.subsampling)
    
    def my_cv(self):
        kf = KFold(n_splits=10, shuffle = False)
        x = self.x
        y = self.y
        score = {}
        std = {}		
        for lam in self.lam_range:
            for L in self.L_range:
                tmp_score = []
                for train_index, test_index in kf.split(x):
                    xTrain, xTest = x[train_index], x[test_index]
                    yTrain, yTest = y[train_index], y[test_index]
                    mu, gTrain = self.get_kernel(xTrain, yTrain, lam=lam, L=L)
                    tmp = self.make_test_kernels(xTrain, xTest, subsampling=self.subsampling)
                    gTest = self.sum_weight_kernels(tmp, mu) ** self.degree
                    classifier = self.get_classifier(c=lam)
                    classifier.fit(gTrain,yTrain)
                    yPredict = classifier.predict(gTest)
                    tmp_score.append(mean_squared_error(yTest,yPredict)) #
                score[lam,L] = np.array(tmp_score).mean()
                std[lam,L] = np.array(tmp_score).std()
        self.best_lam, self.best_L = min(score, key=score.get)
        self.cv_score = score
        self.cv_best_score = np.sqrt(score[self.best_lam, self.best_L])
        self.cv_best_std = std[self.best_lam, self.best_L]
        
if __name__ == '__main__':
    
    data_sets = {1:'ionosphere', 2:'sonar', 3:'breast-cancer', 4:'diabetes', 5:'fourclass', 6:'german',
        7:'heart', 8:'kin8nm', 9:'madelon', 10:'supernova'}
    
    data = 1
    alg = 'pgd' 
    degree = 1
    k = 5
    lam_range = [8.]#[2**(i-k) for i in range(2*k+1)]
    eta = 1.
    L_range = [0.5]#[0.1, 0.25, 0.5, 1.]
    eps = 1e-4
    subsampling = 1
    mu0 = 1.
    mu_init = 1.
    
    dataset = data_sets[data]
    problem = Problem(dataset=dataset, alg=alg, degree=degree, 
        lam_range=lam_range, eta=eta, L_range=L_range, mu0=mu0, mu_init=mu_init, eps=eps, subsampling=subsampling)
    
    problem.my_cv()
    print problem.cv_best_score, '+', problem.cv_best_std
    print problem.best_lam, problem.best_L
