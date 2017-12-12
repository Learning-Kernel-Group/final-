import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
import pgd_l2
import pgd_new
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
        if self.method == 'SVC':
            return SVC(C=c, kernel='precomputed')
        if self.method == 'KRR':
            return KernelRidge(alpha=c, kernel='precomputed')
    
    def get_kernel(self, x, y, lam=1., L=1.):
        return self.find_kernel(x, y, degree=self.degree, lam=lam, eta=self.eta,
            L=L, mu0=self.mu0, mu_init=self.mu_init, eps=self.eps, subsampling=self.subsampling)
        
    def cv(self):
        for lam in self.lam_range:
            for L in self.L_range:
                _, gTrain = self.get_kernel(self.xTrain, self.yTrain, lam=lam, L=L)
                classifier = self.get_classifier(c=lam)
                self.cv_error[L,lam] = - cross_val_score(classifier, gTrain, self.yTrain, cv=10, scoring='neg_mean_squared_error').mean()
        self.best_L, self.best_lam = min(self.cv_error, key=self.cv_error.get)
        self.cv_error_best = self.cv_error[self.best_L,self.best_lam]
        self.classifier = self.get_classifier(c=self.best_lam)
        self.mu, self.gTrain = self.get_kernel(self.xTrain, self.yTrain, lam=self.best_lam, L=self.best_L)
        self.model = self.classifier.fit(gTrain, self.yTrain)
        print 'cv -> ', problem.cv_error_best
    
    def predict(self):
        tmp = self.make_test_kernels(self.xTrain, self.xTest, subsampling=self.subsampling)
        self.gTest = self.sum_weight_kernels(tmp, self.mu) ** self.degree
        self.yPredictR = self.model.predict(self.gTest)
        self.yPredictC = 2 * (self.yPredictR >= 0.) - 1
    
    def score(self):
        self.predict()
        self.mse = np.sqrt(mean_squared_error(self.yTest,self.yPredictR))
        self.msf = np.mean(self.yTest != self.yPredictC)
        print 'mse -> ', problem.mse
        print 'msf -> ', problem.msf
    '''
    def statistical_cv(self):
        for lam in self.lam_range:
            for L in self.L_range:
                _, gTrain = self.get_kernel(self.x, self.y, lam=lam, L=L)
                classifier = self.get_classifier(c=lam)
                self.cv_error[L,lam] = - cross_val_score(classifier, gTrain, self.y, cv=10, scoring='neg_mean_squared_error').mean()
        self.best_L, self.best_lam = min(self.cv_error, key=self.cv_error.get)
        self.cv_error_best = self.cv_error[self.best_L,self.best_lam]
        self.classifier = self.get_classifier(c=self.best_lam)
        self.mu, self.gTrain = self.get_kernel(self.x, self.y, lam=self.best_lam, L=self.best_L)
        self.model = self.classifier.fit(gTrain, self.y)
        print 'cv -> ', problem.cv_error_best
        
    def statistical_score(self):
        tmp = self.make_test_kernels(self.x, self.x, subsampling=self.subsampling)
        self.g = self.sum_weight_kernels(tmp, self.mu) ** self.degree
        self.mse_stat = - cross_val_score(self.classifier, self.g, self.y, scoring='neg_mean_squared_error',
            cv = RepeatedKFold(n_splits=2, n_repeats=30)).mean()
        print 'stat mse -> ', problem.mse_stat
    '''    
    def benchmark(self, method=None):
        print 'benchmark model: ' + method
        classifier = eval(method)
        print 'cv -> ', - cross_val_score(classifier, self.xTrain, self.yTrain, cv=10, scoring='neg_mean_squared_error').mean()
        classifier.fit(self.xTrain,self.yTrain)
        tmp = classifier.predict(self.xTest)
        self.mse_bm = np.sqrt(mean_squared_error(self.yTest,tmp))
        print 'test mse -> ', self.mse_bm
        self.msf_bm = np.mean(self.yTest != (2 * (tmp >= 0.) - 1))
        print 'test msf -> ', self.msf_bm
        
    def uniform(self, method=None):
        tmp = self.make_test_kernels(self.x, self.x, subsampling=self.subsampling)
        mu = np.ones(self.n_features) / self.n_features 
        g = self.sum_weight_kernels(tmp, mu) ** self.degree
        yPredictR = KernelRidge.predict(self.gTest)
        yPredictC = 2 * (self.yPredictR >= 0.) - 1
        
if __name__ == '__main__':
    
    data_sets = {1:'ionosphere', 2:'sonar', 3:'breast-cancer', 4:'diabetes', 5:'fourclass', 6:'german',
        7:'heart', 8:'kin8nm', 9:'madelon', 10:'supernova'}
    
    data = 1
    alg = 'pgd' 
    degree = 1
    k = 3
    lam_range = [0.01]#[2**(i-k) for i in range(2*k+1)]#[0.1,0.2,0.5,1.,2.]#[0.01,0.1,1.,10.,50.,80.,100.]
    eta = 1.
    L_range = [1.]#lam_range#[0.5,1.,2.]#[0.01,0.1,1.,10.,100.]
    eps = 1e-4
    subsampling = 1
    mu0 = 1.
    mu_init = 1.
    
    dataset = data_sets[data]
    problem = Problem(dataset=dataset, alg=alg, degree=degree, 
        lam_range=lam_range, eta=eta, L_range=L_range, mu0=mu0, mu_init=mu_init, eps=eps, subsampling=subsampling)
    '''
    problem.statistical_cv()
    problem.statistical_score()
    
    problem.statistical_benchmark(method='KernelRidge()')
    
    problem.cv()
    problem.score()
    
    problem.benchmark(method='KernelRidge()')
    '''
    mse = []
    msf = []
    mse_bm = []
    msf_bm = []
    for i in range(30):
        preprocess._preprocess(dataset, 10000*i)
        problem = Problem(dataset=dataset, alg=alg, method=method, degree=degree, 
            lam_range=lam_range, eta=eta, L_range=L_range, mu0=mu0, mu_init=mu_init, eps=eps, subsampling=subsampling)
        problem.cv()
        problem.score()
        mse.append(problem.mse)
        msf.append(problem.msf)    
        problem.benchmark()
        mse_bm.append(problem.mse_bm)
        msf_bm.append(problem.msf_bm)
    mse = np.array(mse)
    msf = np.array(msf)    
    mse_bm = np.array(mse_bm)
    msf_bm = np.array(msf_bm)    
            
    print mse.mean(), '+', mse.std()
    print mse_bm.mean(), '+', mse_bm.std()
    print msf.mean(), '+', msf.std()
    print msf_bm.mean(), '+', msf_bm.std()
