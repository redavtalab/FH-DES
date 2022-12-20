"""
====================================================================
Dynamic selection with linear classifiers: Statistical Experiment
====================================================================
"""

import pickle
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from deslib.des import DESKNN, FHDES_JFB, DESFHMW_JFB, DESFHMW_allboxes, FHDES_prior, FHDES_Allboxes_vector, \
    FHDES_JFB_vector, DESFHMW_JFB_vector, FHDES_prior_vector, DESFHMW_prior_vector, DESFHMW_allboxes_vector, METADES

from deslib.static.oracle import Oracle
from deslib.static.single_best import SingleBest
from deslib.util.datasets import make_P2

import sklearn.preprocessing as preprocessing
import scipy.io as sio
import time
import os
import warnings
import xlsxwriter
import math
from myfunctions import *
warnings.filterwarnings("ignore")

"""
====================================================================
Dynamic selection with Complex classifiers: Statistical Experiment
====================================================================
"""
import multiprocessing
import pickle
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from deslib.dcs import LCA
from deslib.dcs import MLA
from deslib.dcs import OLA
from deslib.dcs import MCB
from deslib.dcs import Rank

from deslib.des import DESKNN, FHDES_JFB, DESFHMW_JFB, DESFHMW_allboxes, FHDES_prior, FHDES_Allboxes_vector, \
    FHDES_JFB_vector, DESFHMW_JFB_vector, FHDES_prior_vector, DESFHMW_prior_vector, DESFHMW_allboxes_vector
from deslib.des import KNORAE, KNORAU, KNOP, METADES, DESClustering

from deslib.des import DESFHMW_prior
from deslib.des import FHDES_Allboxes,FHDES_Allboxes_GPU
from deslib.static.oracle import Oracle
from deslib.static.single_best import SingleBest
from deslib.util.datasets import make_P2
# from FMNN.fmnnclassification import FMNNClassification

import sklearn.preprocessing as preprocessing
import scipy.io as sio
import time
import os
import warnings
import math
from myfunctions import *

warnings.filterwarnings("ignore")

# Prepare the DS techniques. Changing k value to 7.

def initialize_ds(pool_classifiers, X_DSEL, y_DSEL, k=7):
    knorau = KNORAU(pool_classifiers, k=k)
    kne = KNORAE(pool_classifiers, k=k)
    desknn = DESKNN(pool_classifiers, k=k)
    ola = OLA(pool_classifiers, k=k)
    lca = LCA(pool_classifiers, k=k)
    mla = MLA(pool_classifiers, k=k)
    mcb = MCB(pool_classifiers, k=k)
    rank = Rank(pool_classifiers, k=k)
    knop = KNOP(pool_classifiers, k=k)
    meta = METADES(pool_classifiers, k=k)

    FH_1v = FHDES_JFB_vector(pool_classifiers, k=k, theta=.6, mu=.1, mis_sample_based=True,
                     doContraction=False, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)

    FH_3v = FHDES_Allboxes_vector(pool_classifiers, k=k, theta=.5, mu=.3, mis_sample_based=True,
                          doContraction=False, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)

    FH_5v = FHDES_JFB_vector(pool_classifiers, k=k, theta=.7, mu=.1, mis_sample_based=True,
                     doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)

    FH_6v = DESFHMW_JFB_vector(pool_classifiers, k=k, theta=.7, mu=.5, mis_sample_based=True,
                       doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)

    FH_8v = DESFHMW_allboxes_vector(pool_classifiers, k=k, theta=.5, mu=.4, mis_sample_based=True,
                            doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)



    oracle = Oracle(pool_classifiers)
    single_best = SingleBest(pool_classifiers, n_jobs=-1)
    majority_voting = pool_classifiers

    list_ds = [FH_1v, FH_3v, FH_5v, FH_6v,  FH_8v]
    # list_ds = [FH_1v, FH_2v, FH_3v, FH_4v, FH_5v, FH_6v, FH_7v, FH_8v, FH_9v, FH_10v]



    return list_ds, methods_names

def save_pool(datasetName,pools):
    path = ExperimentPath + "/Pools/" + datasetName + "_pools.p"
    poolspec = open(path, mode="wb")
    pickle.dump(pools, poolspec)
    poolspec.close()
def load_pool(datasetName):
    path = ExperimentPath + "/Pools/" + datasetName + "_pools.p"
    poolspec = open(path, mode="rb")
    return pickle.load(poolspec)
def save_model(tec_name,datasetName,ds):
    path = ExperimentPath + "/Models/" + tec_name +"_"+datasetName + "_model.p"
    poolspec = open(path, mode="wb")
    pickle.dump(ds, poolspec)
    poolspec.close()
def load_model(tec_name,datasetName):
    path = ExperimentPath + "/Models/" + tec_name +"_"+ datasetName + "_model.p"
    poolspec = open(path, mode="rb")
    return pickle.load(poolspec)
def save_results(tec_name,datasetName,accuracy,labels,yhat,noBoxes):
    folder =  ExperimentPath + "/Results/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = folder + tec_name +"_"+datasetName + "_result.p"
    poolspec = open(path, mode="wb")
    pickle.dump(accuracy, poolspec)
    pickle.dump(labels, poolspec)
    pickle.dump(yhat, poolspec)
    pickle.dump(noBoxes,poolspec)
    poolspec.flush()
    poolspec.close()
def model_setup(datasetName, no_samples):
    global methods_names
    pools = load_pool(datasetName)
    ds_matrix = []
    for itr in range(no_itr):

        pool_classifiers = pools[itr]
        [X_train, X_test, X_DSE, y_train, y_test, y_DSE] = np.load('Datasets3/' + datasetName + str(itr) + '.npy',allow_pickle=True)
        X_DSEL=X_DSE[:no_samples, :]
        y_DSEL = y_DSE[:no_samples]

        list_ds, methods_names = initialize_ds(pool_classifiers,X_DSEL,y_DSEL)
        ds_matrix.append(list_ds)

    for tec in range(NO_techniques):

        # ds_tec = []
        results = []
        labels = []
        yhat = []
        noBoxes = 0
        for itr in range(no_itr):
            [X_train, X_test, X_DSEL, y_train, y_test, y_DSEL] = np.load('Datasets3/' + datasetName + str(itr) + '.npy',  allow_pickle=True)
            X_DSEL = X_DSE[:no_samples, :]
            y_DSEL = y_DSE[:no_samples]

            ds_itr = ds_matrix[itr][tec]
            ds_itr.fit(X_DSEL, y_DSEL)

            labels.append(y_test)
            results.append(ds_itr.score(X_test, y_test) * 100)
            if methods_names[tec] == 'Oracle':
                yhat.append(ds_itr.predict(X_test,y_test))
            else:
                yhat.append(ds_itr.predict(X_test))
            noBoxes += ds_itr.NO_hypeboxes
        noBoxes = noBoxes/no_itr
        print(methods_names[tec], noBoxes)
        save_results(methods_names[tec],datasetName +np.str(no_samples),results,labels,yhat,noBoxes)

theta = .0
NO_Hyperbox_Thereshold = 0
ExperimentPath = "LargeScale1"
NO_classifiers =100
no_itr = 5


# methods_names = ['SB','MV', 'Oracle',]
methods_names = ['FH_1v', 'FH_3v',  'FH_5v',  'FH_6v',  'FH_8v']
# methods_names = [ 'FH_8v']


NO_techniques = len(methods_names)

start = time.time()
n_samples_ = [ 100000] #, 300000, 500000, 700000,900000 ]
datasets ={
            # "Incidents",
            "Data"
           }

for datasetName in datasets:
    # try:
    print(datasetName)
    for n in n_samples_:
        t1 = time.time()
        model_setup(datasetName,n)
        print("Execution time:", n ,time.time()-t1)





