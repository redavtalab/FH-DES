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

    # FH_1p = FHDES_JFB_vector(pool_classifiers, k=k, theta=.27, mu=.99, mis_sample_based=False,
    #                          doContraction=False, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    # FH_2p = FHDES_JFB_vector(pool_classifiers, k=k, theta=0.001, mu=.4, mis_sample_based=False,
    #                          doContraction=True, thetaCheck=False, multiCore_process=True, shuffle_dataOrder=False)

    # FH_3p = FHDES_Allboxes_vector(pool_classifiers, k=k, theta=.8, mu=.1, mis_sample_based=False,
    #                               doContraction=False, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    # FH_4p = FHDES_Allboxes_vector(pool_classifiers, k=k, theta=0.001, mu=.4, mis_sample_based=False,
    #                               doContraction=True, thetaCheck=False, multiCore_process=True, shuffle_dataOrder=False)

    # FH_5p = FHDES_JFB_vector(pool_classifiers, k=k, theta=.8, mu=.3, mis_sample_based=False,
    #                          doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)

    # FH_6p = DESFHMW_JFB_vector(pool_classifiers, k=k, theta=.99, mu=.4, mis_sample_based=False,
    #                            doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)

    # FH_7p = FHDES_Allboxes_vector(pool_classifiers, k=k, theta=.8, mu=.1, mis_sample_based=False,
    #                               doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    # FH_8p = DESFHMW_allboxes_vector(pool_classifiers, k=k, theta=.99, mu=.1, mis_sample_based=False,
    #                                 doContraction=True, thetaCheck=True, multiCore_process=True,
    #                                 shuffle_dataOrder=False)

    # FH_9p = FHDES_prior_vector(pool_classifiers, k=k, theta=.99, mu=.2, mis_sample_based=False,
    #                            doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    # FH_10p = DESFHMW_prior_vector(pool_classifiers, k=k, theta=.8, mu=.6, mis_sample_based=False,
    #                               doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)

    FH_1v = FHDES_JFB_vector(pool_classifiers, k=k, theta=.6, mu=.1, mis_sample_based=True,
                     doContraction=False, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    FH_2v = FHDES_JFB_vector(pool_classifiers, k=k, theta=0.001, mu=.4, mis_sample_based=True,
                     doContraction=True, thetaCheck=False, multiCore_process=True, shuffle_dataOrder=False)

    FH_3v = FHDES_Allboxes_vector(pool_classifiers, k=k, theta=.5, mu=.3, mis_sample_based=True,
                          doContraction=False, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    FH_4v = FHDES_Allboxes_vector(pool_classifiers, k=k, theta=0.001, mu=.1, mis_sample_based=True,
                          doContraction=True, thetaCheck=False, multiCore_process=True, shuffle_dataOrder=False)

    FH_5v = FHDES_JFB_vector(pool_classifiers, k=k, theta=.7, mu=.1, mis_sample_based=True,
                     doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)

    FH_6v = DESFHMW_JFB_vector(pool_classifiers, k=k, theta=.7, mu=.5, mis_sample_based=True,
                       doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)

    FH_7v = FHDES_Allboxes_vector(pool_classifiers, k=k, theta=.4, mu=.4, mis_sample_based=True,
                          doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    FH_8v = DESFHMW_allboxes_vector(pool_classifiers, k=k, theta=.5, mu=.4, mis_sample_based=True,
                            doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)

    FH_9v = FHDES_prior_vector(pool_classifiers, k=k, theta=.7, mu=.7, mis_sample_based=True,
                       doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)
    FH_10v = DESFHMW_prior_vector(pool_classifiers, k=k, theta=.9, mu=.7, mis_sample_based=True,
                          doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=False)


    oracle = Oracle(pool_classifiers)
    single_best = SingleBest(pool_classifiers, n_jobs=-1)
    majority_voting = pool_classifiers

    # list_ds = [majority_voting, oracle]
    # list_ds = [FH_1v, FH_2v, FH_3v, FH_4v, FH_5v, FH_6v, FH_7v, FH_8v, FH_9v, FH_10v]
    # list_ds = [FH_1p, FH_2p, FH_3p, FH_4p, FH_5p, FH_6p, FH_7p, FH_8p, FH_9p, FH_10p]
    list_ds = [knorau, kne, mcb, desknn, ola, rank, knop, meta, FH_4v]

    # fit the ds techniques
    for ds in list_ds:
        if ds != majority_voting:
            ds.fit(X_DSEL, y_DSEL)


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
    poolspec.flush()
    poolspec.close()
def load_model(tec_name,datasetName):
    path = ExperimentPath + "/Models/" + tec_name +"_"+ datasetName + "_model.p"
    poolspec = open(path, mode="rb")
    return pickle.load(poolspec)

def save_results(tec_name,datasetName,accuracy,labels,yhat):
    path = ExperimentPath + "/Results/" + tec_name +"_"+datasetName + "_result.p"
    poolspec = open(path, mode="wb")
    pickle.dump(accuracy, poolspec)
    pickle.dump(labels, poolspec)
    pickle.dump(yhat, poolspec)
    poolspec.flush()
    poolspec.close()

def pool_generator(datasetName):
    state = 0
    pools = []
    for itr in range(0, no_itr):
        rng = np.random.RandomState(state)
        [X_train, X_test, X_DSEL, y_train, y_test, y_DSEL] =  np.load('Datasets3/' + datasetName +str(itr)+'.npy',allow_pickle=True)

        learner = Perceptron(max_iter=100, tol=10e-3, alpha=0.001, penalty=None, random_state=rng)
        model = CalibratedClassifierCV(learner, cv=5,method='isotonic')
        # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, ), random_state=rng)
        pool_classifiers = BaggingClassifier(model, n_estimators=NO_classifiers, bootstrap=True, max_samples=1.0, random_state=rng)
        pool_classifiers.fit(X_train,y_train)
        pools.append(pool_classifiers)

    path = ExperimentPath + "/Pools/" + datasetName + "_pools.p"
    poolspec = open(path, mode="wb")
    pickle.dump(pools, poolspec)

def model_setup(datasetName):
    global methods_names
    pools = load_pool(datasetName)
    ds_matrix = []
    for itr in range(no_itr):
        pool_classifiers = pools[itr]
        [X_train, X_test, X_DSEL, y_train, y_test, y_DSEL] = np.load('Datasets3/' + datasetName + str(itr) + '.npy', allow_pickle=True)
        list_ds, methods_names = initialize_ds(pool_classifiers,X_DSEL,y_DSEL)
        ds_matrix.append(list_ds)

    for tec in range(NO_techniques):
        ds_tec = []
        for itr in range(no_itr):
            ds_tec.append(ds_matrix[itr][tec])
        save_model(methods_names[tec],datasetName,ds_tec)

def evaluate_model(datasetName):
    for tec in range(NO_techniques):
        results = []
        labels = []
        yhat = []
        ds_tec = load_model(methods_names[tec],datasetName)
        for itr in range(no_itr):
            [X_train, X_test, X_DSEL, y_train, y_test, y_DSEL] = np.load('Datasets3/' + datasetName + str(itr) + '.npy',  allow_pickle=True)
            labels.append(y_test)
            results.append(ds_tec[itr].score(X_test, y_test) * 100)
            if methods_names[tec] == 'Oracle':
                yhat.append(ds_tec[itr].predict(X_test,y_test))
            else:
                yhat.append(ds_tec[itr].predict(X_test))

        save_results(methods_names[tec],datasetName,results,labels,yhat)


datasets = {
    #     Data set of DGA1033 report
    "Audit",
    "Banana",
    "Banknote",
    "Blood",
    "Breast",
    "Car",
    "Datausermodeling",
    "Faults",
    "German",
    "Haberman",
    "Heart",
    "ILPD",
    "Ionosphere",
    "Laryngeal1",
    "Laryngeal3",
    "Lithuanian",
    "Liver",
    "Mammographic",
    "Monk2",
    "Phoneme",
    "Pima",
    "Sonar",
    "Statlog",
    "Steel",
    "Thyroid",
    "Vehicle",
    "Vertebral",
    "Voice3",
    "Weaning",
    "Wine"
}
datasets = sorted(datasets)

theta = .0
NO_Hyperbox_Thereshold = 0
ExperimentPath = "Experiment1"
NO_classifiers =100
no_itr = 20
generate_pools = False
do_train = True
do_evaluate = True

# methods_names = ['SB','MV', 'Oracle',]
methods_names = ['FH_1-M', 'FH_2-M', 'FH_3-M', 'FH_4-M', 'FH_5-M', 'FH_6-M', 'FH_7-M', 'FH_8-M', 'FH_9-M', 'FH_10-M']
# methods_names = ['FH_1-C', 'FH_2-C', 'FH_3-C', 'FH_4-C', 'FH_5-C', 'FH_6-C', 'FH_7-C', 'FH_8-C', 'FH_9-C', 'FH_10-C']
NO_techniques = len(methods_names)

start = time.time()

for datasetName in datasets:
    # try:
    print(datasetName)
    if generate_pools:
        pool_generator(datasetName)
    if do_train:
        t1 = time.time()
        model_setup(datasetName)
        print("Train time",time.time()-t1)
    if do_evaluate:
        t1 = time.time()
        evaluate_model(datasetName)
        print("Test time", time.time() - t1)


