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

from deslib.des import DESKNN, FHDES_JFB, DESFHMW_JFB, DESFHMW_allboxes, FHDES_prior
from deslib.des import KNORAE
from deslib.des import KNORAU
from deslib.des import KNOP
from deslib.des import METADES
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

    FH_1 = FHDES_JFB(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                     doContraction=False, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=True)
    FH_2 = FHDES_JFB(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                     doContraction=True, thetaCheck=False, multiCore_process=True, shuffle_dataOrder=True)

    FH_3 = FHDES_Allboxes(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                          doContraction=False, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=True)
    FH_4 = FHDES_Allboxes(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                          doContraction=True, thetaCheck=False, multiCore_process=True, shuffle_dataOrder=True)

    FH_5 = FHDES_JFB(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                     doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=True)
    FH_6 = DESFHMW_JFB(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                       doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=True)

    FH_7 = FHDES_Allboxes(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                          doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=True)
    FH_8 = DESFHMW_allboxes(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                            doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=True)

    FH_9 = FHDES_prior(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                       doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=True)
    FH_10 = DESFHMW_prior(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                          doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=True)

    FH_GG = FHDES_Allboxes_GPU(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True,
                          doContraction=True, thetaCheck=True, multiCore_process=True, shuffle_dataOrder=True)

    oracle = Oracle(pool_classifiers)
    single_best = SingleBest(pool_classifiers, n_jobs=-1)
    majority_voting = pool_classifiers

    list_ds = [FH_1,FH_2,FH_3,FH_4,FH_5,FH_7,FH_8,FH_9]
    # list_ds = [majority_voting, single_best, oracle, knorau, kne, desknn, ola, rank, knop, meta, mcb, fh_4]
              # fhM_cY_tN, fhM_cY_tY]
    # methods_names = ['KNORA-U', 'KNORA-E', 'DESKNN', 'OLA', 'RANK', 'KNOP', 'META-DES', 'MCB']
    # # methods_names = ['MV', 'SB', 'Oracle', 'KNORA-U', 'KNORA-E', 'DESKNN', 'OLA', 'RANK', 'KNOP', 'META-DES', 'MCB', 'FH_4']
    #                 # 'FHm_cN_tY','FHm_cY_tN','FHm_cY_tY' ]

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

theta = .27
NO_Hyperbox_Thereshold = 0.85
ExperimentPath = "Experiment1"
NO_classifiers =100
no_itr = 20
generate_pools = False
do_train = True
do_evaluate = True

methods_names = ['FH_1','FH_2','FH_3','FH_4','FH_5','FH_7','FH_8','FH_9']
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




# datasets = sorted(datasets)
# NO_datasets = len(datasets)
# whole_results = np.zeros([NO_datasets,NO_techniques,no_itr])
# dataset_count = 0
# done_list = []
#
#
# print(f"Time taken = {time.time() - start: .10f}")
# # write_whole_results_into_excel(whole_results, done_list.copy(), methods_names)
# path = ExperimentPath + "WholeResults.p"
# rfile = open(path, mode="wb")
# pickle.dump(whole_results,rfile)
# datasets = done_list
# pickle.dump(datasets,rfile)
# pickle.dump(methods_names,rfile)
# rfile.close()
#
#
# # pdata = np.concatenate((whole_results[:,0:3 ,:],whole_results[:,10 :14,:],whole_results[:,21:22,:]) , axis=1)
# # metName = methods_names[0:3]+ methods_names[10:14] + methods_names[21:22]
# # write_in_latex_table(pdata,done_list,metName,rows="datasets")
# write_in_latex_table(whole_results,done_list,methods_names,rows="datasets")
#
#
# duration = 4  # seconds
# freq = 440  # Hz
# os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
# print("STD:" , np.average(np.std(whole_results,2),0))
#
# # methods_names[0:3]+ methods_names[10:14] + methods_names[21:22]