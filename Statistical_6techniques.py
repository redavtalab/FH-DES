"""
====================================================================
Dynamic selection with linear classifiers: P2 Problem
====================================================================
"""

import pickle
#import pandas as pd
#import openml
#from openml.tasks import TaskType
#import matplotlib.pyplot as plt
#from matplotlib.cm import get_cmap
#from matplotlib.ticker import FuncFormatter
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

from deslib.des import DESKNN
from deslib.des import KNORAE
from deslib.des import KNORAU
from deslib.des import KNOP
from deslib.des import METADES
from deslib.des import DESFH
from deslib.static.oracle import Oracle
from deslib.util.datasets import make_P2

import sklearn.preprocessing as preprocessing
import scipy.io as sio
import time
#import xlsxwriter
import os
#from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
#+##############################################################################

# Prepare the DS techniques. Changing k value to 7.
def initialize_ds(pool_classifiers, X_DSEL, y_DSEL, k=7):
    knorau = KNORAU(pool_classifiers, k=k)
    #kne = KNORAE(pool_classifiers, k=k)
    desknn = DESKNN(pool_classifiers, k=k)
    ola = OLA(pool_classifiers, k=k)
    #lca = LCA(pool_classifiers, k=k)
    #mla = MLA(pool_classifiers, k=k)
    #mcb = MCB(pool_classifiers, k=k)
    #rank = Rank(pool_classifiers, k=k)
    #knop = KNOP(pool_classifiers, k=k)
    meta = METADES(pool_classifiers, k=k)
    desfh_w = DESFH(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=False)
    desfh_m = DESFH(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True)
    #oracle = Oracle(pool_classifiers)
    list_ds = [knorau, ola, desknn,meta, desfh_w,desfh_m]
    methods_names = ['KNORA-U', 'OLA', 'DES-KNN','META-DES', 'FH-DES_W', 'FH-DES_M']

    # fit the ds techniques
    for ds in list_ds:
        ds.fit(X_DSEL, y_DSEL)

    return list_ds, methods_names

def write_results_to_file(result,methods, datasetName):
    path =  "Results/" + datasetName + "Final Results.p"
    rfile = open(path, mode="wb")
    pickle.dump(methods,rfile)
    pickle.dump(result,rfile)
    rfile.close()

def train_phase():
    for taskID, datasetName in datasets.items():

        try:
            # task = openml.tasks.get_task(taskID)
            # dataset = task.get_dataset()
            # X, y = task.get_X_and_y()
            redata = sio.loadmat("DataSets/" + datasetName + ".mat")
            data = redata['dataset']
            X = data[:, 0:-1]
            y = data[:, -1]
            print(datasetName, "is readed.")
            state = 0
            print(datasetName, ': ', X.shape)

            for itr in range(0, no_itr):
                # print("Iteration: ",itr)
                # rand = np.random.randint(1,10000,1)
                rng = np.random.RandomState(state)
                path = "SavedPools/" + datasetName + str(itr) + "-RState-" + np.str(state) + ".p"
                poolspec = open(path, mode="wb")

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y,
                                                                    random_state=rng)  # stratify=y
                X_DSEL, X_test, y_DSEL, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test,
                                                                  random_state=rng)  # stratify=y_test
                #### **** #### **** #### **** #### **** #### **** #### ****
                scaler = preprocessing.MinMaxScaler()
                X = scaler.fit_transform(X)
                X_train = scaler.transform(X_train)
                X_DSEL = scaler.transform(X_DSEL)
                X_test = scaler.transform(X_test)
                #### **** #### **** #### **** #### **** #### **** #### ****

                model = CalibratedClassifierCV(Perceptron(max_iter=100, tol=10e-3, alpha=0.001, penalty=None), cv=5)
                # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, ))
                pool_classifiers = BaggingClassifier(model, n_estimators=NO_classifiers, bootstrap=True,
                                                     max_samples=1.0,
                                                     random_state=rng)

                # print(datasetName," ", state, " is loaded.")
                pool_classifiers.fit(X_train, y_train)
                list_ds, methods_names = initialize_ds(pool_classifiers, X_DSEL, y_DSEL, k=7)

                pickle.dump(state, poolspec)
                pickle.dump(pool_classifiers, poolspec)
                pickle.dump(X_train, poolspec)
                pickle.dump(y_train, poolspec)
                pickle.dump(X_test, poolspec)
                pickle.dump(y_test, poolspec)
                pickle.dump(X_DSEL, poolspec)
                pickle.dump(y_DSEL, poolspec)
                pickle.dump(list_ds, poolspec)
                pickle.dump(methods_names, poolspec)

                del pool_classifiers
                state += 1
        except:
            print(datasetName, "could not be readed")
            continue

def generalization_phase():
    dataset_counter = 0

    baggingScore = 0
    oracleScores = np.zeros((no_itr, len(datasets)))
    #overall_results = np.zeros((len(datasets),No_methods, no_itr))
    methods_names = datasets.values()
    for datasetName in datasets.values():
        state = 0
        starttime = time.time()
        #    try:
        result = np.zeros((No_methods+1, no_itr))
        for itr in range(0, no_itr):
            filepath = "SavedPools/" + datasetName + str(itr) + "-RState-" + np.str(state) + ".p"
            try:
                poolspec = open(filepath, "rb")
                state = pickle.load(poolspec)
                pool_classifiers = pickle.load(poolspec)
                X_train = pickle.load(poolspec)
                y_train = pickle.load(poolspec)
                X_test = pickle.load(poolspec)
                y_test = pickle.load(poolspec)
                X_DSEL = pickle.load(poolspec)
                y_DSEL = pickle.load(poolspec)
                list_ds = pickle.load(poolspec)
                methods_names = pickle.load(poolspec)

                for ind in range(0, No_methods):
                    result[ind, itr] = list_ds[ind].score(X_test, y_test) * 100
                state += 1
            except:
                print(datasetName, "could not be loaded")
                state += 1
                continue
            oracle = Oracle(pool_classifiers).fit(X_train, y_train)
            orasco = oracle.score(X_test, y_test) * 100
            oracleScores[itr, dataset_counter] = orasco
            result[No_methods,itr] = orasco
            baggingScore += pool_classifiers.score(X_test, y_test) * 100


        print("\n\n Results for", datasetName, ":")
        print("Running time: " + str(time.time() - starttime))


        print('Oracle result:', np.average(oracleScores[:, dataset_counter]))
        print('Bagging result:', baggingScore / no_itr)
        baggingScore = 0
        ## without Oracle
        #write_results_to_file(result,methods_names,datasetName)
        #with Oracle
        methods_names.append("Oracle")
        write_results_to_file(result,methods_names,datasetName)
        dataset_counter += 1

theta = .1
NO_Hyperbox_Thereshold = 0.98
NO_classifiers = 100
no_itr = 20

datasets = {
1: "Adult"  ,
#2:"Faults"   ,
3:"Ionosphere" ,
#4:"Mammographic" ,
5:"Thyroid",
#6:"Banana" ,
7:"German"  ,
8:"Laryngeal1" ,
9:"Monk2" ,
10:"Vehicle",
11:"Blood"  ,
12:"Glass"    ,
13:"Laryngeal3" ,
#14:"Phoneme" ,
15:"Vertebral",
16:"Breast" ,
17:"Haberman" ,
18:"Lithuanian" ,
19:"Pima" ,
#20:"WDVG1",
#21:"CTG" ,
22:"Heart" ,
23:"Liver" ,
#24:"Segmentation" ,
25:"Weaning",
#26:"Ecoli"  ,
27:"ILPD"  ,
#28:"Magic"  ,
29:"Sonar" ,
30:"Wine"
}


NO_datasets = len(datasets)
list_ds = []
No_methods = 6
train_phase()
generalization_phase()

duration = 4  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))