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
from deslib.static.single_best import SingleBest
from deslib.util.datasets import make_P2

import sklearn.preprocessing as preprocessing
import scipy.io as sio
import time
#import xlsxwriter
import os
#from datetime import datetime
import warnings

from myfunctions import save_elements, load_elements, write_in_latex_table

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
    mcb = MCB(pool_classifiers, k=k)
    rank = Rank(pool_classifiers, k=k)
    knop = KNOP(pool_classifiers, k=k)
    meta = METADES(pool_classifiers, k=k)
    desfh_w = DESFH(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=False)
    desfh_m = DESFH(pool_classifiers, k=k, theta=theta, mu=NO_Hyperbox_Thereshold, mis_sample_based=True)
    oracle = Oracle(pool_classifiers)
    single_best = SingleBest(pool_classifiers,n_jobs=-1)
    majority_voting = pool_classifiers
    list_ds = [majority_voting,single_best, oracle, knorau, ola, desknn,mcb, rank, knop,meta, desfh_w,desfh_m]
    methods_names = ['Majority-Voting', 'Single-Best', 'Oracle', 'KNORA-U', 'OLA', 'DES-KNN','MCB', 'Rank', 'KNOP','META-DES', 'FH-DES_W', 'FH-DES_M']
    methodsnames = methods_names
    # fit the ds techniques
    for ds in list_ds[1:]:
        ds.fit(X_DSEL, y_DSEL)

    return list_ds, methods_names

def write_results_to_file(result,methods, datasetName):
    path =  "Results/" + datasetName + "Final Results.p"
    rfile = open(path, mode="wb")
    pickle.dump(methods,rfile)
    pickle.dump(result,rfile)
    rfile.close()

def run_process(datasetName):
    redata = sio.loadmat("DataSets/" + datasetName + ".mat")
    data = redata['dataset']
    X = data[:, 0:-1]
    y = data[:, -1]
    print(datasetName, "is readed.")
    state = 0
    print(datasetName, ': ', X.shape)
    ### ### ### ### ### ### ### ### ###
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    #### **** #### **** #### **** #### **** #### **** #### ****
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    #### **** #### **** #### **** #### **** #### **** #### ****
    result_one_dataset = np.zeros((NO_techniques, no_itr))
    for itr in range(0, no_itr):
        if do_train:
            # rand = np.random.randint(1,10000,1)
            rng = np.random.RandomState(state)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y,
                                                                random_state=rng)  # stratify=y
            X_DSEL, X_test, y_DSEL, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test,
                                                              random_state=rng)  # stratify=y_test

            ###########################################################################
            #                               Training                                  #
            ###########################################################################
            model = CalibratedClassifierCV(Perceptron(max_iter=100, tol=10e-3, alpha=0.001, penalty=None), cv=5,method='isotonic')
            # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, ))
            pool_classifiers = BaggingClassifier(model, n_estimators=NO_classifiers, bootstrap=True,
                                                 max_samples=1.0,
                                                 random_state=rng)

            # print(datasetName," ", state, " is loaded.")
            pool_classifiers.fit(X_train, y_train)
            list_ds, methods_names = initialize_ds(pool_classifiers, X_DSEL, y_DSEL, k=7)
            if(save_all_results):
                save_elements(datasetName,itr,state,pool_classifiers,X_train,y_train,X_test,y_test,X_DSEL,y_DSEL,list_ds,methods_names)
        else: # do_not_train
            pool_classifiers,X_train,y_train,X_test,y_test,X_DSEL,y_DSEL,list_ds,methods_names = load_elements(datasetName,itr,state)
        ###########################################################################
        #                               Generalization                            #
        ###########################################################################

        for ind in range(0, len(list_ds)):
            result_one_dataset[ind, itr] = list_ds[ind].score(X_test, y_test) * 100
        state += 1
        write_results_to_file(result_one_dataset, methods_names, datasetName)
    return result_one_dataset,methods_names,list_ds

theta = .25
NO_Hyperbox_Thereshold = 0.99
NO_classifiers = 100
no_itr = 20
save_all_results = True
do_train = True
NO_techniques = 12
datasets = {
#30 Dataset "Ionosphere", "Adult", "Glass",
"Seeds",  "Magic", "CTG", "Faults", "Segmentation", "WDVG1", "Ecoli", "Phoneme","Liver",
 "German", "Laryngeal1",  "Vertebral", "Banana",
"Laryngeal3",  "Pima", "Blood",  "Haberman",  "Lithuanian",    "Weaning",  "Breast",
"Heart",     "Wine",    "ILPD",   "Sonar",
"Mammographic",  "Thyroid",  "Iris",  "Monk2",  "Vehicle"
}
datasets = sorted(datasets)
list_ds = []
methods_names = []

NO_datasets = len(datasets)
whole_results = np.zeros([NO_datasets,NO_techniques,no_itr])


dataset_count = 0
done_list = []
for datasetName in datasets:
    try:
        result,methods_names,list_ds = run_process(datasetName)
        whole_results[dataset_count,:,:] = result
        dataset_count +=1
        done_list.append(datasetName)

    except:
        print(datasetName, "could not be readed")
pdata = np.concatenate((whole_results[:,0:3 ,:],whole_results[:,10: ,:]) , axis=1)
metName = methods_names[0:3] + methods_names[10:]
write_in_latex_table(pdata,done_list,metName,rows="datasets")
write_in_latex_table(whole_results,done_list,methods_names,rows="datasets")

duration = 4  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
