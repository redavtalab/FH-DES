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

from deslib.des import DESKNN
from deslib.des import KNORAE
from deslib.des import KNORAU
from deslib.des import KNOP
from deslib.des import METADES


from deslib.des import DESFHMW_JFB
from deslib.des import FHDES_JFB
from deslib.des import DESFHMW_allboxes
from deslib.des import FHDES_Allboxes
from deslib.des import DESFHMW_prior
from deslib.des import FHDES_prior

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


#+##############################################################################


# Prepare the DS techniques. Changing k value to 7.


# def process_tasks(task_queue):
#     while not task_queue.empty():
#         datasetName = task_queue.get()
#         result,methods_names,list_ds = run_process(datasetName)
#     return result,methods_names,list_ds

def convert_datasets(datasetName):
    redata = sio.loadmat("DataSets/" + datasetName + ".mat")
    data = redata['dataset']
    X = data[:, 0:-1]
    y = data[:, -1]
    print(datasetName, "is readed.")
    state = 0
    print(datasetName, ': ', X.shape)


    # ### ### ### ### ### ### ### ### ###
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    X[np.isnan(X)] = 0
    # #### **** #### **** #### **** #### **** #### **** #### ****
    # scaler = preprocessing.MinMaxScaler()
    # X = scaler.fit_transform(X)
    # #### **** #### **** #### **** #### **** #### **** #### ****


    yhat = np.zeros((no_itr, math.ceil(len(y) / 4)))
    for itr in range(0, no_itr):
        # rand = np.random.randint(1,10000,1)
        rng = np.random.RandomState(itr)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y,
                                                            random_state=rng)  # stratify=y
        X_DSEL, X_test, y_DSEL, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test,
                                                          random_state=rng)  # stratify=y_test
        yhat[itr, :] = y_test


        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_DSEL = scaler.transform(X_DSEL)

        np.save('Datasets3/'+ datasetName+ str(itr) ,[X_train,X_test,X_DSEL,y_train,y_test,y_DSEL])


no_itr = 20
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
# "Iris"
    # "Wholesale",
    #  "Transfusion", low oracle accuracy

    # 30 Dataset
    # Has problem: "Adult", "Glass",  "Ecoli",    "Seeds",         "Voice9"
    # Large: "Magic", "CTG",  "Segmentation", "WDVG1",
}


###
for datasetName in datasets:
    # try:
    convert_datasets(datasetName)




