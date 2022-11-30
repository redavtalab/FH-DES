
import pickle

import pandas as pd

import myfunctions as mf
from scipy.stats import rankdata
#import pandas as pd
#import openml
#from openml.tasks import TaskType
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import FuncFormatter
import numpy as np
import math
from itertools import chain
#methods_names = ['KNORA-U', 'OLA', 'DES-KNN','META-DES', 'FH-DES_W', 'FH-DES_M']
#datasets = {"Circle", "P2"}
#datasets = { "Thyroid","Breast","Wine","Heart"}
def read_results(tec_name,datasetName):
    path = ExperimentPath + "/Results/" + tec_name +"_"+datasetName + "_result.p"
    poolspec = open(path, mode="rb")
    accuracy = pickle.load(poolspec)
    labels = pickle.load(poolspec)
    yhat = pickle.load(poolspec)
    no_boxes = pickle.load(poolspec)

    return accuracy,labels,yhat, no_boxes

datasets = {
    "Data100",
    "Data1000",
    "Data10000",
    "Sensor100",
    "Sensor1000",
    "Sensor10000",
    # "Sensor100000",
    "Incidents100",
    "Incidents1000",
    "Incidents10000",
    "Agrawal1100",
    "Agrawal11000",
    "Agrawal110000",
}


datasets = sorted(datasets)
# mf.print_dataset_table(datasets)

no_itr = 5

ExperimentPath = "LargeScale1"
# methods_names = ['KNORA-U', 'KNORA-E', 'MCB', 'DESKNN', 'OLA', 'RANK', 'KNOP', 'META-DES','FH_2v']
methods_names = ['FH_1v', 'FH_2v', 'FH_3v', 'FH_4v', 'FH_5v', 'FH_6v', 'FH_7v', 'FH_8v', 'FH_9v', 'FH_10v']
# methods_names = ['FH_1p', 'FH_2p', 'FH_3p', 'FH_4p', 'FH_5p', 'FH_6p', 'FH_7p', 'FH_8p', 'FH_9p', 'FH_10p']
# methods_names = ['MV', 'SB', 'FMM', 'FH_2v', 'FH_4v','FH_9v']
No_methods = len(methods_names)


##### Reading Pickle File
dataset_counter = 0
overall_results = np.zeros((len(datasets), No_methods, no_itr))
boxes = np.zeros((len(datasets), No_methods, no_itr))
for dataInd, datasetName in enumerate(datasets):
    result = []
    for  tecInd, tecName in enumerate(methods_names):
        accuracy, labels, yhat,no_boxes = read_results(tecName,datasetName)
        overall_results[dataInd,tecInd,:] = accuracy
        boxes[dataInd,tecInd,:] = no_boxes


####################################    Write in latex      ###################################################
mf.write_in_latex_table(overall_results,datasets,methods_names)
mf.write_in_latex_table(boxes,datasets,methods_names)

################ plot no_box chart ##############3333333
no_samples = [100,1000,10000]
ybox = np.zeros((len(no_samples),))
for method_ind , tecName in enumerate(methods_names):
    ybox[0] = np.average([boxes[0,method_ind,0], boxes[3,method_ind,0], boxes[6,method_ind,0], boxes[9,method_ind,0]], )
    ybox[1] = np.average([boxes[1, method_ind, 0], boxes[4, method_ind, 0], boxes[7, method_ind, 0], boxes[10,method_ind,0]])
    ybox[2] = np.average([boxes[2, method_ind, 0], boxes[5, method_ind, 0], boxes[8, method_ind, 0], boxes[11,method_ind,0]])
    plt.plot(no_samples,ybox,label=tecName)
plt.legend(methods_names)
plt.xlabel("Number of Samples")
plt.ylabel("Number of Hyperboxes")
plt.xticks(no_samples)
plt.ylim((50,35000))
plt.xscale("log")

plt.show()



#############################################         Win_Tie_Loss           #####################################
compared_index = -1
# ind_list = list(chain (range(0,3), range(3,len(methods_names)-1)))
ind_list = range(len(methods_names))

win = np.zeros((len(ind_list)-1,))
tie = np.zeros((len(ind_list)-1,))
loss = np.zeros((len(ind_list)-1,))
no_exp = overall_results[:, 0 , :].shape[0]
dataset_methods_acc = np.average(overall_results,2)
## win-tie-loss FH-DES-M
kk = 0
for ind in ind_list[:-1]:
    no_win = np.sum(dataset_methods_acc[:, ind]  < (dataset_methods_acc[:, compared_index] ))
    no_loss = np.sum(dataset_methods_acc[:, ind] > (dataset_methods_acc[:, compared_index] ))
    win[kk] = no_win
    loss[kk] = no_loss
    tie[kk] = no_exp - no_win - no_loss
    kk +=1

nc = no_exp/2 + 1.645*np.sqrt(no_exp)/2
met_list = [methods_names[i] for i in ind_list[:-1]]
mf.plot_winloss(met_list ,win,tie,loss,nc, without_tie = False)


############################ Ranking - CD diagram  ############################

d_ranks = np.zeros_like(overall_results[:,0:-1,0])
# errors_mat = 100 - np.delete(dataset_methods_acc,2,1)
errors_mat = 100 - dataset_methods_acc

d_ranks = rankdata(errors_mat,axis=1)

ranks = np.average(d_ranks,axis=0) - 1
# m_list = [methods_names[i] for i in chain(range(2),range(3,No_methods))]
m_list = [methods_names[i] for i in range(No_methods)]
mf.plot_CD(m_list,ranks,len(datasets))


np.average(rankdata(np.average(overall_results, axis=2), axis=1), axis=0)
np.average(rankdata(np.average(overall_results, axis=2), axis=1), axis=0)
print("Overal Accuracy:")
scores = np.average(np.average(overall_results, axis=2), axis=0)
# sc = np.concatenate((scores[:2],scores[3:]))
# scores[0] = 81.89
mf.plot_overallresult(scores, methods_names)
print(np.round(ranks,2))
