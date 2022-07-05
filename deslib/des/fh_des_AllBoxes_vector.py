# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from deslib.des.base import BaseDES
from deslib.util.fuzzy_hyperbox import Hyperbox
import sklearn.preprocessing as preprocessing
from deslib.util.instance_hardness import *
import multiprocessing
from sklearn.utils import shuffle

class FHDES_Allboxes_vector(BaseDES):

    def __init__(self, pool_classifiers=None,
                 k=7, DFP=False,
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30,
                 random_state=None,
                 knn_classifier='knn',
                 DSEL_perc=0.5,
                 HyperBoxes=[],
                 theta=0.05,
                 mu=0.991,
                 mis_sample_based = True,
                 doContraction = True,
                 thetaCheck = True,
                 multiCore_process = False,
                 shuffle_dataOrder = False):
        self.theta = theta
        self.mu = mu
        self.mis_sample_based = mis_sample_based
        self.HBoxes = []
        self.NO_hypeboxes = 0
        self.doContraction = doContraction
        self.thetaCheck = thetaCheck
        self.multiCore_process = multiCore_process
        self.shuffle_dataOrder = shuffle_dataOrder

        ############### it should be based on Clustering #############################
        super(FHDES_Allboxes_vector, self).__init__(pool_classifiers=pool_classifiers,
                                    with_IH=with_IH,
                                    safe_k=safe_k,
                                    IH_rate=IH_rate,
                                    mode='hybrid',  # hybrid,weighting
                                    random_state=random_state,
                                    DSEL_perc=DSEL_perc)


    def add_boxes(self, hboxV, hboxW, bV, bW):
        hboxV = np.concatenate((hboxV, bV))
        hboxW = np.concatenate((hboxW, bW))
        return hboxV, hboxW

    def expand_box(self, hboxV, hboxW, boxInd, x):
        hboxV[boxInd] = np.minimum(hboxV[boxInd],x)
        hboxW[boxInd] = np.maximum(hboxW[boxInd], x)
    def is_expandable(self,hboxV, hboxW, boxInd, x):
        candV = np.minimum(hboxV[boxInd], x)
        candW = np.maximum(hboxW[boxInd], x)
        return all((candW-candV) < self.theta)
    def is_inside(self,hboxV, hboxW, boxInd,x):
        return np.all(hboxV[boxInd] < x) and np.all(hboxW[boxInd] > x)
    def membership_boxes(self, hboxV, hboxW, Xq):
        NO_hypeboxes, n_features = hboxV.shape
        hboxC = np.add(hboxV, hboxW) / 2
        boxes_W = hboxW.reshape(NO_hypeboxes, 1, n_features)
        boxes_V = hboxV.reshape(NO_hypeboxes, 1, n_features)
        boxes_center = hboxC.reshape(NO_hypeboxes, 1, n_features)
        halfsize = ((boxes_W - boxes_V) / 2).reshape(NO_hypeboxes, 1, n_features)
        d = np.abs(boxes_center - Xq) - halfsize
        d[d < 0] = 0
        dd = np.linalg.norm(d, axis=2)
        dd = dd / np.sqrt(self.n_features_)
        m = 1 - dd  # m: membership
        m =  np.power(m,6)
        return m
    def will_exceed_samples(self, hboxV, hboxW, boxInd,x, con_samples):
        candidV = np.minimum(hboxV[boxInd], x)
        candidW = np.maximum(hboxW[boxInd], x)
        V = candidV.reshape(1, self.n_features_)
        W = candidW.reshape(1, self.n_features_)
        return np.any(np.all(V <= con_samples, 1) & np.all(W >= con_samples, 1))
    def contract_samplesBased(self, hboxV, hboxW, boxInd, con_samples):
        di = 0
        v = hboxV[boxInd]
        w = hboxW[boxInd]
        mi = con_samples - v
        ma = w - con_samples

        inds1 = np.all(mi > 0, 1)
        inds2 = np.all(ma > 0, 1)
        confilicts_inds = np.where(inds1 & inds2)

        for ind in list(confilicts_inds[0]):
            if np.all(v < con_samples[ind]) and np.all(w > con_samples[ind]):
                mi = con_samples[ind] - v
                ma = w - con_samples[ind]
                if min(mi) < min(ma):
                    d = np.where(mi == min(mi))
                    hboxV[boxInd, d] = con_samples[ind, d]
                else:
                    d = np.where(ma == min(ma))
                    hboxW[boxInd, d] = con_samples[ind, d]


    def fit(self, X, y):
        super(FHDES_Allboxes_vector, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Theta must be between 0 and 1.")

        if self.multiCore_process == False:
            for classifier_index in range(self.n_classifiers_):
                [bV,bW] = self.setup_hyperboxs(classifier_index)
                class_dic =  { "clsr" : classifier_index, "Min" : bV, "Max" : bW }
                self.NO_hypeboxes += len(bV)
                self.HBoxes.append(class_dic)
        else:
            no_processes = int(multiprocessing.cpu_count() /2)+1
            with multiprocessing.Pool(processes=no_processes) as pool:
                list = pool.map(self.setup_hyperboxs, range(self.n_classifiers_))
                for clsr_box in list:
                    class_dic = {"clsr": 0, "Min": clsr_box[0], "Max": clsr_box[1]}
                    self.NO_hypeboxes += len(clsr_box[0])
                    self.HBoxes.append(class_dic)


    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):

        if self.mis_sample_based:
            highest_mems = np.ones([len(query), self.n_classifiers_])
        else:
            highest_mems = np.zeros([len(query), self.n_classifiers_])

        Xq = query.reshape(1,len(query),self.n_features_)

        for clsr in range(len(self.HBoxes)):
            # c_range = range( indices[k], indices[k] + count[k])
            hboxV = self.HBoxes[clsr]["Min"]
            hboxW = self.HBoxes[clsr]["Max"]

            clsrBoxes_m = self.membership_boxes(hboxV,hboxW,Xq)
            if len(hboxV) > 1:
                #bb_indexes = np.argpartition(-clsrBoxes_m, kth=2, axis=0)[:2]
                bb_indexes = np.argsort(-clsrBoxes_m, axis=0)
                b1 = bb_indexes[0,:]
                b2 = bb_indexes[1,:]
                for i in range(0,len(query)):
                    if clsrBoxes_m[b1[i],i]==1 : # if the query sample is located inside or near to the box
                        highest_mems[i,int(clsr)] = 1
                    else:
                        highest_mems[i,int(clsr)] = clsrBoxes_m[b1[i],i] *0.7 + clsrBoxes_m[b2[i],i]*0.3

            else:  # In case that we have only one hyperbox for the classifier
                for i in range(0, len(query)):
                    highest_mems[i, int(clsr)] = clsrBoxes_m[0, i]

        #### was mistake ####
        if self.mis_sample_based:
            competences_ = np.max(highest_mems) - highest_mems
            # competences_ = np.sqrt(self.n_features_)  - competences_

        scaler = preprocessing.MinMaxScaler()
        competences_ = scaler.fit_transform(competences_)

        return competences_

    def setup_hyperboxs(self, classifier ):
        if np.size(classifier) < 0:
            pass

        if(self.mis_sample_based):
            samples_ind = ~self.DSEL_processed_[:, classifier]
            Contraction_ind = self.DSEL_processed_[:, classifier]
        else:
            samples_ind = self.DSEL_processed_[:, classifier]
            Contraction_ind = ~self.DSEL_processed_[:, classifier]

        hboxV = np.zeros((1,self.n_features_)) - 1#np.array()
        hboxW =np.zeros((1,self.n_features_)) - 1

        selected_samples = self.DSEL_data_[samples_ind, :]

        contraction_samples = self.DSEL_data_[Contraction_ind,:]
        ############################################################# Shuffle
        if self.shuffle_dataOrder:
            selected_samples = shuffle(selected_samples,random_state = classifier)

        for ind, X in enumerate(selected_samples):
            # Creation first box
            if hboxV[0,0] == -1:
                hboxV[0, :] = X
                hboxW[0, :] = X
                continue

            # X is in a box?
            is_inBox = False
            for boxInd in range(len(hboxV)):
                if self.is_inside(hboxV, hboxW, boxInd,X):
                    is_inBox = True
                    break
            if is_inBox:
                # nop
                continue

            ######################## Expand ############################
            # Sort boxes by the distances
            hboxC = (hboxV + hboxW) / 2
            expanded = False
            box_list = np.linalg.norm(X-hboxC,axis=1)

            sorted_indexes = np.argsort(box_list)[::-1]
            for ind in sorted_indexes:
                if self.thetaCheck and self.doContraction:
                    if self.is_expandable(hboxV, hboxW, ind, X):
                        self.expand_box(hboxV, hboxW, ind, X)
                        self.contract_samplesBased(hboxV, hboxW, ind, contraction_samples)
                        expanded = True
                        break

                elif self.thetaCheck and not self.doContraction:
                    if self.is_expandable(hboxV, hboxW, ind, X):
                        self.expand_box(hboxV, hboxW, ind, X)
                        expanded = True
                        break

                elif not self.thetaCheck and self.doContraction:
                    if not self.will_exceed_samples(hboxV, hboxW, ind, X, contraction_samples):
                        self.expand_box(hboxV, hboxW, ind, X)
                        expanded = True
                        break

            ######################## Creation ############################
            #            else:
            if expanded == False:
                xt = X.reshape(1, self.n_features_)
                hboxV, hboxW = self.add_boxes(hboxV, hboxW, bV=xt, bW=xt)



        return hboxV, hboxW

    def select(self, competences):

        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        max_value = np.max(competences, axis=1)
        selected_classifiers = (
                competences >= self.mu * max_value.reshape(competences.shape[0], -1))

        return selected_classifiers

#
