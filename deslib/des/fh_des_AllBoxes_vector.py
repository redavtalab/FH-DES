# coding=utf-8

# FH-DES with contraction process
#     1- Check just the nearest hyperbox
#     2- Create a new hyperbox if the nearst hyperboxox has contraction

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
        self.hboxMin = []
        self.hboxMax = []
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
        np.concatenate(hboxV, bV)
        np.concatenate(hboxW, bW)
    def expand_box(self, hboxV, hboxW, boxInd, x):
        hboxV[boxInd] = min(hboxV[boxInd],x)
        hboxW[boxInd] = min(hboxW[boxInd], x)
    def is_expandable(self,hboxV, hboxW, boxInd, x):
        candV = min(hboxV[boxInd], x)
        candW = min(hboxW[boxInd], x)
        return all((candW-candV) < self.theta)
    def is_inside(self,hboxV, hboxW, boxInd,x):
        return np.all(hboxV[boxInd] < x) and np.all(hboxW[boxInd] > x)
    def membership_boxes(self,hboxV, hboxW,Xq):
        NO_hypeboxes, n_features = hboxV.size()
        hboxC = np.add(hboxV, hboxW) / 2
        boxes_W = hboxW .reshape(NO_hypeboxes, 1, n_features)
        boxes_V = hboxV .reshape(NO_hypeboxes, 1, n_features)
        boxes_center = hboxC.reshape(NO_hypeboxes, 1, n_features)

        halfsize = ((boxes_W - boxes_V) / 2).reshape(NO_hypeboxes, 1, n_features)
        d = np.abs(boxes_center - Xq) - halfsize
        d[d < 0] = 0
        dd = np.linalg.norm(d, axis=2)
        dd = dd / np.sqrt(self.n_features_)
        m = 1 - dd  # m: membership
        m =  np.power(m,6)
        return m
    def will_exceed_samples(self, boxInd,x, con_samples):
        candidV = np.minimum(self.hboxV[boxInd], x)
        candidW = np.maximum(self.hboxW[boxInd], x)
        V = candidV.reshape(1, self.n_features_)
        W = candidW.reshape(1, self.n_features_)
        return np.any(np.all(V <= con_samples, 1) & np.all(W >= con_samples, 1))
    def contract_samplesBased(self, boxInd, con_samples):
        di = 0
        v = self.hboxV[boxInd]
        w = self.hboxW[boxInd]
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
                    self.hboxV[boxInd, d] = con_samples[ind, d]
                else:
                    d = np.where(ma == min(ma))
                    self.hboxW[boxInd, d] = con_samples[ind, d]



    def fit(self, X, y):
        super(FHDES_Allboxes_vector, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Theta must be between 0 and 1.")


        if self.multiCore_process == False:
            for classifier_index in range(self.n_classifiers_):
                [bV,bW] = self.setup_hyperboxs(classifier_index)
                np.concatenate(self.hboxMin,bV)
                np.concatenate(self.hboxMax,bW)


        else:
            # classifier_index = range(self.n_classifiers_)
            no_processes = int(multiprocessing.cpu_count() /2)+1
            with multiprocessing.Pool(processes=no_processes) as pool:
                listBox = pool.map(self.setup_hyperboxs, range(self.n_classifiers_))
                for item in listBox:
                    self.hboxMin = np.concatenate(self.hboxMin, item[0])
                    self.hboxMax = np.concatenate(self.hboxMax, item[1])
                    self.NO_hypeboxes += len(listBox)


    # def remove_bad_labels:
    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):
        boxes_classifier = np.zeros((len(self.HBoxes),1))

        ###################################### should be removed after vectorize...
        for i in range(len(self.HBoxes)):
            boxes_classifier[i] = self.HBoxes[i].clsr
            self.hboxMax.append(self.HBoxes[i].Max)
            self.hboxMin.append(self.HBoxes[i].Min)
        ###########################################################################


        if self.mis_sample_based:
            competences_ = np.ones([len(query), self.n_classifiers_])
        else:
            competences_ = np.zeros([len(query), self.n_classifiers_])


        Xq = query.reshape(1,len(query),self.n_features_)

        ## Membership Calculation
        m = self.membership_boxes(Xq)


        classifiers, indices, count = np.unique(boxes_classifier, return_counts = True,return_index = True)
        k = 0
        for clsr in classifiers:
            c_range = range( indices[k], indices[k] + count[k])
            k+=1
            clsrBoxes_m = m[c_range]
            if len(c_range) > 1:
                #bb_indexes = np.argpartition(-clsrBoxes_m, kth=2, axis=0)[:2]
                bb_indexes = np.argsort(-clsrBoxes_m, axis=0)
                b1 = bb_indexes[0,:]
                b2 = bb_indexes[1,:]
                for i in range(0,len(query)):
                    if clsrBoxes_m[b1[i],i]==1 : # if the query sample is located inside or near to the box
                        competences_[i,int(clsr)] = 1
                    else:
                        competences_[i,int(clsr)] = clsrBoxes_m[b1[i],i] *0.7 + clsrBoxes_m[b2[i],i]*0.3


            else:  # In case that we have only one hyperbox for the classifier
                for i in range(0, len(query)):
                    competences_[i, int(clsr)] = clsrBoxes_m[0, i]

        #### was mistake ####
        if self.mis_sample_based:
            competences_ = np.max(competences_) - competences_
            # competences_ = np.sqrt(self.n_features_)  - competences_

        scaler = preprocessing.MinMaxScaler()
        competences_ = scaler.fit_transform(competences_)

        return competences_

    def setup_hyperboxs(self, classifier ):
        #        print(np.size(samples_ind))
        if np.size(classifier) < 0:
            pass

        if(self.mis_sample_based):
            samples_ind = ~self.DSEL_processed_[:, classifier]
            Contraction_ind = self.DSEL_processed_[:, classifier]
        else:
            samples_ind = self.DSEL_processed_[:, classifier]
            Contraction_ind = ~self.DSEL_processed_[:, classifier]


        hboxV = []
        hboxW = []

        selected_samples = self.DSEL_data_[samples_ind, :]

        contraction_samples = self.DSEL_data_[Contraction_ind,:]
        ############################################################# Shuffle
        if self.shuffle_dataOrder:
            selected_samples = shuffle(selected_samples,random_state = classifier)




        for ind, X in enumerate(selected_samples):
            # Creation first box
            if len(hboxV) < 1:
                # Create the first Box
                self.add_boxes(bV=X, bW=X)
                # b = Hyperbox(v=X, w=X, classifier=classifier, theta=self.theta)
                # self.NO_hypeboxes += 1
                # boxes.append(b)
                continue

            # X is in a box?
            inFlage = False
            for boxInd in range(self.NO_hypeboxes):
                if self.is_inside(boxInd,X):
                    inFlage = True
                    break
            if inFlage:
                # nop
                continue

            ######################## Expand ############################
            # Sort boxes by the distances
            expanded = False
            box_list = np.linalg.norm(X-self.hboxC)

            # box_list = np.zeros((self.NO_hyperboxes,))
            # for i, box in enumerate(boxes):
            #     box_list[i] = np.linalg.norm(X - box.Center)
                # box_list[i] = box.membership(X)
            sorted_indexes = np.argsort(box_list)[::-1]
            for ind in sorted_indexes:
                # nearest_box = boxes[ind]

                if self.thetaCheck and self.doContraction:
                    if self.is_expandable(ind, X):
                        self.expand(ind, X)
                        self.contract_samplesBased(ind, contraction_samples)
                        expanded = True
                        break


                elif self.thetaCheck and not self.doContraction:
                    if self.is_expandable(ind, X):
                        self.expand(ind, X)
                        expanded = True
                        break

                elif not self.thetaCheck and self.doContraction:
                    if not self.will_exceed_samples(ind, X, contraction_samples):
                        self.expand(ind, X)
                        expanded = True
                        break

                # if ((not self.thetaCheck) or nearest_box.is_expandable(X)) and (( not self.doContraction) or (not nearest_box.will_exceed_samples(X, contraction_samples))):
                #     nearest_box.expand(X)
                #     nearest_box.contract_samplesBased(contraction_samples)
                #     expanded = True
                #     break

            # nDist = np.inf
            # nearest_box = None
            # for box in boxes:
            #     dist = np.linalg.norm(X - box.Center)
            #     if dist < nDist:
            #         nearest_box = box
            #         nDist = dist
            # if (nearest_box.is_expandable(X) or (not self.thetaCheck)):
            #     nearest_box.expand(X)
            #     if (nearest_box.will_exceed_samples(X, contraction_samples)) and (self.doContraction):
            #         nearest_box.contract_samplesBased(contraction_samples)
            #     continue

                ######################## Creation ############################
            #            else:
            if expanded == False:
                self.add_boxes(hboxV, hboxW, bV=X, bW=W)


        return hboxV, hboxW

    def select(self, competences):

        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        max_value = np.max(competences, axis=1)
        selected_classifiers = (
                competences >= self.mu * max_value.reshape(competences.shape[0], -1))

        return selected_classifiers

#