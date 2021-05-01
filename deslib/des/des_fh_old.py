# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.des.base import BaseDES
from deslib.util.fuzzy_hyperbox import Hyperbox


class DESFH(BaseDES):
    
    HBoxes = []
    def __init__(self, pool_classifiers=None,
                 k=7, DFP=False, 
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30, 
                 random_state=None, 
                 DSEL_perc=0.5,
                 HyperBoxes=[],
                 theta = 0.2,
                 gama = 0.991):
        self.theta = theta
        self.gama = gama
############### it should be based on Clustering #############################
        super(DESFH, self).__init__(pool_classifiers=pool_classifiers,
                                            with_IH=with_IH,
                                            safe_k=safe_k,
                                            IH_rate=IH_rate,
                                            mode='hybrid',
                                            random_state=random_state,
                                            DSEL_perc=DSEL_perc)
#       super(DESFH, self).__init__(pool_classifiers, k,
#                                     DFP=DFP,
#                                     with_IH=with_IH,
#                                     safe_k=safe_k,
#                                     IH_rate=IH_rate,
#                                     mode='weighting',
#                                     random_state=random_state,
#                                     DSEL_perc=DSEL_perc)
       # bb = HyperBoxes
       
       
    
    def fit(self, X, y):
        
        super(DESFH, self).fit(X, y)
        for classifier_index in range(self.n_classifiers_):
            MissSet_indexes = ~self.DSEL_processed_[:,classifier_index]
            self.setup_hyperboxs(MissSet_indexes, classifier_index)
              

    def estimate_competence(self, query,neighbors=None, distances=None,
                            predictions=None):
        competences_ = np.zeros([len(query),self.n_classifiers_]);
        for index , sample in enumerate(query):
            for box in DESFH.HBoxes:
                comp = box.membership(sample)
                if comp > competences_[index,box.clsr]:
                    competences_[index,box.clsr] = comp
        competences_=(1-competences_) * self.k         
        return competences_
    
    
    
    
    def setup_hyperboxs(self,Mis_ind,classifier):
#        print(np.size(Mis_ind))
        if np.size(Mis_ind)<1:
            pass 
#       MisSet = self.DSEL_data_(Mis_ind) 
        boxes = []
        mis_samples = self.DSEL_data_[Mis_ind,:]
        for X in mis_samples:
            # Creation first box
            if len(boxes) < 1:
                #Create the first Box
                b = Hyperbox(v=X, w=X, classifier=classifier,theta =self.theta)
                boxes.append(b)
                continue
            #find nearst boxes
            for box in boxes:
                np
                
            # X is in a box?
            IsInBox = False
            for box in boxes:
                if np.all(box.Min>X) and np.all(box.Max<X):
                    IsInBox = True
                    box.add_sample(X)
            if IsInBox:
                #nop
                continue
           
            ######################## Expand ############################
            # Finding nearest box
            nDist = np.inf
            nearest_box = None
            for  box in boxes:
                 dist = np.linalg.norm(X-box.Center);
                 if dist < nDist:
                     nearest_box = box
                     nDist = dist
            if nearest_box.is_expandable(X):
                nearest_box.expand(X)
                continue 
            
            ######################## Creation ############################
#            else:
            b = Hyperbox(v=X, w=X, classifier=classifier, theta =self.theta )
            boxes.append(b)
       
        DESFH.HBoxes.extend(boxes)
        
        
            
            
    def select(self, competences):
        """Selects all base classifiers that obtained a local accuracy of 100%
        in the region of competence (i.e., local oracle). In the case that no
        base classifiers obtain 100% accuracy, the size of the region of
        competence is reduced and the search for the local oracle is restarted.

        Notes
        ------
        Instead of re-applying the method several times (reducing the size of
        the region of competence), we compute the number of consecutive correct
        classification of each base classifier starting from the closest
        neighbor to the more distant in the estimate_competence function.
        The number of consecutive correct classification represents the size
        of the region of competence in which the corresponding base classifier
        is an Local Oracle. Then, we select all base classifiers with the
        maximum value for the number of consecutive correct classification.
        This speed up the selection process.

        Parameters
        ----------
        competences : array of shape = [n_samples, n_classifiers]
            Competence level estimated for each base classifier and test
            example.

        Returns
        -------
        selected_classifiers : array of shape = [n_samples, n_classifiers]
            Boolean matrix containing True if the base classifier is selected,
            False otherwise.

        """
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        # Checks which was the max value for each sample
        # (i.e., the maximum number of consecutive predictions)
        max_value = np.max(competences, axis=1)

        # Select all base classifiers with the maximum number of
        #  consecutive correct predictions for each sample.
        selected_classifiers = (
                    competences >= self.gama * max_value.reshape(competences.shape[0], -1))

        return selected_classifiers