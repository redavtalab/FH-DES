"""
The :mod:`deslib.des` provides a set of key dynamic ensemble selection
algorithms (DES).
"""

from .base import BaseDES

from .des_clustering import DESClustering
from .des_knn import DESKNN
from .des_mi import DESMI
from .des_p import DESP
from .knop import KNOP
from .knora_e import KNORAE
from .knora_u import KNORAU

from .des_FHMW_JFB import DESFHMW_JFB
from .fh_des_JFB import FHDES_JFB
from .des_FHMW_AllBoxes import DESFHMW_allboxes
from .fh_des_AllBoxes import FHDES_Allboxes
from .fh_des_AllBoxes_GPU import FHDES_Allboxes_GPU
from .des_FHMW_prior import DESFHMW_prior
from .fh_des_prior import FHDES_prior

from .meta_des import METADES
from deslib.des.probabilistic.base import BaseProbabilistic
from deslib.des.probabilistic.minimum_difference import MinimumDifference
from deslib.des.probabilistic.deskl import DESKL
from deslib.des.probabilistic.rrc import RRC
from deslib.des.probabilistic.exponential import Exponential
from deslib.des.probabilistic.logarithmic import Logarithmic

__all__ = ['BaseDES',

           'DESFHMW_JFB',
           'FHDES_JFB',
           'DESFHMW_allboxes',
           'FHDES_Allboxes',
           'FHDES_Allboxes_GPU',
           'DESFHMW_prior',
           'FHDES_prior',

           'METADES',
           'KNORAE',
           'KNORAU',
           'KNOP',
           'DESP',
           'DESKNN',
           'DESClustering',
           'DESMI',
           'BaseProbabilistic',
           'RRC',
           'DESKL',
           'MinimumDifference',
           'Exponential',
           'Logarithmic']


