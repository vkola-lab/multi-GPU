"""
Created on Thu Oct 14 14:47:38 2021

@author: cxue2
"""

# abstract class
from ._metric import Metric

# classification metrics
from .confusion_matrix import ConfusionMatrix
from .accuracy import Accuracy
from .precision import Precision
from .recall import Recall
from .f1 import F1
from .matthews_corr_coef import MatthewsCorrCoef
from .cross_entropy import CrossEntropy

# regression metrics
from .mean_absolute_error import MeanAbsoluteError
from .mean_squared_error import MeanSquaredError

# remove
from ._misc import _detach
del _misc
del _detach
del _metric

del confusion_matrix
del accuracy
del precision
del recall
del f1
del matthews_corr_coef
del cross_entropy

del mean_absolute_error
del mean_squared_error