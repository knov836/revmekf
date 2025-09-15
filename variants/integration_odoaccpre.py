
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy
from mpmath import mpf
from mpmath import mp
import numdifftools as nd 

from .variant_odoacc import Filter

from function_quat import *
from predict_odo_test import predict as predict_tested
from update_test import update0 as update_tested
from proj_func import *


class PredictFilter(Filter):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.variant_update=self.variant_update_f
    def variant_update_f(self,Time, Surface,Accelerometer,Pressure, leftw, rightw,Orient):
        leftw=leftw*self.dt
        rightw=rightw*self.dt
        self.predict(leftw,rightw,Surface)


