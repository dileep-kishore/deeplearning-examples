import os

BASEPATH = os.path.dirname(__file__).rsplit('/', 1)[0]
DATAPATH = os.path.join(BASEPATH, 'data')

def datapath():
    return DATAPATH

from .Churn import Churn
