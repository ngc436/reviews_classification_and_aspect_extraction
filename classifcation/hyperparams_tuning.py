from hyperopt import Trials, STATUS_OK, tpe

from hyperas import optim
from hyperas.distributions import choice, uniform

def tune_cnn():
