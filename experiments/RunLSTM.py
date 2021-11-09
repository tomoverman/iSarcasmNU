#load the LSTMSarcasm class
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'models/LSTMSarcasm.py')))
print(sys.path)
from models.LSTMSarcasm import LSTMSarcasm
#specify our loss function and optimizer
#we use cross entropy loss function with regularization of 10^-8

#use RMSprop with initial step size of 0.001

#train for 30 epochs

