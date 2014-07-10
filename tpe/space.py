import sys
sys.path.insert(0, "/home/domhant/Projects/caffe-fr-dev/python")
import unittest
import traceback
from cappuccino.tpesearchspace import convnet_space_to_tpe
from cappuccino.tpesearchspace import tpe_sample_to_caffenet
from cappuccino.convnetsearchspace import ConvNetSearchSpace
from cappuccino.caffeconvnet import CaffeConvNet
import hyperopt.pyll.stochastic
from cappuccino.convnetsearchspace import ImagenetSearchSpace
from cappuccino.caffeconvnet import ImagenetConvNet
from cappuccino.smacsearchspace import convnet_space_to_smac, smac_space_to_str

space = ImagenetSearchSpace(parameterize_winit=True)
space = convnet_space_to_tpe(space)
