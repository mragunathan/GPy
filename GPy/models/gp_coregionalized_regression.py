# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern
from .. import util
from paramz import ObsAr

class GPCoregionalizedRegression(GP):
    """
    Gaussian Process model for heteroscedastic multioutput regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X_list: list of input observations corresponding to each output
    :type X_list: list of numpy arrays
    :param Y_list: list of observed values related to the different noise models
    :type Y_list: list of numpy arrays
    :param kernel: a GPy kernel ** Coregionalized, defaults to RBF ** Coregionalized
    :type kernel: None | GPy.kernel defaults
    :likelihoods_list: a list of likelihoods, defaults to list of Gaussian likelihoods
    :type likelihoods_list: None | a list GPy.likelihoods
    :param name: model name
    :type name: string
    :param W_rank: number tuples of the corregionalization parameters 'W' (see coregionalize kernel documentation)
    :type W_rank: integer
    :param kernel_name: name of the kernel
    :type kernel_name: string
    """
    def __init__(self, X_list, Y_list, kernel=None, likelihoods_list=None, name='GPCR',W_rank=1,kernel_name='coreg'):

        #Input and Output
        X,Y,self.output_index = util.multioutput.build_XY(X_list,Y_list)
        Ny = len(Y_list)

        #Kernel
        if kernel is None:
            kernel = kern.RBF(X.shape[1]-1)
            
            kernel = util.multioutput.ICM(input_dim=X.shape[1]-1, num_outputs=Ny, kernel=kernel, W_rank=1,name=kernel_name)

        #Likelihood
        likelihood = util.multioutput.build_likelihood(Y_list,self.output_index,likelihoods_list)

        super(GPCoregionalizedRegression, self).__init__(X,Y,kernel,likelihood, Y_metadata={'output_index':self.output_index})

    def set_XY(self, X=None, Y=None):
        # print("set_XY: ")
        # print("X.shape: ",X.shape)
        # print("Y.shape: ",Y.shape)
        """
        Set the input / output data of the model
        This is useful if we wish to change our existing data but maintain the same model

        :param X: input observations
        :type X: np.ndarray
        :param Y: output observations
        :type Y: np.ndarray
        """
        X_list = []
        Y_list = []
        for i in np.arange(Y.shape[1]):
           X_list.append(X.copy())
           Y_list.append(np.atleast_2d(Y[:,i]).T)
        
        # print("len(X_list): ",len(X_list))
        # print("len(Y_list): ",len(Y_list))

        X,Y,self.output_index = util.multioutput.build_XY(X_list,Y_list)
        self.Y_metadata = {'output_index':self.output_index}

        # print("after build_XY: ")
        # print("X.shape: ",X.shape)
        # print("Y.shape: ",Y.shape)
        # print("self.output_index: ",self.output_index)

        self.update_model(False)
        if Y is not None:
            if self.normalizer is not None:
                self.normalizer.scale_by(Y)
                self.Y_normalized = ObsAr(self.normalizer.normalize(Y))
                self.Y = Y
            else:
                self.Y = ObsAr(Y)
                self.Y_normalized = self.Y
        if X is not None:
            if self.X in self.parameters:
                # LVM models
                if isinstance(self.X, VariationalPosterior):
                    assert isinstance(X, type(self.X)), "The given X must have the same type as the X in the model!"
                    index = self.X._parent_index_
                    self.unlink_parameter(self.X)
                    self.X = X
                    self.link_parameter(self.X, index=index)
                else:
                    index = self.X._parent_index_
                    self.unlink_parameter(self.X)
                    from ..core import Param
                    self.X = Param('latent mean', X)
                    self.link_parameter(self.X, index=index)
            else:
                self.X = ObsAr(X)
        self.update_model(True)