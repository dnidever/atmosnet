#!/usr/bin/env python

"""MODELS.PY - Model for model atmosphere ANN

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20211205'  # yyyymmdd

# Some of the software is from Yuan-Sen Ting's The_Payne repository
# https://github.com/tingyuansen/The_Payne

import os
import numpy as np
import warnings
from glob import glob
from scipy.interpolate import interp1d
from dlnpyutils import (utils as dln, bindata, astro)
import copy
import logging
import contextlib, io, sys
import time
import dill as pickle
from . import utils
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
    
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

# Get print function to be used locally, allows for easy logging
#print = utils.getprintfunc() 

    
def leaky_relu(z):
    '''
    This is the activation function used by default in all our neural networks.
    '''

    return z*(z > 0) + 0.01*z*(z < 0)
    


# Load the default Atmosnet model
def load_model():
    """
    Load the default Atmosnet model.
    """

    datadir = utils.datadir()
    files = glob(datadir+'atmosnet_*.pkl')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No Atmosnet model files in "+datadir)
    if nfiles>1:
        return AtmosModelSet.read(files)
    else:
        return AtmosModel.read(files)


# Load a single or list of atmosnet models
def load_atmosnet_model(mfile):
    """
    Load an atmosnet model from file.

    Returns
    -------
    mfiles : string
       File name (or list of filenames) of atmosnet models to load.

    Examples
    --------
    model = load_atmosnet_model()

    """

    if os.path.exists(mfile) == False:
        raise ValueError(mfile+' not found')

    
    # read in the weights and biases parameterizing a particular neural network.
    
    #tmp = np.load(mfile)
    #w_array_0 = tmp["w_array_0"]
    #w_array_1 = tmp["w_array_1"]
    #w_array_2 = tmp["w_array_2"]
    #b_array_0 = tmp["b_array_0"]
    #b_array_1 = tmp["b_array_1"]
    #b_array_2 = tmp["b_array_2"]
    #x_min = tmp["x_min"]
    #x_max = tmp["x_max"]
    #if 'labels' in tmp.files:
    #    labels = list(tmp["labels"])
    #else:
    #    print('WARNING: No label array')
    #    labels = [None] * w_array_0.shape[1]
    #coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    #tmp.close()
    #return coeffs, labels

    with open(mfile, 'rb') as f: 
        data = pickle.load(f)
    return data
        
def load_models(mtype='c3k'):
    """
    Load all Atmosnet models from the atmosnet data/ directory
    and return as a AtmosModel.

    Parameters
    ----------
    mtype : str
        Model type.  Currently only "c3k" is supported.

    Returns
    -------
    models : AtmosModel
        AtmosModel for all Atmosnet models in the
        atmosnet /data directory.

    Examples
    --------
    models = load_models()

    """    
    datadir = utils.datadir()
    files = glob(datadir+'atmosnet_'+mtype+'*.pkl')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No "+mtype+" atmosnet model files in "+datadir)
    models = []
    for f in range(nfiles):
        am = AtmosModel.read(files[f])
        models.append(am)
    return AtmosModelSet(models)

def check_params(model,params):
    """ Check input fit or fixed parameters against Atmosnet model labels."""
    # Check the input labels against the Paybe model labels

    if isinstance(params,dict):
        paramdict = params.copy()
        params = list(paramdict.keys())
        isdict = True
    else:
        isdict = False

    # Check for duplicates
    uparams = np.unique(np.array(params))
    if len(uparams)!=len(params):
        raise ValueError('There are duplicates in '+','.join(params))
        
    # Loop over parameters
    for i,par in enumerate(params):
        # replace VROT with VSINI        
        if par=='VROT' and 'VSINI' in model.labels:
            print('Replacing VROT -> VSINI')
            params[i] = 'VSINI'
            par = 'VSINI'
        # replace VMICRO with VTURB            
        elif par=='VMICRO' and 'VTURB' in model.labels:
            print('Replacing VMICRO -> VTURB')
            params[i] = 'VTURB'
            par = 'VTURB'
        # check against model labels
        if (par != 'ALPHA_H') and (not par in model.labels):
            raise ValueError(par+' NOT a Atmosnet label. Available labels are '+','.join(model.labels)+' and ALPHA_H')

    # Return "adjusted" params
    if isdict==True:
        paramdict = dict(zip(params,paramdict.values()))
        return paramdict
    else:    
        return params


class AtmosModel(object):
    """
    A class to represent a model atmosphere Artificial Neural Network model.

    Parameters
    ----------
    coeffs : list
        List of coefficient arrays.
    labels : list
        List of Atmosnet label names.

    """
    
    def __init__(self,data):
        """ Initialize AtmosModel object. """
        if type(data) is list:
            self.ncolumns = len(data)
            self._data = data
        else:
            self.ncolumns = 1
            self._data = list(data)
        self.labels = self._data[0]['labels']
        self.nlabels = len(self.labels)
        self.npix = self._data[0]['w_array_2'].shape[0]
        # Label ranges
        ranges = np.zeros((self.nlabels,2),float)
        training_labels = self._data[0]['training_labels']
        for i in range(self.nlabels):
            ranges[i,0] = np.min(training_labels[:,i])
            ranges[i,1] = np.max(training_labels[:,i])
        self.ranges = ranges
        
    def __call__(self,labels,column=None):
        """
        Create the model atmosphere given the input label values.

        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.
        column : int
            Only do a specific column.

        Returns
        -------
        model : numpy array
            The output model atmosphere array.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """
        
        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')

        # Check the labels against the ranges
        if self._check_labels(labels)==False:
            raise ValueError('Labels are out of range.')
        
        # Loop over the columns
        if column is not None:
            columns = [column]
        else:
            columns = np.arange(self.ncolumns)
        atmos = np.zeros((len(columns),self.npix),float)
        # Loop over the columns
        for i,col in enumerate(columns):
            data = self._data[col]
            # assuming your NN has two hidden layers.
            x_min, x_max = data['x_min'],data['x_max']
            w_array_0, w_array_1, w_array_2 = data['w_array_0'],data['w_array_1'],data['w_array_2']
            b_array_0, b_array_1, b_array_2 = data['b_array_0'],data['b_array_1'],data['b_array_2']            
            scaled_labels = (labels-x_min)/(x_max-x_min) - 0.5   # scale the labels
            inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
            outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
            model = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
            atmos[i,:] = model

        # Exponentiate, we originally took the log of the values
        atmos = np.exp(atmos)
            
        return atmos

    def _check_labels(self,labels):
        """ Check labels against ranges."""
        inside = True
        for i in range(self.nlabels):
            inside &= (labels[i]>=self.ranges[i,0]) & (labels[i]<=self.ranges[i,1])
        return inside
        
    def label_arrayize(self,labeldict):
        """
        Convert labels from a dictionary or numpy structured array to array.

        Parameters
        ----------
        labeldict : dictionary
            Dictionary of label values.  Values for all model labels need to be given.

        Returns
        -------
        arr : numpy array
            Array of label values.
        
        Example
        -------
        .. code-block:: python

             labelarr = model.label_arrayize(labeldict)

        """
        arr = np.zeros(len(self.labels),np.float64)
        for i in range(len(self.labels)):
            val = labeldict.get(self.labels[i])
            if val == None:
                raise ValueError(self.labels[i]+' NOT FOUND')
            arr[i] = val
        return arr
    

    def copy(self):
        """ Make a full copy of the AtmosModel object. """
        new_coeffs = []
        for c in self._coeffs:
            new_coeffs.append(c.copy())
        new = AtmosModel(new_coeffs,self._dispersion.copy(),self.labels.copy())
        return new

    
    @classmethod
    def read(cls,mfile):
        """ Read in a single Atmosnet Model."""
        data = load_atmosnet_model(mfile)
        return AtmosModel(data)

    def write(self,mfile):
        """ Write out a single Atmosnet Model."""
        with open(mfile, 'wb') as f:
            pickle.dump(self._data, f)
    
        
class AtmosModelSet(object):
    """
    A class to represent a set of model atmosphere Artificial Neural Network models.  This is used
    when separate models are used to cover a different "chunk" of parameter space.

    Parameters
    ----------
    models : list of AtmosModel objects
        List of AtmosModel objects.

    """
    
    def __init__(self,models):
        """ Initialize AtmosModelSet object. """
        # Make sure it's a list
        if type(models) is not list:
            models = [models]
        # Check that the input is Atmosnet models
        if not isinstance(models[0],AtmosModel):
            raise ValueError('Input must be list of AtmosModel objects')
            
        self.nmodels = len(models)
        self._data = models
        self.npix = self._data[0].npix
        self.labels = self._data[0].labels
        self.nlabels = self._data[0].nlabels
        self.ncolumns = self._data[0].ncolumns
        self.npix = self._data[0].npix
        # Label ranges
        ranges = np.zeros((self.nlabels,2),float)
        ranges[:,0] = np.inf
        ranges[:,1] = -np.inf        
        for i in range(self.nlabels):
            for j in range(self.nmodels):
                ranges[i,0] = np.minimum(ranges[i,0],self._data[j].ranges[i,0])
                ranges[i,1] = np.maximum(ranges[i,1],self._data[j].ranges[i,1])
        self.ranges = ranges

        
    def __call__(self,labels,column=None):
        """
        Create the Atmosnet model spectrum given the input label values.


        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.
        column : int
            Only do a specific column.

        Returns
        -------
        model : numpy array
            The output model atmosphere array.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """

        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')

        # Get correct AtmosModel that covers this range
        model = self.get_best_model(labels)
        if model is None:
            return None

        return model(labels,column=column)
 

    def get_best_model(self,labels):
        """ This returns the first AtmosModel instance that has the right range."""
        for m in self._data:
            ranges = m.ranges
            inside = True
            for i in range(self.nlabels):
                inside &= (labels[i]>=ranges[i,0]) & (labels[i]<=ranges[i,1])
            if inside:
                return m
        return None
    
    def __setitem__(self,index,data):
        self._data[index] = data
    
    def __getitem__(self,index):
        # Return one of the Atmosnet models in the set
        return self._data[index]

    def __len__(self):
        return self.nmodel
    
    def __iter__(self):
        self._count = 0
        return self
        
    def __next__(self):
        if self._count < self.nmodel:
            self._count += 1            
            return self._data[self._count-1]
        else:
            raise StopIteration

    def copy(self):
        """ Make a copy of the AtmosModelSet."""
        new_models = []
        for d in self._data:
            new_models.append(d.copy())
        new = AtmosModelSet(new_models)
        return new

    @classmethod
    def read(cls,mfiles):
        """ Read a set of model files."""
        n = len(mfiles)
        models = []
        for i in range(n):
            models.append(AtmosModel.read(mfiles[i]))
        # Sort by wavelength
        def minwave(m):
            return m.dispersion[0]
        models.sort(key=minwave)
        return AtmosModelSet(models)
    
    #def write(self,mfile):
    #    """ Write out a single Atmosnet Model."""
    #    with open(mfile, 'wb') as f:
    #        pickle.dump(self._data, f)
