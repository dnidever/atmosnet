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
    files = glob(datadir+'atmosnet_*.npz')
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
    tmp = np.load(mfile)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    if 'labels' in tmp.files:
        labels = list(tmp["labels"])
    else:
        print('WARNING: No label array')
        labels = [None] * w_array_0.shape[1]
    coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    return coeffs, labels

def load_models():
    """
    Load all Atmosnet models from the atmosnet data/ directory
    and return as a AtmosModel.

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
    files = glob(datadir+'atmosnet_*.npz')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No atmosnet model files in "+datadir)
    return AtmosModel.read(files)

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


class AtmodModel(object):
    """
    A class to represent a model atmosphere Artificial Neural Network model.

    Parameters
    ----------
    coeffs : list
        List of coefficient arrays.
    labels : list
        List of Atmosnet labels.

    """
    
    def __init__(self,coeffs,labels):
        """ Initialize AtmosModel object. """
        self._coeffs = coeffs
        self.labels = list(labels)
        self.ncol = 8
        self.npix = 80
        #self.npix = len(self._dispersion)


    def __call__(self,labels):
        """
        Create the model atmosphere given the input label values.

        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.

        Returns
        -------
        model : numpy array
            The output model atmosphere.


        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """
        
        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')

        '''
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        '''

        # Loop over the columns
        atmos = np.zeros((self.ncol,self.npix),float)
        for i in range(self.ncol):
        
            # assuming your NN has two hidden layers.
            w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = self._coeffs
            scaled_labels = (labels-x_min)/(x_max-x_min) - 0.5
            inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
            outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
            model = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
            atmos[i,:] = model

        return atmos

    
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
        coeffs, wavelength, labels, wavevac = load_atmosnet_model(mfile)
        return AtmosModel(coeffs, wavelength, labels, wavevac=wavevac)

        
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
        """ Initialize AtmosModel object. """
        # Make sure it's a list
        if type(models) is not list:
            models = [models]
        # Check that the input is Atmosnet models
        if not isinstance(models[0],AtmosModel):
            raise ValueError('Input must be list of Atmosnet models')
            
        self.nmodel = len(models)
        self._data = models
        self.npix = len(self._dispersion)
        self.labels = self._data[0].labels
        
    
    def __call__(self,labels):
        """
        Create the Atmosnet model spectrum given the input label values.

        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.

        Returns
        -------
        mspec : numpy array or Spec1D object
            The output model Atmosnet spectrum.  If fluxonly=True then only the flux array is returned,
            otherwise a Spec1D object is returned.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """

        
        '''
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        '''

        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')

        # Prepare the spectrum
        if spec is not None:
            out = self.prepare(labels,spec=spec,rv=rv,vsini=vsini,vmacro=vmacro,wave=wave)
            if fluxonly is True:
                return out.flux
            return out

        # Only a subset of wavelenths requested
        if wr is not None:
            # Check that we have pixels in this range
            lo, = np.where(self._dispersion >= wr[0])
            hi, = np.where(self._dispersion <= wr[1])          
            if (len(lo)==0) | (len(hi)==0):
                raise Exception('No pixels between '+str(wr[0])+' and '+str(wr[1]))
            # Get the chunks that we need
            #gg, = np.where( (self._wrarray[0,:] >= wr[0]) & (self._wrarray[1,:] <= wr[1]) )
            gg, = np.where( ((self._wrarray[1,:] >= wr[0]) & (self._wrarray[1,:] <= wr[1])) |
                            ((self._wrarray[0,:] <= wr[1]) & (self._wrarray[0,:] >= wr[0])) )
            ngg = len(gg)
            npix = 0
            for i in range(ngg):
                npix += self._data[gg[i]].npix
            spectrum = np.zeros(npix,np.float64)
            wavelength = np.zeros(npix,np.float64)
            cnt = 0
            for i in range(ngg):
                spec1 = self._data[gg[i]](labels,fluxonly=True)
                nspec1 = len(spec1)
                spectrum[cnt:cnt+nspec1] = spec1
                wavelength[cnt:cnt+nspec1] = self._data[gg[i]].dispersion               
                cnt += nspec1
            # Now trim a final time
            ggpix, = np.where( (wavelength >= wr[0]) & (wavelength <= wr[1]) )
            wavelength = wavelength[ggpix]
            spectrum = spectrum[ggpix]

        # all pixels
        else:
            spectrum = np.zeros(self.npix,np.float64)
            cnt = 0
            for i in range(self.nmodel):
                spec1 = self._data[i](labels,fluxonly=True)
                nspec1 = len(spec1)
                spectrum[cnt:cnt+nspec1] = spec1
                cnt += nspec1
            wavelength = self._dispersion.copy()

                
        return mspec   

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
