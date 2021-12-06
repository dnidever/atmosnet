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

def make_header(labels,abu=None):
    """
    Make Kurucz model atmosphere header
    """

    
    # TEFF   3500.  GRAVITY 0.00000 LTE 
    #TITLE  [-1.5] N(He)/Ntot=0.0784 VTURB=2  L/H=1.25 NOVER                         
    # OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0
    # CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00
    #ABUNDANCE SCALE   0.03162 ABUNDANCE CHANGE 1 0.92150 2 0.07843
    # ABUNDANCE CHANGE  3 -10.94  4 -10.64  5  -9.49  6  -3.52  7  -4.12  8  -3.21
    # ABUNDANCE CHANGE  9  -7.48 10  -3.96 11  -5.71 12  -4.46 13  -5.57 14  -4.49
    # ABUNDANCE CHANGE 15  -6.59 16  -4.71 17  -6.54 18  -5.64 19  -6.92 20  -5.68
    # ABUNDANCE CHANGE 21  -8.87 22  -7.02 23  -8.04 24  -6.37 25  -6.65 26  -4.54
    # ABUNDANCE CHANGE 27  -7.12 28  -5.79 29  -7.83 30  -7.44 31  -9.16 32  -8.63
    # ABUNDANCE CHANGE 33  -9.67 34  -8.63 35  -9.41 36  -8.73 37  -9.44 38  -9.07
    # ABUNDANCE CHANGE 39  -9.80 40  -9.44 41 -10.62 42 -10.12 43 -20.00 44 -10.20
    # ABUNDANCE CHANGE 45 -10.92 46 -10.35 47 -11.10 48 -10.27 49 -10.38 50 -10.04
    # ABUNDANCE CHANGE 51 -11.04 52  -9.80 53 -10.53 54  -9.87 55 -10.91 56  -9.91
    # ABUNDANCE CHANGE 57 -10.87 58 -10.46 59 -11.33 60 -10.54 61 -20.00 62 -11.03
    # ABUNDANCE CHANGE 63 -11.53 64 -10.92 65 -11.69 66 -10.90 67 -11.78 68 -11.11
    # ABUNDANCE CHANGE 69 -12.04 70 -10.96 71 -11.98 72 -11.16 73 -12.17 74 -10.93
    # ABUNDANCE CHANGE 75 -11.76 76 -10.59 77 -10.69 78 -10.24 79 -11.03 80 -10.91
    # ABUNDANCE CHANGE 81 -11.14 82 -10.09 83 -11.33 84 -20.00 85 -20.00 86 -20.00
    # ABUNDANCE CHANGE 87 -20.00 88 -20.00 89 -20.00 90 -11.95 91 -20.00 92 -12.54
    # ABUNDANCE CHANGE 93 -20.00 94 -20.00 95 -20.00 96 -20.00 97 -20.00 98 -20.00
    # ABUNDANCE CHANGE 99 -20.00
    #READ DECK6 72 RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB, FLXCNV,VCONV,VELSND
    # 1.75437086E-02   1995.0 1.754E-02 1.300E+04 7.601E-06 1.708E-04 2.000E+05 0.000E+00 0.000E+00 1.177E+06
    # 2.26928500E-02   1995.0 2.269E-02 1.644E+04 9.674E-06 1.805E-04 2.000E+05 0.000E+00 0.000E+00 9.849E+05
    # 2.81685925E-02   1995.0 2.816E-02 1.999E+04 1.199E-05 1.919E-04 2.000E+05 0.000E+00 0.000E+00 8.548E+05
    # 3.41101002E-02   1995.0 3.410E-02 2.374E+04 1.463E-05 2.043E-04 2.000E+05 0.000E+00 0.000E+00 7.602E+05

    # Construct the header
    header = ['TEFF   ' +  labels[0] + '.  GRAVITY  ' + labels[1] + ' LTE \n',
              'TITLE ATLAS12                                                                   \n',
              ' OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 0\n',
              ' CONVECTION ON   1.25 TURBULENCE OFF  0.00  0.00  0.00  0.00\n',
              'ABUNDANCE SCALE   1.00000 ABUNDANCE CHANGE 1 ' + renormed_H_5s + ' 2 ' + solar_He_5s + '\n',
              ' ABUNDANCE CHANGE  3 ' + abu[ 2] + '  4 ' + abu[ 3] + '  5 ' + abu[ 4] + '  6 ' + abu[ 5] + '  7 ' + abu[ 6] + '  8 ' + abu[ 7] + '\n',
              ' ABUNDANCE CHANGE  9 ' + abu[ 8] + ' 10 ' + abu[ 9] + ' 11 ' + abu[10] + ' 12 ' + abu[11] + ' 13 ' + abu[12] + ' 14 ' + abu[13] + '\n',
              ' ABUNDANCE CHANGE 15 ' + abu[14] + ' 16 ' + abu[15] + ' 17 ' + abu[16] + ' 18 ' + abu[17] + ' 19 ' + abu[18] + ' 20 ' + abu[19] + '\n',
              ' ABUNDANCE CHANGE 21 ' + abu[20] + ' 22 ' + abu[21] + ' 23 ' + abu[22] + ' 24 ' + abu[23] + ' 25 ' + abu[24] + ' 26 ' + abu[25] + '\n',
              ' ABUNDANCE CHANGE 27 ' + abu[26] + ' 28 ' + abu[27] + ' 29 ' + abu[28] + ' 30 ' + abu[29] + ' 31 ' + abu[30] + ' 32 ' + abu[31] + '\n',
              ' ABUNDANCE CHANGE 33 ' + abu[32] + ' 34 ' + abu[33] + ' 35 ' + abu[34] + ' 36 ' + abu[35] + ' 37 ' + abu[36] + ' 38 ' + abu[37] + '\n',
              ' ABUNDANCE CHANGE 39 ' + abu[38] + ' 40 ' + abu[39] + ' 41 ' + abu[40] + ' 42 ' + abu[41] + ' 43 ' + abu[42] + ' 44 ' + abu[43] + '\n',
              ' ABUNDANCE CHANGE 45 ' + abu[44] + ' 46 ' + abu[45] + ' 47 ' + abu[46] + ' 48 ' + abu[47] + ' 49 ' + abu[48] + ' 50 ' + abu[49] + '\n',
              ' ABUNDANCE CHANGE 51 ' + abu[50] + ' 52 ' + abu[51] + ' 53 ' + abu[52] + ' 54 ' + abu[53] + ' 55 ' + abu[54] + ' 56 ' + abu[55] + '\n',
              ' ABUNDANCE CHANGE 57 ' + abu[56] + ' 58 ' + abu[57] + ' 59 ' + abu[58] + ' 60 ' + abu[59] + ' 61 ' + abu[60] + ' 62 ' + abu[61] + '\n',
              ' ABUNDANCE CHANGE 63 ' + abu[62] + ' 64 ' + abu[63] + ' 65 ' + abu[64] + ' 66 ' + abu[65] + ' 67 ' + abu[66] + ' 68 ' + abu[67] + '\n',
              ' ABUNDANCE CHANGE 69 ' + abu[68] + ' 70 ' + abu[69] + ' 71 ' + abu[70] + ' 72 ' + abu[71] + ' 73 ' + abu[72] + ' 74 ' + abu[73] + '\n',
              ' ABUNDANCE CHANGE 75 ' + abu[74] + ' 76 ' + abu[75] + ' 77 ' + abu[76] + ' 78 ' + abu[77] + ' 79 ' + abu[78] + ' 80 ' + abu[79] + '\n',
              ' ABUNDANCE CHANGE 81 ' + abu[80] + ' 82 ' + abu[81] + ' 83 ' + abu[82] + ' 84 ' + abu[83] + ' 85 ' + abu[84] + ' 86 ' + abu[85] + '\n',
              ' ABUNDANCE CHANGE 87 ' + abu[86] + ' 88 ' + abu[87] + ' 89 ' + abu[88] + ' 90 ' + abu[89] + ' 91 ' + abu[90] + ' 92 ' + abu[91] + '\n',
              ' ABUNDANCE CHANGE 93 ' + abu[92] + ' 94 ' + abu[93] + ' 95 ' + abu[94] + ' 96 ' + abu[95] + ' 97 ' + abu[96] + ' 98 ' + abu[97] + '\n',   
              ' ABUNDANCE CHANGE 99 ' + abu[98] + '\n',
              ' ABUNDANCE TABLE\n',
              '    1H   ' + renormed_H_5s + '0       2He  ' + solar_He_5s + '0\n',
              '    3Li' + abu3[ 2] + ' 0.000    4Be' + abu3[ 3] + ' 0.000    5B ' + abu3[ 4] + ' 0.000    6C ' + abu3[ 5] + ' 0.000    7N ' + abu3[ 6] + ' 0.000\n',
              '    8O ' + abu3[ 7] + ' 0.000    9F ' + abu3[ 8] + ' 0.000   10Ne' + abu3[ 9] + ' 0.000   11Na' + abu3[10] + ' 0.000   12Mg' + abu3[11] + ' 0.000\n',
              '   13Al' + abu3[12] + ' 0.000   14Si' + abu3[13] + ' 0.000   15P ' + abu3[14] + ' 0.000   16S ' + abu3[15] + ' 0.000   17Cl' + abu3[16] + ' 0.000\n',
              '   18Ar' + abu3[17] + ' 0.000   19K ' + abu3[18] + ' 0.000   20Ca' + abu3[19] + ' 0.000   21Sc' + abu3[20] + ' 0.000   22Ti' + abu3[21] + ' 0.000\n',
              '   23V ' + abu3[22] + ' 0.000   24Cr' + abu3[23] + ' 0.000   25Mn' + abu3[24] + ' 0.000   26Fe' + abu3[25] + ' 0.000   27Co' + abu3[26] + ' 0.000\n',
              '   28Ni' + abu3[27] + ' 0.000   29Cu' + abu3[28] + ' 0.000   30Zn' + abu3[29] + ' 0.000   31Ga' + abu3[30] + ' 0.000   32Ge' + abu3[31] + ' 0.000\n',
              '   33As' + abu3[32] + ' 0.000   34Se' + abu3[33] + ' 0.000   35Br' + abu3[34] + ' 0.000   36Kr' + abu3[35] + ' 0.000   37Rb' + abu3[36] + ' 0.000\n',
              '   38Sr' + abu3[37] + ' 0.000   39Y ' + abu3[38] + ' 0.000   40Zr' + abu3[39] + ' 0.000   41Nb' + abu3[40] + ' 0.000   42Mo' + abu3[41] + ' 0.000\n',
              '   43Tc' + abu3[42] + ' 0.000   44Ru' + abu3[43] + ' 0.000   45Rh' + abu3[44] + ' 0.000   46Pd' + abu3[45] + ' 0.000   47Ag' + abu3[46] + ' 0.000\n',
              '   48Cd' + abu3[47] + ' 0.000   49In' + abu3[48] + ' 0.000   50Sn' + abu3[49] + ' 0.000   51Sb' + abu3[50] + ' 0.000   52Te' + abu3[51] + ' 0.000\n',
              '   53I ' + abu3[52] + ' 0.000   54Xe' + abu3[53] + ' 0.000   55Cs' + abu3[54] + ' 0.000   56Ba' + abu3[55] + ' 0.000   57La' + abu3[56] + ' 0.000\n',
              '   58Ce' + abu3[57] + ' 0.000   59Pr' + abu3[58] + ' 0.000   60Nd' + abu3[59] + ' 0.000   61Pm' + abu3[60] + ' 0.000   62Sm' + abu3[61] + ' 0.000\n',
              '   63Eu' + abu3[62] + ' 0.000   64Gd' + abu3[63] + ' 0.000   65Tb' + abu3[64] + ' 0.000   66Dy' + abu3[65] + ' 0.000   67Ho' + abu3[66] + ' 0.000\n',
              '   68Er' + abu3[67] + ' 0.000   69Tm' + abu3[68] + ' 0.000   70Yb' + abu3[69] + ' 0.000   71Lu' + abu3[70] + ' 0.000   72Hf' + abu3[71] + ' 0.000\n',
              '   73Ta' + abu3[72] + ' 0.000   74W ' + abu3[73] + ' 0.000   75Re' + abu3[74] + ' 0.000   76Os' + abu3[75] + ' 0.000   77Ir' + abu3[76] + ' 0.000\n',
              '   78Pt' + abu3[77] + ' 0.000   79Au' + abu3[78] + ' 0.000   80Hg' + abu3[79] + ' 0.000   81Tl' + abu3[80] + ' 0.000   82Pb' + abu3[81] + ' 0.000\n',
              '   83Bi' + abu3[82] + ' 0.000   84Po' + abu3[83] + ' 0.000   85At' + abu3[84] + ' 0.000   86Rn' + abu3[85] + ' 0.000   87Fr' + abu3[86] + ' 0.000\n',
              '   88Ra' + abu3[87] + ' 0.000   89Ac' + abu3[88] + ' 0.000   90Th' + abu3[89] + ' 0.000   91Pa' + abu3[90] + ' 0.000   92U ' + abu3[91] + ' 0.000\n',
              '   93NP' + abu3[92] + ' 0.000   94Pu' + abu3[93] + ' 0.000   95Am' + abu3[94] + ' 0.000   96Cm' + abu3[95] + ' 0.000   97Bk' + abu3[96] + ' 0.000\n',
              '   98Cf' + abu3[97] + ' 0.000   99Es' + abu3[98] + ' 0.000\n',
              'READ DECK6 72 RHOX,T,P,XNE,ABROSS,ACCRAD,VTURB, FLXCNV,VCONV,VELSND\n']
    return header

    

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

        # Exponentiate, we originally took the log of the values after adding a small offset
        atmos = np.exp(atmos)
        atmos -= 1e-16
        
        # output the header as well
        
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

    def tofile(self,mfile):
        """ Write the model to a file."""
        pass

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

    def tofile(self,mfile):
        """ Write the model to a file."""
        pass
        
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
