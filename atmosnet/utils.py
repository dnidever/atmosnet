#!/usr/bin/env python

"""UTILS.PY - Utility functions

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20210605'  # yyyymmdd                                                                                                                           

import os
import numpy as np
import warnings
from scipy import sparse
from scipy.interpolate import interp1d
from dlnpyutils import utils as dln
import matplotlib.pyplot as plt
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
        
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

def getprintfunc(inplogger=None):
    """ Allows you to modify print() locally with a logger."""
    
    # Input logger
    if inplogger is not None:
        return inplogger.info  
    # Check if a global logger is defined
    elif hasattr(builtins,"logger"):
        return builtins.logger.info
    # Return the buildin print function
    else:
        return builtins.print
    

# The atmosnet data directory
def datadir():
    """ Return the atmosnet data/ directory."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir

# Split a filename into directory, base and fits extensions
def splitfilename(filename):
    """ Split filename into directory, base and extensions."""
    fdir = os.path.dirname(filename)
    base = os.path.basename(filename)
    exten = ['.fit','.fits','.fit.gz','.fits.gz','.fit.fz','.fits.fz']
    for e in exten:
        if base[-len(e):]==e:
            base = base[0:-len(e)]
            ext = e
            break
    return (fdir,base,ext)

def model_abund(pars):
    """
    Model atmosphere abundances.
    """

    # Create the input 99-element abundance array
    pertab = Table.read('/home/dnidever/payne/periodic_table.txt',format='ascii')
    #inpabund = np.zeros(99,np.float64)
    #g, = np.where(np.char.array(labels.dtype.names).find('_H') != -1)
    #ind1,ind2 = dln.match(np.char.array(labels.dtype.names)[g],np.char.array(pertab['symbol']).upper()+'_H')
    #inpabund[ind2] = np.array(labels[0])[g[ind1]]
    #feh = inpabund[25]

    #read model atmosphere
    atmostype, teff, logg, vmicro2, mabu, nd, atmos = synple.read_model(modelfile)
    mlines = dln.readlines(modelfile)

    # solar abundances
    # first two are Teff and logg
    # last two are Hydrogen and Helium
    solar_abund = np.array([ 4750., 2.5, 
                            -10.99, -10.66,  -9.34,  -3.61,  -4.21,
                            -3.35,  -7.48,  -4.11,  -5.80,  -4.44,
                            -5.59,  -4.53,  -6.63,  -4.92,  -6.54,
                            -5.64,  -7.01,  -5.70,  -8.89,  -7.09,
                            -8.11,  -6.40,  -6.61,  -4.54,  -7.05,
                            -5.82,  -7.85,  -7.48,  -9.00,  -8.39,
                            -9.74,  -8.70,  -9.50,  -8.79,  -9.52,
                            -9.17,  -9.83,  -9.46, -10.58, -10.16,
                           -20.00, -10.29, -11.13, -10.47, -11.10,
                           -10.33, -11.24, -10.00, -11.03,  -9.86,
                           -10.49,  -9.80, -10.96,  -9.86, -10.94,
                           -10.46, -11.32, -10.62, -20.00, -11.08,
                           -11.52, -10.97, -11.74, -10.94, -11.56,
                           -11.12, -11.94, -11.20, -11.94, -11.19,
                           -12.16, -11.19, -11.78, -10.64, -10.66,
                           -10.42, -11.12, -10.87, -11.14, -10.29,
                           -11.39, -20.00, -20.00, -20.00, -20.00,
                           -20.00, -20.00, -12.02, -20.00, -12.58,
                           -20.00, -20.00, -20.00, -20.00, -20.00,
                           -20.00, -20.00])

    # scale global metallicity
    abu = solar_abund.copy()
    abu[2:] += feh
    # Now offset the elements with [X/Fe],   [X/Fe]=[X/H]-[Fe/H]
    g, = np.where(np.char.array(labels.dtype.names).find('_H') != -1)
    ind1,ind2 = dln.match(np.char.array(labels.dtype.names)[g],np.char.array(pertab['symbol']).upper()+'_H')
    abu[ind2] += (np.array(labels[0])[g[ind1]]).astype(float) - feh
    # convert to linear
    abu[2:] = 10**abu[2:]
    # Divide by N(H)
    g, = np.where(np.char.array(mlines).find('ABUNDANCE SCALE') != -1)
    nhtot = np.float64(mlines[g[0]].split()[6])
    abu[2:] /= nhtot
    # use model values for H and He
    abu[0:2] = mabu[0:2]

    return abu
    

def elements(husser=False):
  
    """
    Reads the solar elemental abundances
    
    From Carlos Allende Prieto's synple package.

    Parameters
    ----------
     husser: bool, optional
        when set the abundances adopted for Phoenix models by Huser et al. (2013)
        are adopted. Otherwise Asplund et al. (2005) are used -- consistent with
        the MARCS (Gustafsson et al. 2008) models and and Kurucz (Meszaros et al. 2012)
        Kurucz model atmospheres.
        
    Returns
    -------
     symbol: numpy array of str
        element symbols
     mass: numpy array of floats
        atomic masses (elements Z=1-99)
     sol: numpy array of floats
        solar abundances N/N(H)
  
    """

    symbol = [
        'H' ,'He','Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne', 
        'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar','K' ,'Ca', 
        'Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni','Cu','Zn', 
        'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y' ,'Zr', 
        'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn', 
        'Sb','Te','I' ,'Xe','Cs','Ba','La','Ce','Pr','Nd', 
        'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb', 
        'Lu','Hf','Ta','W' ,'Re','Os','Ir','Pt','Au','Hg', 
        'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th', 
        'Pa','U' ,'Np','Pu','Am','Cm','Bk','Cf','Es' ]

    mass = [ 1.00794, 4.00260, 6.941, 9.01218, 10.811, 12.0107, 14.00674, 15.9994,
             18.99840, 20.1797, 22.98977, 24.3050, 26.98154, 28.0855, 30.97376, 
             32.066, 35.4527, 39.948, 39.0983, 40.078, 44.95591, 47.867, 50.9415, 
             51.9961, 54.93805, 55.845, 58.93320, 58.6934, 63.546, 65.39, 69.723, 
             72.61, 74.92160, 78.96, 79.904, 83.80, 85.4678, 87.62, 88.90585, 
             91.224, 92.90638, 95.94, 98., 101.07, 102.90550, 106.42, 107.8682, 
             112.411, 114.818, 118.710, 121.760, 127.60, 126.90447, 131.29, 
             132.90545, 137.327, 138.9055, 140.116, 140.90765, 144.24, 145, 150.36, 
             151.964, 157.25, 158.92534, 162.50, 164.93032, 167.26, 168.93421, 
             173.04, 174.967, 178.49, 180.9479, 183.84, 186.207, 190.23, 192.217, 
             195.078, 196.96655, 200.59, 204.3833, 207.2, 208.98038, 209., 210., 
             222., 223., 226., 227., 232.0381, 231.03588, 238.0289, 237., 244., 
             243., 247., 247., 251., 252. ]

    if not husser:
        #Asplund, Grevesse and Sauval (2005), basically the same as 
        #Grevesse N., Asplund M., Sauval A.J. 2007, Space Science Review 130, 205
        sol = [  0.911, 10.93,  1.05,  1.38,  2.70,  8.39,  7.78,  8.66,  4.56,  7.84, 
                 6.17,  7.53,  6.37,  7.51,  5.36,  7.14,  5.50,  6.18,  5.08,  6.31, 
                 3.05,  4.90,  4.00,  5.64,  5.39,  7.45,  4.92,  6.23,  4.21,  4.60, 
                 2.88,  3.58,  2.29,  3.33,  2.56,  3.28,  2.60,  2.92,  2.21,  2.59, 
                 1.42,  1.92, -9.99,  1.84,  1.12,  1.69,  0.94,  1.77,  1.60,  2.00, 
                 1.00,  2.19,  1.51,  2.27,  1.07,  2.17,  1.13,  1.58,  0.71,  1.45, 
                 -9.99,  1.01,  0.52,  1.12,  0.28,  1.14,  0.51,  0.93,  0.00,  1.08, 
                 0.06,  0.88, -0.17,  1.11,  0.23,  1.45,  1.38,  1.64,  1.01,  1.13,
                 0.90,  2.00,  0.65, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99,  0.06,   
                 -9.99, -0.52, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99 ]
              
        sol[0] = 1.

    else:
        #a combination of meteoritic/photospheric abundances from Asplund et al. 2009
        #chosen for the Husser et al. (2013) Phoenix model atmospheres
        sol = [  12.00, 10.93,  3.26,  1.38,  2.79,  8.43,  7.83,  8.69,  4.56,  7.93, 
                 6.24,  7.60,  6.45,  7.51,  5.41,  7.12,  5.50,  6.40,  5.08,  6.34, 
                 3.15,  4.95,  3.93,  5.64,  5.43,  7.50,  4.99,  6.22,  4.19,  4.56, 
                 3.04,  3.65,  2.30,  3.34,  2.54,  3.25,  2.36,  2.87,  2.21,  2.58, 
                 1.46,  1.88, -9.99,  1.75,  1.06,  1.65,  1.20,  1.71,  0.76,  2.04, 
                 1.01,  2.18,  1.55,  2.24,  1.08,  2.18,  1.10,  1.58,  0.72,  1.42, 
                 -9.99,  0.96,  0.52,  1.07,  0.30,  1.10,  0.48,  0.92,  0.10,  0.92, 
                 0.10,  0.85, -0.12,  0.65,  0.26,  1.40,  1.38,  1.62,  0.80,  1.17,
                 0.77,  2.04,  0.65, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99,  0.06,   
                 -9.99, -0.54, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99 ]
      
    sol[0] = 1.
    for i in range(len(sol)-1):
        sol[i+1] = 10.**(sol[i+1]-12.0)

    return (symbol,mass,sol)
