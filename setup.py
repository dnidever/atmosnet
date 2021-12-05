#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='atmosnet',
      version='1.0.0',
      description='ANN trained on model atmospheres',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/atmosnet',
      packages=find_packages(exclude=["tests"]),
      #scripts=['bin/doppler'],
      install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)'],
      include_package_data=True,
)
