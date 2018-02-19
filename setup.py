__author__ = 'max tkachenko'

from distutils.core import setup
from setuptools import find_packages

required = []

setup(name='tfmicro',
      version='0.1',
      description='Micro framework for tensorflow',
      author='Max Tkachenko',
      author_email='makseq@gmail.com',
      packages=find_packages(),
      install_requires=required)
