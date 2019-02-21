from __future__ import print_function

import tfmicro as package
from distutils.core import setup
from setuptools import find_packages

# read side packages
with open('requirements.txt') as f:
    required = f.read().splitlines()

# check if everything is in com__description__mit
if 'dirty' in package.__version__:
    print('Current git description: %s \nWarning: please commit your changes' % package.__version__)
    exit(-101)

setup(name=package.__name__,
      version=package.__version__,
      author=package.__author__,
      author_email=package.__email__,
      description=package.__description__,
      packages=find_packages(),
      install_requires=required)
