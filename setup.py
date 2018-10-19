import subprocess
from distutils.core import setup
from setuptools import find_packages
# from distutils.core import setup
# from setuptools import find_packages

# take version from git
desc = subprocess.check_output('git describe --long --tags --always --dirty --broken', shell=True).strip()
version = desc.replace('-', '+', 1).replace('-', '.')
version = version.lstrip('v')
print 'Version:', version
# check if everything is commited
if 'dirty' in desc or 'broken' in desc:
    print '! Error: please commit your changes or fix repository corruptions! Current version from git: ' + desc
    exit(-100)

__author__ = 'Max Tkachenko'
required = open('requirements.txt').read().split('\n')

setup(name='tfmicro',
      version=version,
      description='Micro framework for tensorflow',
      author='Max Tkachenko',
      author_email='makseq@gmail.com',
      packages=find_packages(),
      install_requires=required)
