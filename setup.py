from subprocess import STDOUT, CalledProcessError, check_output as run
from distutils.core import setup
from setuptools import find_packages


def get_version():
    # take version from git
    try:
        desc = run('git describe --long --tags --always --dirty', stderr=STDOUT, shell=True)
    except CalledProcessError as e:
        desc = None
        print "cmd: %s\nError %s: %s" % (e.cmd, e.returncode, e.output)
        exit(-101)

    # check if everything is commited
    if 'dirty' in desc:
        print 'Current git description: %sError: please commit your changes' % desc
        exit(-100)

    # take os name
    keys = ('ID=', 'VERSION_ID=', 'RELEASE=')
    with open('/etc/os-release') as f:
        os = ''.join(s.split("=", 1)[1].rstrip().strip('"').replace('.', '') for s in f if s.startswith(keys))

    # create package version
    v = desc.lstrip('v').rstrip().replace('-', '+', 1).replace('-', '.') + '.' + os
    print 'Version:', v
    return v

__author__ = 'Max Tkachenko'
version = get_version()
required = open('requirements.txt').read().split('\n')

setup(name='tfmicro',
      version=version,
      description='Micro framework for tensorflow',
      author='Max Tkachenko',
      author_email='makseq@gmail.com',
      packages=find_packages(),
      install_requires=required)
