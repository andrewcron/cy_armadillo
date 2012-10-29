'''
Created on October 5, 2012

@author: Andrew Cron
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
from cyarma import include_dir as arma_inc_dir

setup(name='example',
      version='0.1',
      packages=['example'],
      package_dir={'example': 'example'},
      description='Wrapper to Armadillo',
      package_data={'cyarma': ['*.pyx','*.pxd']},
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("example", 
                [#arma_inc_dir+"/cyarma.cpp",
                                   "example/example.pyx"],
                include_dirs = [get_include(), '/usr/include',
				'/usr/local/include',
                                arma_inc_dir,'./example', '.'],
                library_dirs = ['/usr/lib', '/usr/local/lib'],
                libraries=["armadillo"],
                language='c++')],
      )
