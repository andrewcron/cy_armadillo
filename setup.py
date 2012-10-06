'''
Created on October 5, 2012

@author: Andrew Cron
'''

from distutils.core import setup
from distutils.extension import Extension
from numpy import get_include

setup(name='cyarma',
      version='0.1',
      packages=['cyarma'],
      package_dir={'cyarma': 'cyarma'},
      description='Wrapper to Armadillo',
      maintainer='Andrew Cron',
      maintainer_email='andrew.j.cron@gmail.com',
      author='Andrew Cron',
      author_email='andrew.j.cron@gmail.com',
      url='https://github.com/andrewcron/cyarma',
      requires=['numpy (>=1.3.0)',
                'scipy (>=0.6)',
                'matplotlib (>=1.0)',
                'cython (>=0.15.1)'],
      package_data={'cyarma': ['*.pyx']},
      )
