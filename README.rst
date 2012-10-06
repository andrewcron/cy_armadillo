============
cy_armadillo
============

*This is still under heavy development.*

This is simply a set of cython definitions for the Armadillo C++
library. It also includes conversion utilities to and from numpy
arrays. For now, the best working model is to inline them using the
"include" cython statement. "Installing" the package only stores the
definitions in site_packages so that distutils can find it later.

Requirements
------------

* Armadillo C++ library
* NumPy
* Cython

Installation and testing
------------------------

Install via

::

    python setup.py install

There is an example program and setup.py file in the "example"
folder. Build it --inplace and run test_example.py to try it out.
You may need to change the include_dirs to point to your installation 
of armadillo. Furthermore, lapack and blas must be installed and linked
to armadillo through the library wrapper. 



