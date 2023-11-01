from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("core", ["core.pyx"])]
setup(name='monotonic_align',
        # ext_modules=cythonize("core.pyx"),
        ext_modules=cythonize(extensions),
        include_dirs=[numpy.get_include()])
