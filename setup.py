from setuptools import find_packages, setup, Extension

import numpy as np
import os


with open("README.md", "r") as f:
    long_description = f.read()


def get_eigen_include():
    EIGEN = "thirdparty/eigen"
    DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(DIRECTORY, EIGEN)
    return [path]


extra_compile_args = [
    '--std=c++14',
    '-fPIC',
    '-Wall',
    '-Werror',
    '-pedantic',
    # '-Wshadow', # Cython creates its own shadow mess
    '-Wextra',
    '-faligned-new',
    '-O3',
    # '-march=native', # Building for specific arch makes it 30 % faster on amd but 100 % slower on intel
    '-DNDEBUG',
    '-DEIGEN_NO_DEBUG',
    '-funroll-loops',
    '-fomit-frame-pointer',
]


setup(
    name="fnnlsEigen",
    version="1.0.0",
    packages=find_packages(),
    author="Mikael Twengström",
    description="A fast nnls solver for python implemented in C++ using Eigen",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="Url to the project on github",
    ext_modules=[
        Extension(
            name='fnnlsEigen',
            sources=['fnnlsEigen/eigen_fnnls.pyx'],
            language='c++',
            include_dirs=[np.get_include()] + get_eigen_include(),
            extra_compile_args=extra_compile_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ],
    install_requires=[
        'numpy==1.20.2',
        'Cython==0.29.23',
    ]
)
