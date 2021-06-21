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


extension = Extension(name='fnnlsEigen',
                      sources=['fnnlsEigen/eigen_fnnls.pyx'],
                      language='c++',
                      include_dirs=[np.get_include()] + get_eigen_include(),
                      extra_compile_args=extra_compile_args,
                      define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                      )


setup(
    name="fnnlsEigen",
    version="1.0.0",
    packages=find_packages(),
    author="Mikael Twengström",
    author_email="m.twengstrom@gmail.com",
    description="A fast nnls solver for python implemented in C++ using Eigen",
    long_description=long_description,
    license="MIT",
    platforms="Linux",
    long_description_content_type='text/markdown',
    url="https://github.com/mikaeltw/fnnlsEigen",
    ext_modules=[extension],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        'numpy==1.20.2',
        'Cython==0.29.23',
    ],
    python_requires=">=3.8.5",
)
