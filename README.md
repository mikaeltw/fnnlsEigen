# fnnlsEigen
fnnlsEigen implements the fast non-negativity-constrained least squares algorithm (fnnls, [Link to article](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1099-128X(199709/10)11:5%3C393::AID-CEM483%3E3.0.CO;2-L)). The algorithm is implemented in C++ using Eigen ([Link](https://eigen.tuxfamily.org/index.php?title=Main_Page)) and is callable from Python using a Cython interface.

# The nnls and the fnnls algorithms
The non-negative least squares algorithm finds solves the following problem for a given matrix `Z` and vector `x`:

<img src="https://render.githubusercontent.com/render/math?math=\LARGE\displaystyle\min\limits_{d\geq0}\left|\left|x-Zd\right|\right|_2">

where `d` is the solution.

The fnnls algorithm in this repository is optimised for intermediary sparsed matrices and is considerably faster than e.g. SciPy's
nnls algorithm for large matrices.

# Installing
``` bash
$ python3 -m pip install fnnlsEigen
```

# API

**Direct usage**

The fnnls solver accepts both `np.float32` and `np.float64` precision `dtypes` via `fnnls` and `fnnlsf` respectively.

``` python
>>> import numpy as np
>>> import fnnlsEigen as fe

>>> Z = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64)
>>> x = np.array([1.0, 1.0, 0.0], dtype=np.float64)
>>> d = fe.fnnls(Z, x)
>>> d
array([0.66666667, 0.66666667])
```

Optionally, the breaking tolerance (`tolerance`) and maximum number of iterations (`max_iterations`) can be adjusted.
```
    max_iterations : int
        Defaults to 3 * array_Z.shape[1].
    tolerance : float32 / float64
        Defaults to machine epsilon of float32 / float64 times columns in the given matrix.
```

**Batched usage**

Should the provided matrix `Z` change very little or not at all between calls to the solver; Performance can be enhanced by caching
the `Z.transpose() * Z` product since this product is expensive. Two classes (float32 / float64) are provided as to enable memory mapping for the
`Z.transpose() * Z` product:

```
CachePreComputeNNLS
```

``` python
>>> import numpy as np
>>> import fnnlsEigen as fe

>>> pc = fe.CachePreComputeNNLS()

>>> Z = np.abs(np.random.rand(500, 1000))

>>> for _ in range(0, 100):
>>>     x = np.abs(np.random.rand(500))
>>>     d = pc.fnnls(Z, x)
```

# Contributing
Clone the project from github. Create a topic branch and issue a pull request for said branch. Wait for review.

**Setup**

On Linux platforms the environment can be easily set up as follows (after entering the cloned directory):

``` bash
$ ./install_setup.sh
```
This creates a virtual environment and installs the python dependencies.

Enable the environment:
``` bash
$ source env.sh
```

Compile the project:
``` bash
$ make build
```

Tests and style checks can be accessed via:
``` bash
$ make test
```

and

``` bash
$ make check
```