# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - dacpp.py

 Description:
  - Eigen based implementation of dual annealing.
  - Derived from [2].
  - Local search is fixed to LBFGS-B.


 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] https://github.com/scipy/scipy/blob/master/scipy/optimize/_dual_annealing.py

 Documentation:
  -


=============================================================================
"""



import sys
import os
import ctypes as ct
import numpy as np
from numpy.random import PCG64DXSM, Generator
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.evaluator import _check_bounds, call_back_type, callback, libcmalib

from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float], 
             bounds: Optional[Bounds] = None, 
             x0: Optional[ArrayLike] = None,
             max_evaluations: Optional[int] = 100000, 
             use_local_search: Optional[bool] = True,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             runid: Optional[int] = 0) -> OptimizeResult:

    """
    Minimizes a given function using Differential Annealing (DA) algorithm with optional
    local search. This function is a Python interface to an underlying C implementation.

    Args:
        fun: The objective function to be minimized. It should accept a 1-D array-like
            object as input and return a float.
        bounds: Optional bounds for the variables as an instance of `scipy.optimize.Bounds`.
            This defines the lower and upper bounds of the search space.
        x0: Optional initial guess for the solution as a 1-D array-like object.
            If not provided, it will be generated randomly within the bounds.
        max_evaluations: Maximum number of function evaluations allowed. Default is 100000.
        use_local_search: Whether to perform local search after the main optimization
            (True) or not (False). Default is True.
        rg: Random number generator instance for reproducibility. Defaults to
            `numpy.random.Generator(PCG64DXSM())`.
        runid: Optional identifier for the optimization run. Defaults to 0.

    Returns:
        OptimizeResult: The optimization result represented as a `scipy.optimize.OptimizeResult` object.
            This object includes the found solution, function value at the solution, number of
            function evaluations (nfev), number of iterations (nit), the status of the optimization,
            and a success flag.

    Raises:
        Exception: If an unexpected error occurs during the optimization process.
    """
                
    lower, upper, guess = _check_bounds(bounds, x0, rg)   
    dim = guess.size
    array_type = ct.c_double * dim   
    c_callback = call_back_type(callback(fun))
    seed = int(rg.uniform(0, 2**32 - 1))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeDA_C(runid, c_callback, dim, seed,
                    array_type(*guess), 
                    None if lower is None else array_type(*lower), 
                    None if upper is None else array_type(*upper), 
                    max_evaluations, use_local_search, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)

if not libcmalib is None: 
          
    optimizeDA_C = libcmalib.optimizeDA_C
    optimizeDA_C.argtypes = [ct.c_long, call_back_type, ct.c_int, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.c_int, ct.c_bool, ct.POINTER(ct.c_double)]

