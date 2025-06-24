# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - bitecpp.py

 Description:
  - This module implements a stochastic non-linear bound-constrained derivative-free optimization method.
  - It is a Python wrapper for the C++ implementation of the BiteOpt algorithm.
  - The BiteOpt algorithm is designed for efficient optimization in high-dimensional spaces.

 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] https://github.com/avaneev/biteopt

 Documentation:
  -


=============================================================================
"""



import sys
import os
import math
import ctypes as ct
import numpy as np
from numpy.random import PCG64DXSM, Generator
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.evaluator import _check_bounds, mo_call_back_type, callback_so, libcmalib

from typing import Optional, Callable
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float],
             bounds: Optional[Bounds] = None,
             x0: Optional[ArrayLike] = None,
             max_evaluations: Optional[int] = 100000,
             stop_fitness: Optional[float] = -np.inf,
             M: Optional[int] = 1,
             popsize: Optional[int] = 0,
             stall_criterion: Optional[int]  = 0,
             rg: Optional[Generator]  = Generator(PCG64DXSM()),
             runid: Optional[int] = 0) -> OptimizeResult:
    """
    Minimize an objective function using the optimizer.

    This function performs optimization on a given callable objective function by utilizing specified
    bounds, constraints, and other parameters. The optimizer iteratively adjusts the input variables
    to reach an optimal solution that minimizes the objective function.

    Args:
        fun: A callable objective function that accepts an array-like input and returns a float value
            representing the function value to be minimized.
        bounds: Optional bounds for the input variables, which must be consistent with the search space.
        x0: Optional initial guess for the input variables; used to initialize the search process.
        max_evaluations: Maximum number of function evaluations allowed during optimization.
        stop_fitness: Optional stopping criterion based on achieving a particular fitness value.
        M: Optional parameter for additional optimization configuration.
        popsize: Optional population size parameter for optimization algorithms requiring population-based
            computations.
        stall_criterion: Optional criterion to stop the search when no significant improvement is seen.
        rg: Optional random number generator, used for ensuring reproducibility and randomness in the
            optimization.
        runid: Optional identifier for the specific optimization run; used for tracking and reporting.

    Returns:
        OptimizeResult: A data structure containing the optimization results, including the best solution
            found, its corresponding function value, number of function evaluations, number of iterations
            performed, status, and a boolean indicating success or failure.

    Raises:
        Exception: Raised for any unexpected errors encountered during the optimization process.
    """
    
    lower, upper, guess = _check_bounds(bounds, x0, rg)      
    dim = guess.size   
    array_type = ct.c_double * dim 
    c_callback = mo_call_back_type(callback_so(fun, dim))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeBite_C(runid, c_callback, dim, int(rg.uniform(0, 2**32 - 1)), 
                           None if x0 is None else array_type(*guess), 
                           None if lower is None else array_type(*lower), 
                           None if upper is None else array_type(*upper), 
                           max_evaluations, stop_fitness, M, popsize, stall_criterion, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)

if not libcmalib is None: 
    
    optimizeBite_C = libcmalib.optimizeBite_C
    optimizeBite_C.argtypes = [ct.c_long, mo_call_back_type, ct.c_int, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.c_int, ct.c_double, ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_double)]
       


