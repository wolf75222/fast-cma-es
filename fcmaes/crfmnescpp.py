# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - crfmnescpp.py

 Description:
  - Eigen based implementation of Fast Moving Natural Evolution Strategy
    for High-Dimensional Problems (CR-FM-NES), see [2].
  - Derived from [3].


 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] https://arxiv.org/abs/2201.11422
  - [3] https://github.com/nomuramasahir0/crfmnes

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
from fcmaes.evaluator import _check_bounds, _get_bounds, callback_par, parallel, call_back_par, libcmalib

from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float], 
             bounds: Optional[Bounds] = None, 
             x0: Optional[ArrayLike] = None,
             input_sigma = 0.3, 
             popsize = 32, 
             max_evaluations = 100000, 
             workers = None,
             stop_fitness = -np.inf, 
             rg = Generator(PCG64DXSM()),
             runid=0,
             normalize = False,
             use_constraint_violation = True,
             penalty_coef = 1E5
             ) -> OptimizeResult:

    """
    Minimizes a given objective function using the Covariance Matrix Adaptation Evolution
    Strategy with Constraint Handling (CR-FM-NES). The optimization adjusts variables within the
    provided bounds to achieve the minimum value of the given objective function.

    Args:
        fun: The objective function to be minimized. It should be a callable that takes an
            array-like structure as input and returns a float value as output.
        bounds: Optional bounds for the variables as a `Bounds` object. If not specified,
            the search is unbounded.
        x0: Optional initial guess for the independent variables as an array-like structure.
            If not provided, it is created randomly within the bounds.
        input_sigma: Initial step size(s) for the search. It can be a float or callable
            providing an initial sigma value. If multi-dimensional, the mean is used.
        popsize: The population size for the evolution. Must be an even number. If not
            specified, defaults to 32.
        max_evaluations: The maximum number of function evaluations allowed during optimization.
        workers: Number of parallel workers to use for evaluation. If `None` or `workers` <= 1,
            no parallelism is applied.
        stop_fitness: The fitness value at which the optimization halts if surpassed.
            Default is negative infinity.
        rg: A random generator to control stochastic behavior. Defaults to an instance of
            `Generator` with a `PCG64DXSM` bit generator.
        runid: An identifier for this particular optimization run.
        normalize: A boolean indicating whether to normalize the variables to the [0, 1] range
            during optimization.
        use_constraint_violation: Whether to apply penalty-based handling for constraint
            violations during optimization. Defaults to True.
        penalty_coef: Penalty coefficient for constraint violation handling. Default is 1E5.

    Returns:
        OptimizeResult: An object containing the optimization results, such as the optimized
        variables (`x`), the function value at the solution (`fun`), the number of function
        evaluations performed (`nfev`), the number of iterations (`nit`), the exit status code
        (`status`), and whether the optimization was successful (`success`).

    Raises:
        Exception: Raises an exception if the optimization failed, and an empty `OptimizeResult`
        is returned with default attributes indicating failure.
    """
    
    lower, upper, guess = _check_bounds(bounds, x0, rg)      
    dim = guess.size   
    if popsize is None:
        popsize = 32      
    if popsize % 2 == 1: # requires even popsize
        popsize += 1
    if callable(input_sigma):
        input_sigma=input_sigma()
    if np.ndim(input_sigma) > 0:
        input_sigma = np.mean(input_sigma)   
    array_type = ct.c_double * dim   
    parfun = None if (workers is None or workers <= 1) else parallel(fun, workers)  
    c_callback_par = call_back_par(callback_par(fun, parfun))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeCRFMNES_C(runid, c_callback_par, dim, array_type(*guess), 
                None if lower is None else array_type(*lower), 
                None if upper is None else array_type(*upper), 
                input_sigma, max_evaluations, stop_fitness,
                popsize, int(rg.uniform(0, 2**32 - 1)), penalty_coef, 
                use_constraint_violation, normalize, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        res = OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        res = OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)
    if not parfun is None:
        parfun.stop()
    return res

class CRFMNES_C:
    """
    Minimization of a scalar function of one or more variables using a C++ CR-FM-NES
    implementation, interfaced via Python using ctypes.

    This class implements the Covariance Matrix Adaptation Evolution Strategy
    (CR-FM-NES) algorithm for numerical optimization problems. It relies on the
    underlying C++ implementation for the actual optimization and provides a Python
    wrapper for convenient usage. The algorithm aims to find the minimum of a scalar
    objective function and supports various configurations like bounds on variables,
    initialization parameters, and constraint handling.

    Attributes:
        ptr (ctypes.POINTER): Pointer to the underlying C++ object handling the optimization.
        popsize (int): Population size used by the CMA-ES algorithm.
        dim (int): Dimension of the decision variable vector being optimized.
    """
    def __init__(self,
                dim: int, 
                bounds: Optional[Bounds] = None, 
                x0: Optional[ArrayLike] = None,
                input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3, 
                popsize: Optional[int] = 32,   
                rg: Optional[Generator] = Generator(PCG64DXSM()),
                runid: Optional[int] = 0,
                normalize: Optional[bool] = False,
                use_constraint_violation: Optional[bool] = True,
                penalty_coef: Optional[float] = 1E5
                ):

        """
        Initializes an evolutionary optimization algorithm with constrained bounds and penalty-based constraint
        handling. Uses randomized initial guess for optimization and allows configuration of the population size
        and mutation parameters.

        Args:
            dim (int): The dimensionality of the optimization problem.
            bounds (Optional[Bounds]): The search space boundaries. If not specified, no limits are imposed.
            x0 (Optional[ArrayLike]): Initial guess for the optimization problem. If None, a random guess is used.
            input_sigma (Optional[Union[float, ArrayLike, Callable]]): Initial distribution width for mutation.
                Default is 0.3. Callable functions are invoked and their result used.
            popsize (Optional[int]): Size of the population. Default is 32. If an odd value is provided, it is
                incremented to ensure compatibility.
            rg (Optional[Generator]): Random generator for sampling. Default is Generator(PCG64DXSM()).
            runid (Optional[int]): Unique identifier for the optimization run. Default is 0.
            normalize (Optional[bool]): Indicates whether the search space should be normalized. Default is False.
            use_constraint_violation (Optional[bool]): Enables constraint violation handling during optimization.
                Default is True.
            penalty_coef (Optional[float]): Coefficient of the penalty term for constraints. Default is 1E5.

        """

        lower, upper, guess = _get_bounds(dim, bounds, x0, rg)      
        if popsize is None:
            popsize = 32      
        if popsize % 2 == 1: # requires even popsize
            popsize += 1
        if lower is None:
            lower = [0]*dim
            upper = [0]*dim
        if callable(input_sigma):
            input_sigma=input_sigma()
        if np.ndim(input_sigma) > 0:
            input_sigma = np.mean(input_sigma)   
        array_type = ct.c_double * dim   
        try:
            self.ptr = initCRFMNES_C(runid, dim, array_type(*guess), 
                           array_type(*lower), array_type(*upper), 
                    input_sigma, popsize, int(rg.uniform(0, 2**32 - 1)), penalty_coef, 
                    use_constraint_violation, normalize)
            self.popsize = popsize
            self.dim = dim            
        except Exception as ex:
            print (ex)
            pass
    
    def __del__(self):
        """
        Destroys the current instance and releases associated resources.

        This destructor is called when the object is deleted, or goes out of scope
        to ensure proper cleanup of resources tied to the object.

        Raises:
            Any destruction-related error that may arise during the resource
            release process.
        """
        destroyCRFMNES_C(self.ptr)
            
    def ask(self) -> np.ndarray:
        """
        Generates a population of candidate solutions using the CR-FM-NES algorithm.

        This method computes a new set of candidate solutions for the optimization
        problem using internal state and algorithm properties. It uses the specified
        population size and dimensionality of the problem.

        Returns:
            np.ndarray: A 2D array where each row corresponds to a candidate solution.

        Raises:
            Exception: If there is an error while generating the candidate solutions.
        """
        try:
            lamb = self.popsize
            n = self.dim
            res = np.empty(lamb*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            askCRFMNES_C(self.ptr, res_p)
            xs = np.empty((lamb, n))
            for p in range(lamb):
                xs[p,:] = res[p*n : (p+1)*n]
            return xs
        except Exception as ex:
            print (ex)
            return None

    def tell(self, ys: np.ndarray):
        """
        Provides functionality to send an array of numerical values to a specific C-based function, with error
        handling in place to detect and notify when issues occur during execution. This method is particularly
        useful for communicating with lower-level systems or libraries that require data in specific formats.

        Args:
            ys (np.ndarray): An array of numerical values to be sent to the underlying C function.

        Returns:
            int: Returns the result from the C function upon successful execution, or -1 if an exception occurs.

        Raises:
            Exception: Captures and prints the exception message when an error is encountered during operation.
        """
        try:
            array_type_ys = ct.c_double * len(ys)
            return tellCRFMNES_C(self.ptr, array_type_ys(*ys))
        except Exception as ex:
            print (ex)
            return -1        

    def population(self) -> np.ndarray:
        """
        Generates and retrieves the current population of candidate solutions.

        This method computes the population of candidate solutions for the
        problem space using the `populationCRFMNES_C` function. The population
        data is organized as a two-dimensional NumPy array where each row
        represents a candidate solution.

        Returns:
            np.ndarray: A two-dimensional array where each row corresponds to
                a candidate solution in the population. Returns `None` in case
                of an exception.

        Raises:
            Exception: If an error occurs during computation or while calling
                `populationCRFMNES_C`.
        """
        try:
            lamb = self.popsize
            n = self.dim
            res = np.empty(lamb*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            populationCRFMNES_C(self.ptr, res_p)
            xs = np.array(lamb, n)
            for p in range(lamb):
                xs[p] = res[p*n : (p+1)*n]
                return xs
        except Exception as ex:
            print (ex)
            return None
        
    def result(self) -> OptimizeResult:
        """
        Computes the optimization result and returns it as an `OptimizeResult` object.

        The function retrieves the optimization output, including the optimized variables, function value,
        number of evaluations, number of iterations, and the status of the optimization. If an error
        occurs during the computation, an `OptimizeResult` object indicating failure is returned.

        Returns:
            OptimizeResult: An object containing details of the optimization result, including the
            optimized variables (`x`), the function value at the optimized point (`fun`), the number
            of function evaluations (`nfev`), the number of iterations performed (`nit`), the optimization
            status (`status`), and whether the optimization was successful (`success`).

        Raises:
            Exception: If an error occurs during the computation process, leading to the creation of a
            failure `OptimizeResult`.
        """
        res = np.empty(self.dim+4)
        res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
        try:
            resultCRFMNES_C(self.ptr, res_p)
            x = res[:self.dim]
            val = res[self.dim]
            evals = int(res[self.dim+1])
            iterations = int(res[self.dim+2])
            stop = int(res[self.dim+3])
            res = OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
        except Exception as ex:
            res = OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)
        return res

if not libcmalib is None: 

    optimizeCRFMNES_C = libcmalib.optimizeCRFMNES_C
    optimizeCRFMNES_C.argtypes = [ct.c_long, call_back_par, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.c_double, ct.c_int, ct.c_double, ct.c_int, 
                ct.c_long, ct.c_double, 
                ct.c_bool, ct.c_bool, ct.POINTER(ct.c_double)]
          
    initCRFMNES_C = libcmalib.initCRFMNES_C
    initCRFMNES_C.argtypes = [ct.c_long, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.c_double, ct.c_int,
                ct.c_long, ct.c_double, 
                ct.c_bool, ct.c_bool]
    
    initCRFMNES_C.restype = ct.c_void_p   
    
    destroyCRFMNES_C = libcmalib.destroyCRFMNES_C
    destroyCRFMNES_C.argtypes = [ct.c_void_p]
    
    askCRFMNES_C = libcmalib.askCRFMNES_C
    askCRFMNES_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    tellCRFMNES_C = libcmalib.tellCRFMNES_C
    tellCRFMNES_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    tellCRFMNES_C.restype = ct.c_int
    
    populationCRFMNES_C = libcmalib.populationCRFMNES_C
    populationCRFMNES_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    resultCRFMNES_C = libcmalib.resultCRFMNES_C
    resultCRFMNES_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
