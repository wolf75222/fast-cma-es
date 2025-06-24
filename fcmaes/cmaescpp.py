# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - cmaescpp.py

 Description:
  - Eigen based implementation of active CMA-ES.


 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] http://cma.gforge.inria.fr/cmaes.m
  - [3] https://www.researchgate.net/publication/227050324_The_CMA_Evolution_Strategy_A_Comparing_Review

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
from fcmaes.evaluator import _check_bounds, _get_bounds, mo_call_back_type, callback_so, callback_par, call_back_par, parallel, libcmalib

from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float], 
             bounds: Optional[Bounds] = None, 
             x0: Optional[ArrayLike] = None,
             input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3,
             popsize: Optional[int] = 31,
             max_evaluations: Optional[int]  = 100000,
             accuracy: Optional[float] = 1.0, 
             stop_fitness: Optional[float] = -np.inf, 
             stop_hist: Optional[float] = -1,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             runid: Optional[int] = 0,
             workers: Optional[int] = 1, 
             normalize: Optional[bool] = True,
             delayed_update: Optional[bool] = True,
             update_gap: Optional[int] = None
             ) -> OptimizeResult:

    """
    Optimizes a given objective function using an adaptive CMA-ES algorithm under specified constraints
    and parameters. This method allows for advanced customization in algorithm behavior, enabling
    stochastic optimization of complex, possibly non-linear functions.

    Args:
        fun (Callable[[ArrayLike], float]): The objective function to be minimized. It must accept a
            single argument, the candidate solution, and return a scalar fitness value.
        bounds (Optional[Bounds]): The bounds within which the optimization search is conducted. If None,
            no boundaries are enforced.
        x0 (Optional[ArrayLike]): Initial guess for the solution. If None, an initial guess will be randomly generated.
        input_sigma (Optional[Union[float, ArrayLike, Callable]]): Initial step size for the search, defining
            strategy parameter for standard deviations. Can also be an array indicating sigma per dimension
            or a callable returning sigma.
        popsize (Optional[int]): Size of the population per generation. Affects the sampling process in each iteration.
        max_evaluations (Optional[int]): Maximum number of function evaluations to perform before termination.
        accuracy (Optional[float]): Desired accuracy of the solution or tolerance level for convergence checks.
        stop_fitness (Optional[float]): Threshold fitness value. If reached, the optimization process halts.
        stop_hist (Optional[float]): Historical improvement-based stopping condition. If None or negative, this
            criterion is disabled.
        rg (Optional[Generator]): Random number generator for stochastic sampling. Provides reproducibility
            when a specific generator seed is used.
        runid (Optional[int]): Unique identifier for the optimization run, allowing for comparisons across runs.
        workers (Optional[int]): Number of worker threads/processes for parallel evaluation of the objective
            function. If set to 0, it defaults to non-parallel execution.
        normalize (Optional[bool]): Indicates whether solutions are normalized during the optimization process.
        delayed_update (Optional[bool]): Specifies whether the algorithm updates the internal state of the
            covariance matrix immediately or after a delay.
        update_gap (Optional[int]): Gap between covariance matrix updates. If None, a default gap is used.

    Returns:
        OptimizeResult: An object containing the optimization result, including the best solution `x`, the
        value of the objective function `fun` at `x`, the number of function evaluations `nfev`, the number
        of iterations performed `nit`, the optimizer's termination status `status`, and a boolean `success`
        indicating whether the optimization was successful.
    """
    
    lower, upper, guess = _check_bounds(bounds, x0, rg)      
    dim = guess.size   
    if workers is None:
        workers = 0
    mu = int(popsize/2)
    if callable(input_sigma):
        input_sigma=input_sigma()
    if np.ndim(input_sigma) == 0:
        input_sigma = [input_sigma] * dim
    if stop_hist is None:
        stop_hist = -1;
    array_type = ct.c_double * dim 
    c_callback = mo_call_back_type(callback_so(fun, dim))
    parfun = None if delayed_update == True or workers is None or workers <= 1 else parallel(fun, workers)
    c_callback_par = call_back_par(callback_par(fun, parfun))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeACMA_C(runid, c_callback, c_callback_par, 
                dim, array_type(*guess), 
                None if lower is None else array_type(*lower), 
                None if upper is None else array_type(*upper), 
                array_type(*input_sigma), max_evaluations, stop_fitness, stop_hist, mu, 
                popsize, accuracy, int(rg.uniform(0, 2**32 - 1)), 
                normalize, delayed_update, -1 if update_gap is None else update_gap,
                workers, res_p)
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

class ACMA_C:
    """
    Implementation of the ACMA-C algorithm for optimization.

    This class provides functionalities for using the ACMA-C optimization algorithm,
    enabling the user to define the problem's dimensions, bounds, initial guesses,
    population size, and more. It integrates with ctypes to leverage C-based
    implementations for efficiency. The algorithm supports customizable stopping criteria
    such as fitness value thresholds and history-based progress termination limits.

    Attributes:
        popsize (int): CMA-ES population size.
        dim (int): Dimensionality of the optimization problem.
    """
    def __init__(self,
        dim, 
        bounds: Optional[Bounds] = None, 
        x0: Optional[ArrayLike] = None,
        input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3, 
        popsize: Optional[int] = 31,  
        max_evaluations: Optional[int] = 100000, 
        accuracy: Optional[float] = 1.0, 
        stop_fitness: Optional[float] = -np.inf, 
        stop_hist: Optional[float] = -1,
        rg: Optional[Generator] = Generator(PCG64DXSM()),
        runid: Optional[int] = 0,
        normalize: Optional[bool] = True,
        delayed_update: Optional[bool] = True,
        update_gap: Optional[int] = None
     ):

        """
        Initializes the object with parameters for the Adaptive Covariance Matrix Evolution Strategy (A-CMA).

        This initialization method configures the bounds, population size, adaptive mechanisms,
        termination criteria, and various other parameters essential for the A-CMA optimization process.
        The initialization sets up the internal configuration and prepares the underlying pointer
        to interface with the A-CMA C backend.

        Args:
            dim (int): Dimensionality of the problem.
            bounds (Optional[Bounds]): The boundary constraints for each dimension. It can be None if
                no bounds are specified.
            x0 (Optional[ArrayLike]): The initial guess or starting point for the optimizer. Can be None
                if a default starting point is desired.
            input_sigma (Optional[Union[float, ArrayLike, Callable]]): Initial step size for the optimizer.
                It can be a float, an array of floats (one per dimension), or a callable returning a float.
            popsize (Optional[int]): Population size for the optimization process. Defaults to 31.
            max_evaluations (Optional[int]): The maximum number of objective function evaluations allowed for
                the optimization process.
            accuracy (Optional[float]): Accuracy level for convergence. A smaller value indicates stricter
                convergence criteria.
            stop_fitness (Optional[float]): The fitness value at which the optimization process should stop.
                Defaults to negative infinity.
            stop_hist (Optional[float]): Number of historical steps to consider for stagnation checks. Defaults
                to -1.
            rg (Optional[Generator]): Random number generator to use for sampling. Defaults to a PCG64DXSM-based
                generator.
            runid (Optional[int]): Unique identifier for the optimization run. Useful for debugging or
                differentiating among runs.
            normalize (Optional[bool]): Whether or not to normalize the input coordinates. Defaults to True.
            delayed_update (Optional[bool]): Enable or disable delayed covariance matrix updates for computational
                efficiency. Defaults to True.
            update_gap (Optional[int]): Number of iterations between updates to the covariance matrix. If None,
                updates are performed without a fixed gap.

        Raises:
            Exception: If there is an error during the initialization of the A-CMA C backend.
        """
             
        lower, upper, guess = _get_bounds(dim, bounds, x0, rg)
        mu = int(popsize/2)
        if callable(input_sigma):
            input_sigma=input_sigma()
        if np.ndim(input_sigma) == 0:
            input_sigma = [input_sigma] * dim
        if stop_hist is None:
            stop_hist = -1;
        array_type = ct.c_double * dim 
        try:
            self.ptr = initACMA_C(runid,
                dim, array_type(*guess), 
                None if lower is None else array_type(*lower), 
                None if upper is None else array_type(*upper), 
                array_type(*input_sigma), max_evaluations, stop_fitness, stop_hist, mu, 
                popsize, accuracy, int(rg.uniform(0, 2**32 - 1)), 
                normalize, delayed_update, -1 if update_gap is None else update_gap)
            self.popsize = popsize
            self.dim = dim            
        except Exception as ex:
            print (ex)
            pass
    
    def __del__(self):
        """
        Deletes the instance and performs necessary cleanup.

        This destructor method ensures that the resources allocated or managed by the
        class are properly released. It is automatically invoked when the object is
        no longer in use or explicitly deleted.

        Raises:
            Any exceptions that might be raised during the execution of the cleanup
            process are dependent on the underlying `destroyACMA_C` function.

        """
        destroyACMA_C(self.ptr)
            
    def ask(self) -> np.array:
        """
        Generates and retrieves a population of candidate solutions.

        This method calls a low-level C function to generate new candidate solutions
        based on the current state of the optimization process. These solutions are
        returned as a numpy array. If an error occurs during the process, the method
        prints the exception and returns None.

        Returns:
            np.array: A 2D array of generated candidate solutions with shape
                (popsize, dim), where `popsize` is the population size and `dim`
                is the dimensionality of each candidate solution. Returns None
                if an exception occurs.

        Raises:
            Exception: Prints the exception message if an error occurs during the
                process.
        """
        try:
            popsize = self.popsize
            n = self.dim
            res = np.empty(popsize*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            askACMA_C(self.ptr, res_p)
            xs = np.empty((popsize, n))
            for p in range(popsize):
                xs[p,:] = res[p*n : (p+1)*n]
            return xs
        except Exception as ex:
            print (ex)
            return None

    def tell(self, 
             ys: np.ndarray, 
             xs: Optional[np.ndarray] = None) -> int:
        """
        Provides functionality to handle processed arrays and manage operations
        using the given inputs. It determines the execution path based on the
        presence of optional parameters and interacts with external components.

        Args:
            ys (np.ndarray): A numpy array with processed data that serves
                as the primary input for internal operations.
            xs (Optional[np.ndarray]): An optional numpy array that, if provided,
                modifies the internal operation and enables additional functionality.

        Returns:
            int: Represents the status or result of the operation. A successful
                execution returns a non-negative integer, whereas a failure or
                exception may result in a negative value.
        """
        if not xs is None:
            return self.tell_x_(ys, xs)
        try:
            array_type_ys = ct.c_double * len(ys)
            return tellACMA_C(self.ptr, array_type_ys(*ys))
        except Exception as ex:
            print (ex)
            return -1    

    def tell_x_(self, ys: np.ndarray, xs: np.ndarray):
        """
        Passes flattened numpy arrays and interacts with an external C function using ctypes.

        This method takes numpy arrays as input, flattens one of them, converts both arrays
        to ctypes-compatible types, and then passes them to a C function. Any exceptions
        encountered during this process are handled and logged.

        Args:
            ys (np.ndarray): A numpy array representing the first input data.
            xs (np.ndarray): A numpy array representing the second input data, which gets flattened
                before being passed to the external C function.

        Returns:
            int: The return value from the external C function `tellXACMA_C` or -1 in case of an
            exception.
        """
        try:
            flat_xs = xs.flatten()
            array_type_xs = ct.c_double * len(flat_xs)
            array_type_ys = ct.c_double * len(ys)
            return tellXACMA_C(self.ptr, array_type_ys(*ys), array_type_xs(*flat_xs))
        except Exception as ex:
            print (ex)
            return -1 
        
    def population(self) -> np.array:
        """
        Retrieve the current population of solutions.

        This method fetches the entire population of solutions from the underlying
        ACMA instance, represented as a 2D numpy array. Each row in the array corresponds
        to an individual solution in the population.

        Raises:
            Exception: If an error occurs in retrieving the population, the exception
                details are printed and None is returned.

        Returns:
            np.array: A 2D numpy array where each row is an individual solution in
                the current population.
        """
        try:
            popsize = self.popsize
            n = self.dim
            res = np.empty(popsize*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            populationACMA_C(self.ptr, res_p)
            xs = np.array(popsize, n)
            for p in range(popsize):
                xs[p] = res[p*n : (p+1)*n]
                return xs
        except Exception as ex:
            print (ex)
            return None

    def result(self) -> OptimizeResult:
        """
        Computes and returns the result of an optimization process.

        This function utilizes a low-level C library function to process optimization
        results. It retrieves the optimization results, such as the optimized variable,
        objective function value, evaluation count, iteration count, and stop status,
        and packages them into an `OptimizeResult` object.

        Returns:
            OptimizeResult: An object containing the results of the optimization
            process, including the optimized variable (`x`), objective function value
            (`fun`), evaluation count (`nfev`), iteration count (`nit`), stop
            status (`status`), and a success flag (`success`).
        """
        res = np.empty(self.dim+4)
        res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
        try:
            resultACMA_C(self.ptr, res_p)
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

    optimizeACMA_C = libcmalib.optimizeACMA_C
    optimizeACMA_C.argtypes = [ct.c_long, mo_call_back_type, call_back_par, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.POINTER(ct.c_double), ct.c_int, ct.c_double, ct.c_double, ct.c_int, ct.c_int, \
                ct.c_double, ct.c_long, ct.c_bool, ct.c_bool, ct.c_int, 
                ct.c_int, ct.POINTER(ct.c_double)]
    
    initACMA_C = libcmalib.initACMA_C
    initACMA_C.argtypes = [ct.c_long, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.POINTER(ct.c_double), ct.c_int, ct.c_double, ct.c_double, ct.c_int, 
                ct.c_int, ct.c_double, ct.c_long, ct.c_bool, ct.c_bool, ct.c_int]
                    
    initACMA_C.restype = ct.c_void_p   
    
    destroyACMA_C = libcmalib.destroyACMA_C
    destroyACMA_C.argtypes = [ct.c_void_p]
    
    askACMA_C = libcmalib.askACMA_C
    askACMA_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    tellACMA_C = libcmalib.tellACMA_C
    tellACMA_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    tellACMA_C.restype = ct.c_int
    
    tellXACMA_C = libcmalib.tellXACMA_C
    tellXACMA_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
    tellXACMA_C.restype = ct.c_int
    
    populationACMA_C = libcmalib.populationACMA_C
    populationACMA_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    resultACMA_C = libcmalib.resultACMA_C
    resultACMA_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
