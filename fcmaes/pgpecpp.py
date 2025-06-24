# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - pgpecpp.py

 Description:
  - Eigen based implementation of PGPE see [2] derived from [3].


 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] http://mediatum.ub.tum.de/doc/1099128/631352.pdf
  - [3] https://github.com/google/evojax/blob/main/evojax/algo/pgpe.py

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
             input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.1,
             popsize: Optional[int] = 32,
             max_evaluations: Optional[int] = 100000,
             workers: Optional[int] = None,
             stop_fitness: Optional[float] = -np.inf,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             runid: Optional[int] = 0,
             normalize: Optional[bool] = True,
             lr_decay_steps: Optional[int] = 1000,
             use_ranking: Optional[bool] = True,
             center_learning_rate: Optional[float] = 0.15,
             stdev_learning_rate: Optional[float] = 0.1,
             stdev_max_change: Optional[float] = 0.2,
             b1: Optional[float] = 0.9,
             b2: Optional[float] = 0.999,
             eps: Optional[float] = 1e-8,
             decay_coef: Optional[float] = 1.0,
             ) -> OptimizeResult:

    """
    Optimize a given objective function by minimizing its value using the PGPE
    (Policy Gradient with Parameter-based Exploration) method. This optimization
    approach uses parallel processing and supports both optional parameter
    normalization and adaptive learning rates for center and standard deviation.

    Args:
        fun (Callable[[ArrayLike], float]): Objective function to minimize. Must
            take a single input of type ArrayLike and return a float.
        bounds (Optional[Bounds]): Bounds for the variables. Should be specified
            as a tuple (lower_bounds, upper_bounds) or None for unbounded variables.
        x0 (Optional[ArrayLike]): Initial guess for the variables. If None,
            random initialization will be applied within bounds if specified.
        input_sigma (Optional[Union[float, ArrayLike, Callable]]): Initial
            standard deviation for parameter sampling. Defaults to 0.1. Can be
            a scalar, array, or callable returning an array.
        popsize (Optional[int]): Population size for sampling. Defaults to 32.
            If not specified or odd, it will be adjusted to the next even number.
        max_evaluations (Optional[int]): Maximum number of function evaluations
            to perform. Defaults to 100000.
        workers (Optional[int]): Number of parallel workers to use for evaluation.
            Defaults to None (no parallelism).
        stop_fitness (Optional[float]): Value of fitness to stop early. If the
            objective reaches this value, the algorithm will terminate early.
            Defaults to -infinity.
        rg (Optional[Generator]): Random number generator to use during
            optimization. Defaults to Generator(PCG64DXSM()).
        runid (Optional[int]): Unique identifier for the run. Useful for
            distinguishing runs in logging or debugging.
        normalize (Optional[bool]): Whether to normalize the input parameters for the
            optimizer. Defaults to True.
        lr_decay_steps (Optional[int]): Number of steps for learning rate decay.
            Defaults to 1000.
        use_ranking (Optional[bool]): Whether to use ranking over raw fitness
            when calculating updates. Defaults to True.
        center_learning_rate (Optional[float]): Learning rate used to update
            the center. Defaults to 0.15.
        stdev_learning_rate (Optional[float]): Learning rate used to update the
            standard deviation. Defaults to 0.1.
        stdev_max_change (Optional[float]): Maximum allowed change for standard
            deviation updates. Defaults to 0.2.
        b1 (Optional[float]): Exponential moving average factor for the first
            moment estimate during adaptive updates. Defaults to 0.9.
        b2 (Optional[float]): Exponential moving average factor for the second
            moment estimate during adaptive updates. Defaults to 0.999.
        eps (Optional[float]): Small term added to avoid division by zero in
            adaptive algorithms. Defaults to 1e-8.
        decay_coef (Optional[float]): Coefficient that controls decay in the
            learning rate updates. Defaults to 1.0.

    Returns:
        OptimizeResult: An object containing optimization results, including
            the best parameters found (`x`), the objective value (`fun`) at
            those parameters, the number of function evaluations (`nfev`), the
            number of iterations (`nit`), the stopping status as an integer code
            (`status`), and a boolean success flag (`success`).
    """
    
    lower, upper, guess = _check_bounds(bounds, x0, rg)      
    dim = guess.size   
    if popsize is None:
        popsize = 32      
    if popsize % 2 == 1: # requires even popsize
        popsize += 1
    if lower is None:
        lower = [0]*dim
        upper = [0]*dim
    if callable(input_sigma):
        input_sigma=input_sigma()
    if np.ndim(input_sigma) == 0:
        input_sigma = [input_sigma] * dim 
    parfun = None if (workers is None or workers <= 1) else parallel(fun, workers)
    array_type = ct.c_double * dim   
    c_callback_par = call_back_par(callback_par(fun, parfun))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizePGPE_C(runid, c_callback_par, dim, array_type(*guess), 
                array_type(*lower), array_type(*upper), 
                array_type(*input_sigma), max_evaluations, stop_fitness,
                popsize, int(rg.uniform(0, 2**32 - 1)), 
                lr_decay_steps, use_ranking, center_learning_rate,
                stdev_learning_rate, stdev_max_change, b1, b2, eps, decay_coef,
                normalize, res_p)
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

class PGPE_C:
    """
    Optimization class using a C++ CR-FM-NES implementation via `ctypes`.

    Minimizes a scalar objective function of one or more variables using
    a population-based evolutionary strategy. This class allows interaction
    with the underlying C++ implementation to run optimization tasks with
    high performance and flexibility.

    Attributes:
        ptr (ctypes.c_void_p): Pointer to the C++ PGPE object.
        popsize (int): Population size used for evolutionary optimization.
        dim (int): Dimensionality of the problem's search space.
    """
    def __init__(self,
        dim: int,
        bounds: Optional[Bounds] = None,
        x0: Optional[ArrayLike] = None,
        input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.1,
        popsize: Optional[int] = 32,
        rg: Optional[Generator] = Generator(PCG64DXSM()),
        runid: Optional[int] = 0,
        normalize: Optional[bool] = True,
        lr_decay_steps: Optional[int] = 1000,
        use_ranking: Optional[bool] = False, 
        center_learning_rate: Optional[float] = 0.15,
        stdev_learning_rate: Optional[float] = 0.1, 
        stdev_max_change: Optional[float] = 0.2, 
        b1: Optional[float] = 0.9,
        b2: Optional[float] = 0.999, 
        eps: Optional[float] = 1e-8, 
        decay_coef: Optional[float] = 1.0, 
        ):

        """
        Initializes an instance of the class with parameters for a PGPE (Policy Gradients
        with Parameter-based Exploration) optimization algorithm.

        This constructor sets various hyperparameters and configuration options for
        running the algorithm. These include the problem dimension, bounds, initial
        guess, population size, learning rates, and other factors that control the
        optimization process.

        Args:
            dim (int): Dimensionality of the optimization problem.
            bounds (Optional[Bounds]): Input bounds defining the feasible region.
            x0 (Optional[ArrayLike]): Initial guess for the algorithm.
            input_sigma (Optional[Union[float, ArrayLike, Callable]]): Initial standard
                deviation for exploration, either as a scalar or array or function returning
                a value.
            popsize (Optional[int]): Size of the population. Ensures even number if not
                already.
            rg (Optional[Generator]): Random number generator to control randomization in
                the algorithm. Defaults to a PCG64DXSM generator.
            runid (Optional[int]): Unique identifier for the optimization run.
            normalize (Optional[bool]): Whether to normalize the input bounds.
            lr_decay_steps (Optional[int]): Number of iterations over which the learning
                rate decays.
            use_ranking (Optional[bool]): If True, enables ranking-based updates.
            center_learning_rate (Optional[float]): Learning rate for the mean or center
                of the distribution.
            stdev_learning_rate (Optional[float]): Learning rate for the standard
                deviation of the search distribution.
            stdev_max_change (Optional[float]): Maximum allowable change for standard
                deviation in an iteration.
            b1 (Optional[float]): Exponential decay rate for first moment estimates in
                adaptive learning.
            b2 (Optional[float]): Exponential decay rate for second moment estimates in
                adaptive learning.
            eps (Optional[float]): Small constant to prevent division by zero in
                adaptive learning.
            decay_coef (Optional[float]): Coefficient controlling overall decay in
                learning adjustments.

        Raises:
            Exception: Propagates exceptions encountered during the initialization of
                the algorithm backend.
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
        if np.ndim(input_sigma) == 0:
            input_sigma = [input_sigma] * dim 
        array_type = ct.c_double * dim   
        try:
            self.ptr = initPGPE_C(runid, dim, array_type(*guess), 
                           array_type(*lower), array_type(*upper), 
                    array_type(*input_sigma), popsize, int(rg.uniform(0, 2**32 - 1)),
                    lr_decay_steps, use_ranking, center_learning_rate,
                    stdev_learning_rate, stdev_max_change, b1, b2, eps, decay_coef, 
                    normalize)
            self.popsize = popsize
            self.dim = dim            
        except Exception as ex:
            print (ex)
            pass
    
    def __del__(self):
        """
        Handles the destruction of the PGPE_C object to manage resources effectively.

        This method is invoked automatically when the instance is about to be destroyed,
        allowing for proper cleanup of associated resources.

        Raises:
            None
        """
        destroyPGPE_C(self.ptr)
            
    def ask(self) -> np.array:
        """
        Generates and returns a population of samples based on the current state of the algorithm.

        This method interacts with the C library function `askPGPE_C` to generate a new set of samples
        for the population. The returned samples are organized into a 2D NumPy array.

        Returns:
            np.array: A 2D array where each row corresponds to a sample in the population.

        Raises:
            Exception: If an error occurs during the population generation process with specific details
                       printed to the console.
        """
        
        try:
            lamb = self.popsize
            n = self.dim
            res = np.empty(lamb*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            askPGPE_C(self.ptr, res_p)
            xs = np.empty((lamb, n))
            for p in range(lamb):
                xs[p,:] = res[p*n : (p+1)*n]
            return xs
        except Exception as ex:
            print (ex)
            return None

    def tell(self, 
             ys: np.ndarray) -> int:
        """
        Executes the PGPE (Policy Gradient with Parameter-based Exploration) algorithm by interfacing
        with native code through ctypes. The method passes the given numpy array to the
        underlying PGPE implementation.

        Args:
            ys (np.ndarray): A 1D numpy array containing the parameters to be used by the PGPE algorithm.

        Returns:
            int: The result returned by the PGPE native implementation. Returns -1 in case of an error.
        """
        
        try:
            array_type_ys = ct.c_double * len(ys)
            return tellPGPE_C(self.ptr, array_type_ys(*ys))
        except Exception as ex:
            print (ex)
            return -1        

    def population(self) -> np.array:
        """
        Retrieves the population from a population PGPE algorithm, processes it, and returns it
        as a NumPy array. This method interacts with external C code for handling population
        data and converts the resulting data into a structured array.

        Returns:
            np.array: A NumPy array containing the processed population data, where each
            entry corresponds to a subset of individuals in the population as defined
            by the `popsize` and `dim` attributes.

        Raises:
            Exception: If an error occurs during the execution of the underlying
            population retrieval process or data processing.
        """
        try:
            lamb = self.popsize
            n = self.dim
            res = np.empty(lamb*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            populationPGPE_C(self.ptr, res_p)
            xs = np.array(lamb, n)
            for p in range(lamb):
                xs[p] = res[p*n : (p+1)*n]
                return xs
        except Exception as ex:
            print (ex)
            return None

    def result(self) -> OptimizeResult:
        """
        Fetches the optimization result.

        The method retrieves the solution obtained from the optimization procedure
        using a C-based backend. The result includes the optimized parameters,
        objective function value at the solution, the number of function
        evaluations, the number of iterations, and the exit status of the optimizer.

        In case of any exception during the retrieval process, a default result
        indicating failure is returned.

        Returns:
            OptimizeResult: An object containing the optimization result. It includes
            the following fields:
                - x: ndarray of the optimized parameters.
                - fun: float value of the objective function at the solution.
                - nfev: int count of function evaluations.
                - nit: int count of iterations performed.
                - status: int exit status of the optimization.
                - success: bool indicating the success (True) or failure (False) of the optimization.
        """
        res = np.empty(self.dim+4)
        res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
        try:
            resultPGPE_C(self.ptr, res_p)
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

    optimizePGPE_C = libcmalib.optimizePGPE_C
    optimizePGPE_C.argtypes = [ct.c_long, call_back_par, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.POINTER(ct.c_double), ct.c_int, ct.c_double, ct.c_int, 
                ct.c_long, ct.c_int, 
                ct.c_bool, ct.c_double, ct.c_double, ct.c_double, 
                ct.c_double, ct.c_double, ct.c_double, ct.c_double, 
                ct.c_bool, ct.POINTER(ct.c_double)]
                
    initPGPE_C = libcmalib.initPGPE_C
    initPGPE_C.argtypes = [ct.c_long, ct.c_int, ct.POINTER(ct.c_double), 
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), 
                ct.c_int, ct.c_long, ct.c_int, 
                ct.c_bool, ct.c_double, ct.c_double, ct.c_double, 
                ct.c_double, ct.c_double, ct.c_double, ct.c_double, 
                ct.c_bool]
    
    initPGPE_C.restype = ct.c_void_p   
    
    destroyPGPE_C = libcmalib.destroyPGPE_C
    destroyPGPE_C.argtypes = [ct.c_void_p]
    
    askPGPE_C = libcmalib.askPGPE_C
    askPGPE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    tellPGPE_C = libcmalib.tellPGPE_C
    tellPGPE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    tellPGPE_C.restype = ct.c_int
    
    populationPGPE_C = libcmalib.populationPGPE_C
    populationPGPE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    resultPGPE_C = libcmalib.resultPGPE_C
    resultPGPE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
