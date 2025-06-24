# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - decpp.py

 Description:
  - Eigen based implementation of differential evolution using the DE/best/1 strategy.
    Uses three deviations from the standard DE algorithm:
    a) temporal locality introduced in [2].
    b) reinitialization of individuals based on their age.
    c) oscillating CR/F parameters.

    The ints parameter is a boolean array indicating which parameters are discrete integer values. This
    parameter was introduced after observing non optimal results for the ESP2 benchmark problem:
    [3]
    If defined it causes a "special treatment" for discrete variables: They are rounded to the next integer value and
    there is an additional mutation to avoid getting stuck to local minima.


 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] https://www.researchgate.net/publication/309179699_Differential_evolution_for_protein_folding_optimization_based_on_a_three-dimensional_AB_off-lattice_model
  - [3] https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/DockerCFDBenchmark.py

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
from fcmaes.evaluator import mo_call_back_type, callback_so, libcmalib
from fcmaes.de import _check_bounds

from typing import Optional, Callable, Tuple, Union
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float],
             dim: Optional[int] = None,
             bounds: Optional[Bounds] = None,
             popsize: Optional[int] = 31,
             max_evaluations: Optional[int] = 100000,
             stop_fitness: Optional[float] = -np.inf,
             keep: Optional[int] = 200,
             f: Optional[float] = 0.5,
             cr: Optional[float] = 0.9,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             ints: Optional[ArrayLike] = None,
             min_mutate: Optional[float] = 0.1,
             max_mutate: Optional[float] = 0.5,
             workers: Optional[int] = 1,
             is_terminate: Optional[Callable[[ArrayLike, float], bool]] = None,
             x0: Optional[ArrayLike] = None,
             input_sigma: Optional[Union[float, ArrayLike, Callable]] = None,
             min_sigma: Optional[float] = 0,
             runid: Optional[int] = 0) -> OptimizeResult:

    """
    Performs optimization using the Differential Evolution algorithm.

    This function seeks to find the global minimum of a given objective function
    by employing the Differential Evolution strategy. It supports various
    configurable parameters, including bounds, population size, mutation factors,
    and customization of the optimization process through callbacks and other
    options.

    Args:
        fun (Callable[[ArrayLike], float]): The objective function to minimize.
            Must take an array-like input and return a scalar float value.
        dim (Optional[int]): Dimensionality of the input space. If bounds are
            provided, this should match the bounds' dimensionality.
        bounds (Optional[Bounds]): Bounds on the input parameters. Should be
            provided as a sequence of tuples (min, max) or equivalent.
        popsize (Optional[int]): Size of the population for evolution. Defaults
            to 31.
        max_evaluations (Optional[int]): Maximum number of function evaluations
            allowed. Defaults to 100000.
        stop_fitness (Optional[float]): Fitness threshold at which optimization
            stops. Defaults to -infinity.
        keep (Optional[int]): Number of best individuals to retain in the current
            population for elitism purposes. Defaults to 200.
        f (Optional[float]): Differential weight for mutation [0, 2]. Defaults to
            0.5.
        cr (Optional[float]): Crossover probability in the range [0, 1].
            Defaults to 0.9.
        rg (Optional[Generator]): Random number generator instance. Defaults
            to Generator(PCG64DXSM()).
        ints (Optional[ArrayLike]): Boolean array indicating which dimensions
            should be treated as integers during optimization.
        min_mutate (Optional[float]): Minimum mutation factor. Defaults to 0.1.
        max_mutate (Optional[float]): Maximum mutation factor. Defaults to 0.5.
        workers (Optional[int]): Number of parallel threads for evaluation.
            Defaults to 1. Use 0 for single-threaded execution.
        is_terminate (Optional[Callable[[ArrayLike, float], bool]]): Callback to
            determine whether to terminate the optimization early. Takes the best
            solution and its fitness as input.
        x0 (Optional[ArrayLike]): Initial positions of the population. If not
            provided, it is randomly initialized within bounds.
        input_sigma (Optional[Union[float, ArrayLike, Callable]]): Standard
            deviation for initializing the population. If scalar, applies to all
            dimensions. If callable, it should generate the array.
        min_sigma (Optional[float]): Minimum allowable standard deviation for the
            population. Defaults to 0.
        runid (Optional[int]): Identifier for this particular optimization run.
            Defaults to 0.

    Returns:
        OptimizeResult: Object containing optimization results such as the best
        solution (`x`), function value (`fun`), number of evaluations (`nfev`),
        number of iterations (`nit`), exit status (`status`), and success flag
        (`success`).
    """
    
    dim, lower, upper = _check_bounds(bounds, dim)
    if popsize is None:
        popsize = 31
    if not input_sigma is None: 
        if callable(input_sigma):
            input_sigma=input_sigma()
        if np.ndim(input_sigma) == 0:
            input_sigma = [input_sigma] * dim
    if workers is None:
        workers = 0 
    array_type = ct.c_double * dim   
    bool_array_type = ct.c_bool * dim 
    c_callback = mo_call_back_type(callback_so(fun, dim, is_terminate))
    seed = int(rg.uniform(0, 2**32 - 1))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeDE_C(runid, c_callback, dim, seed,
                            None if lower is None else array_type(*lower), 
                            None if upper is None else array_type(*upper), 
                            None if x0 is None else array_type(*x0), 
                            None if input_sigma is None else array_type(*input_sigma), 
                            min_sigma,
                            None if ints is None else bool_array_type(*ints), 
                            max_evaluations, keep, stop_fitness,  
                            popsize, f, cr, min_mutate, max_mutate, workers, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)  
      
class DE_C:
    """
    A class to perform Differential Evolution (DE) algorithm using a C implementation backend.

    This class provides an interface to perform Differential Evolution optimization. The
    underlying algorithms are implemented in C for performance. Users can configure the
    optimization parameters such as population size, mutation factors, bounds, and
    constraints. It also provides methods to query results, manage population and perform
    optimization steps using an ask-tell interface for iterative optimization processes.

    Attributes:
        dim (int): Dimension of the optimization problem.
        bounds (Bounds): Bounds for each dimension of the search space.
        popsize (int): Size of the population used in the DE optimization.
        keep (int): Number of individuals to keep across generations.
        f (float): Mutation factor used in the DE optimization.
        cr (float): Crossover probability used in the DE optimization.
        rg (Generator): Random number generator for initializing the DE process.
        ints (ArrayLike): Array specifying whether each dimension is integer-constrained.
        min_mutate (float): Minimum mutation factor during the DE process.
        max_mutate (float): Maximum mutation factor during the DE process.
        x0 (ArrayLike): Initial guess for the solution to the optimization problem.
        input_sigma (Union[float, ArrayLike, Callable]): Standard deviation for generating
            the initial population.
        min_sigma (float): Minimum standard deviation allowed during initialization.
    """
    def __init__(self,                
                 dim: Optional[int] = None,
                 bounds: Optional[Bounds] = None,
                 popsize: Optional[int] = 31,
                 keep: Optional[int] = 200,
                 f: Optional[float] = 0.5,
                 cr: Optional[float] = 0.9,
                 rg: Optional[Generator] = Generator(PCG64DXSM()),
                 ints: Optional[ArrayLike] = None,
                 min_mutate: Optional[float] = 0.1,
                 max_mutate: Optional[float] = 0.5,
                 x0: Optional[ArrayLike] = None,
                 input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3,
                 min_sigma: Optional[float] = 0,
        ):
        """
        Initializes an object for Differential Evolution-based optimization.

        This implementation is designed to manage optimization parameters such as
        dimensionality, bounds, population size, mutation rates, and crossover
        probabilities, as well as additional customization options including random
        number generation, variable constraints, and initial starting points.

        Args:
            dim (Optional[int]): Dimensionality of the optimization problem.
            bounds (Optional[Bounds]): Bounds of the variables for the optimization.
            popsize (Optional[int]): Population size of the Differential Evolution
                algorithm. Default is 31.
            keep (Optional[int]): Number of iterations to retain historical data. Default is 200.
            f (Optional[float]): Mutation factor, which determines the amplitude of
                the differential variation. Default is 0.5.
            cr (Optional[float]): Crossover probability, controlling the level of blending
                among individuals. Default is 0.9.
            rg (Optional[Generator]): Random number generator for initialization
                and stochastic operations. Default generator uses the PCG64DXSM algorithm.
            ints (Optional[ArrayLike]): Indicates which variables are integers.
            min_mutate (Optional[float]): Minimum mutation rate for the algorithm. Default is 0.1.
            max_mutate (Optional[float]): Maximum mutation rate for the algorithm. Default is 0.5.
            x0 (Optional[ArrayLike]): Initial guess or starting point for the optimization.
            input_sigma (Optional[Union[float, ArrayLike, Callable]]): Standard deviation
                for initialization. Could be a constant, array, or callable function. Default is 0.3.
            min_sigma (Optional[float]): Minimum limit for the standard deviation. Default is 0.
        """
        dim, lower, upper = _check_bounds(bounds, dim)     
        if popsize is None:
            popsize = 31
        if callable(input_sigma):
            input_sigma=input_sigma()
        if np.ndim(input_sigma) == 0:
            input_sigma = [input_sigma] * dim
        array_type = ct.c_double * dim   
        bool_array_type = ct.c_bool * dim 
        seed = int(rg.uniform(0, 2**32 - 1))
        try:
            self.ptr = initDE_C(0, dim, seed,
                            None if lower is None else array_type(*lower), 
                            None if upper is None else array_type(*upper), 
                            None if x0 is None else array_type(*x0), 
                            None if input_sigma is None else array_type(*input_sigma), 
                            min_sigma,
                            None if ints is None else bool_array_type(*ints),
                            keep, popsize, f, cr, min_mutate, max_mutate)
            self.popsize = popsize
            self.dim = dim            
        except Exception as ex:
            print (ex)
            pass
 
    def __del__(self):
        """
        Deletes the object and performs cleanup by invoking the corresponding native method.

        The destructor is responsible for freeing any resources or performing
        necessary cleanup operations associated with the object when it is
        deallocated.

        Raises:
            Exception: If there is an issue during the cleanup process, it may
                raise an exception.
        """
        destroyDE_C(self.ptr)
            
    def ask(self) -> np.array:
        """
        Generates a new population of candidate solutions using the DE (Differential Evolution)
        algorithm.

        The `ask` method retrieves a set of candidate solutions generated by the DE algorithm from
        a C extension. The solutions are returned as a NumPy array with dimensions corresponding to
        the population size and the dimensionality of the search space.

        Raises:
            Exception: If the underlying procedure fails, an exception is raised and handled
            internally.

        Returns:
            np.array: A NumPy array where each row represents a candidate solution. Returns
            `None` if an exception occurs during the process.
        """
        try:
            popsize = self.popsize
            n = self.dim
            res = np.empty(popsize*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            askDE_C(self.ptr, res_p)
            xs = np.empty((popsize, n))
            for p in range(popsize):
                xs[p,:] = res[p*n : (p+1)*n]
            return xs
        except Exception as ex:
            print (ex)
            return None

    def tell(self, ys: np.ndarray):
        """
        Updates the object with input data and performs a specific operation using
        the provided numerical array.

        Args:
            ys (np.ndarray): A NumPy array of numerical data used as input for
                the operation.

        Returns:
            int: Returns an integer value indicating the result of the operation
                or -1 in case of an exception.

        Raises:
            Exception: Catches and prints any exceptions that occur during the
                execution process.
        """
        try:
            array_type_ys = ct.c_double * len(ys)
            return tellDE_C(self.ptr, array_type_ys(*ys))
        except Exception as ex:
            print (ex)
            return -1        

    def population(self) -> np.array:
        """
        Generates and retrieves the population of individuals in a multi-dimensional
        optimization context.

        This function interacts with a native library to populate a numpy array
        representing the population in a given algorithm. It processes the
        results returned from the library and organizes them into a structured
        numpy array to return to the caller.

        Returns:
            np.array: A 2D numpy array where each row represents an individual in
                      the population, and each column corresponds to an individual's
                      dimension.

        Raises:
            Exception: If an error occurs during population retrieval or processing.
        """
        try:
            popsize = self.popsize
            n = self.dim
            res = np.empty(popsize*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            populationDE_C(self.ptr, res_p)
            xs = np.array(popsize, n)
            for p in range(popsize):
                xs[p] = res[p*n : (p+1)*n]
                return xs
        except Exception as ex:
            print (ex)
            return None

    def result(self) -> OptimizeResult:
        """
        Executes the optimization process and returns the result as an OptimizeResult object.

        This method carries out the computation for the given optimization problem,
        wraps the results into an OptimizeResult object, and handles exceptions by
        providing a default failure result.

        Returns:
            OptimizeResult: An object containing the results of the optimization process,
            including the solution vector, function value at the solution, number of
            function evaluations, number of iterations, stopping status, and success flag.

        Raises:
            Exception: If an error occurs during the optimization computation, it returns a
            default OptimizeResult indicating failure.
        """
        res = np.empty(self.dim+4)
        res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
        try:
            resultDE_C(self.ptr, res_p)
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
    
    optimizeDE_C = libcmalib.optimizeDE_C
    optimizeDE_C.argtypes = [ct.c_long, mo_call_back_type, ct.c_int, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_double, \
                ct.POINTER(ct.c_bool), \
                ct.c_int, ct.c_double, ct.c_double, ct.c_int, \
                ct.c_double, ct.c_double, ct.c_double, ct.c_double, 
                ct.c_int, ct.POINTER(ct.c_double)]
        
    initDE_C = libcmalib.initDE_C
    initDE_C.argtypes = [ct.c_long, ct.c_int, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_double, \
                ct.POINTER(ct.c_bool), \
                ct.c_double, ct.c_int, \
                ct.c_double, ct.c_double, ct.c_double, ct.c_double]
    
    initDE_C.restype = ct.c_void_p   
    
    destroyDE_C = libcmalib.destroyDE_C
    destroyDE_C.argtypes = [ct.c_void_p]
    
    askDE_C = libcmalib.askDE_C
    askDE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    tellDE_C = libcmalib.tellDE_C
    tellDE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    tellDE_C.restype = ct.c_int
    
    populationDE_C = libcmalib.populationDE_C
    populationDE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    resultDE_C = libcmalib.resultDE_C
    resultDE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]

