# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - optimizer.py

 Description:
  - Provides different optimization methods for use with parallel retry.
  - Implements a sequence of optimizers, random choice of optimizers,
  and different optimizers like CRFMNES, CMA-ES, Differential Evolution,
  Dual Annealing, Bite, and PGPE.
  - Provides a wrapper for fitness functions to use with parallel retry.
  - Implements utility functions for scaling, fitting, and generating random values.
  - Implements a base class for optimizers and derived classes for specific optimizers.
  - Provides functions to create common optimizer sequences like DE -> CMA-ES,

 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es

 Documentation:
  -

=============================================================================
"""

import numpy as np
from numpy.random import PCG64DXSM, Generator
from scipy.optimize import Bounds, minimize, shgo, differential_evolution, dual_annealing, basinhopping
import sys
import time
from loguru import logger
import ctypes as ct
import multiprocessing as mp 
from fcmaes.evaluator import serial, parallel
from fcmaes import crfmnes, crfmnescpp, pgpecpp, cmaes, de, cmaescpp, decpp, dacpp, bitecpp

from typing import Optional, Callable, Tuple, Union
from numpy.typing import ArrayLike

def eprint(*args, **kwargs):
    """
    Prints the provided arguments to the standard error stream.

    This function behaves similarly to the built-in print function but redirects
    the output to the standard error stream (sys.stderr) instead of the standard
    output stream. This function is useful for logging errors or debugging
    information without interfering with normal program output.

    Args:
        *args: Variable-length positional arguments to be printed. The arguments
            are converted to strings using str() before being printed.
        **kwargs: Variable-length keyword arguments that are passed to the
            built-in print function. Common options include `sep`, `end`, and
            `flush`.
    """
    print(*args, file=sys.stderr, **kwargs)

def scale(lower: ArrayLike, 
          upper: ArrayLike) -> np.ndarray:
    """
    Scales the difference between the upper and lower bounds by a factor of 0.5.

    This function calculates the scaled difference between two array-like inputs,
    representing the lower and upper bounds, respectively. The result is computed by
    taking the difference of the two inputs and multiplying by 0.5.

    Args:
        lower (ArrayLike): The lower bounds as an array-like structure.
        upper (ArrayLike): The upper bounds as an array-like structure.

    Returns:
        np.ndarray: A NumPy array representing the scaled difference between the
        upper and lower bounds.
    """
    return 0.5 * (np.asarray(upper) - np.asarray(lower))

def typical(lower: ArrayLike, 
            upper: ArrayLike) -> np.ndarray:
    """
    Computes the midpoint of the bounds by taking the average of the lower
    and upper bounds.

    Args:
        lower: Lower bounds specified as an array-like structure.
        upper: Upper bounds specified as an array-like structure.

    Returns:
        A NumPy array containing the computed midpoints of the corresponding
        lower and upper bounds.
    """
    return 0.5 * (np.asarray(upper) + np.asarray(lower))

def fitting(guess: ArrayLike, 
            lower: ArrayLike, 
            upper: ArrayLike) -> np.ndarray:
    """
    Clip an input array-like to ensure it remains within specified bounds.

    This function ensures that the values in the input array-like `guess` are
    restricted to a range defined by `lower` (minimum bound) and `upper` (maximum
    bound). The input arrays are converted into NumPy arrays before applying the
    clipping operation.

    Args:
        guess: Input array-like containing the values to be clipped.
        lower: Array-like specifying the lower bounds for clipping.
        upper: Array-like specifying the upper bounds for clipping.

    Returns:
        np.ndarray: A NumPy array with values clipped to lie within the boundaries
        defined by `lower` and `upper`.
    """
    return np.clip(np.asarray(guess), np.asarray(upper), np.asarray(lower))

def is_terminate(runid: int, 
                 iterations: int, 
                 val: float) -> bool:
    """
    Determines whether a process should terminate based on the given parameters.

    The function evaluates specific conditions using the provided parameters to
    determine if the process should terminate. Returns a boolean value indicating
    the termination status.

    Args:
        runid (int): A unique identifier representing the current run.
        iterations (int): The number of iterations executed so far in a process.
        val (float): A numeric value used to evaluate termination conditions.

    Returns:
        bool: True if the process meets the termination condition, False otherwise.
    """
    return False    

def random_x(lower: ArrayLike, upper: ArrayLike) -> np.ndarray:
    """
    Generates a random numpy array of values within the specified bounds.

    This function takes lower and upper bounds as input arrays and returns
    a numpy array with randomized values scaled and translated to lie
    within the specified bounds. The size and shape of the output array
    are determined by the size of the `lower` input.

    Args:
        lower: Lower bounds for the random values. Should be array-like
            and convertible to a numpy array.
        upper: Upper bounds for the random values. Should be array-like
            and convertible to a numpy array.

    Returns:
        np.ndarray: A numpy array containing random values within the
        specified lower and upper bounds.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return lower + np.multiply(upper - lower, np.random.rand(lower.size))
    
def dtime(t0: float) -> float:
    """
    Calculates the elapsed time in seconds since the provided starting time `t0`.

    This function computes the difference between the current time (as measured by
    `time.perf_counter`) and the given starting time `t0`. The result is rounded
    to two decimal places for precision.

    Args:
        t0: A float representing the starting time, typically obtained from
            `time.perf_counter`.

    Returns:
        A float representing the elapsed time in seconds, rounded to two decimal
        places.
    """
    return round(time.perf_counter() - t0, 2)

class wrapper(object):
    """Wrapper for optimization evaluation and logging.

    This class serves as a wrapper around a user-defined objective function
    (`fit`). It tracks the number of function evaluations, the best evaluation
    value found so far, and provides logging functionality. The wrapper is
    typically used in optimization tasks to facilitate performance tracking
    and debug information.

    Attributes:
        fit (Callable[[ArrayLike], float]): The objective function to be
            evaluated.
        evals (multiprocessing.sharedctypes.Synchronized): A shared integer
            that tracks the total number of evaluations.
        best_y (multiprocessing.sharedctypes.Synchronized): A shared double
            used to store the best result value found so far.
        t0 (float): The timestamp (in seconds) when the wrapper is
            instantiated. Used for tracking elapsed time.
    """

    def __init__(self, 
                 fit: Callable[[ArrayLike], float]):
        """
        Initializes the class with a fitness evaluation function, shared evaluation counts,
        best result achieved, and initial time measurement.

        Args:
            fit (Callable[[ArrayLike], float]): A callable function that evaluates the fitness of
                an array-like input and returns a float score.
        """
        self.fit = fit
        self.evals = mp.RawValue(ct.c_int, 0) 
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.t0 = time.perf_counter()

    def __call__(self, x: ArrayLike) -> float:
        """
        Executes the callable object with the given input `x`, tracking its evaluations and logging improvements.

        This method evaluates the function with the provided input array-like `x`, updates the best result seen so far,
        and logs the details if an improvement occurs. If an exception is raised during execution, it prints the error
        and returns the maximum float value.

        Args:
            x (ArrayLike): The input to be evaluated by the callable object.

        Returns:
            float: The result of the evaluation, or the maximum float value in case of an exception.
        """
        try:
            self.evals.value += 1
            y = self.fit(x)
            y0 = y if np.isscalar(y) else sum(y)
            if y0 < self.best_y.value:
                self.best_y.value = y0
                logger.info( 
                    f'{dtime(self.t0)} {self.evals.value} {self.evals.value/(1E-9 + dtime(self.t0)):.1f} {self.best_y.value} {list(x)}'      
                )
            return y
        except Exception as ex:
            print(str(ex))  
            return sys.float_info.max  
    
class Optimizer(object):
    """
    Provides functionalities for optimization tasks.

    This class is designed to handle optimization operations, allowing users
    to set a maximum number of evaluations and assign a specific name to the
    optimizer. It also has methods for retrieving the maximum evaluation number
    and counting runs, optionally integrating with a `store` object.

    Attributes:
        max_evaluations (int, optional): Maximum number of evaluations allowed for
            the optimizer. Default is 50000.
        name (str, optional): The name assigned to the optimizer. Default is an
            empty string.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000, 
                 name: Optional[str] = ''):
        """
        Initializes the class with specific parameters including the maximum number of
        evaluations and the name.

        Args:
            max_evaluations: Optional[int]. The maximum number of evaluations allowed. Defaults to 50000.
            name: Optional[str]. The name or identifier for the instance. Defaults to an empty string.
        """
        self.max_evaluations = max_evaluations
        self.name = name  

    def max_eval_num(self, store=None):
        """
        Evaluates and returns the maximum number of evaluations based on the provided store
        or the object's default value.

        Args:
            store (optional): A store object that provides the `eval_num` method.

        Returns:
            int: The maximum number of evaluations, as determined by the store if provided,
            or the object's default maximum evaluations value.
        """
        return self.max_evaluations if store is None else \
                store.eval_num(self.max_evaluations)
                
    def get_count_runs(self, store=None):
        """
        Gets the count of runs from the provided store. If no store is provided,
        returns 0.

        Args:
            store: An object with a method `get_count_runs`. If None, the method
                   returns 0.

        Returns:
            int: The count of runs retrieved from the store, or 0 if the store
                 is not provided.
        """
        return 0 if store is None else \
                store.get_count_runs()
                        
class Sequence(Optimizer):
    """
    A class for sequentially combining multiple optimizers.

    The Sequence class is designed to execute a series of optimizers sequentially.
    Each optimizer in the provided sequence takes over from the result of the
    previous one, allowing for a composite approach to optimization. The class
    inherits from the Optimizer base class.

    Attributes:
        optimizers (ArrayLike): List of optimizer instances to be executed
            sequentially. Each optimizer must inherit from the base Optimizer
            class.
        max_evaluations (int): Total number of evaluations across all optimizers
            in the sequence.
        name (str): Concatenated name of all optimizers in the sequence, separated
            by arrows (' -> ').
    """
    
    def __init__(self, optimizers: ArrayLike):
        """
        Initializes the class with a list of optimizers, calculating the total number of evaluations and preparing the combined
        optimizer name.

        Each optimizer from the provided list contributes its name and maximum evaluations to compose a combined metadata
        representation for the initialized object.

        Args:
            optimizers (ArrayLike): A list or array-like object containing optimizer instances. Each optimizer should have a
                `name` attribute and a `max_evaluations` attribute.
        """
        Optimizer.__init__(self)
        self.optimizers = optimizers 
        self.max_evaluations = 0 
        for optimizer in self.optimizers:
            self.name += optimizer.name + ' -> '
            self.max_evaluations += optimizer.max_evaluations
        self.name = self.name[:-4]

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Bounds, 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike, Callable]] = None, 
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store=None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given objective function using multiple optimizers and evaluates
        respective results to find the optimal solution. This function iteratively
        utilizes different optimizers to search for the minimum value of the objective
        function within specified bounds while updating the best guess during the
        process. The total function evaluations across all optimizers are accumulated.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to minimize.
                Must accept a variable of type ArrayLike and return a float.
            bounds (Bounds): The bounds within which the optimization is performed.
            guess (Optional[ArrayLike]): An optional initial guess for the optimization.
            sdevs (Optional[Union[float, ArrayLike, Callable]]): Optional standard
                deviations or a callable defining standard deviation that supports
                optimization.
            rg (Optional[Generator]): Random number generator for stochastic techniques,
                defaults to numpy's Generator(PCG64DXSM).
            store: Optional parameter to store intermediate results or data related
                to the optimization process.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing:
                - The optimal parameters as a numpy ndarray.
                - The minimum value of the function found.
                - Total function evaluations across all optimizers.
        """
        evals = 0
        y = np.inf
        for optimizer in self.optimizers:
            ret = optimizer.minimize(fun, bounds, guess, sdevs, rg, store)
            if ret[1] < y:
                y = ret[1]
                x = ret[0]
            guess = x
            evals += ret[2]
        return x, y, evals
                  
class Choice(Optimizer):
    """A class representing a choice-based optimizer.

    The Choice class allows selecting a random optimizer from a list of provided
    optimizers. It facilitates optimization by delegating the minimize function
    to one of the contained optimizers. The purpose of this class is to provide
    flexibility in switching between different optimization strategies dynamically.

    Attributes:
        optimizers (ArrayLike): A list of optimizers to be used by the choice
            optimizer.
        max_evaluations (int): The maximum number of evaluations allowed,
            determined from the first optimizer in the list.
    """
    
    def __init__(self, optimizers: ArrayLike):
        """
        Combines multiple optimizers to operate as a single optimizer. This class
        allows multiple optimization algorithms to be combined and treated as
        a unified optimizer. Each individual optimizer contributes to the
        effectiveness of the overall optimization process.

        Args:
            optimizers (ArrayLike): A collection of optimizer instances that
                will be combined into a unified optimizer.

        Attributes:
            optimizers (ArrayLike): Collection of optimizer instances.
            max_evaluations (int): Maximum number of evaluations defined by
                the first optimizer in the collection.
            name (str): Concatenated names of all optimizers separated by " | ".
        """
        Optimizer.__init__(self)
        self.optimizers = optimizers 
        self.max_evaluations = optimizers[0].max_evaluations 
        for optimizer in self.optimizers:
            self.name += optimizer.name + ' | '
        self.name = self.name[:-3]
                  
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Bounds, 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike, Callable]] = None, 
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store=None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given function using one of the available optimizers chosen at random.

        This method selects an optimizer from the available pool of optimizers randomnly,
        then uses the selected optimizer to minimize the given function. The function to
        be minimized, along with other parameters required for the optimization, are passed
        to the chosen optimizer. The result of the optimization, including the solution,
        minimum value, and number of iterations, is returned.

        Args:
            fun (Callable[[ArrayLike], float]): The function to minimize. This should take
                a single argument, which is a set of parameter values, and return a scalar
                value representing the objective to minimize.
            bounds (Bounds): The bounds for the optimization, specifying the feasible
                range for each parameter in the input space.
            guess (Optional[ArrayLike]): Initial guess for the optimization. Provides
                a starting point for the optimization process. If not provided, defaults
                to None.
            sdevs (Optional[Union[float, ArrayLike, Callable]]): Standard deviations or
                a function to generate them, influencing the exploration during the
                optimization process. If not provided, defaults to None.
            rg (Optional[Generator]): Random number generator instance for selecting a
                random optimizer and potentially generating random values for the optimization.
                Defaults to a `Generator` using `PCG64DXSM`.
            store: Optional storage object for logging or saving optimization history. The
                structure and usage are dependent on the specific optimizers used.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing the optimization results:
                - A NumPy array with the parameters corresponding to the solution.
                - A float denoting the minimum value of the objective function found.
                - An integer indicating the number of iterations used in the optimization.
        """
        choice = rg.integers(0, len(self.optimizers))
        opt = self.optimizers[choice]
        return opt.minimize(fun, bounds, guess, sdevs, rg, store)

def de_cma(max_evaluations: Optional[int] = 50000, 
           popsize: Optional[int] = 31, 
           stop_fitness: Optional[float] = -np.inf, 
           de_max_evals: Optional[int] = None, 
           cma_max_evals: Optional[int] = None, 
           ints: Optional[ArrayLike] = None, 
           workers: Optional[int]  = None) -> Sequence:
    """
    Creates a sequence of optimization algorithms combining Differential Evolution (DE) and Covariance Matrix Adaptation (CMA).

    This function allows the user to specify the maximum number of evaluations, population size, and stopping fitness, and it
    distributes the evaluations proportionally between DE and CMA. The parameters for DE and CMA can also be individually
    customized. It returns a sequence of optimization instances for further use in optimization tasks.

    Args:
        max_evaluations (Optional[int]): The maximum number of evaluations to be distributed between DE and CMA optimization
            algorithms. Defaults to 50000.
        popsize (Optional[int]): Population size for the optimizers. Defaults to 31.
        stop_fitness (Optional[float]): Fitness threshold to stop the optimization process. Defaults to -infinity.
        de_max_evals (Optional[int]): Specific maximum number of evaluations for the DE optimizer. If None, it is calculated
            as a proportion of max_evaluations. Defaults to None.
        cma_max_evals (Optional[int]): Specific maximum number of evaluations for the CMA optimizer. If None, it is calculated
            as a proportion of max_evaluations. Defaults to None.
        ints (Optional[ArrayLike]): Array-like parameter to indicate integer constraints for optimization. Defaults to None.
        workers (Optional[int]): Number of workers to use for optimization. Defaults to None.

    Returns:
        Sequence: A sequence containing instances of DE and CMA optimizers with specified configurations.
    """

    de_evals = np.random.uniform(0.1, 0.5)
    if de_max_evals is None:
        de_max_evals = int(de_evals*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int((1.0-de_evals)*max_evaluations)
    opt1 = De_cpp(popsize=popsize, max_evaluations = de_max_evals, 
                  stop_fitness = stop_fitness, ints=ints, workers = workers)
    opt2 = Cma_cpp(popsize=popsize, max_evaluations = cma_max_evals, 
                   stop_fitness = stop_fitness, workers = workers)
    return Sequence([opt1, opt2])

def de_cma_py(max_evaluations: Optional[int] = 50000, 
           popsize: Optional[int] = 31, 
           stop_fitness: Optional[float] = -np.inf, 
           de_max_evals: Optional[int] = None, 
           cma_max_evals: Optional[int] = None, 
           ints: Optional[ArrayLike] = None, 
           workers: Optional[int]  = None) -> Sequence:
    """
    Creates and returns a sequence of optimizers configured with DE (Differential
    Evolution) and CMA (Covariance Matrix Adaptation) algorithms. The function
    allows customization of population size, maximum evaluations, stopping
    fitness, and the balance between DE and CMA evaluation limits.

    It is useful for processes requiring hybrid optimization and provides a
    flexible interface for setting up the optimization procedure.

    Args:
        max_evaluations: The total number of evaluations allowed for the
            optimization procedure. Default is 50000.
        popsize: The population size for both DE and CMA optimization methods.
            Default is 31.
        stop_fitness: The stopping criteria based on fitness value. Default is
            -inf (no stopping fitness).
        de_max_evals: Maximum number of evaluations allowed for the DE algorithm.
            If None, it is calculated dynamically based on `max_evaluations`.
            Default is None.
        cma_max_evals: Maximum number of evaluations allowed for the CMA algorithm.
            If None, it is calculated dynamically based on `max_evaluations`.
            Default is None.
        ints: Array-like input of integer constraints (if any). Used specifically in
            the DE algorithm. Default is None.
        workers: Number of parallel processes or threads allowed for computations.
            Default is None.

    Returns:
        A sequence containing two elements:
            1. A DE optimizer configured with given or calculated parameters.
            2. A CMA optimizer configured with given or calculated parameters.
    """

    de_evals = np.random.uniform(0.1, 0.5)
    if de_max_evals is None:
        de_max_evals = int(de_evals*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int((1.0-de_evals)*max_evaluations)
    opt1 = De_python(popsize=popsize, max_evaluations = de_max_evals, 
                     stop_fitness = stop_fitness, ints=ints, workers = workers)
    opt2 = Cma_python(popsize=popsize, max_evaluations = cma_max_evals, 
                   stop_fitness = stop_fitness, workers = workers)
    return Sequence([opt1, opt2])

def da_cma(max_evaluations: Optional[int] = 50000, 
           popsize: Optional[int] = 31, 
           da_max_evals: Optional[int] = None, 
           cma_max_evals: Optional[int] = None, 
           stop_fitness: Optional[float] = -np.inf) -> Sequence:
    """
    Combines Differential Algorithm (DA) and Covariance Matrix Adaptation (CMA)
    optimization techniques into a sequential process, dividing the computational
    budget between the two algorithms and returning the resulting sequence.

    This function allows for parameterized customization of each algorithm's
    maximum evaluations, population size, and stopping fitness to suit specific
    optimization problems.

    Args:
        max_evaluations (Optional[int]): The total number of evaluations to
            allocate between the DA and CMA algorithms. Default is 50000.
        popsize (Optional[int]): Population size for the CMA algorithm. Default
            is 31.
        da_max_evals (Optional[int]): The maximum evaluations for the DA
            algorithm. If None, it is calculated as a fraction of
            `max_evaluations`. Default is None.
        cma_max_evals (Optional[int]): The maximum evaluations for the CMA
            algorithm. If None, it is calculated as the remaining budget from
            `max_evaluations`. Default is None.
        stop_fitness (Optional[float]): The minimum fitness value at which to
            terminate optimization. Default is -np.inf.

    Returns:
        Sequence: A sequence of DA and CMA optimization processes configured with
        the provided or default parameters.
    """

    da_evals = np.random.uniform(0.1, 0.5)
    if da_max_evals is None:
        da_max_evals = int(da_evals*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int((1.0-da_evals)*max_evaluations)
    opt1 = Da_cpp(max_evaluations = da_max_evals, stop_fitness = stop_fitness)
    opt2 = Cma_cpp(popsize=popsize, max_evaluations = cma_max_evals, 
                   stop_fitness = stop_fitness)
    return Sequence([opt1, opt2])

def de_crfmnes(max_evaluations: Optional[int] = 50000, 
               popsize: Optional[int] = 32, 
               stop_fitness: Optional[float] = -np.inf, 
               de_max_evals: Optional[int] = None, 
               crfm_max_evals: Optional[int] = None, 
               ints: Optional[ArrayLike] = None, 
               workers: Optional[int]  = None) -> Sequence:
    """
    Combines Differential Evolution (DE) and Covariance Matrix Adaptation Evolution
    Strategies (CR-FM-NES) optimization algorithms in sequence.

    This function initializes and configures two optimization algorithms, DE and
    CR-FM-NES, with specified parameters and returns a sequence of the two optimizers.
    The maximum evaluations for each optimizer are proportionally allocated based
    on a random variable unless explicitly specified.

    Args:
        max_evaluations: The total number of evaluations available for both
            optimization algorithms (default: 50000).
        popsize: The population size for both optimization algorithms (default: 32).
        stop_fitness: The fitness threshold to stop optimization when achieved
            (default: -infinity).
        de_max_evals: Maximum evaluations explicitly allocated for DE. If None, the
            evaluations are derived from a random proportion of max_evaluations.
        crfm_max_evals: Maximum evaluations explicitly allocated for CR-FM-NES. If
            None, the evaluations are derived as the remainder of max_evaluations.
        ints: An optional array-like structure indicating integer constraints in the
            optimization process (default: None).
        workers: The number of workers for parallel processing in optimization
            algorithms (default: None).

    Returns:
        Sequence: A sequence containing two optimizers, the first implementing DE
            and the second implementing CR-FM-NES.
    """

    de_evals = np.random.uniform(0.1, 0.5)
    if de_max_evals is None:
        de_max_evals = int(de_evals*max_evaluations)
    if crfm_max_evals is None:
        crfm_max_evals = int((1.0-de_evals)*max_evaluations)
    opt1 = De_cpp(popsize=popsize, max_evaluations = de_max_evals, 
                  stop_fitness = stop_fitness, ints=ints, workers = workers)
    opt2 = Crfmnes_cpp(popsize=popsize, max_evaluations = crfm_max_evals, 
                   stop_fitness = stop_fitness, workers = workers)
    return Sequence([opt1, opt2])

def crfmnes_bite(max_evaluations: Optional[int] = 50000, 
                popsize: Optional[int] = 31, 
                stop_fitness: Optional[float] = -np.inf, 
                crfm_max_evals: Optional[int] = None, 
                bite_max_evals: Optional[int] = None, 
                M: Optional[int] = 1) -> Sequence:
    """
    Creates a sequence of two optimization methods, Crfmnes_cpp and Bite_cpp,
    with configurable parameters for evaluations, population size, stopping
    fitness criteria, and others. The evaluation budgets for each
    optimizer are adjusted proportionally based on the given parameters or
    their defaults.

    Args:
        max_evaluations: Maximum number of allowed evaluations for the overall
            optimization process. Default is 50000.
        popsize: Population size for the optimization process. Default is 31.
        stop_fitness: Fitness value at which the optimization stops if reached.
            Default is negative infinity.
        crfm_max_evals: Maximum evaluations assigned to the Crfmnes_cpp
            optimization method. If not provided, it is calculated based on a
            proportion of `max_evaluations`.
        bite_max_evals: Maximum evaluations assigned to the Bite_cpp
            optimization method. If not provided, it is calculated as the
            remaining proportion of `max_evaluations`.
        M: Parameter specific to the Bite_cpp optimization method which may
            influence its behavior. Default is 1.

    Returns:
        Sequence: A sequence containing two initialized optimizers, Crfmnes_cpp
            and Bite_cpp, configured with the provided or default parameters.
    """

    crfmnes_evals = np.random.uniform(0.1, 0.5)
    if crfm_max_evals is None:
        crfm_max_evals = int(crfmnes_evals*max_evaluations)
    if bite_max_evals is None:
        bite_max_evals = int((1.0-crfmnes_evals)*max_evaluations)
    opt1 = Crfmnes_cpp(popsize=popsize, max_evaluations = crfm_max_evals, 
                  stop_fitness = stop_fitness)
    opt2 = Bite_cpp(popsize=popsize, max_evaluations = bite_max_evals, 
                   stop_fitness = stop_fitness, M=M)
    return Sequence([opt1, opt2])

def bite_cma(max_evaluations: Optional[int] = 50000, 
            popsize: Optional[int] = 31, 
            stop_fitness: Optional[float] = -np.inf,
            bite_max_evals: Optional[int] = None,  
            cma_max_evals: Optional[int] = None, 
            M: Optional[int] = 1) -> Sequence:
    """
    Generates a sequence of optimization strategies using the Biogeography-based
    optimization technique (Bite) and the Covariance Matrix Adaptation (CMA)
    algorithm. The function allows customization of evaluations, population size,
    and stopping criteria to balance between the two strategies.

    Args:
        max_evaluations (Optional[int]): Maximum number of evaluations allowed
            for the combined Bite and CMA optimization. Defaults to 50000.
        popsize (Optional[int]): Population size for the optimization
            strategies. Defaults to 31.
        stop_fitness (Optional[float]): Minimal fitness threshold to stop the
            optimization early. Defaults to negative infinity.
        bite_max_evals (Optional[int]): Maximum number of evaluations allocated
            for the Bite component. If None, it is computed as a proportion of
            max_evaluations based on a random factor. Defaults to None.
        cma_max_evals (Optional[int]): Maximum number of evaluations allocated
            for the CMA component. If None, it is derived as the remainder of
            max_evaluations after the Bite component evaluations. Defaults to None.
        M (Optional[int]): Number of parallel optimization runs for the Bite
            strategy. Defaults to 1.

    Returns:
        Sequence: A sequence containing two optimization strategies, where the
            first element uses the Bite algorithm and the second element uses
            the CMA algorithm.
    """

    bite_evals = np.random.uniform(0.1, 0.5)
    if bite_max_evals is None:
        bite_max_evals = int(bite_evals*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int((1.0-bite_evals)*max_evaluations)
    opt1 = Bite_cpp(popsize=popsize, max_evaluations = bite_max_evals, 
                   stop_fitness = stop_fitness, M=M)
    opt2 = Cma_cpp(popsize=popsize, max_evaluations = cma_max_evals, 
                  stop_fitness = stop_fitness)
    return Sequence([opt1, opt2])

def cma_bite(max_evaluations: Optional[int] = 50000, 
            popsize: Optional[int] = 32, 
            stop_fitness: Optional[float] = -np.inf, 
            cma_max_evals: Optional[int] = None, 
            bite_max_evals: Optional[int] = None, 
            M: Optional[int] = 1) -> Sequence:
    """
    Combines CMA-ES and BITE optimization algorithms into a single sequence of
    operations, utilizing their respective strengths for optimization.

    This function initializes instances of CMA-ES (Covariance Matrix Adaptation
    Evolution Strategy) and BITE (Binary Teaching–Learning-Based Optimization)
    algorithms with specified parameters, calculates their respective evaluation
    budgets, and returns a sequence of these algorithms for further processing.

    The CMA-ES algorithm is used for global optimization tasks, and the BITE
    algorithm focuses on discrete or binary optimization. The combination allows
    for more versatile optimization.

    Args:
        max_evaluations: Optional parameter specifying the total number of
            evaluations available for both algorithms. Default is 50000.
        popsize: Optional parameter specifying the population size for the
            optimization algorithms. Default is 32.
        stop_fitness: Optional parameter indicating the fitness value at which
            optimization should stop. The default is -np.inf.
        cma_max_evals: Optional parameter specifying the evaluation budget for
            the CMA-ES algorithm. If not provided, it is calculated as a fraction
            of max_evaluations.
        bite_max_evals: Optional parameter specifying the evaluation budget for
            the BITE algorithm. If not provided, it is calculated as the remainder
            of the total evaluation budget after allocating to CMA-ES.
        M: Optional parameter specifying the additional configuration for the
            BITE algorithm. Default is 1.

    Returns:
        Sequence: A sequence containing two optimization objects, the first
            configured for CMA-ES and the second for BITE.
    """

    cma_evals = np.random.uniform(0.1, 0.5)
    if cma_max_evals is None:
        cma_max_evals = int(cma_evals*max_evaluations)
    if bite_max_evals is None:
        bite_max_evals = int((1.0-cma_evals)*max_evaluations)
    opt1 = Cma_cpp(popsize=popsize, max_evaluations = cma_max_evals, 
                  stop_fitness = stop_fitness, stop_hist = 0)
    opt2 = Bite_cpp(popsize=popsize, max_evaluations = bite_max_evals, 
                   stop_fitness = stop_fitness, M=M)
    return Sequence([opt1, opt2])

class Crfmnes(Optimizer):
    """
    Implements the CR-FM-NES (Covariance Matrix Adaptation Evolution Strategy with Full Matrix)
    optimization algorithm.

    This class provides functionality to perform constrained function minimization using
    the CR-FM-NES optimization approach. It supports features such as population-based
    search, optional fitness stopping criteria, and multi-threaded evaluations. The optimizer
    is designed for use in scenarios requiring efficient searching in complex objective
    functions with optional boundary constraints.

    Attributes:
        popsize (int): Size of the population used in each generation.
        stop_fitness (float): Fitness threshold for stopping the optimization process. The
            optimization stops if a solution with this fitness or better is found.
        guess (ArrayLike): Initial guess for the starting position of the optimization.
            If not provided, a default initialization is used.
        sdevs (float): Initial standard deviation for the covariance matrix in the
            optimization process. It determines the scale of the search.
        workers (int): Number of parallel workers used for function evaluations. A higher
            number can speed up optimization for computationally intensive objective functions.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 32, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None, 
                 workers: Optional[int] = None):
        """
        Initializes the optimizer with specific parameters for the CR-FM-NES algorithm.

        This constructor sets up the configuration for the optimizer, including
        maximum evaluations, population size, initial guess, stopping fitness,
        standard deviations, and the number of workers for parallel evaluations.

        Args:
            max_evaluations: Optional[int]
                Maximum number of evaluations allowed for the optimization process.
                Defaults to 50000.
            popsize: Optional[int]
                Size of the population for each generation in the optimization.
                Defaults to 32.
            guess: Optional[ArrayLike]
                Initial guess or starting point for the optimization process.
                Defaults to None.
            stop_fitness: Optional[float]
                Target fitness value at which the optimization stops. Defaults
                to negative infinity.
            sdevs: Optional[float]
                Standard deviations used in the search distribution. Defaults to None.
            workers: Optional[int]
                Number of workers to use for parallel evaluations. Defaults to None.
        """
        Optimizer.__init__(self, max_evaluations, 'crfmnes')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = 0.3, 
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given objective function using a specific optimization strategy.

        This function executes a constrained optimization method to find the minimum value
        of the provided objective function within explicit boundaries.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to be minimized.
                It takes an input of type ArrayLike and returns a float representing the
                scalar value of the objective.
            bounds (Optional[Bounds]): Constraints on the optimization variables
                defined as lower and upper bounds.
            guess (Optional[ArrayLike]): Initial guess for the optimization. If not
                provided, the class's internal guess attribute is used.
            sdevs (Optional[float]): Initial standard deviation for the optimization.
                Defaults to 0.3 if not specified and overrides the class's internal
                sdevs attribute if provided.
            rg (Optional[Generator]): Random number generator used during optimization.
                Defaults to a Generator instance of PCG64DXSM.
            store: An optional argument for storing optimization-related data. The usage
                depends on the implementation of the optimization method.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing the best solution found
                (as a NumPy array), the corresponding function value, and the number of
                function evaluations performed.
        """
        ret = crfmnes.minimize(fun, bounds, 
                self.guess if not self.guess is None else guess,
                input_sigma = self.sdevs if not self.sdevs is None else sdevs,
                max_evaluations = self.max_eval_num(store), 
                popsize=self.popsize, 
                stop_fitness = self.stop_fitness,
                rg=rg, runid=self.get_count_runs(store),
                workers = self.workers)     
        return ret.x, ret.fun, ret.nfev

class Crfmnes_cpp(Optimizer):
    """
    An implementation of the CR-FM-NES (Covariance-Rescaling Fast Matrix-Normal Evolution Strategies)
    optimizer designed for solving real-valued optimization problems.

    This class serves as a wrapper to a lower-level implementation, providing a user-friendly
    interface for evolutionary optimization tasks. The main purpose of this optimizer is to
    enable efficient optimization of noisy objective functions in high-dimensional spaces
    through the use of a covariance rescaling mechanism.

    Attributes:
        popsize (int): The population size for the evolutionary algorithm.
        stop_fitness (float): The fitness value at which the optimization process halts if
            achieved. Defaults to negative infinity.
        guess (ArrayLike, optional): An initial guess value for the optimization process.
            If not provided at initialization, it must be set during the call to `minimize`.
        sdevs (float, optional): Initial standard deviations for the optimization process.
            If not provided at initialization, it must be set during the call to `minimize`.
        workers (int, optional): The number of parallel workers to use during optimization.
            If None, single-threaded execution is assumed.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 32, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None, 
                 workers: Optional[int] = None):
        """
        Initializes the optimizer with the specified parameters to control the behavior of
        the evolutionary algorithm.

        Args:
            max_evaluations: The maximum number of evaluations allowed during the optimization
                process. Defaults to 50000.
            popsize: The population size used in the algorithm. Defaults to 32.
            guess: The optional initial guess for the solution vector. If not provided, it
                will be set to None.
            stop_fitness: The threshold fitness value at which the optimization stops. The
                process terminates when this value is reached or exceeded. Defaults to -np.inf.
            sdevs: The optional initial standard deviations for the solution vector. If not
                provided, it will be set to None.
            workers: The number of parallel workers used for evaluations during the optimization
                process. Defaults to None.
        """
        Optimizer.__init__(self, max_evaluations, 'crfmnes cpp')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = 0.3, 
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimize the provided objective function under given constraints.

        This method performs function minimization using the provided objective
        function, optimization bounds, and optional parameters such as an initial guess,
        standard deviations, a random generator, or a data storage object. It delegates
        the computation to an underlying implementation and returns the results of
        the minimization process.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to be minimized.
                It must accept a single argument (input array) and return a scalar value.
            bounds (Optional[Bounds]): The bounds within which the minimization is to be
                performed. This must specify valid limits for the input values.
            guess (Optional[ArrayLike]): The initial guess for the minimization process.
                If not provided, it defaults to None.
            sdevs (Optional[float]): The standard deviation for the optimization
                process. If not provided, defaults to 0.3.
            rg (Optional[Generator]): A random number generator to ensure consistent and
                reproducible behavior. Defaults to `Generator(PCG64DXSM())`.
            store: An optional object for storing the state or results of the
                minimization process. Its behavior depends on its implementation.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing:
                - np.ndarray: The optimal parameters minimizing the objective function.
                - float: The minimum value of the objective function at the optimal point.
                - int: The total number of function evaluations performed during the
                  minimization process.

        """
        ret = crfmnescpp.minimize(fun, bounds, 
                self.guess if not self.guess is None else guess,
                input_sigma = self.sdevs if not self.sdevs is None else sdevs,
                max_evaluations = self.max_eval_num(store), 
                popsize=self.popsize, 
                stop_fitness = self.stop_fitness,
                rg=rg, runid=self.get_count_runs(store), 
                workers = self.workers)     
        return ret.x, ret.fun, ret.nfev

class Pgpe_cpp(Optimizer):
    """
    Performs optimization using the PGPE (Policy Gradient with Parameter Exploration)
    algorithm implemented in C++.

    This class acts as a Python interface to leverage an efficient C++ implementation
    of the PGPE algorithm for solving optimization problems. It inherits from the
    base `Optimizer` class and provides the ability to minimize a function using
    population-based search strategies. The optimizer uses given constraints and
    parameters to iteratively converge towards the optimal solution.

    Attributes:
        popsize (int): Number of individuals in the population per generation.
        stop_fitness (float): Fitness value at which the optimization stops if achieved.
        guess (ArrayLike): Initial guess for the parameter values.
        sdevs (float): Initial standard deviation for parameter exploration.
        workers (int): Number of parallel processes for evaluation.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 500000,
                 popsize: Optional[int] = 640, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None, 
                 workers: Optional[int] = None):
        """
        Initializes the optimizer with the specified parameters. This optimizer is
        intended for use in scenarios requiring parameter guessing and optimized
        population-based performance evaluation.

        Args:
            max_evaluations: An optional integer specifying the maximum number of
                function evaluations allowed during the optimization process.
                Defaults to 500000.
            popsize: An optional integer specifying the population size for the
                optimizer. Defaults to 640.
            guess: An optional parameter defining the initial guess or starting
                point for the optimization as an array-like object. Defaults to None.
            stop_fitness: An optional float representing the stopping threshold for
                fitness. Optimization ends when this fitness value is reached.
                Defaults to negative infinity.
            sdevs: An optional float specifying the standard deviation for the
                guessed parameters. Defaults to None.
            workers: An optional integer specifying the number of worker threads
                or processes to be used during the optimization process. Defaults
                to None.
        """
        Optimizer.__init__(self, max_evaluations, 'pgpe cpp')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = 0.1, 
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """Minimizes a given function using a defined optimization algorithm.

        This method uses a stochastic optimization algorithm to find the minimum
        value of the given function within specified bounds. It allows for an
        optional initial guess, standard deviation for the input parameters,
        and can utilize parallel processing through workers.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to minimize.
                It should accept an array-like input and return a scalar value.
            bounds (Optional[Bounds]): The bounds within which the optimization
                will be performed. It defines the valid parameter space.
            guess (Optional[ArrayLike], optional): Optional initial guess for
                the optimization. If not provided, a default guess may be used.
                Defaults to None.
            sdevs (Optional[float], optional): The standard deviation for the input
                parameters. This is used in the optimization algorithm to explore
                the parameter space. Defaults to 0.1.
            rg (Optional[Generator], optional): A random generator for the
                optimization algorithm. Defaults to Generator(PCG64DXSM()).
            store: Optional storage object to store intermediate optimization
                results or states. Its usage depends on the specific optimization
                procedure.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing:
                - The optimal parameters as a numpy array.
                - The minimum value of the objective function.
                - The number of function evaluations performed.
        """
        ret = pgpecpp.minimize(fun, bounds, 
                self.guess if not self.guess is None else guess,
                input_sigma = self.sdevs if not self.sdevs is None else sdevs,
                max_evaluations = self.max_eval_num(store), 
                popsize=self.popsize, 
                stop_fitness = self.stop_fitness,
                rg=rg, runid=self.get_count_runs(store), 
                workers = self.workers)     
        return ret.x, ret.fun, ret.nfev

class Cma_python(Optimizer):
    """
    Cma_python optimizer class.

    This class implements the CMA-ES optimization algorithm with additional
    customizable parameters. It is designed to minimize objective functions
    within provided bounds and constraints.

    Attributes:
        popsize (int): Population size used by the optimization algorithm.
        stop_fitness (float): Target fitness value. If reached, the optimization
            process will stop early.
        update_gap (int or None): Number of iterations between covariance matrix
            updates. None means default behavior.
        guess (ArrayLike or None): Initial guess for the solution in the search
            space.
        sdevs (float or ArrayLike or None): Standard deviations for the search.
            If set to None, default values will be used.
        normalize (bool): Whether to normalize input data during the optimization.
        workers (int or None): Number of workers for parallelized evaluations.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None, 
                 workers: Optional[int] = None,        
                 update_gap: Optional[int] = None, 
                 normalize: Optional[bool] = True):
        """
        Initializes the CMA-ES optimizer with its configuration parameters.

        This method sets up the optimizer by initializing its key attributes based on the
        user-specified or default values. It inherits the generic optimizer functionality
        and extends it with CMA-ES-specific parameters.

        Args:
            max_evaluations: Optional maximum number of evaluations allowed (default 50000).
            popsize: Optional population size for the CMA-ES algorithm (default 31).
            guess: Optional initial guess for the solution.
            stop_fitness: Optional fitness value at which optimization stops (default -inf).
            sdevs: Optional standard deviations for the distribution.
            workers: Optional number of workers for parallel evaluations.
            update_gap: Optional number of generations between updates.
            normalize: Optional boolean to enable normalization of input parameters
                (default True).
        """
        Optimizer.__init__(self, max_evaluations, 'cma py')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.update_gap = update_gap
        self.guess = guess
        self.sdevs = sdevs
        self.normalize = normalize
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike, Callable]] = 0.1, 
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a provided objective function using the CMA-ES optimization algorithm.

        This method utilizes the Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
        to perform function minimization. It supports optional parameters such as bounds,
        initial guesses, and standard deviations for sampling, among others. The method
        returns the optimal solution, the corresponding function value, and the number
        of function evaluations performed.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to be minimized. It
                should take an input array and return a scalar function value.
            bounds (Optional[Bounds]): The bounds within which the solution is searched.
                Can be None for unbounded optimization.
            guess (Optional[ArrayLike], optional): The initial guess for the solution.
                If not provided, an internal guess will be used.
            sdevs (Optional[Union[float, ArrayLike, Callable]], optional): The standard
                deviations used for sampling around the guess. Defaults to 0.1.
            rg (Optional[Generator], optional): The random number generator for the
                optimization process. Defaults to Generator(PCG64DXSM()).
            store: An optional object used for storing additional metadata or state
                during optimization.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing:
                - The optimal solution as an array.
                - The minimum function value achieved.
                - The number of function evaluations performed.
        """
        ret = cmaes.minimize(fun, bounds, 
                self.guess if not self.guess is None else guess,
                input_sigma= self.sdevs if not self.sdevs is None else sdevs,
                max_evaluations = self.max_eval_num(store), 
                popsize=self.popsize, 
                stop_fitness = self.stop_fitness,
                rg=rg, runid=self.get_count_runs(store),
                normalize = self.normalize,
                update_gap = self.update_gap,
                workers = self.workers)     
        return ret.x, ret.fun, ret.nfev

class Cma_cpp(Optimizer):
    """
    This class implements the CMA-ES optimization algorithm with customization options.

    The Cma_cpp class extends the functionality of a generic optimizer by introducing
    CMA-ES specific features. This includes options for population size, convergence
    criteria, standard deviations for mutations, and the ability to parallelize the
    optimization process. It is designed for optimizing objective functions with or
    without constraints.

    Attributes:
        popsize (int): The population size for the optimization process.
        stop_fitness (float): Fitness value threshold for stopping the optimization process early.
        stop_hist (int): Number of recent fitness values considered for convergence checks.
        guess (ArrayLike): Initial guess for the optimization variables.
        sdevs (float): Initial standard deviations for the distribution used in CMA-ES.
        update_gap (int): Interval (in terms of generations) between updates to the distribution.
        delayed_update (bool): Indicates whether updates to the distribution are delayed until
            certain criteria are met.
        normalize (bool): Specifies whether the input parameters should be normalized for the
            optimization process.
        workers (int): The number of workers used to parallelize computations and evaluations.
    """
   
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None, 
                 workers: Optional[int] = None,        
                 update_gap: Optional[int] = None, 
                 normalize: Optional[bool] = True,                 
                 delayed_update: Optional[bool] = True,   
                 stop_hist: Optional[int] = -1):
        """
        Initializes the CMA-ES optimization algorithm with the specified parameters. This
        method extends an existing optimizer by incorporating properties specific to
        CMA-ES. It allows fine-tuning of the optimization process through various parameters.

        Args:
            max_evaluations (Optional[int]): The maximum number of evaluations allowed for the optimizer.
            popsize (Optional[int]): The population size for the optimization process.
            guess (Optional[ArrayLike]): Initial guess for the optimization variables.
            stop_fitness (Optional[float]): Fitness value threshold for stopping the optimization.
            sdevs (Optional[float]): Initial standard deviations for the distribution.
            workers (Optional[int]): The number of workers to parallelize computations.
            update_gap (Optional[int]): Interval (in terms of generations) between updates to the distribution.
            normalize (Optional[bool]): Specifies whether the input parameters should be normalized.
            delayed_update (Optional[bool]): Indicates whether updates to the distribution are delayed.
            stop_hist (Optional[int]): Number of recent fitness values to consider for convergence checks.
        """
        Optimizer.__init__(self, max_evaluations, 'cma cpp')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.stop_hist = stop_hist
        self.guess = guess
        self.sdevs = sdevs
        self.update_gap = update_gap
        self.delayed_update = delayed_update
        self.normalize = normalize
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike, Callable]] = 0.1, 
                 rg=Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes an objective function using the CMA-ES optimization algorithm.

        This method utilizes the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to
        find the minima of the given objective function within specified bounds. The algorithm
        supports customizable standard deviations for mutations, an optional random number generator,
        and the ability to store optimization progress.

        Args:
            fun: A callable objective function to minimize. It accepts an array-like object as input
                and returns a scalar value.
            bounds: The bounds within which the optimization will be performed. This is optional
                and can be used to constrain the search space.
            guess: Initial guess for the optimization. If not provided, a default initial guess is used.
            sdevs: Optional standard deviations for mutations. It can be provided as a float,
                an array-like object, or a callable. Default is 0.1.
            rg: A random number generator instance. By default, it uses a generator from the PCG64DXSM
                algorithm.
            store: Optional. A store to save progress or retrieve optimization state.

        Returns:
            Tuple[np.ndarray, float, int]:
            - The first element is the array containing the optimal solution.
            - The second element is the corresponding minimum function value.
            - The third element is the number of function evaluations performed.

        """
        ret = cmaescpp.minimize(fun, bounds,
                self.guess if not self.guess is None else guess,
                input_sigma = self.sdevs if not self.sdevs is None else sdevs,
                max_evaluations =self.max_eval_num(store),
                popsize = self.popsize,
                stop_fitness = self.stop_fitness,
                stop_hist = self.stop_hist,
                rg = rg, runid = self.get_count_runs(store),
		        update_gap = self.update_gap,
                normalize = self.normalize,
                delayed_update = self.delayed_update,
                workers = self.workers)   
        return ret.x, ret.fun, ret.nfev

class Cma_orig(Optimizer):
    """
    Represents an optimizer specifically designed for the CMA-ES (Covariance Matrix
    Adaptation Evolution Strategy) algorithm.

    This class is used to solve sophisticated optimization problems by utilizing the CMA-ES
    methodology to iteratively refine solutions. It inherits from a base Optimizer class and
    provides additional attributes and functionality associated with CMA-ES for advanced
    optimization use cases.

    Attributes:
        popsize (Optional[int]): The population size, determining the number of candidate
            solutions to evaluate per generation.
        stop_fitness (Optional[float]): A threshold fitness value; the optimization process
            halts if this fitness value is achieved.
        guess (Optional[ArrayLike]): The initial guess or starting point in the search space
            for the optimization algorithm.
        sdevs (Optional[float]): Initial standard deviations for constructing a variance-covariance
            matrix in the adaptation process.
    """
   
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None):
        """
        Initializes an optimizer instance specifically designed for the CMA-ES (Covariance Matrix
        Adaptation Evolution Strategy) algorithm. This is used for solving optimization problems
        by iteratively improving upon solutions in search space. The class inherits core functionality
        from a base Optimizer class and initializes additional parameters related to the CMA-ES algorithm.

        Args:
            max_evaluations (Optional[int]): Maximum number of evaluations allowed for the optimizer.
            popsize (Optional[int]): Population size, representing the number of candidate solutions
                processed per generation.
            guess (Optional[ArrayLike]): Initial guess for the solution, acting as a starting point
                for the optimizer.
            stop_fitness (Optional[float]): Value of fitness to stop the optimization process if achieved.
            sdevs (Optional[float]): Standard deviations to initialize the search distribution.
        """
        Optimizer.__init__(self, max_evaluations, 'cma orig')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike]] = 0.3, 
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes the given objective function using the CMA-ES (Covariance Matrix Adaptation
        Evolution Strategy) algorithm.

        This method aims to find the optimal set of parameters that minimize the output
        value of a provided function. It utilizes the CMA-ES optimization routine,
        which is particularly effective for non-linear and non-convex optimization problems.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to minimize.
                It should accept an array-like object as input and return a float as the
                cost or fitness value.
            bounds (Optional[Bounds]): The bounds for the variables in the optimization.
                The bounds provide lower (bounds.lb) and upper (bounds.ub) limits for
                the parameters.
            guess (Optional[ArrayLike], optional): An initial guess for the optimization
                process. If None, a random guess within the bounds will be generated.
                Defaults to None.
            sdevs (Optional[Union[float, ArrayLike]], optional): Standard deviation or
                scale of the search distribution. If None, a default value of 0.3 will
                be used. Defaults to 0.3.
            rg (Optional[Generator], optional): A random generator instance, defaulting
                to Generator(PCG64DXSM()). This is used for generating random values
                during optimization when necessary.
            store (any): A storage mechanism used to manage or obtain related optimization
                metadata like limiting the maximum evaluations.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing:
                - Best solution (`np.ndarray`): The array representing the optimal parameters.
                - Best objective value (`float`): The minimal value obtained from the
                  objective function.
                - Number of evaluations (`int`): The total number of evaluations performed.

        Raises:
            ImportError: If the required `cma` library is not installed.
            Exception: Any error that arises during the optimization process.
        """
        lower = bounds.lb
        upper = bounds.ub
        guess = self.guess if not self.guess is None else guess
        if guess is None:
            guess = rg.uniform(lower, upper)
        max_evaluations = self.max_eval_num(store)   
        input_sigma= self.sdevs if not self.sdevs is None else sdevs
        try:
            import cma
        except ImportError as e:
            raise ImportError("Please install CMA (pip install cma)") 
        try: 
            es = cma.CMAEvolutionStrategy(guess, 0.1,  {'bounds': [lower, upper], 
                                                             'typical_x': guess,
                                                             'scaling_of_variables': scale(lower, upper),
                                                             'popsize': self.popsize,
                                                             'CMA_stds': input_sigma,
                                                             'verbose': -1,
                                                             'verb_disp': -1})
            evals = 0
            for i in range(max_evaluations):
                X, Y = es.ask_and_eval(fun)
                es.tell(X, Y)
                evals += self.popsize
                if es.stop():
                    break 
                if evals > max_evaluations:
                    break    
            return es.result.xbest, es.result.fbest, evals
        except Exception as ex:
            print(ex)

class Cma_lw(Optimizer):
    """CMA-ES lightweight optimizer implementation.

    This class implements the Covariance Matrix Adaptation Evolution Strategy
    (CMA-ES) algorithm in a lightweight manner. It is designed to perform
    black-box optimization for continuous problems. The optimizer iteratively
    searches for the minimum of a given objective function within provided
    bounds, using strategies for population sampling and updates.

    Attributes:
        popsize (int): Population size parameter that determines the number of
            candidate solutions sampled per iteration.
        stop_fitness (float): Threshold for the optimization stopping if the
            fitness value of the best solution reaches or falls below this
            value.
        guess (ArrayLike, optional): Initial guess for the optimization; if not
            provided, it can be randomly generated within bounds.
        sdevs (Union[float, ArrayLike], optional): Standard deviations used for
            initializing the search distribution; it can be a scalar or an
            array matching the dimensionality of the problem.
        workers (int, optional): Number of worker processes used for parallel
            evaluation of the objective function; defaults to serial execution
            if None or a value ≤ 1.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[Union[float, ArrayLike]] = None, 
                 workers: Optional[int] = None):
        """
        Initializes the CMA-LW optimizer with specified parameters to configure its behavior.
        This optimizer is designed to perform Covariance Matrix Adaptation Evolution Strategy
        (CMA-ES) with a lightweight implementation. Parameters like `max_evaluations`, `popsize`,
        and `workers` control the optimization process while `guess` and `sdevs` define
        starting points and variability. This initialization prepares the object with
        configuration settings.

        Args:
            max_evaluations: Optional; Maximum number of evaluations allowed for the optimizer.
            popsize: Optional; Population size used for the optimization process.
            guess: Optional; Initial guess or starting point for the optimization.
            stop_fitness: Optional; Stopping criterion based on target fitness value.
            sdevs: Optional; Standard deviations for the initial distribution.
            workers: Optional; Number of parallel workers to use during optimization.
        """
        Optimizer.__init__(self, max_evaluations, 'cma_lw')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike]] = 0.3, 
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given objective function using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

        This method employs an iterative optimization algorithm to find the minimum of the provided objective
        function. It supports parallel execution of the function evaluations if enabled. The method allows for
        the specification of initial guesses, bounds, standard deviations, and random number generators for
        flexible and customizable optimization.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to minimize.
            bounds (Optional[Bounds]): The bounds of the search domain for each dimension.
            guess (Optional[ArrayLike]): Initial guess for the optimization. If not provided, it is generated
                randomly within the bounds.
            sdevs (Optional[Union[float, ArrayLike]]): Initial standard deviation(s) for the CMA-ES algorithm.
                The default is 0.3.
            rg (Optional[Generator]): Random number generator used for initialization. Defaults to
                a PCG64DXSM generator instance.
            store: Used to store intermediate results during the optimization process. Its usage is optional.

        Returns:
            Tuple[np.ndarray, float, int]: Returns a tuple containing the best solution found (as a numpy array),
            the associated objective function value, and the total number of function evaluations performed.

        Raises:
            ImportError: If the required 'cmaes' library is not installed.
        """
        try:
            import cmaes
        except ImportError as e:
            raise ImportError("Please install cmaes (pip install cmaes)") 

        if guess is None:
            guess = self.guess
        if guess is None:    
            guess = rg.uniform(bounds.lb, bounds.ub)
        bds = np.array([t for t in zip(bounds.lb, bounds.ub)])
        seed = int(rg.uniform(0, 2**32 - 1))
        optimizer = cmaes.CMA(mean=guess, sigma=np.mean(sdevs), bounds=bds, seed=seed, population_size=self.popsize)
        best_y = np.inf
        evals = 0
        fun = serial(fun) if (self.workers is None or self.workers <= 1) else parallel(fun, self.workers)  
        while evals < self.max_evaluations and not optimizer.should_stop():
            xs = [optimizer.ask() for _ in range(optimizer.population_size)]
            ys = fun(xs)
            solutions = []
            for i in range(optimizer.population_size):
                x = xs[i]
                y = ys[i]
                solutions.append((x, y))
                if y < best_y:
                    best_y = y
                    best_x = x
            optimizer.tell(solutions)
            evals += optimizer.population_size           
        if isinstance(fun, parallel):
            fun.stop()
        return best_x, best_y, evals

class Cma_awm(Optimizer):
    """
    Cma_awm optimizer using the CMA-ES algorithm tailored for problems with both continuous
    and discrete decision spaces.

    This class extends the Optimizer base class to implement the CMA-ES (Covariance Matrix
    Adaptation Evolution Strategy) algorithm with customized parameters. It is suitable for
    optimization tasks in mixed search spaces where continuous and discrete variables coexist.

    Attributes:
        popsize (Optional[int]): Size of the population used in the optimization process.
        stop_fitness (Optional[float]): The fitness value at which the optimization process
            will stop if reached.
        guess (Optional[ArrayLike]): Initial guess or solution provided to the optimizer.
        sdevs (Optional[Union[float, ArrayLike]]): The standard deviations of the initial
            search distribution for the optimizer.
        workers (Optional[int]): Number of parallel workers utilized during optimization.
        continuous_space: Continuous constraints or ranges for the problem space.
        discrete_space: Discrete constraints or options for the problem space.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[Union[float, ArrayLike]] = None, 
                 continuous_space = None, 
                 discrete_space = None, 
                 workers: Optional[int] = None):
        """
        Initializes the CMA-ES optimizer with specific parameters for optimization tasks.

        Args:
            max_evaluations (Optional[int]): Maximum number of function evaluations allowed
                during the optimization process. Defaults to 50000.
            popsize (Optional[int]): Size of the population used in the optimization.
                Defaults to 31.
            guess (Optional[ArrayLike]): Initial guess or solution for the optimizer.
                Defaults to None.
            stop_fitness (Optional[float]): Desired fitness value to stop the optimization
                when achieved. Defaults to -np.inf.
            sdevs (Optional[Union[float, ArrayLike]]): Standard deviations for the
                initial search distribution. Defaults to None.
            continuous_space: Continuous space constraints for the optimization.
                Defaults to None.
            discrete_space: Discrete space constraints for the optimization.
                Defaults to None.
            workers (Optional[int]): Number of parallel workers to be used during
                optimization. Defaults to None.
        """
        Optimizer.__init__(self, max_evaluations, 'cma_awm')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers
        self.continuous_space = continuous_space
        self.discrete_space = discrete_space

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike]] = 0.3, 
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given objective function using the CMAwM optimization algorithm.

        This method uses the Covariance Matrix Adaptation with Mirrored Sampling (CMAwM)
        to efficiently find the minimum of the given objective function within the defined
        bounds. It supports continuous and discrete spaces and allows for parallel evaluation
        of function calls when multiple workers are specified.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to minimize.
                It takes an input array and returns a scalar fitness value.
            bounds (Optional[Bounds]): The bounds within which to search for a solution.
                Must define lower (lb) and upper (ub) bounds for each dimension.
            guess (Optional[ArrayLike], optional): Initial guess for the optimization algorithm.
                If not provided, defaults to the value defined in the class or is sampled uniformly
                within the bounds. Defaults to None.
            sdevs (Optional[Union[float, ArrayLike]], optional): Standard deviations for initial
                distribution of solutions. Either a float for uniform deviations or an array
                matching the dimensionality of the problem. Defaults to 0.3.
            rg (Optional[Generator], optional): A random generator for sampling and reproducibility.
                Defaults to a predefined PCG64DXSM generator.
            store (optional): Optional object or mechanism to store intermediate results; the
                specific usage of this parameter depends on the implementation. Defaults to None.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing the best solution found (as an array),
            the corresponding fitness value (float), and the total number of evaluations performed (int).

        Raises:
            ImportError: If the required "cmaes" library is not installed.
        """
        try:
            import cmaes
        except ImportError as e:
            raise ImportError("Please install cmaes (pip install cmaes)") 
              
        if guess is None:
            guess = self.guess
        if guess is None:    
            guess = rg.uniform(bounds.lb, bounds.ub)
        seed = int(rg.uniform(0, 2**32 - 1))
        optimizer = cmaes.CMAwM(mean=guess, sigma=np.mean(sdevs),       
                         continuous_space=self.continuous_space,  
                         discrete_space=self.discrete_space, 
                         seed=seed, population_size=self.popsize)
        best_y = 1E99
        evals = 0
        fun = serial(fun) if (self.workers is None or self.workers <= 1) else parallel(fun, self.workers)  
        while evals < self.max_evaluations and not optimizer.should_stop():
            asks = [optimizer.ask() for _ in range(optimizer.population_size)]
            ys = fun([x[0] for x in asks])
            solutions = []
            for i in range(optimizer.population_size):
                x = asks[i][1]
                y = ys[i]
                solutions.append((x, y))
                if y < best_y:
                    best_y = y
                    best_x = x
            optimizer.tell(solutions)
            evals += optimizer.population_size           
        if isinstance(fun, parallel):
            fun.stop()
        return best_x, best_y, evals

class Cma_sep(Optimizer):
    """
    Cma_sep is an optimizer class utilizing the CMA-ES (Covariance Matrix Adaptation
    Evolution Strategy) in a separable form for high-dimensional optimization tasks.

    The Cma_sep class enables optimization over a multivariate space efficiently using
    the separable CMA-ES by supporting configurable parameters. It integrates easily
    with parallel or serial computation of fitness functions. Users can input a starting
    guess, specify standard deviations, and set termination criteria based on
    fitness or evaluations. This class is especially suitable for optimization problems
    where separable approximations are beneficial.

    Attributes:
        popsize (int): Population size for the optimization process. Higher values
            may provide better diversity but increase computation cost.
        stop_fitness (float): Stop criterion for fitness. Optimization halts if a
            solution achieves this fitness or better.
        guess (ArrayLike): Initial guess for the optimization process. If None, the
            algorithm initializes randomly within bounds.
        sdevs (Union[float, ArrayLike]): Standard deviations for the initial search
            distribution. Scalar or array to control exploration magnitude.
        workers (int): Number of parallel workers for evaluating the fitness
            function. If None or <= 1, fitness evaluations are serialized.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[Union[float, ArrayLike]] = None, 
                 workers: Optional[int] = None):
        """
        Initializes an evolutionary optimization class with specific parameters for
        the Covariance Matrix Adaptation with Separable Functions (CMA-SEP).

        Args:
            max_evaluations (Optional[int]): The maximum number of evaluations allowed
                during the optimization process. Defaults to 50000.
            popsize (Optional[int]): The population size for the optimization process.
                Defaults to 31.
            guess (Optional[ArrayLike]): The initial guess for the optimization. If
                None, it must be provided during optimization setup. Defaults to None.
            stop_fitness (Optional[float]): The fitness value at which the optimization
                stops. If -np.inf, optimization will not stop based on fitness. Defaults
                to -np.inf.
            sdevs (Optional[Union[float, ArrayLike]]): The standard deviations used for
                sampling during the optimization process. If None, a default set of
                standard deviations should be used or initialized later. Defaults to None.
            workers (Optional[int]): The number of worker threads or processes to use
                during parallelization. Defaults to None.
        """
        Optimizer.__init__(self, max_evaluations, 'cma_sep')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike]] = 0.3, 
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given objective function using the CMA-ES optimization algorithm.

        This method leverages the Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
        to optimize the given function, which is useful for solving non-linear and non-convex
        optimization problems. It supports bounds constraints and parallel execution to
        accelerate the optimization process.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to minimize. It takes
                an input array and outputs a float representing the function value.
            bounds (Optional[Bounds]): The parameter bounds for the optimization. Must be
                of type `Bounds` with lower and upper limits defined.
            guess (Optional[ArrayLike]): Optional initial guess for the starting point of the
                optimization. If not provided, it defaults to `self.guess` or samples randomly
                from the bounds.
            sdevs (Optional[Union[float, ArrayLike]]): Standard deviation(s) for the initial
                search distribution. Can be a float or an array-like structure. Default is 0.3.
            rg (Optional[Generator]): Optional random number generator to ensure consistency
                in sampling. Defaults to a Generator instance with a PCG64DXSM source.
            store (Optional): Unspecified storage parameter for additional usage or output.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing:
                - np.ndarray: The best solution vector found during optimization.
                - float: The corresponding minimum function value.
                - int: The total number of evaluations performed.
        """
        try:
            import cmaes
        except ImportError as e:
            raise ImportError("Please install cmaes (pip install cmaes)") 

        if guess is None:
            guess = self.guess
        if guess is None:    
            guess = rg.uniform(bounds.lb, bounds.ub)
        bds = np.array([t for t in zip(bounds.lb, bounds.ub)])
        seed = int(rg.uniform(0, 2**32 - 1))
        optimizer = cmaes.SepCMA(mean=guess, sigma=np.mean(sdevs), bounds=bds, seed=seed, population_size=self.popsize)
        best_y = np.inf
        evals = 0
        fun = serial(fun) if (self.workers is None or self.workers <= 1) else parallel(fun, self.workers)  
        while evals < self.max_evaluations and not optimizer.should_stop():
            xs = [optimizer.ask() for _ in range(optimizer.population_size)]
            ys = fun(xs)
            solutions = []
            for i in range(optimizer.population_size):
                x = xs[i]
                y = ys[i]
                solutions.append((x, y))
                if y < best_y:
                    best_y = y
                    best_x = x
            optimizer.tell(solutions)
            evals += optimizer.population_size          
        if isinstance(fun, parallel):
            fun.stop()
        return best_x, best_y, evals
      
class De_cpp(Optimizer):
    """
    Differential Evolution optimizer using C++ backend.

    This class implements a Differential Evolution (DE) optimization algorithm.
    It interfaces with a C++ backend to efficiently optimize a given objective
    function within specified boundaries. The optimizer supports several
    parameters such as population size, crossover rate, mutation factor, and
    others to control the optimization process.

    Attributes:
        popsize (int, optional): Size of the population for the DE algorithm.
        guess (ArrayLike, optional): Initial guess for the optimization variables.
        stop_fitness (float): Fitness value at which optimization will stop early.
        keep (int): Number of top solutions to retain during certain steps.
        f (float): Differential weight or mutation factor in the DE algorithm.
        cr (float): Crossover probability in the DE algorithm.
        ints (ArrayLike, optional): Specifies which variables are integers, if any.
        workers (int, optional): Number of workers for parallel evaluation.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = None, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 keep: Optional[int] = 200, 
                 f: Optional[float] = 0.5, 
                 cr: Optional[float] = 0.9, 
                 ints: Optional[ArrayLike] = None, 
                 workers: Optional[int] = None):
        """
        Initializes the differential evolution optimizer with the given parameters.

        This optimizer adapts the DE/rand/1/bin strategy and performs optimization
        by iteratively improving the candidate solutions based on differential
        evolution algorithm parameters and constraints.

        Args:
            max_evaluations: Optional; Maximum number of function evaluations
                to perform before terminating.
            popsize: Optional; Population size of the candidate solutions.
            guess: Optional; Initial guess or starting point for the optimization.
            stop_fitness: Optional; If achieved, optimization stops early. Defaults
                to negative infinity.
            keep: Optional; Number of top individuals to retain for the next
                generation.
            f: Optional; Differential weight factor for scaling the mutation
                vector. Defaults to 0.5.
            cr: Optional; Crossover probability for recombination. Defaults
                to 0.9.
            ints: Optional; Specifies whether the variables being optimized
                should be treated as integers or not.
            workers: Optional; Number of workers for parallel processing during
                optimization.
        """
        Optimizer.__init__(self, max_evaluations, 'de cpp')
        self.popsize = popsize
        self.guess = guess
        self.stop_fitness = stop_fitness
        self.keep = keep
        self.f = f
        self.cr = cr
        self.ints = ints
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None,  # ignored
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given objective function using a differential evolution optimization
        algorithm. This method attempts to find the global minimum of the objective
        function within a bounded search space.

        Args:
            fun (Callable[[ArrayLike], float]): Objective function to be minimized.
                It takes an array-like input and returns a float as the function's
                value at that input point.
            bounds (Optional[Bounds]): An optional boundary specification for the
                optimization. Defines the search space for the optimization process.
            guess (Optional[ArrayLike]): Initial guess for the optimization algorithm,
                if provided. If not supplied, fallback to the default guess.
            sdevs (Optional[float]): Standard deviations for the process. This
                parameter is ignored in the implementation.
            rg (Optional[Generator]): Random number generator to be used during the
                optimization. Defaults to Generator(PCG64DXSM()).
            store: An optional parameter to store additional computation details, if
                necessary.

        Returns:
            Tuple[numpy.ndarray, float, int]: A tuple containing the optimized parameter set,
                the objective function's value at the optimized parameters (fitness),
                and the total number of function evaluations performed during the
                optimization process.
        """
        if guess is None:
            guess = self.guess
            
        ret = decpp.minimize(fun, None, bounds, 
                popsize=self.popsize, 
                max_evaluations = self.max_eval_num(store), 
                stop_fitness = self.stop_fitness,
                keep = self.keep, f = self.f, cr = self.cr, ints=self.ints,
                rg=rg, runid = self.get_count_runs(store), 
                workers = self.workers, x0 = guess)
        return ret.x, ret.fun, ret.nfev

class De_python(Optimizer):
    """
    Differential Evolution (DE) optimizer implemented in Python.

    This class provides an implementation of the Differential Evolution optimization
    algorithm. DE is a stochastic, population-based optimization algorithm suitable
    for solving complex optimization problems, often used when no gradient information
    is available. This implementation supports parallelism and configurable constraints.

    Attributes:
        popsize (int, optional): The population size, i.e., the number of candidate
            solutions evaluated in each generation.
        stop_fitness (float, optional): The fitness value at which the optimization
            stops if reached. Default is negative infinity.
        keep (int, optional): The number of best-performing individuals retained
            between generations during optimization.
        f (float, optional): The mutation factor, controlling the amplification of
            differential variations. Ranges typically between 0 and 2.
        cr (float, optional): The crossover probability, indicating the probability
            of recombination. Ranges typically between 0 and 1.
        ints (ArrayLike, optional): Indices indicating which variables to constrain
            to integer values.
        workers (int, optional): The number of worker threads used for parallel
            evaluation of the population.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = None,
                 stop_fitness: Optional[float] = -np.inf,
                 keep: Optional[int] = 200, 
                 f: Optional[float] = 0.5, 
                 cr: Optional[float] = 0.9, 
                 ints: Optional[ArrayLike] = None, 
                 workers: Optional[int] = None):
        """
        Initializes an instance of the Differential Evolution (DE) optimizer with the given parameters.

        This class is a specialized optimizer extending the base `Optimizer` functionality and is used
        to perform optimization using the DE algorithm. The parameters configure the specifics of the
        optimization process such as population size, mutation and crossover probabilities, stopping
        criteria, and other optional configurations.

        Args:
            max_evaluations (Optional[int]): Maximum number of evaluations to perform during the optimization
                process. Default is 50000.
            popsize (Optional[int]): Size of the population used in the DE algorithm. If not provided, it is
                automatically determined based on the problem configuration.
            stop_fitness (Optional[float]): The threshold for the fitness value to stop the optimization run.
                Defaults to -np.inf, meaning the optimization will continue until max evaluations are exhausted.
            keep (Optional[int]): Number of best solutions to retain between iterations. Default is 200.
            f (Optional[float]): Differential weight used in the mutation process. Default is 0.5.
            cr (Optional[float]): Crossover rate probability used in the recombination step. Default is 0.9.
            ints (Optional[ArrayLike]): Array-like object specifying whether certain decision variables in
                the optimization problem should be treated as integers. Default is None.
            workers (Optional[int]): Number of parallel workers or threads used for computation. Default is None,
                which uses the system's available resources.

        """
        Optimizer.__init__(self, max_evaluations, 'de py')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.keep = keep
        self.f = f
        self.cr = cr
        self.ints = ints
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes the given objective function within the specified bounds using
        differential evolution optimization.

        This function uses a differential evolution optimization algorithm to find
        the minimum of a given objective function. Users can specify initial
        parameters such as bounds, population size, and maximum evaluations.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to minimize. This
                function should accept an array-like input and return a float.
            bounds (Optional[Bounds]): The bounds within which the optimization is
                performed. Each dimension should have a lower and upper bound defined.
            guess (Optional[ArrayLike]): An optional initial guess for the optimization.
            rg (Optional[Generator]): A random number generator to control stochastic
                elements of the optimization. Default is Generator(PCG64DXSM()).
            store: A mechanism to store intermediate results or progress during the
                optimization. This is passed directly but its internal behavior
                is not described here.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing:
                - The best set of parameters found during optimization (np.ndarray).
                - The objective function value corresponding to the best parameters
                  (float).
                - The number of function evaluations performed (int).
        """
        ret = de.minimize(fun, None, 
                bounds, self.popsize, self.max_eval_num(store),
                stop_fitness = self.stop_fitness,
                keep = self.keep, f = self.f, cr = self.cr, ints=self.ints,
                rg=rg, workers = self.workers)
        return ret.x, ret.fun, ret.nfev

class Cma_ask_tell(Optimizer):
    """CMA-ES Ask-Tell based optimizer.

    This class implements the Covariance Matrix Adaptation Evolution Strategy
    (CMA-ES) using an ask-tell interface for optimization tasks. It is designed
    to minimize objective functions over specified domains with control over
    population size and stopping criteria. The optimizer maintains an iterative
    mechanism for asking candidate solutions, evaluating their quality, and
    updating its models to converge to an optimal solution.

    The optimizer integrates seamlessly with the CMA-ES library and allows for
    various customization options like user-defined initial guesses, population
    sizes, and stopping fitness.

    Attributes:
        popsize (int): Size of the population used in the CMA-ES optimization.
        stop_fitness (float): Threshold for stopping if the fitness value
            reaches or exceeds this value.
        guess (Optional[ArrayLike]): Initial guess for the optimization
            process. Defaults to None.
        sdevs (Optional[float]): Standard deviation for the initial sampling.
            If None, defaults are used.
    """
    
    def __init__(self, max_evaluations=50000,
                 popsize = 31, guess=None, stop_fitness = -np.inf, sdevs = None):
        """
        Initializes the CMA-ES optimizer with specified configurations.

        This constructor initializes an instance of the CMA-ES optimizer with user-defined
        or default values. It sets the number of maximum evaluations, population size,
        initial guess, stop fitness value, and standard deviations for the optimizer to
        function as per the provided or default arguments.

        Args:
            max_evaluations: int
                The maximum number of evaluations the optimizer is allowed to perform.
            popsize: int
                The population size indicating the number of individuals per generation.
            guess: Optional[float]
                An optional initial guess for the initial search point.
            stop_fitness: float
                The stop criterion based on achieving the target fitness value.
            sdevs: Optional[float]
                An optional standard deviation array for initializing the optimizer.
        """
        Optimizer.__init__(self, max_evaluations, 'cma at')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given objective function within specified bounds using CMA-ES optimization algorithm.

        This method applies the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to find
        the minimum of a given objective function. The optimization process involves multiple
        iterations where candidate solutions are generated, evaluated, and subsequently refined
        until a stopping criterion is met.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to minimize. This
                function should take an array-like input and return a scalar value as the
                function's output.
            bounds (Optional[Bounds]): The boundary constraints for the optimization. Specifies
                the lower and upper bounds for the input values.
            guess (Optional[ArrayLike], optional): An initial guess for the solution. If not
                specified, the optimizer will choose an initial value automatically.
            sdevs (Optional[float], optional): Standard deviations for the CMA-ES algorithm.
                Ignored if already specified in the class instance. Defaults to None.
            rg (Optional[Generator], optional): A random number generator to use for sampling
                in the CMA-ES algorithm. Defaults to `Generator(PCG64DXSM())`.
            store: A storage mechanism to track evaluations and optimization states.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing the best solution found as an
                array (`es.best_x`), its corresponding objective function value (`es.best_value`),
                and the total number of function evaluations performed (`evals`).
        """
        es = cmaes.Cmaes(bounds,
                popsize = self.popsize, 
                input_sigma = self.sdevs if not self.sdevs is None else sdevs, 
                rg = rg)       
        iters = self.max_eval_num(store) // self.popsize
        evals = 0
        for j in range(iters):
            xs = es.ask()
            ys = [fun(x) for x in xs]
            evals += len(xs)
            stop = es.tell(ys)
            if stop != 0:
                break 
        return es.best_x, es.best_value, evals

class De_ask_tell(Optimizer):
    """
    Differential Evolution optimizer for numerical optimization tasks.

    This class implements the Differential Evolution (DE) optimizer, which is
    a population-based, stochastic optimization technique suitable for solving
    complex continuous optimization problems. The optimizer uses operations
    like mutation, crossover, and selection to evolve a population of
    candidate solutions towards an optimal solution over a number of iterations.

    Attributes:
        popsize (Optional[int]): Number of individuals in the population, which
            defines the optimization's search capacity.
        stop_fitness (Optional[float]): Fitness threshold for early stopping
            if an individual in the population achieves this value.
        keep (Optional[int]): Number of top-performing individuals retained for
            the next generation to stabilize the evolution process.
        f (Optional[float]): Differential weighting factor for mutation, controlling
            the scaling of the difference vectors.
        cr (Optional[float]): Crossover probability that determines the rate at which
            components are exchanged between solutions.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = None,
                 stop_fitness: Optional[float] = -np.inf,
                 keep: Optional[int] = 200, 
                 f: Optional[float] = 0.5, 
                 cr: Optional[float] = 0.9):
        """
        Initializes the Differential Evolution (DE) optimizer with specified configuration options.
        The DE optimizer is part of evolutionary algorithms and works by iteratively improving a
        candidate solution with regard to a measure of quality (fitness). This initialization
        method allows setting the core parameters of the DE optimizer.

        Args:
            max_evaluations (Optional[int]): Maximum number of evaluations that the optimizer
                can perform before stopping. Default is 50000.
            popsize (Optional[int]): Number of individuals in the population. Determines the
                population size for optimization. Default is None (uses a default or dynamically
                calculated value).
            stop_fitness (Optional[float]): Fitness value at which the optimizer will stop early
                if reached. Default is -np.inf (no early stopping based on fitness).
            keep (Optional[int]): Number of individuals or solutions to keep from one generation
                to the next within the optimization process. Default is 200.
            f (Optional[float]): Differential weighting factor, a scaling constant used to scale
                the difference between two individuals in the population. Default is 0.5.
            cr (Optional[float]): Crossover probability, the rate at which components are
                swapped between solutions during the recombination process. Default is 0.9.
        """
        Optimizer.__init__(self, max_evaluations, 'de at')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.keep = keep
        self.f = f
        self.cr = cr

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """Minimizes a given objective function using a Differential Evolution (DE) algorithm.

        The algorithm works by iteratively evolving a population of potential solutions to
        optimize the given objective function within specified bounds. It supports features
        like random seeds, population size customization, and optional constraint handling.
        The `minimize` method evaluates the convergence of the algorithm based on specified
        maximum evaluations, and returns the best solution, its value, and the number of
        evaluations conducted.

        Args:
            fun (Callable[[ArrayLike], float]): Objective function to minimize. This is a callable
                that takes an array-like input and returns a scalar output representing the
                function value.
            bounds (Optional[Bounds]): Boundary constraints for the search space. Specifies the
                lower and upper bounds for each dimension of the input array.
            guess (Optional[ArrayLike]): Initial guess for the solution. Optional and can be left
                as None, in which case the default initialization strategy is used.
            sdevs (Optional[float]): Placeholder parameter. This argument is ignored in the optimization
                process.
            rg (Optional[Generator]): Random generator for stochastic operations. By default,
                it uses a PCG64DXSM generator instance.
            store (object): Optional storage handler for tracking or persisting evaluation metrics
                during the optimization process. This parameter is passed to internal utility
                methods for handling storage requirements.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing three elements:
                - np.ndarray: The best solution found by the optimizer (array with dimensions
                  defined by the bounds).
                - float: The value of the objective function at the best solution.
                - int: The total number of function evaluations performed.

        """
        dim = len(bounds.lb)
        popsize = 31 if self.popsize is None else self.popsize
        es = de.DE(dim, bounds, popsize = popsize, rg = rg, keep = self.keep, F = self.f, Cr = self.cr)  
        es.fun = fun  #remove
        max_evals = self.max_eval_num(store)
        while es.evals < max_evals:
            xs = es.ask()
            ys = [fun(x) for x in xs]
            stop = es.tell(ys, xs)
            if stop != 0:
                break 
        return es.best_x, es.best_value, es.evals

class random_search(Optimizer):
    """
    A class for performing optimization using a random search strategy.

    This class implements a random search optimization algorithm, which
    is a straightforward method that randomly samples candidate solutions
    within the specified bounds to minimize the objective function. It is
    useful for exploring high-dimensional or complicated search spaces
    and does not rely on gradient information.

    Attributes:
        max_evaluations (int): The maximum number of evaluations for the optimization
            process. Determines the computational budget for the search.
    """
   
    def __init__(self, max_evaluations=50000):
        """
        Initializes an instance of the optimizer with a random search strategy.

        Args:
            max_evaluations (int, optional): The maximum number of evaluations for
                the optimization process. Defaults to 50000.
        """
        Optimizer.__init__(self, max_evaluations, 'random')
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg: Optional[Generator] = Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given function within the specified bounds using a random sampling
        method. The method repeatedly samples points within the bounds, evaluates the
        function at those points, and keeps track of the point with the minimum function value.

        Args:
            fun (Callable[[ArrayLike], float]): Objective function to be minimized.
            bounds (Optional[Bounds]): Optimization bounds specifying the lower (lb)
                and upper (ub) limits for the variables.
            guess (Optional[ArrayLike]): Initial guess for the solution (ignored by
                this implementation).
            sdevs (Optional[float]): Not used in the implementation (ignored).
            rg (Optional[Generator]): Random number generator to sample
                points within bounds, defaults to `Generator(PCG64DXSM())`.
            store: Auxiliary data structure to track the state or additional
                information (details not specified).

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing the best input point
            found (x_min), the minimum function value (y_min), and the total number
            of function evaluations performed (evals).

        Raises:
            None explicitly raised, but function evaluations or sampling may fail
            due to improper configuration or input.
        """
        dim, x_min, y_min = len(bounds.lb), None, None
        max_chunk_size = 1 + 4e4 / dim
        evals = self.max_eval_num(store)
        budget = evals
        while budget > 0:
            chunk = int(max([1, min([budget, max_chunk_size])]))
            X = rg.uniform(bounds.lb, bounds.ub, size = [chunk, dim])
            F = [fun(x) for x in X]
            index = np.argmin(F) if len(F) else None
            if index is not None and (y_min is None or F[index] < y_min):
                x_min, y_min = X[index], F[index]
            budget -= chunk
        return x_min, y_min, evals

    
class Da_cpp(Optimizer):
    """
    Represents the Da_cpp optimization algorithm, a variant of the Optimizer class.

    This class provides an interface to a differential evolution-based optimization
    algorithm implemented in the `dacpp` library. It allows customization of certain
    parameters such as the maximum number of evaluations, stopping fitness criteria,
    and whether local search should be applied. The optimizer can be used to find the
    minimum of a given objective function within specified bounds.

    Attributes:
        stop_fitness (float): Defines the stopping fitness value. If the objective
            function achieves a fitness less than or equal to this value, the
            optimization is terminated.
        use_local_search (bool): Indicates whether local search should be utilized
            during the optimization process to refine results.
        guess (ArrayLike or None): Provides an optional initial guess for the
            optimization. If unspecified, no initial guess is used.
    """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 stop_fitness: Optional[float] = -np.inf,
                 use_local_search: Optional[bool] = True,
                 guess: Optional[ArrayLike] = None):
        """
        Initializes the optimizer with given parameters and defaults.

        This constructor sets up the optimizer with options for a maximum number of
        evaluations, a stopping fitness threshold, a toggle for local search, and
        an optional initial guess. These parameters guide the optimizer's behavior
        during its execution.

        Args:
            max_evaluations: Maximum number of evaluations permissible for the
                optimization process. Defaults to 50000.
            stop_fitness: Fitness value at which the optimization process should
                stop. Defaults to negative infinity.
            use_local_search: Boolean indicating whether to enable local search.
                Defaults to True.
            guess: Initial guess or starting point for the optimization process.
                Can be provided as an array-like structure. Defaults to None.
        """
        Optimizer.__init__(self, max_evaluations, 'da cpp',)
        self.stop_fitness = stop_fitness
        self.use_local_search = use_local_search
        self.guess = guess
 
    def minimize(self, 
                fun: Callable[[ArrayLike], float], 
                bounds: Optional[Bounds], 
                guess: Optional[ArrayLike] = None, 
                sdevs: Optional[float] = None, # ignored
                rg=Generator(PCG64DXSM()), 
                store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes an objective function within given bounds using a stochastic optimization
        method.

        This function attempts to find the minimum of the provided objective function by
        exploring the parameter space defined by the given bounds. An optional initial guess
        can be provided to start the optimization at a specific point. The function uses a
        random number generator for stochasticity during optimization. The result includes
        the optimal point, the optimal function value, and the number of function evaluations.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to minimize.
                It must take an array-like input and return a float.
            bounds (Optional[Bounds]): The bounds for the variables as a Bounds object. Defines
                the search space for the optimization.
            guess (Optional[ArrayLike]): An optional initial guess for the optimization. If not
                provided, a default guess specified by the class will be used.
            sdevs (Optional[float]): Standard deviation for some internal process. Ignored in
                the current implementation.
            rg: A random number generator instance for stochasticity during the optimization.
            store: Storage or logging object used to keep track of evaluation counts or other
                optimization parameters.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing:
                - The best solution found (as a numpy.ndarray).
                - The objective function value at the best solution.
                - The total number of function evaluations performed.
        """
        ret = dacpp.minimize(fun, bounds, 
                             self.guess if guess is None else guess,
                            max_evaluations = self.max_eval_num(store), 
                            use_local_search = self.use_local_search,
                            rg=rg, runid = self.get_count_runs(store))
        return ret.x, ret.fun, ret.nfev

class Bite_cpp(Optimizer):
    """
    A hybrid optimization algorithm (BITE++ algorithm) developed for solving
    complex optimization problems.

    The `Bite_cpp` class is derived from the `Optimizer` base class and implements
    a specific optimization algorithm. It is especially suitable for multi-variable
    optimization tasks and allows fine control over the algorithm's behavior through
    configurable parameters. This class serves as an interface to the underlying
    optimization algorithm implemented in the BITE++ library.

    Attributes:
        guess (ArrayLike): Initial guess for the optimization process. If None,
            the algorithm will start with random initialization.
        stop_fitness (float): Target fitness value at which the optimization
            process stops. Default is negative infinity.
        M (int): Optional parameter for algorithm-specific settings. Defaults to 1
            if not explicitly defined.
        popsize (int): Population size used by the optimization algorithm. Defaults
            to 0 if not explicitly configured.
        stall_criterion (int): Number of iterations over which lack of improvement
            halts the optimization process. Defaults to 0.
    """
   
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,                
                 M: Optional[int] = None,
                 popsize: Optional[int] = None,
                 stall_criterion: Optional[int] = None):
        """
        Initializes the optimization algorithm with parameters specific to the optimization process.

        Args:
            max_evaluations: Optional; Maximum number of evaluations allowed for the optimizer to perform. Default is 50000.
            guess: Optional; Initial guess for the optimization process. Default is None.
            stop_fitness: Optional; Fitness value at which to stop the optimization. Default is negative infinity (-np.inf).
            M: Optional; Parameter related to the optimization configuration. Default value is 1 if not provided.
            popsize: Optional; Population size for the optimization process. Default value is 0 if not provided.
            stall_criterion: Optional; Criterion related to the stall condition in optimization. Default value is 0 if not provided.
        """
        Optimizer.__init__(self, max_evaluations, 'bite cpp')
        self.guess = guess
        self.stop_fitness = stop_fitness
        self.M = 1 if M is None else M 
        self.popsize = 0 if popsize is None else popsize 
        self.stall_criterion = 0 if stall_criterion is None else stall_criterion 

    def minimize(self,
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a scalar function of multiple variables using a given optimization algorithm.

        This method applies an algorithm to minimize the objective function `fun` within the
        specified bounds. It optionally uses an initial guess for the solution, random generator
        for sampling, and allows storing intermediate results. The function is particularly
        designed to handle optimization tasks efficiently with various configuration options.

        Args:
            fun (Callable[[ArrayLike], float]): Objective function to be minimized.
                It should accept an array-like input and return a float value.
            bounds (Optional[Bounds]): Bounds for the optimization variables.
                Specifies the range of the variables during optimization.
            guess (Optional[ArrayLike]): Initial guess for the solution. If None,
                the default guess will be used.
            sdevs (Optional[float]): Ignored. Value provided to this parameter will have no effect.
            rg: Random generator instance for sampling during the optimization process.
                Default is Generator(PCG64DXSM()).
            store: Optional storage for storing intermediate results such as run history or
                diagnostics information.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing:
                - The optimized solution as a NumPy array.
                - The value of the objective function at the optimized solution.
                - The total number of function evaluations performed.

        """
        ret = bitecpp.minimize(fun, bounds, 
                self.guess if guess is None else guess,
                max_evaluations = self.max_eval_num(store), 
                stop_fitness = self.stop_fitness, M = self.M, popsize = self.popsize, 
                stall_criterion = self.stall_criterion,
                rg=rg, runid = self.get_count_runs(store))     
        return ret.x, ret.fun, ret.nfev
        
class Dual_annealing(Optimizer):
    """Dual annealing optimization algorithm implementation.

    This class facilitates performing optimization using the dual-annealing
    algorithm, as implemented in SciPy. It provides functionality for finding
    the global minimum of a function within given bounds. Local search can also
    be enabled or disabled for refinement after the global optimization phase.

    Attributes:
        max_evaluations (Optional[int]): Maximum number of function evaluations
            allowed during optimization.
        no_local_search (bool): A flag indicating whether local search is disabled.
            If True, local search is not applied.
    """
 
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 use_local_search: Optional[bool] = True):
        """
        Initializes an optimizer instance that uses the scipy differential annealing
        algorithm. This class inherits from the base `Optimizer` class.

        It allows customization of the maximum number of evaluations and whether
        local search should be employed as part of the optimization process.

        Args:
            max_evaluations: Optional; maximum number of evaluations allowed during
                optimization. Defaults to 50000 if not provided.
            use_local_search: Optional; flag indicating whether to use local search
                during optimization. Defaults to True if not provided.
        """
        Optimizer.__init__(self, max_evaluations, 'scipy da')
        self.no_local_search = not use_local_search
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a scalar function of one or more variables using the Dual Annealing
        optimization method.

        This method minimizes an objective function over a domain defined by bounds
        and an optional initial guess. It allows specifying a random number generator
        and provides an option for storing intermediate results. The optimization is
        done using the Dual Annealing algorithm, which is particularly effective for
        global optimization problems.

        Args:
            fun: Callable function to minimize. It must accept a single argument, which
                is the optimization variable(s), and return a scalar value.
            bounds: Bounds object defining the lower and upper limits for each
                optimization variable. The bounds must be specified for all dimensions.
            guess: Optional initial guess for the optimization variables. It provides
                a starting point for the optimization algorithm.
            sdevs: Optional float intended for standard deviations. This parameter is
                currently ignored in the implementation.
            rg: Random number generator instance used for seeding the optimization
                process to ensure reproducibility.
            store: Optional storage used to manage intermediate results or post-process
                optimization details. If provided, results may be customized based on
                storage behavior.

        Returns:
            Tuple containing:
                - The optimized variable(s) `np.ndarray`, representing the location of
                  the minimum.
                - The scalar value of the objective function at the minimum.
                - The total number of function evaluations performed during optimization.
        """
        ret = dual_annealing(fun, bounds=list(zip(bounds.lb, bounds.ub)),
            maxfun = self.max_eval_num(store), 
            no_local_search = self.no_local_search,
            x0=guess,
            seed = int(rg.uniform(0, 2**32 - 1)))
        return ret.x, ret.fun, ret.nfev

class Differential_evolution(Optimizer):
    """A class for performing optimization using the Differential Evolution algorithm.

    Differential Evolution is a global optimization algorithm suited for optimizing
    real-valued, multi-modal functions. This class is built upon the `scipy.optimize`
    implementation and extends it with additional functionality specific to its purpose.

    Attributes:
        popsize (int): The population size used by the Differential Evolution algorithm.
    """
 
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31):
        """
        Initializes an instance of the DE optimizer.

        The optimizer class is responsible for configuring and managing settings
        specific to the differential evolution (DE) algorithm. Upon initialization,
        it sets up the maximum evaluations allowed and the population size for the
        DE routine.

        Args:
            max_evaluations: Optional; The maximum number of evaluations allotted for
                the DE optimization algorithm. Defaults to 50000 if not provided.
            popsize: Optional; The population size used by the DE algorithm. Defaults
                to 31 if not provided.
        """
        Optimizer.__init__(self, max_evaluations, 'scipy de')
        self.popsize = popsize
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given objective function using the Differential Evolution algorithm.

        The method optimizes a given objective function within specified bounds, leveraging
        stochastic parallel computation to approximate a global minimum. It uses a population-based
        approach evolved through random sampling.

        Args:
            fun (Callable[[ArrayLike], float]): Objective function to minimize. It takes an array-like
                input and returns a float output representing the objective to minimize.
            bounds (Optional[Bounds]): Constraints for the optimization. The lower (`bounds.lb`) and
                upper (`bounds.ub`) bounds define the search space for the optimization algorithm.
            guess (Optional[ArrayLike]): Initial guess for the starting position in the parameter space.
                Defaults to None.
            rg (Generator): Random number generator instance. Determines random seed for reproducibility.
                Default is `Generator(PCG64DXSM())`.
            store: Optional storage object to record evaluation history or intermediate results.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing:
                - The best solution (ndarray) found by the algorithm.
                - The value of the objective function at the found solution (float).
                - The number of function evaluations performed (int).
        """
        popsize = self.popsize 
        maxiter = int(self.max_eval_num(store) / (popsize * len(bounds.lb)) - 1)
        ret = differential_evolution(fun, bounds=bounds, maxiter=maxiter,
                                      seed = int(rg.uniform(0, 2**32 - 1)))
        return ret.x, ret.fun, ret.nfev

class CheckBounds(object):
    """
    Validates whether a given set of values falls within specified bounds.

    This class is designed to check if a new set of parameters provided fits
    within the upper and lower bounds defined. It can be called with keyword
    arguments to perform the validation. Primarily useful in optimization
    problems or parameter validations where constraints on variable limits are
    required.

    Attributes:
        bounds (Bounds): The boundary constraints, containing the upper (`ub`)
            and lower (`lb`) bounds used for validation.
    """
    def __init__(self, bounds: Bounds):
        """
        Represents an object with defined bounds.

        This class encapsulates an object that requires bounds, likely for geometric
        or spatial calculations. It is initialized with a Bounds instance.
        The Bounds parameter describes the necessary spatial boundary details to
        construct the object.

        Attributes:
            bounds: The bounds object specifying spatial or defined boundaries.
        """
        self.bounds = bounds
        
    def __call__(self, **kwargs):
        """
        Checks if the given 'x_new' value falls within the defined bounds.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            bool: True if 'x_new' is within bounds, False otherwise.
        """
        x = kwargs["x_new"]
        inb =  np.less_equal(x, self.bounds.ub).all() and \
            np.greater_equal(x, self.bounds.lb).all()
        return inb

class Basin_hopping(Optimizer):
    """
    Optimization class implementing the basin hopping algorithm from SciPy.

    This class is used to solve optimization problems using the basin hopping
    algorithm, which is a global optimization technique. It combines random
    perturbation of the input parameters with local optimization, making it
    suitable for finding global minimums in problems with multiple local
    minima. The optimizer stops after a maximum number of evaluations or when
    a suitable solution is found.

    Attributes:
        max_evaluations (int): Maximum number of function evaluations allowed
            for the optimizer.
        name (str): Name of the optimization algorithm being used.
    """
 
    def __init__(self, max_evaluations=50000, store=None):
        """
        Initializes the optimizer, which employs the SciPy basin-hopping algorithm to perform optimization.

        This constructor sets up the optimizer with the maximum allowable evaluations and an optional storage system
        for retaining optimization state or results. The optimization process employs a global optimization algorithm
        that effectively navigates through a rough landscape to find the global minimum.

        Args:
            max_evaluations: int
                The maximum number of iterations or evaluations the optimizer is allowed to perform.

            store: Optional
                An optional storage mechanism to save or track the state/results of the optimization process.

        """
        Optimizer.__init__(self, max_evaluations, 'scipy basin hopping')
         
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given objective function using the basinhopping optimization
        algorithm with local search.

        This method performs optimization by generating random starting points
        within the specified bounds and repeatedly applies local search algorithms
        to find the minimum of the function.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to minimize.
                It must accept an input array and return a scalar value as output.
            bounds (Optional[Bounds]): The bounds or constraints on the optimization
                problem. This should define the lower and upper limits for each
                dimension of the input.
            guess (Optional[ArrayLike]): The optional initial guess for the starting
                point of the optimization. If not provided, a uniform random starting
                point within the bounds is generated.
            sdevs (Optional[float]): Ignored parameter for compatibility purposes.
            rg (Generator): A random number generator for providing random points
                within the bounds during the optimization process.
            store: Additional storage object or parameter, used for handling the
                maximum allowable evaluations during the optimization.

        Returns:
            Tuple[np.ndarray, float, int]: Returns a tuple containing:
                - The array of input values (`np.ndarray`) that minimize the objective
                  function.
                - The minimum objective value (`float`).
                - The number of function evaluations performed (`int`).
        """
        localevals = 200
        maxiter = int(self.max_eval_num(store) / localevals)         
        if guess is None:
            guess = rg.uniform(bounds.lb, bounds.ub)
        
        ret = basinhopping(fun, guess, niter=maxiter, 
                           minimizer_kwargs={"method": 'SLSQP', 
                                             "bounds":bounds},
                           accept_test=CheckBounds(bounds),
                           seed = int(rg.uniform(0, 2**32 - 1)))
        return ret.x, ret.fun, ret.nfev

class Minimize(Optimizer):
    """Optimization utilizing the scipy minimize function.

    This class is designed to perform optimization tasks using the scipy
    minimize method. It allows for specifying bounds, initial guesses,
    and handles optimization with lower-level control.

    Attributes:
        max_evaluations (int): Maximum number of function evaluations
            allowed during optimization.
        name (str): Name of the optimizer, set to 'scipy minimize'.
    """
 
    def __init__(self, max_evaluations=50000, store=None):
        """
        Initializes an optimizer using the scipy minimize method with a specified maximum number
        of evaluations and an optional store for collected data.

        Args:
            max_evaluations: int, optional
                The maximum number of evaluations allowed during optimization. Defaults to 50000.
            store: Any, optional
                Optional storage for collected data during optimization. Defaults to None.
        """
        Optimizer.__init__(self, max_evaluations, 'scipy minimize')
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given function within specified bounds and optionally using an initial
        guess.

        This method uses a specified random generator to generate an initial guess if none
        is provided. The optimization process is carried out within the provided bounds,
        and the optimal solution, objective function value at the optimum,
        and the number of function evaluations are returned.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to be minimized.
                Accepts an array-like input and returns a floating-point value.
            bounds (Optional[Bounds]): The boundary constraints for the optimization
                variables in the form of lower and upper bounds.
            guess (Optional[ArrayLike], optional): The initial guess for the optimization,
                if provided. If None, a random value within bounds will be generated.
                Defaults to None.
            sdevs (Optional[float], optional): Ignored.
            rg (Generator): A random generator for creating random numbers, used if no
                guess is provided. Defaults to Generator(PCG64DXSM()).
            store: Reserved for potential future usage. Defaults to None.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing:
                - The optimal solution as a NumPy array.
                - The objective function value at the optimum.
                - The number of function evaluations during the optimization.
        """
        if guess is None:
            guess = rg.uniform(bounds.lb, bounds.ub)
        ret = minimize(fun, x0=guess, bounds=bounds)
        return ret.x, ret.fun, ret.nfev
 
class Shgo(Optimizer):
    """Shgo optimizer for mathematical function minimization.

    This class leverages the scipy `shgo` optimization algorithm to minimize
    a given mathematical function within a specified set of bounds. It is
    particularly suitable for global optimization problems.

    The optimizer evaluates the function iteratively and searches for its
    minimum value, returning the optimal solution, the function's value at
    the solution, and the number of evaluations performed.

    Attributes:
        max_evaluations (int): Maximum number of function evaluations allowed
            in the optimization process.
    """

    def __init__(self, max_evaluations=50000, store=None):
        """
        Initializes a Scipy SHGO (Simplicial Homology Global Optimization) optimizer.

        This constructor sets up the optimizer with a maximum number of evaluations
        and an optional storage to manage optimization runs. It leverages the
        `Optimizer` base class for initialization and specifies the use of the
        SHGO algorithm.

        Args:
            max_evaluations: int
                The maximum number of function evaluations allowed during the
                optimization process. Defaults to 50000.
            store: object or None
                An optional parameter to store or manage optimization results.
                If not provided, defaults to None.
        """
        Optimizer.__init__(self, max_evaluations, 'scipy shgo')
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes a given callable function within specified bounds using a particular optimization algorithm.

        This function leverages the SHGO (Simplicial Homology Global Optimization) algorithm to
        identify the minimum of a user-provided objective function over given bounds. The optimization
        process involves constraints, initial guesses, and additional configuration. The optimizer
        returns the minimum parameters, the minimum function value, and the number of objective
        function evaluations performed.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to minimize. It should be a
                callable that accepts an array-like input and returns a scalar.
            bounds (Optional[Bounds]): The bounds for the search space, typically containing lower
                and upper limits for each dimension of the input.
            guess (Optional[ArrayLike], optional): An optional array providing an initial guess for
                the optimization. Defaults to None.
            sdevs (Optional[float], optional): A parameter intended for standard deviations, currently
                ignored in this implementation. Defaults to None.
            rg (Generator, optional): A random number generator used for reproducible optimization
                processes. Defaults to a PCG64DXSM generator instance.
            store (optional): A storage option to track evaluation or state during the optimization
                process. The exact format and use depend on user implementation.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing the following:
                - np.ndarray: The parameter values at the minimum point.
                - float: The minimum value of the objective function.
                - int: The total number of function evaluations performed during optimization.
        """
        ret = shgo(fun, bounds=list(zip(bounds.lb, bounds.ub)), 
                   options={'maxfev': self.max_eval_num(store)})
        return ret.x, ret.fun, ret.nfev

class single_objective:
    """Represents a wrapper for a single-objective optimization problem.

    This class encapsulates a single-objective optimization problem provided
    via a Pagmo problem instance. It provides a fitness function for evaluating
    solutions and handles problem-specific details such as bounds.

    Attributes:
        pagmo_prob: Pagmo problem instance containing the optimization problem.
        name (str): A string representing the name of the optimization problem.
        fun (function): A function that evaluates the fitness of a solution.
        bounds (Bounds): An object defining the lower and upper bounds for the
            decision variables in the problem.
    """
      
    def __init__(self, pagmo_prob):
        """
        Initializes an instance of the class with properties derived from the provided pagmo
        problem.

        Args:
            pagmo_prob: A pagmo problem instance used to initialize the class. It is expected
                to contain methods `get_name()` and `get_bounds()`, which provide the name of
                the problem and its boundary values, respectively.

        Attributes:
            pagmo_prob: The provided pagmo problem instance.
            name: Name of the problem as retrieved from the `get_name` method of the
                provided pagmo problem instance.
            fun: Fitness function associated with the initialized class instance.
            bounds: Bounds object created using the lower and upper bounds of the problem as
                retrieved from the `get_bounds` method of the provided pagmo problem instance.
        """
        self.pagmo_prob = pagmo_prob
        self.name = pagmo_prob.get_name() 
        self.fun = self.fitness
        lb, ub = pagmo_prob.get_bounds()
        self.bounds = Bounds(lb, ub)
         
    def fitness(self,X):
        """
        Calculates the fitness value for a given input vector X using the problem's
        fitness function. If any exception occurs during the calculation, it returns
        the maximum floating-point value as a fallback.

        Args:
            X (list[float]): The input vector for which the fitness value is to be
                calculated.

        Returns:
            float: The fitness value calculated for the input vector X. Returns
                sys.float_info.max in case of an exception.
        """
        try:
            return self.pagmo_prob.fitness(X)[0]
        except Exception as ex:
            return sys.float_info.max

class NLopt(Optimizer):
    """
    Optimizer class utilizing NLopt algorithms for optimization tasks.

    This class is designed to perform optimization using the NLopt library,
    which provides algorithms for nonlinear optimization with constraints and bounds.
    The NLopt optimizer supports various optimization algorithms and is utilized
    through this wrapper to interact with objective functions, bounds, and initial
    guess settings.

    Attributes:
        algo: The NLopt algorithm object that defines the optimization algorithm to use.
    """

    def __init__(self, algo, max_evaluations=50000, store=None):
        """
        Initializes the optimizer with the specified algorithm, maximum evaluation limit, and storage option.

        Args:
            algo: Instance of the optimization algorithm to be used.
            max_evaluations: Maximum number of evaluations to perform during optimization.
                             Defaults to 50000.
            store: Optional parameter for storage of intermediate results. Defaults to
                   None.
        """
        Optimizer.__init__(self, max_evaluations, 'NLopt ' + algo.get_algorithm_name())
        self.algo = algo
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(PCG64DXSM()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        """
        Minimizes an objective function within given bounds using an optimizer object.

        This method leverages the optimization algorithm set on the optimizer object
        to minimize the provided objective function. It also allows setting initial
        guess values and configuration through the specified parameters.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function that takes an
                array-like input and returns a float as its value to be minimized.
            bounds (Optional[Bounds]): The bounds for the optimization. It defines
                the lower and upper limits for the parameters of the target function.
            guess (Optional[ArrayLike]): Initial guess values for the optimization
                parameters. If not provided, a random guess within bounds will be used.
            sdevs (Optional[float]): Standard deviation initial step size for the
                optimizer. This parameter may not always be utilized and is ignored
                in this method.
            rg (Generator): Random number generator instance for generating random
                initial guesses if `guess` is not provided.
            store (Any): Optional object to store information during the evaluation
                process. This parameter may affect the behavior of `max_eval_num`.

        Returns:
            Tuple[np.ndarray, float, int]: A tuple containing the optimal parameter
            values (array-like), the function value at the minimum, and the number
            of function evaluations performed.
        """
        self.fun = fun
        opt = self.algo
        opt.set_min_objective(self.nlfunc)
        opt.set_lower_bounds(bounds.lb)
        opt.set_upper_bounds(bounds.ub)
        opt.set_maxeval(self.max_eval_num(store))
        opt.set_initial_step(sdevs)
        if guess is None:
            guess = rg.uniform(bounds.lb, bounds.ub)
        x = opt.optimize(guess)
        y = opt.last_optimum_value()
        return x, y, opt.get_numevals()
    
    def nlfunc(self, x, _):
        """
        Executes a computation using the provided function. If an exception occurs during
        execution, it returns the maximum float value from sys.float_info.

        Args:
            x: Input value passed to the function for computation.
            _: Unused parameter.

        Returns:
            The result of the function computation if successful, or the maximum float
            value from sys.float_info in case of an exception.
        """
        try:
            return self.fun(x)
        except Exception as ex:
            return sys.float_info.max
    
