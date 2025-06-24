# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - evaluator.py

 Description:
  - This module provides parallel objective function evaluation
  and a serial objective function wrapper for cmaes.minimize.
  - The Evaluator class manages the parallel processes and pipes.
  - Parallel objective function evaluator.
    Uses pipes to avoid re-spawning new processes for each eval_parallel call.
    the objective function is distributed once to all processes and
    reused for all eval_parallel calls. Evaluator(fun) needs to be stopped after the
    whole optimization is finished to avoid a resource leak.



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



from __future__ import annotations



from multiprocessing import Process, Pipe
import multiprocessing as mp
import ctypes as ct
import numpy as np
import sys, math, os  
from loguru import logger
from typing import Optional, Callable, Tuple
from numpy.typing import ArrayLike

pipe_limit = 64 # higher values can cause issues

def is_log_level_active(level):
    """
    Determines if the specified log level is currently active.

    This function checks whether the provided logging level is active by iterating
    through the logger handlers and comparing their level settings.

    Args:
        level: The logging level to check.

    Returns:
        bool: True if the specified logging level is active, otherwise False.
    """
    try: # nasty but currently there is no other way
        for handler in logger._core.handlers.values():
            if handler._levelno <= logger.level(level).no:
                return True
    except Exception as ex:   
        pass
    return False

def is_debug_active():
    """
    Determines if the debug log level is currently active.

    This function checks whether the "DEBUG" log level is active by invoking
    the `is_log_level_active` function with the "DEBUG" string as an argument.
    It is useful for determining if debug-level logging is enabled in the
    application.

    Returns:
        bool: True if the "DEBUG" log level is active, False otherwise.
    """
    return is_log_level_active("DEBUG")

def is_trace_active():
    """
    Checks if the TRACE log level is currently active.

    This function determines whether the TRACE level of logging is enabled,
    typically indicating if detailed debug information should be logged.

    Returns:
        bool: True if the TRACE log level is active, False otherwise.
    """
    return is_log_level_active("TRACE")

def eval_parallel(xs: ArrayLike, 
                  evaluator: Evaluator):
    """
    Evaluates a set of inputs in parallel using a provided evaluator function.

    This function processes a sequence of elements in chunks, using a custom
    evaluator. It supports efficiently evaluating large data by dividing the
    inputs into smaller segments and processing them iteratively in a pipeline.

    Args:
        xs (ArrayLike): Array-like collection of inputs to be evaluated.
        evaluator (Evaluator): An evaluator callable or function responsible
            for computing the results of the inputs provided.

    Returns:
        np.ndarray: Array containing the evaluation results corresponding to the
        input elements.
    """
    popsize = len(xs)
    ys = np.empty(popsize)
    i0 = 0
    i1 = min(popsize, pipe_limit)
    while True:
        _eval_parallel_segment(xs, ys, i0, i1, evaluator)
        if i1 >= popsize:
            break;
        i0 += pipe_limit
        i1 = min(popsize, i1 + pipe_limit)
    return ys
        
def eval_parallel_mo(xs: ArrayLike, 
                     evaluator: Evaluator, 
                     nobj: int):
    """
    Evaluates a population of solutions in parallel for a multi-objective optimization problem.

    This function takes a population of solutions and evaluates them in parallel to compute
    objective values. It is designed to handle large populations by dividing them into
    segments and processing each segment iteratively. The computed objective values are
    returned in a 2-dimensional array.

    Args:
        xs (ArrayLike): The input population of solutions to evaluate. Each solution should
            be represented as an array-like structure.
        evaluator (Evaluator): The evaluator object that computes objective values for
            the given solutions.
        nobj (int): The number of objectives being evaluated.

    Returns:
        np.ndarray: A 2-dimensional array where each row corresponds to the computed
            objective values of a solution in the input population.
    """
    popsize = len(xs)
    ys = np.empty((popsize,nobj))
    i0 = 0
    i1 = min(popsize, pipe_limit)
    while True:
        _eval_parallel_segment(xs, ys, i0, i1, evaluator)
        if i1 >= popsize:
            break;
        i0 += pipe_limit
        i1 = min(popsize, i1 + pipe_limit)
    return ys
        
class Evaluator(object):
    """
    Evaluator class to manage the parallel evaluation of a function across multiple workers.

    The Evaluator class facilitates the distribution and parallel execution of a specified
    objective function using worker processes. It enables efficient evaluation of the
    function with shared resources such as pipes and mutex locks to ensure synchronization.
    Workers can be dynamically started based on the number of available CPUs.

    Attributes:
        fun (Callable[[ArrayLike], float]): Objective function to be evaluated by workers.
        pipe (Pipe): Interprocess communication pipe for sending and receiving data
            between workers.
        read_mutex (Lock): Mutex lock to ensure safe reading from the pipe.
        write_mutex (Lock): Mutex lock to ensure safe writing to the pipe.
    """
    def __init__(self, 
                 fun: Callable[[ArrayLike], float], # objective function
                ):
        """
        Initializes the instance of the class with the provided objective function, and
        sets up the necessary communication and synchronization mechanisms.

        Args:
            fun (Callable[[ArrayLike], float]): Objective function to be used by the
                instance.
        """
        self.fun = fun 
        self.pipe = Pipe()
        self.read_mutex = mp.Lock() 
        self.write_mutex = mp.Lock() 
            
    def start(self, workers: Optional[int] = mp.cpu_count()):
        """
        Starts the multiprocessing environment with a specified number of workers and
        initializes the processes required to execute the given function.

        Args:
            workers (Optional[int]): The number of worker processes to spawn. If not
                specified, defaults to the system's CPU count.
        """
        self.workers = workers
        self.proc=[Process(target=_evaluate, args=(self.fun, 
                self.pipe, self.read_mutex, self.write_mutex)) for _ in range(workers)]
        [p.start() for p in self.proc]
        
    def stop(self): # shutdown all workers 
        """
        Stops all workers and frees resources properly.

        This method ensures a graceful shutdown of all initiated workers by notifying
        them through the pipe and waiting for their termination. Once all workers have
        been joined, it also closes all communication pipes.

        Raises:
            OSError: Raised if there are issues while closing any of the pipes.

        """
        for _ in range(self.workers):
            self.pipe[0].send(None)
        [p.join() for p in self.proc]    
        for p in self.pipe:
            p.close()

def _eval_parallel_segment(xs, ys, i0, i1, evaluator):
    """
    Evaluates a segment of data in parallel by sending and receiving data through a pipe.

    This function processes a segment of input data specified by the indices i0 and i1
    using the evaluator's pipeline. It sends values from the input `xs` to the evaluator,
    and receives processed results to update the output `ys`.

    Args:
        xs: A sequence of input data to be evaluated.
        ys: A sequence to store the results after evaluation.
        i0: The starting index of the segment to be processed (inclusive).
        i1: The ending index of the segment to be processed (exclusive).
        evaluator: An object with a communication pipe for parallel processing.

    Returns:
        The updated `ys` sequence containing the evaluated results.
    """
    for i in range(i0, i1):
        evaluator.pipe[0].send((i, xs[i]))
    for _ in range(i0, i1):        
        i, y = evaluator.pipe[0].recv()
        ys[i] = y
    return ys

def _evaluate(fun, pipe, read_mutex, write_mutex): # worker
    """
    Execute a given function on inputs received from a communication pipe in a
    thread-safe manner and send results back through the pipe. Handles potential
    exceptions during function execution and ensures proper synchronization
    using the provided mutex locks.

    Args:
        fun (Callable): Function to be executed on input data received from the pipe.
        pipe (Tuple[Connection, Connection]): A pair of connection objects for inter-process
            communication, used for receiving inputs and sending results.
        read_mutex (Lock): A threading lock ensuring safe reading from the input pipe.
        write_mutex (Lock): A threading lock ensuring safe writing to the output pipe.
    """
    while True:
        with read_mutex:
            msg = pipe[1].recv() # Read from the input pipe
        if msg is None: 
            break # shutdown worker
        try:
            i, x = msg
            y = fun(x)
        except Exception as ex:
            y =  sys.float_info.max
        with write_mutex:            
            pipe[1].send((i, y)) # Send result

def _check_bounds(bounds, guess, rg):
    """
    Checks and processes bounds, guesses, and random generator inputs for optimization.

    This function ensures that either bounds or guesses are properly defined, as they
    are necessary to perform any optimization task. If `bounds` is not provided, but
    `guess` is, it returns a processed version of the guess. If `bounds` is provided
    but `guess` is absent, it generates a random guess within the bounds using the
    provided random generator. Ultimately, it arranges and returns bounds and guesses
    as numpy arrays.

    Args:
        bounds: Object containing lower (`lb`) and upper (`ub`) bounds. Can be None
            if guess is provided directly.
        guess: Initial guess of the parameters. Can be None if bounds are provided.
        rg: A random number generator instance used to generate guesses when they
            are not explicitly provided.

    Returns:
        Tuple containing:
            - Lower bounds array (numpy.ndarray) or None if bounds are not provided.
            - Upper bounds array (numpy.ndarray) or None if bounds are not provided.
            - Guess array (numpy.ndarray) processed from the provided inputs.

    Raises:
        ValueError: If both bounds and guess are None.
    """
    if bounds is None and guess is None:
        raise ValueError('either guess or bounds need to be defined')
    if bounds is None:
        return None, None, np.asarray(guess)
    if guess is None:
        guess = rg.uniform(bounds.lb, bounds.ub)
    return np.asarray(bounds.lb), np.asarray(bounds.ub), np.asarray(guess)

def _get_bounds(dim, bounds, guess, rg):
    """
    Configures and validates bounds, initial guesses, and dimensionality for a process.

    This function processes and ensures the compatibility of dimensionality, bounds,
    and guesses for an optimization or computational routine. It also generates
    default values where absent, ensuring all return values align with the given
    inputs.

    Args:
        dim (int): Dimensionality of the problem.
        bounds (Optional[Bounds]): The lower and upper bounds for each dimension.
        guess (Optional[np.ndarray]): Initial guess for the optimization variable(s).
        rg (np.random.Generator): Random number generator used for generating
            uniformly distributed guesses if `guess` is not provided.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]: A tuple
            containing the processed lower bounds, upper bounds, and initial guess.
            Returns (None, None, guess) when bounds is not specified.
    """
    if bounds is None:
        if guess is None:
            guess = np.asarray(np.zeros(dim))
        return None, None, guess
    if guess is None:
        guess = rg.uniform(bounds.lb, bounds.ub)
    return np.asarray(bounds.lb), np.asarray(bounds.ub), np.asarray(guess)

class _fitness(object):
    """
    Represents a fitness evaluation system for optimization problems.

    This class encapsulates a fitness evaluation system that supports parallel evaluation,
    manages scaling and normalization of input values based on given bounds, and
    provides encoding and decoding mechanisms for the input values. It tracks the
    number of fitness evaluations performed and ensures that inputs conform to
    specified feasible bounds.

    Attributes:
        fun (callable): The fitness function used to evaluate fitness values.
        evaluation_counter (int): Counter tracking the number of fitness evaluations performed.
        lower: The lower bound for input values. Can be `None` if no lower bound is defined.
        upper: The upper bound for input values. Can be `None` if no upper bound is defined.
        normalize (bool): Indicates whether normalization of input values is applied.
    """
     
    def __init__(self, fun, lower, upper, normalize = None):
        """
        Initializes the class with a given function, lower and upper bounds, and an
        optional normalization flag.

        Args:
            fun: A function object that defines the operation or behavior related to
                the instance of the class.
            lower: A numeric value or an iterable defining the lower bounds for the
                function.
            upper: A numeric value or an iterable defining the upper bounds for the
                function.
            normalize: Optional; A boolean flag indicating whether the function's
                behavior should be normalized. Defaults to False.
        """
        self.fun = fun
        self.evaluation_counter = 0
        self.lower = lower
        self.normalize = False
        if not (lower is None or normalize is None):
            self.normalize = normalize
        if not lower is None:
            self.upper = upper
            self.scale = 0.5 * (upper - lower)
            self.typx = 0.5 * (upper + lower)

    def values(self, Xs): #enables parallel evaluation
        """
        Evaluates the given input values through the provided function while
        tracking the number of evaluations performed.

        Args:
            Xs: Input values for which the function is evaluated.

        Returns:
            np.ndarray: An array containing the evaluated results.
        """
        values = self.fun(Xs)
        self.evaluation_counter += len(Xs)
        return np.array(values)
    
    def closestFeasible(self, X):
        """
        Finds and returns the closest feasible value(s) within the specified boundaries.

        If lower and upper bounds are defined, the value(s) are clipped between those
        bounds. If normalization is enabled, the values are clipped between -1.0 and
        1.0. Otherwise, the original value(s) are returned.

        Args:
            X: The value(s) to be adjusted to the closest feasible within the
                specified boundaries.

        Returns:
            The value(s) clipped to the closest feasible within the boundaries.
        """
        if self.lower is None:
            return X    
        else:
            if self.normalize:
                return np.clip(X, -1.0, 1.0)
            else:
                return np.clip(X, self.lower, self.upper)

    def encode(self, X):
        """
        Encodes the input data X by either normalizing it based on provided scaling
        factors or leaving it unchanged depending on the normalization setting.

        Args:
            X: Input data to encode.

        Returns:
            The encoded data, normalized if the `normalize` attribute is set to True,
            otherwise the original input.
        """
        if self.normalize:
            return (X - self.typx) / self.scale
        else:
            return X
   
    def decode(self, X):
        """
        Decodes the given input using an optional normalization method.

        If normalization is enabled, it applies a transformation to the input based on the provided
        scale and typx values. Otherwise, it returns the input as is.

        Args:
            X: Input data to decode.

        Returns:
            The decoded data. If normalization is enabled, it will be adjusted using the defined
            scale and typx values.
        """
        if self.normalize:
            return (X * self.scale) + self.typx
        else:
            return X
         
def serial(fun):
    """
    Creates a function that applies a given function to each element in a list.

    This decorator takes a function as input and returns a new function. The
    returned function, when called with a list of elements, applies the input
    function to each element in the list sequentially. If the input function fails
    for an element, the resulting behavior is handled by an internal mechanism.

    Args:
        fun: The function to be applied to each element in a list.

    Returns:
        A new function that takes a list of elements and applies 'fun' to each
        element, returning a list of results.
    """
  
    return lambda xs : [_tryfun(fun, x) for x in xs]
        
def _func_serial(fun, num, pid, xs, ys):
    """
    Applies a function to elements of a list in a serial manner based on a processing
    strategy using process ID and total number of processes.

    This function modifies the `ys` list in place, applying the `fun` function
    to elements in `xs` at specific indices determined by the process ID (`pid`) and
    total number of processes (`num`). It ensures elements are processed in a
    distributed order across processes.

    Args:
        fun: Callable function to apply to elements of the input list.
        num: int. Total number of processes.
        pid: int. Process ID, indicating the current process index starting at 0.
        xs: list. Input list of elements to process.
        ys: list. Output list where results are stored at corresponding indices.
    """
    for i in range(pid, len(xs), num):
        ys[i] = _tryfun(fun, xs[i])

def _tryfun(fun, x):
    """
    Attempts to evaluate the specified function with a given input and handle errors
    gracefully. If the result is a finite value, it is returned. Otherwise, or
    if the evaluation raises an exception, the maximum floating-point value is returned.

    Args:
        fun: A callable that represents the function to be evaluated.
        x: The input value passed to the function for evaluation.

    Returns:
        The result of the function evaluation if it produces a finite value.
        Otherwise, returns the maximum floating-point value.
    """
    try:
        fit = fun(x)
        return fit if math.isfinite(fit) else sys.float_info.max
    except Exception:
        return sys.float_info.max
    
class parallel(object):
    """
    Executes functions in parallel using a multi-worker setup.

    This class facilitates the parallel execution of functions across multiple
    processes for improved performance and efficiency, particularly for
    computationally expensive or repetitive tasks. It leverages an evaluator
    to manage function calls and starts job execution across specified workers.

    Attributes:
        evaluator (Evaluator): The evaluator instance responsible for managing
            parallel execution of function calls.
    """
        
    def __init__(self, 
                 fun: Callable[[ArrayLike], float], 
                 workers: Optional[int] = mp.cpu_count()):
        """
        Initializes a new instance of the class.

        Args:
            fun (Callable[[ArrayLike], float]): A callable function that accepts an argument of
                type ArrayLike and returns a float. This function will be used as the evaluator.
            workers (Optional[int]): The number of worker processes to use. Defaults to the
                number of CPUs available on the machine.
        """
        self.evaluator = Evaluator(fun)
        self.evaluator.start(workers)
    
    def __call__(self, xs: ArrayLike) -> np.ndarray:
        """
        Evaluates the input using the provided evaluator function in parallel and returns the result.

        Args:
            xs (ArrayLike): Input data to be evaluated.

        Returns:
            np.ndarray: The result of the evaluation.
        """
        return eval_parallel(xs, self.evaluator)

    def stop(self):
        """
        Stops the evaluation process.

        This method halts the ongoing evaluation process managed by the evaluator
        to prevent further execution or processing.

        Raises:
            RuntimeError: If the evaluation process cannot be stopped.
        """
        self.evaluator.stop()

class parallel_mo(object):
    """
    Manages parallel multi-objective evaluations.

    This class handles the evaluation of multi-objective tasks in a parallelized
    manner by utilizing multiple workers. It initializes with a given function
    to be evaluated and starts the parallel evaluation process with the specified
    or default number of workers.

    Attributes:
        nobj (int): Number of objectives to be evaluated.
        evaluator (Evaluator): Instance of the Evaluator class used to perform
            parallel evaluations.
    """
    def __init__(self, 
                 fun: Callable[[ArrayLike], ArrayLike], 
                 nobj: int, 
                 workers: Optional[int] = mp.cpu_count()):
        """
        Initializes a class instance designed for handling function evaluation in a parallel
        manner across multiple workers. The class takes a user-defined function,
        the number of objectives, and an optional number of workers to allocate
        for parallel computation. It leverages an external evaluator to perform
        this task efficiently.

        Args:
            fun (Callable[[ArrayLike], ArrayLike]): A callable function that takes
                an input array-like object and returns an array-like object as output.
            nobj (int): The number of objectives for the function being evaluated.
            workers (Optional[int]): The number of workers to allocate for parallel
                function execution. Defaults to the number of CPU cores in the system.
        """
        self.nobj = nobj
        self.evaluator = Evaluator(fun)
        self.evaluator.start(workers)
    
    def __call__(self, xs: ArrayLike) -> np.ndarray:
        """
        Evaluates a set of inputs using a specified evaluator in a parallelized manner, producing a multi-objective
        output array.

        Args:
            xs (ArrayLike): Input data to be evaluated, typically a collection of elements suitable for
                multi-objective evaluation.

        Returns:
            np.ndarray: Computed multi-objective evaluation results in the form of an array.

        """
        return eval_parallel_mo(xs, self.evaluator, self.nobj)

    def stop(self):
        """
        Stops the evaluator process.

        This method halts the execution of the evaluator process. It ensures that
        any continuous operations linked to the evaluator instance are terminated.

        Raises:
            Any exceptions raised by the `evaluator.stop()` method will propagate.
        """
        self.evaluator.stop()

class callback(object):
    """Represents a callable object to evaluate a function with input processing.

    This class serves as a wrapper for a given function, enabling it to be
    called with processed inputs. It evaluates the function on a subset of
    inputs, applies type conversion, and ensures that non-finite results are
    replaced with a fallback maximum float value.

    Attributes:
        fun (Callable[[ArrayLike], float]): The function to be evaluated,
            which takes an ArrayLike input and returns a float.
    """
    def __init__(self, fun: Callable[[ArrayLike], float]):
        """
        Initializes a callable function to be used.

        Args:
            fun (Callable[[ArrayLike], float]): A function that operates on an ArrayLike
                object and returns a float.
        """
        self.fun = fun
    
    def __call__(self, n: int, x: ArrayLike) -> float:
        """
        Evaluates a function with input values derived from an array slice,
        returning a numerical result. If the result is not finite or an
        error occurs during computation, it returns the maximum finite
        float value.

        Args:
            n (int): The number of elements to consider from the array `x`.
            x (ArrayLike): An array-like object containing input values.

        Returns:
            float: The computed result of the function or the maximum finite
            float value in case of an error or non-finite result.

        Raises:
            Exception: Any exception encountered during the computation of the
            function will be caught, and the maximum float value will be returned
            instead.
        """
        try:
            fit = self.fun(np.fromiter((x[i] for i in range(n)), dtype=float))
            return fit if math.isfinite(fit) else sys.float_info.max
        except Exception as ex:
            return sys.float_info.max
        
class callback_so(object):
    """
    Handles callback functionality for single-objective optimization purposes.

    This class facilitates communication between external and Python-based
    optimization routines. It processes input vectors, evaluates the
    objective function, handles memory using ctypes, and determines if a
    termination condition has been met.

    Attributes:
        fun (Callable[[ArrayLike], float]): Objective function that takes an
            input array and returns a fitness value.
        dim (int): Dimensionality of the input array passed to the objective
            function.
        nobj (int): Number of objectives handled by the callback. Set to 1 for
            single-objective optimization.
        is_terminate (Optional[Callable[[ArrayLike, float], bool]]): Optional
            termination callback function. It evaluates the termination
            condition given the input array and fitness value.
    """
    def __init__(self, 
                 fun: Callable[[ArrayLike], float], 
                 dim: int, 
                 is_terminate: Optional[Callable[[ArrayLike, float], bool]] = None):
        """
        Initializes the instance with the provided objective function, dimensionality, and optional
        termination condition. The `fun` parameter defines the objective function to optimize, `dim`
        specifies the number of dimensions for the optimization, and `is_terminate` is an optional
        callback to evaluate termination criteria.

        Args:
            fun (Callable[[ArrayLike], float]): The objective function to optimize.
            dim (int): The number of dimensions for the optimization problem.
            is_terminate (Optional[Callable[[ArrayLike, float], bool]]): Optional callback that
                determines if optimization should terminate based on the solution and its evaluation.
        """
        self.fun = fun
        self.dim = dim
        self.nobj = 1
        self.is_terminate = is_terminate
    
    def __call__(self, dim, x, y):
        """
        Invokes the callable object with given dimensions and inputs, processes the input
        buffers, computes the function output, and updates the output buffer.

        This method handles finite computations, manages buffer conversions, and determines
        whether a termination condition is met.

        Args:
            dim: The dimensional size of the input array.
            x: A ctypes object representing the input buffer.
            y: A ctypes object representing the output buffer.

        Returns:
            bool: False if there is no termination condition, or the result of the termination
            condition check if it exists.

        Raises:
            Exception: Generic exception raised for any unexpected issues during execution.
        """
        try:
            arrTypeX = ct.c_double*(self.dim)
            xaddr = ct.addressof(x.contents)
            xbuf = np.frombuffer(arrTypeX.from_address(xaddr))
            arrTypeY = ct.c_double*(self.nobj)
            yaddr = ct.addressof(y.contents)   
            ybuf = np.frombuffer(arrTypeY.from_address(yaddr))  
            fit = self.fun(xbuf)
            ybuf[0] = fit if math.isfinite(fit) else sys.float_info.max
            return False if self.is_terminate is None else self.is_terminate(xbuf, ybuf) 
        except Exception as ex:
            print (ex)
            return False

class callback_mo(object):
    """
    Callable object for multi-objective optimization.

    This class is designed to act as a callable callback for multi-objective
    optimization problems, allowing the evaluation of objective functions at
    given points and the computation of related output values. It also supports
    an optional termination condition.

    Attributes:
        fun (Callable[[ArrayLike], ArrayLike]): The objective function to be called.
            It takes an input array-like structure and returns an array-like
            structure.
        dim (int): The dimensionality of the input space.
        nobj (int): The number of objectives in the optimization problem.
        is_terminate (Optional[bool]): A callable to determine termination conditions.
            If provided, it should take two array-like arguments (inputs and outputs)
            and return a boolean indicating whether the optimization should be
            terminated.
    """
    def __init__(self, 
                 fun: Callable[[ArrayLike], ArrayLike], 
                 dim: int, 
                 nobj: int, 
                 is_terminate: Optional[bool] = None):
        """
        Initializes the class with the given function, dimension, number of objectives,
        and optional termination indicator.

        Args:
            fun: Callable[[ArrayLike], ArrayLike]. A function that takes an
                array-like input and returns an array-like output.
            dim: int. The dimension of the input space.
            nobj: int. The number of objectives.
            is_terminate: Optional[bool]. A flag indicating whether termination
                should be enabled or not. Defaults to None.
        """
        self.fun = fun
        self.dim = dim
        self.nobj = nobj
        self.is_terminate = is_terminate
    
    def __call__(self, dim: int, x, y):
        """
        Executes a callable object that processes given input, evaluates a function,
        and optionally checks a termination condition.

        Args:
            dim (int): The dimensionality of the input data.
            x: A ctypes pointer to a buffer containing input data.
            y: A ctypes pointer to a buffer where the output data will be stored.

        Returns:
            bool: False if the processing completes successfully without termination or
                if the termination condition is not met. True if the termination
                condition is met.

        Raises:
            Exception: If an error occurs during the execution process, details will
                be printed, and False will be returned.
        """
        try:
            arrTypeX = ct.c_double*(dim)
            xaddr = ct.addressof(x.contents)
            xbuf = np.frombuffer(arrTypeX.from_address(xaddr))
            arrTypeY = ct.c_double*(self.nobj)
            yaddr = ct.addressof(y.contents)   
            ybuf = np.frombuffer(arrTypeY.from_address(yaddr))  
            ybuf[:] = self.fun(xbuf)[:]
            return False if self.is_terminate is None else self.is_terminate(xbuf, ybuf) 
        except Exception as ex:
            print (ex)
            return False

class callback_par(object):
    """Wrapper class for callable objects with functionality to process
    populations of data using provided functions.

    This class provides a mechanism to evaluate a function or a parallelized
    function (`fun` or `parfun`) on a specified population's data. It
    accommodates use cases where either single or parallelized computation is
    required.

    Attributes:
        fun (Callable[[ArrayLike], float]): A function that processes a
            single element of a population and returns a float result.
        parfun (Callable[[ArrayLike], ArrayLike]): A parallelizable function
            that processes multiple elements of a population and returns
            an array of results.
    """
    def __init__(self, 
                 fun: Callable[[ArrayLike], float], 
                 parfun: Callable[[ArrayLike], ArrayLike]):
        """
        Initializes the object with the provided function and parameter function.

        Args:
            fun (Callable[[ArrayLike], float]): A callable function that takes an
                ArrayLike input and returns a float.
            parfun (Callable[[ArrayLike], ArrayLike]): A callable parameter function
                that takes an ArrayLike input and returns an ArrayLike output.
        """
        self.fun = fun
        self.parfun = parfun
    
    def __call__(self, popsize, n, xs_, ys_):
        """
        Calls the function or parallel function to evaluate the array of input values
        and store the results.

        This method operates on the input arrays provided and evaluates them using the
        given function (`fun`) or a parallel function (`parfun`) if available. The
        results are then stored in the output array.

        Args:
            popsize: int
                The population size; determines how many sets of inputs are processed.
            n: int
                The length of each input array segment.
            xs_: ctypes.POINTER(ctypes.c_double)
                A pointer to the shared memory that contains the input numeric data.
                It points to an array of double-precision floating-point numbers.
            ys_: ctypes.POINTER(ctypes.c_double)
                A pointer to the shared memory where the evaluated results will
                be stored. It points to an array of double-precision floating-point
                numbers.

        Raises:
            Exception: If the evaluation process of the inputs fails for any reason,
                it will print the exception message to the standard output.
        """
        try:
            arrType = ct.c_double*(popsize*n)
            addr = ct.addressof(xs_.contents)
            xall = np.frombuffer(arrType.from_address(addr))
            
            if self.parfun is None:
                for p in range(popsize):
                    ys_[p] = self.fun(xall[p*n : (p+1)*n])
            else:    
                xs = []
                for p in range(popsize):
                    x = xall[p*n : (p+1)*n]
                    xs.append(x)
                ys = self.parfun(xs)
                for p in range(popsize):
                    ys_[p] = ys[p]
        except Exception as ex:
            print (ex)

basepath = os.path.dirname(os.path.abspath(__file__))

try: 
    if sys.platform.startswith('linux'):
        libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.so')  
    elif 'mac' in sys.platform or 'darwin' in sys.platform:
        libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dylib')  
    else:
        os.environ['PATH'] = (basepath + '/lib') + os.pathsep + os.environ['PATH']
        libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dll')
except Exception as ex:
    libcmalib = None
    
mo_call_back_type = ct.CFUNCTYPE(ct.c_bool, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))
  
call_back_type = ct.CFUNCTYPE(ct.c_double, ct.c_int, ct.POINTER(ct.c_double))  

call_back_par = ct.CFUNCTYPE(None, ct.c_int, ct.c_int, \
                                  ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))  

