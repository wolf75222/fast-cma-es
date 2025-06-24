# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - modecpp.py

 Description:
  - Eigen based implementation of multi objective
    Differential Evolution using the DE/pareto/1 strategy.
    Derived and adapted for MO from its C++ counterpart
    [2]

    Can switch to NSGA-II like population update via parameter 'nsga_update'.
    Then it works essentially like NSGA-II but instead of the tournament selection
    the whole population is sorted and the best individuals survive. To do this
    efficiently the crowd distance ordering is slightly inaccurate.

    Supports parallel fitness function evaluation.

    Features enhanced multiple constraint ranking [3]
    improving its performance in handling constraints for engineering design optimization.

    Enables the comparison of DE and NSGA-II population update mechanism with everything else
    kept completely identical.

    Requires python 3.5 or higher.

    Uses the following deviation from the standard DE algorithm:
    a) oscillating CR/F parameters.

    You may keep parameters F and CR at their defaults since this implementation works well with the given settings for most problems,
    since the algorithm oscillates between different F and CR settings.

    For expensive objective functions (e.g. machine learning parameter optimization) use the workers
    parameter to parallelize objective function evaluation. This causes delayed population update.
    It is usually preferrable if popsize > workers and workers = mp.cpu_count() to improve CPU utilization.

    The ints parameter is a boolean array indicating which parameters are discrete integer values. This
    parameter was introduced after observing non optimal DE-results for the ESP2 benchmark problem:
    [4]
    If defined it causes a "special treatment" for discrete variables: They are rounded to the next integer value and
    there is an additional mutation to avoid getting stuck to local minima.

    See [5] for a detailed description.

 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/deoptimizer.cpp
  - [3] https://www.jstage.jst.go.jp/article/tjpnsec/11/2/11_18/_article/-char/en/
  - [4] https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/DockerCFDBenchmark.py
  - [5] https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/MODE.adoc

 Documentation:
  -


=============================================================================
"""

import os
import time
import threadpoolctl
import ctypes as ct
import multiprocessing as mp 
from multiprocessing import Process
import numpy as np
from scipy.optimize import Bounds
from fcmaes import mode, moretry
from fcmaes.mode import _filter, store
from numpy.random import Generator, PCG64DXSM, SeedSequence
from fcmaes.optimizer import dtime
from fcmaes.evaluator import mo_call_back_type, callback_mo, parallel_mo, libcmalib
from fcmaes.de import _check_bounds
from fcmaes.evaluator import is_debug_active
from loguru import logger
from typing import Optional, Callable, Tuple
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(mofun: Callable[[ArrayLike], ArrayLike],
             nobj: int,
             ncon: int,
             bounds: Bounds,
             guess: Optional[np.ndarray] = None,
             popsize: Optional[int] = 64,
             max_evaluations: Optional[int] = 100000,
             workers: Optional[int] = 1,
             f: Optional[float] = 0.5,
             cr: Optional[float] = 0.9,
             pro_c: Optional[float] = 0.5,
             dis_c: Optional[float] = 15.0,
             pro_m: Optional[float] = 0.9,
             dis_m: Optional[float] = 20.0,
             nsga_update: Optional[bool] = True,
             pareto_update: Optional[int] = 0,
             ints: Optional[ArrayLike] = None,
             min_mutate: Optional[float] = 0.1,
             max_mutate: Optional[float] = 0.5,           
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             store: Optional[store] = None,
             runid: Optional[int] = 0) -> Tuple[np.ndarray, np.ndarray]:

    """
    Minimizes a multi-objective optimization problem using evolutionary strategies.

    This function implements a multi-objective optimization routine that supports
    various configurations such as population size, mutation and crossover probabilities,
    and constraints. It provides options for parallel computation and supports custom
    random number generation. Additionally, results can be stored in the provided storage
    object if specified.

    Args:
        mofun (Callable[[ArrayLike], ArrayLike]): The objective function to be minimized.
            It must accept a single numpy array as input and return an array of objective
            values.
        nobj (int): The number of objectives in the problem.
        ncon (int): The number of constraints in the problem.
        bounds (Bounds): The bounds for the decision variables.
        guess (Optional[np.ndarray]): Optional initial guess for the decision variables.
        popsize (Optional[int]): Size of the population. Defaults to 64.
        max_evaluations (Optional[int]): Maximum number of function evaluations allowed.
            Defaults to 100,000.
        workers (Optional[int]): Number of workers for parallel computation. Defaults to 1 (serial).
        f (Optional[float]): Differential weight used in mutation step. Defaults to 0.5.
        cr (Optional[float]): Crossover probability. Defaults to 0.9.
        pro_c (Optional[float]): Probability of crossover operation. Defaults to 0.5.
        dis_c (Optional[float]): Distribution index for crossover. Defaults to 15.0.
        pro_m (Optional[float]): Probability of mutation. Defaults to 0.9.
        dis_m (Optional[float]): Distribution index for mutation. Defaults to 20.0.
        nsga_update (Optional[bool]): Whether to apply NSGA-II updates. Defaults to True.
        pareto_update (Optional[int]): Interval for Pareto front updates. Defaults to 0.
        ints (Optional[ArrayLike]): Specifies which decision variables are integer-valued.
            Defaults to None.
        min_mutate (Optional[float]): Minimum mutation step size as fraction of variable range.
            Defaults to 0.1.
        max_mutate (Optional[float]): Maximum mutation step size as fraction of variable range.
            Defaults to 0.5.
        rg (Optional[Generator]): Random number generator. Defaults to `Generator(PCG64DXSM())`.
        store (Optional[store]): Storage object to store results if specified. Defaults to None.
        runid (Optional[int]): Identifier for the run. Can be used for logging or tracking.
            Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the population of solutions
        (decision variable values) and their objective function values.

    Raises:
        Exception: If any error occurs during the optimization process.

    """
    
    try:
        mode = MODE_C(nobj, ncon, bounds, popsize, f, cr, pro_c, dis_c, pro_m, dis_m, 
                        nsga_update, pareto_update, ints, min_mutate, max_mutate, rg, runid)
        mode.set_guess(guess, mofun, rg)     
        if workers <= 1:
            x, y = mode.minimize_ser(mofun, max_evaluations)           
        else:
            x, y = mode.minimize_par(mofun, max_evaluations, workers)             
        if not store is None:
            store.create_views()
            store.add_results(x, y)
        return x, y
    except Exception as ex:
        print(str(ex))  
        return None, None
  
def retry(mofun: Callable[[ArrayLike], ArrayLike], 
            nobj: int,
            ncon: int, 
            bounds: Bounds,
            guess: Optional[np.ndarray] = None,
            num_retries: Optional[int] = 64,
            popsize: Optional[int] = 64, 
            max_evaluations: Optional[int] = 100000, 
            workers: Optional[int] = mp.cpu_count(),
            nsga_update: Optional[bool] = False,
            pareto_update: Optional[int] = 0,
            ints: Optional[ArrayLike] = None,
            capacity: Optional[int] = None):

    """
    Retries a multi-objective optimization process in parallel to optimize given
    objective functions and constraints.

    This function orchestrates the process of executing an optimization task
    multiple times across multiple workers, each working with different random
    seeds. It uses a population-based approach to iteratively search for optimal
    solutions for a given multi-objective problem, leveraging parallel computing
    to efficiently handle large workloads.

    Args:
        mofun (Callable[[ArrayLike], ArrayLike]): The objective function to optimize,
            mapping input parameters to objective and constraint values.
        nobj (int): Number of objective functions in the optimization problem.
        ncon (int): Number of constraints in the optimization problem.
        bounds (Bounds): The bounds for the decision variables. This defines the
            lower and upper bounds for optimization.
        guess (Optional[np.ndarray]): Initial guess for the input variables.
            Default is None.
        num_retries (Optional[int]): Number of retries allowed for each worker.
            Default is 64.
        popsize (Optional[int]): The population size for the optimization algorithm.
            Default is 64.
        max_evaluations (Optional[int]): Maximum number of function evaluations
            allowed. Default is 100000.
        workers (Optional[int]): Number of workers to run in parallel. If not
            specified, it defaults to the number of CPUs available on the machine.
        nsga_update (Optional[bool]): If True, enables an additional NSGA update
            step in the optimization. Default is False.
        pareto_update (Optional[int]): Frequency of updating the Pareto front
            during optimization. Default is 0 (no updates).
        ints (Optional[ArrayLike]): Indices of decision variables that are integers.
            Default is None.
        capacity (Optional[int]): Capacity of the storage system for maintaining
            results during optimization. If not provided, it defaults to
            2048 times the population size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - xs: The input configurations corresponding to the Pareto-optimal solutions.
            - ys: The objective and constraint evaluation results for the
              Pareto-optimal solutions.
    """
    
    dim, _, _ = _check_bounds(bounds, None)
    if capacity is None:
        capacity = 2048*popsize
    store = mode.store(dim, nobj + ncon, capacity)
    sg = SeedSequence()
    rgs = [Generator(PCG64DXSM(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
           args=(num_retries, pid, rgs, mofun, nobj, ncon, bounds, guess, popsize, 
                 max_evaluations, workers, nsga_update, pareto_update, 
                 store, ints))
                for pid in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    xs, ys = store.get_front()            
    return xs, ys

def _retry_loop(num_retries, pid, rgs, mofun, nobj, ncon, bounds, guess, popsize, 
                max_evaluations, workers, nsga_update, pareto_update, 
                store, ints):
    """
    Executes a retry loop for parallel optimization tasks, ensuring multiple
    minimization attempts are conducted until a sufficient number of results are
    added to the storage.

    Args:
        num_retries: Number of retry attempts for the optimization loop.
        pid: Process identifier used for dealing with random generators.
        rgs: List of random number generators for each process.
        mofun: Multi-objective function to be minimized.
        nobj: Number of objectives in the optimization problem.
        ncon: Number of constraints in the optimization problem.
        bounds: Boundaries for the decision variables in the optimization problem.
        guess: Initial guess values for the optimization variables.
        popsize: Population size for the optimization algorithm.
        max_evaluations: Maximum number of evaluations for each optimization attempt.
        workers: Number of worker processes available for parallelization.
        nsga_update: Callback or function for handling NSGA updates during optimization.
        pareto_update: Callback or function for managing Pareto updates.
        store: Storage object for managing results and tracking progress.
        ints: Indices of decision variables that are integer-constrained.
    """
    store.create_views()
    t0 = time.perf_counter()
    num = max(1, num_retries - workers)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        while store.num_added.value < num: 
            minimize(mofun, nobj, ncon, bounds, guess, popsize,
                        max_evaluations = max_evaluations, 
                        nsga_update=nsga_update, pareto_update=pareto_update,
                        rg = rgs[pid], store = store, ints=ints) 
            if is_debug_active():
                logger.debug("retries = {0}: time = {1:.1f} i = {2}"
                            .format(store.num_added.value, dtime(t0), store.num_stored.value))

class MODE_C:
    """
    A class for managing the MODE-C optimization algorithm.

    This class provides an interface for the multi-objective differential evolution
    (MODE) algorithm and supports handling of objective functions, constraints, parallel
    evaluation, and population management. It is designed to work with scenarios that
    require solving optimization problems with multiple competing objectives, bounded
    variables, and potentially integer-constrained decision variables.

    Attributes:
        popsize (int): The size of the population used in the optimization process.
        dim (int): The dimensionality of the decision variable space.
        nobj (int): The number of objective functions.
        ncon (int): The number of constraints in the optimization problem.
        bounds (Bounds): The bounds on variables, specified as a sequence of (min, max) bounds or
            using the `scipy.optimize.Bounds` class.
    """
    def __init__(self,
             nobj: int,
             ncon: int, 
             bounds: Bounds,
             popsize: Optional[int] = 64, 
             f: Optional[float] = 0.5, 
             cr: Optional[float] = 0.9, 
             pro_c: Optional[float] = 0.5,
             dis_c: Optional[float] = 15.0,
             pro_m: Optional[float] = 0.9,
             dis_m: Optional[float] = 20.0,
             nsga_update: Optional[bool] = True,
             pareto_update: Optional[int] = 0,
             ints: Optional[ArrayLike] = None,
             min_mutate: Optional[float] = 0.1,
             max_mutate: Optional[float] = 0.5,           
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             runid: Optional[int] = 0):

        """
        Initializes an instance of the optimization problem using the specified parameters.

        This constructor sets up the necessary parameters for a multi-objective differential
        evolution (MODE) algorithm, with options for NSGA-II-inspired updates and Pareto
        front approximations. It handles checks on bounds, dimensions, and population size,
        and initializes the required internal structures for the optimization process.

        Args:
            nobj (int): Number of objectives in the optimization problem.
            ncon (int): Number of constraints in the optimization problem.
            bounds (Bounds): Object defining the lower and upper bounds for the decision variables.
            popsize (Optional[int]): Size of the population. Defaults to 64.
            f (Optional[float]): Differential weight utilized in mutation. Defaults to 0.5.
            cr (Optional[float]): Crossover probability. Defaults to 0.9.
            pro_c (Optional[float]): Probability of crossover used in simulated binary crossover. Defaults to 0.5.
            dis_c (Optional[float]): Distribution index for crossover. Defaults to 15.0.
            pro_m (Optional[float]): Probability of mutation. Defaults to 0.9.
            dis_m (Optional[float]): Distribution index for mutation. Defaults to 20.0.
            nsga_update (Optional[bool]): Flag to enable NSGA-II-inspired update rules. Defaults to True.
            pareto_update (Optional[int]): Mode for Pareto front update. Defaults to 0.
            ints (Optional[ArrayLike]): Binary array denoting whether each variable is an integer (True) or continuous (False).
                Defaults to None.
            min_mutate (Optional[float]): Minimum mutation rate for adaptive mutation. Defaults to 0.1.
            max_mutate (Optional[float]): Maximum mutation rate for adaptive mutation. Defaults to 0.5.
            rg (Optional[Generator]): Random number generator instance. Defaults to Generator(PCG64DXSM()).
            runid (Optional[int]): Unique identifier for the model run. Defaults to 0.

        """

        dim, lower, upper = _check_bounds(bounds, None)
        if popsize is None:
            popsize = 64
        if popsize % 2 == 1 and nsga_update: # nsga update requires even popsize
            popsize += 1
        if lower is None:
            lower = [0]*dim
            upper = [0]*dim  
        if ints is None or nsga_update: # nsga update doesn't support mixed integer
            ints = [False]*dim   
        array_type = ct.c_double * dim   
        bool_array_type = ct.c_bool * dim 
        seed = int(rg.uniform(0, 2**32 - 1))
        try:
            self.ptr = initMODE_C(runid, dim, nobj, ncon, seed,
                               array_type(*lower), array_type(*upper), bool_array_type(*ints), 
                               popsize, f, cr, 
                               pro_c, dis_c, pro_m, dis_m,
                               nsga_update, pareto_update, min_mutate, max_mutate)
            self.popsize = popsize
            self.dim = dim    
            self.nobj = nobj  
            self.ncon = ncon
            self.bounds = bounds        
        except Exception as ex:
            print (str(ex))
            pass
     
    def __del__(self):
        """
        Handles the cleanup and destruction of resources managed by an instance of this class.

        This method is automatically called when the instance is about to be destroyed. It ensures
        that any resources tied to the instance are released properly to avoid memory leaks or
        resource contention.

        Raises:
            Exception: If the clean-up or destruction process encounters an error.
        """
        destroyMODE_C(self.ptr)
        
    def set_guess(self, guess, mofun, rg = None):
        """
        Set the initial guess values for optimization along with
        corresponding function evaluations.

        This function initializes guesses and their associated computed values based
        on the input guess and the provided function evaluator.

        If a random generator is not provided, a default PCG64DXSM-based generator
        is created and used to randomly select a subset of guesses and evaluations.

        Args:
            guess (Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]): Initial guesses
                for the optimization process. It can either be a numpy array of
                guesses, or a tuple containing both the guesses and their
                pre-computed evaluations.
            mofun (Callable): A callable function used to compute the output of
                each guess value.
            rg (Optional[Generator]): A numpy random generator for sampling. Defaults
                to None, in which case a new generator instance is created.
        """
        if not guess is None:
            if isinstance(guess, np.ndarray):
                ys = np.array([mofun(x) for x in guess])
            else:
                guess, ys = guess
            if rg is None:
                rg = Generator(PCG64DXSM())
            choice = rg.choice(len(ys), self.popsize, 
                                    replace = (len(ys) < self.popsize))
            self.tell(ys[choice], guess[choice])
          
    def ask(self) -> np.ndarray:
        """
        Generates and retrieves a new population of candidate solutions.

        This function interacts with a C library to generate a new population of
        solutions for an optimization task. It ensures that the results are
        retrieved and formatted appropriately in a NumPy array for further
        processing or evaluation.

        Returns:
            np.ndarray: A 2D NumPy array containing the generated population of
            solutions. Each row corresponds to a candidate solution, and each
            column corresponds to a dimension in the solution space.
        """
        try:
            popsize = self.popsize
            n = self.dim
            res = np.empty(popsize*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            askMODE_C(self.ptr, res_p)
            xs = np.empty((popsize, n))
            for p in range(popsize):
                xs[p,:] = res[p*n : (p+1)*n]
            return xs
        except Exception as ex:
            print (str(ex))
            return None

    def tell(self, ys: np.ndarray, xs: Optional[np.ndarray] = None) -> int:
        """
        Submits new candidate solutions and their corresponding function values to the optimizer.

        This function allows reporting of new solution candidates (`xs`) along with their associated
        function values (`ys`) to the optimization process. The optimizer will use this information
        to update its state and proceed with optimization.

        Args:
            ys (np.ndarray): A NumPy array of the function values corresponding to the given candidate
                solutions. This must be a one-dimensional array or will be flattened internally.
            xs (Optional[np.ndarray]): A NumPy array of candidate solutions. This must be a
                two-dimensional array with each row representing a candidate solution. If not provided,
                only the function values (`ys`) are reported.

        Returns:
            int: A status code. The status code indicates successful reporting of the solutions (e.g.,
                positive values) or failure due to an exception encountered during processing (negative
                values like -1).
        """
        try:
            flat_ys = ys.flatten()
            array_type_ys = ct.c_double * len(flat_ys)           
            if xs is None:
                return tellMODE_C(self.ptr, array_type_ys(*flat_ys))
            else:            
                flat_xs = xs.flatten()
                array_type_xs = ct.c_double * len(flat_xs)
                return setPopulationMODE_C(self.ptr, len(ys), 
                                            array_type_xs(*flat_xs), array_type_ys(*flat_ys))
        except Exception as ex:
            print (str(ex))
            return -1          
    
    def tell_switch(self, ys: np.ndarray, 
                        nsga_update: Optional[bool] = True,
                        pareto_update: Optional[int] = 0) -> int:
        """
        Updates information to a switching mechanism based on the input array.

        This function performs an operation to update internal mechanisms using the provided
        data. It interacts with an external library or module through a C function call,
        processing the input array into a flattened format before submission.

        Args:
            ys (np.ndarray): A NumPy array containing input data. The array will be flattened
                before use.
            nsga_update (Optional[bool]): Indicates whether an NSGA update mechanism is enabled.
                Default is True.
            pareto_update (Optional[int]): Specifies whether a Pareto-based update mechanism
                is triggered. Default is 0.

        Returns:
            int: The result of the external function call. On successful operation, it
                will likely be a status code or effect-based response. In case of an error,
                it returns -1.
        """
        try:
            flat_ys = ys.flatten()
            array_type_ys = ct.c_double * len(flat_ys)
            return tellMODE_switchC(self.ptr, array_type_ys(*flat_ys), nsga_update, pareto_update)
        except Exception as ex:
            print (ex)
            return -1        
 
    def population(self) -> np.ndarray:
        """
        Generates and retrieves the current population of individuals in a population-based
        algorithm.

        This method computes the population from an internal representation and
        returns it as a numpy array. The population matrix is reconstructed
        by splitting and reshaping raw flat data fetched from an external C function.

        Returns:
            np.ndarray: A 2D numpy array where each row represents an individual
            in the population and columns represent their feature values. If an error
            occurs during processing, the method returns None.

        Raises:
            Exception: If an internal error occurs during computation or interfacing
            with external C functions, an exception is raised and error information
            is printed to standard output.
        """
        try:
            popsize = self.popsize
            n = self.dim
            res = np.empty(popsize*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            populationMODE_C(self.ptr, res_p)
            xs = np.empty((popsize, n))
            for p in range(popsize):
                xs[p,:] = res[p*n : (p+1)*n]
            return xs
        except Exception as ex:
            print (str(ex))
            return None

    def minimize_ser(self, 
                     fun: Callable[[ArrayLike], ArrayLike], 
                     max_evaluations: Optional[int] = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Minimizes the given scalar function using a serial optimization method. The function
        iteratively evaluates the objective function on the candidate solutions, updates the
        internal state of the optimizer, and stops when the termination criterion is met or the
        maximum number of evaluations is reached.

        Args:
            fun (Callable[[ArrayLike], ArrayLike]): The objective function to be minimized.
                It should take an input of type ArrayLike and return a value of type ArrayLike.
            max_evaluations (Optional[int]): The maximum number of function evaluations allowed.
                Defaults to 100000.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the final candidate solutions as a
                numpy array and their corresponding evaluated objective function values as a numpy array.
        """
        evals = 0
        stop = 0
        while stop == 0 and evals < max_evaluations:
            xs = self.ask()
            ys = np.array([fun(x) for x in xs])
            stop = self.tell(ys)
            evals += self.popsize
        return xs, ys
        
    def minimize_par(self, 
                     fun: Callable[[ArrayLike], ArrayLike], 
                     max_evaluations: Optional[int] = 100000, 
                     workers: Optional[int] = mp.cpu_count()) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a parallel multi-objective optimization process.

        This method leverages parallel processing to perform multi-objective
        optimizations using a given objective function. It runs multiple iterations
        until the stopping criteria are met or the specified maximum evaluations are
        reached.

        Args:
            fun (Callable[[ArrayLike], ArrayLike]): Objective function to minimize.
            max_evaluations (Optional[int]): Maximum number of evaluations allowed.
                Defaults to 100000.
            workers (Optional[int]): Number of parallel workers to use. Defaults to
                the number of CPU cores.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the array of evaluated
                input solutions (`xs`) and their corresponding objective values (`ys`).

        """
        fit = parallel_mo(fun, self.nobj + self.ncon, workers)
        evals = 0
        stop = 0
        while stop == 0 and evals < max_evaluations:
            xs = self.ask()
            ys = fit(xs)
            stop = self.tell(ys)
            evals += self.popsize
        fit.stop()
        return xs, ys
    
if not libcmalib is None: 
        
    initMODE_C = libcmalib.initMODE_C
    initMODE_C.argtypes = [ct.c_long, ct.c_int, ct.c_int, \
                ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_bool), \
                ct.c_int, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, 
                ct.c_bool, ct.c_double, ct.c_double, ct.c_double]
    
    initMODE_C.restype = ct.c_void_p   
    
    destroyMODE_C = libcmalib.destroyMODE_C
    destroyMODE_C.argtypes = [ct.c_void_p]
    
    askMODE_C = libcmalib.askMODE_C
    askMODE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    tellMODE_C = libcmalib.tellMODE_C
    tellMODE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    tellMODE_C.restype = ct.c_int
    
    tellMODE_switchC = libcmalib.tellMODE_switchC
    tellMODE_switchC.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double), ct.c_bool, ct.c_double]
    tellMODE_switchC.restype = ct.c_int
    
    populationMODE_C = libcmalib.populationMODE_C
    populationMODE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]

    setPopulationMODE_C = libcmalib.setPopulationMODE_C
    setPopulationMODE_C.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
    setPopulationMODE_C.restype = ct.c_int

