# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - pygmoretry.py

 Description:
  - This module provides a retry mechanism for the PYGMO/PAGMO optimization framework.
  - It allows for parallel retries of optimization problems using the PYGMO/PAGMO library.
  - It is designed to work with problems that have constraints or multiple objectives,
  which cannot be handled by the standard fcmaes.retry module.
  - The retry mechanism uses multiprocessing to perform multiple optimization attempts
  in parallel, improving efficiency and scalability.

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
import math
import os
import sys
import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence
from scipy.optimize import OptimizeResult, Bounds
import multiprocessing as mp
from multiprocessing import Process
from fcmaes.retry import Store

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def minimize(prob, 
             algo,
             value_limit = np.inf,
             num_retries = 100*mp.cpu_count(),
             workers = mp.cpu_count(),
             popsize = 1, 
             ) -> OptimizeResult:
    """
    Minimizes an optimization problem using given algorithm and configurations.

    This function attempts to find the minimum of the given problem by applying
    the specified algorithm. It retries the operation a specified number of
    times with multiple workers and uses given population size for optimization.

    Args:
        prob: The optimization problem that defines the objective function and
            constraints.
        algo: The optimization algorithm to be applied.
        value_limit: An optional upper limit on the value of the objective
            function. Defaults to positive infinity.
        num_retries: The number of retries allowed for the optimization process.
            Defaults to 100 times the number of CPU cores available.
        workers: The number of worker processes to use for parallel computation.
            Defaults to the number of CPU cores available.
        popsize: The size of the population for optimization. Defaults to 1.

    Returns:
        OptimizeResult: The result of the optimization process, including the
        solution and other relevant information about the optimization.
    """

    lb, ub = prob.get_bounds()
    bounds = Bounds(lb, ub)
    store = Store(bounds)
    return retry(store, prob, algo, num_retries, value_limit, popsize, workers)
                 
def retry(store, prob, algo, num_retries, value_limit = np.inf, popsize=1, workers=mp.cpu_count()):
    """
    Retries optimization over multiple attempts across parallel workers to find
    the best result. The function orchestrates the parallel execution, manages
    random number generators for each worker, and aggregates the results.

    Args:
        store: An object responsible for storing and managing the optimization
            results.
        prob: The optimization problem to be solved.
        algo: The algorithm used to perform optimization.
        num_retries: The number of retries to attempt for the optimization process.
        value_limit: The upper limit for the values considered valid in the
            optimization result (default is np.inf).
        popsize: The population size used in the optimization process
            (default is 1).
        workers: The number of parallel workers to use for the optimization
            (default is the number of CPU cores available).

    Returns:
        OptimizeResult: An object containing the best solution (`x`), the best
            objective value (`fun`), the number of function evaluations performed
            (`nfev`), and the success status of the optimization (`success`).

    Raises:
        ImportError: If the Pygmo library is not installed.
    """
    try:
        import pygmo as pg
    except ImportError as e:
        raise ImportError("Please install PYGMO (pip install pygmo) to use PAGMO optimizers") from e
    sg = SeedSequence()
    rgs = [Generator(PCG64DXSM(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
            args=(pid, rgs, store, prob, algo, num_retries, value_limit, popsize, pg)) for pid in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    store.sort(store.get_xs())
    store.dump()
    return OptimizeResult(x=store.get_x_best(), fun=store.get_y_best(), 
                          nfev=store.get_count_evals(), success=True)
        
def _retry_loop(pid, rgs, store, prob, algo, num_retries, value_limit, popsize, pg):
    """
    Executes a retry loop for a given probabilistic algorithm to attempt finding a feasible solution.

    The function runs multiple attempts to generate solutions using a probabilistic algorithm. It retrieves a random seed,
    evolves a population, and evaluates the best solution. Feasible solutions meeting specified criteria are added to a
    result store.

    Args:
        pid (int): Identifier for the process or individual task.
        rgs (list): List of random generators corresponding to each identifier.
        store (object): Object responsible for storing results and managing retries.
        prob (object): Problem object defining the optimization problem.
        algo (object): Algorithm used to evolve populations.
        num_retries (int): Maximum number of retries to evolve a solution.
        value_limit (float): Feasibility threshold for solutions.
        popsize (int): Size of the population used in each evolutionary attempt.
        pg (object): External module used for handling evolutionary population and problem definition.
    """
    while store.get_runs_compare_incr(num_retries):      
        try:            
            seed = int(rgs[pid].uniform(0, 2**32 - 1))
            pop = pg.population(prob, popsize, seed=seed)
            pop = algo.evolve(pop)
        except Exception:
            pass  # ignore "Maximum number of iteration reached"      
        sol = pop.champion_x
        y = pop.champion_f
        evals = pop.problem.get_fevals()
         
        _feasible = prob.feasibility_x(pop.champion_x)
        if _feasible:
            store.add_result(y[0], sol, evals, value_limit)
            store.dump()
