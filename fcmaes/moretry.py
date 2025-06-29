# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - moretry.py

 Description:
  - Parallel multi objective optimization retry using CMA-ES and differential evolution.

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
import math, sys, time, warnings, threadpoolctl
import multiprocessing as mp
from multiprocessing import Process
from scipy.optimize import Bounds
from numpy.random import Generator, PCG64DXSM, SeedSequence
from fcmaes.optimizer import de_cma, dtime, Optimizer
from fcmaes import retry, advretry
from loguru import logger

from typing import Optional, Callable, Tuple
from numpy.typing import ArrayLike

def minimize(fun: Callable[[ArrayLike], float],
             bounds: Bounds,
             weight_bounds: Bounds,
             ncon: Optional[int] = 0,
             value_exp: Optional[float] = 2.0,
             value_limits: Optional[ArrayLike] = None,
             num_retries: Optional[int] = 1024,
             workers: Optional[int] = mp.cpu_count(),
             popsize: Optional[int] = 31, 
             max_evaluations: Optional[int] = 50000, 
             capacity: Optional[int] = None,
             optimizer: Optional[Optimizer] = None,
             statistic_num: Optional[int] = 0,
             plot_name: Optional[str] = None
              ) -> Tuple[np.ndarray, np.ndarray]:   
    """Minimization of a multi objective function of one or more variables using parallel 
     optimization retry.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x) -> float``
        where ``x`` is an 1-D array with shape (n,)
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    weight_bounds : `Bounds`, optional
        Bounds on objective weights.
    ncon : int, optional
        number of constraints
    value_exp : float, optional
        exponent applied to the objective values for the weighted sum. 
    value_limits : sequence of floats, optional
        Upper limit for optimized objective values to be stored. 
    num_retries : int, optional
        Number of optimization retries.    
    workers : int, optional
        number of parallel processes used. Default is mp.cpu_count()
    popsize = int, optional
        CMA-ES population size used for all CMA-ES runs. 
        Not used for differential evolution. 
        Ignored if parameter optimizer is defined. 
    max_evaluations : int, optional
        Forced termination of all optimization runs after ``max_evaluations`` 
        function evaluations. Only used if optimizer is undefined, otherwise
        this setting is defined in the optimizer. 
    capacity : int, optional
        capacity of the evaluation store.
    optimizer : optimizer.Optimizer, optional
        optimizer to use. Default is a sequence of differential evolution and CMA-ES.
    plot_name : plot_name, optional
        if defined the pareto front is plotted during the optimization to monitor progress
     
    Returns
    -------
    xs, ys: list of argument vectors and corresponding value vectors of the optimization results. """

    if optimizer is None:
        optimizer = de_cma(max_evaluations, popsize)  
    if capacity is None: 
        capacity = num_retries
    store = retry.Store(fun, bounds, capacity = capacity, 
                        statistic_num = statistic_num)
    store.plot_name = plot_name
    xs = np.array(mo_retry(fun, weight_bounds, ncon, value_exp, 
                           store, optimizer.minimize, num_retries, value_limits, workers))
    ys = np.array([fun(x) for x in xs])
    return xs, ys
    
def mo_retry(fun: Callable[[ArrayLike], float], 
             weight_bounds: Bounds, 
             ncon: int, 
             y_exp: float, 
             store, 
             optimize: Callable, 
             num_retries: int, 
             value_limits: ArrayLike, 
             workers: Optional[int] = mp.cpu_count()):
    """
    Executes a multi-objective optimization with retry functionality by distributing work across
    multiple processes.

    This function performs optimization by spawning multiple processes, each executing a retry
    loop targeting a specific optimizer and constraints. The retries ensure robustness against
    failures or subpar outcomes in individual optimization attempts across the workers.

    Args:
        fun (Callable[[ArrayLike], float]): The objective function to optimize. It must accept
            an input vector and return a scalar cost or objective value.
        weight_bounds (Bounds): Bounds for the weights, defining constraints on the optimization
            variables.
        ncon (int): Number of constraints to enforce during the optimization process.
        y_exp (float): The expected value used as a threshold or benchmark for optimization outcomes.
        store: An object for storing and managing results during optimization. Must support `sort`
            and `dump` methods, among others.
        optimize (Callable): Optimization function or solver to be used within the retry loop. It
            must be callable and compatible with the function and constraints provided.
        num_retries (int): The number of retries per process, ensuring multiple attempts for robust
            optimization results.
        value_limits (ArrayLike): Array-like structure specifying the acceptable limits for values
            during optimization.
        workers (Optional[int]): Number of processes to spawn for parallel optimization. Defaults
            to the total available CPU count.

    Returns:
        xs_view: A view or representation of the sorted results stored in the `store` after the
        optimization process is complete.
    """
    sg = SeedSequence()
    rgs = [Generator(PCG64DXSM(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
            args=(pid, rgs, fun, weight_bounds, ncon, y_exp, 
                  store, optimize, num_retries, value_limits)) for pid in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    store.sort()
    store.dump()
    return store.xs_view

def _retry_loop(pid, rgs, fun, weight_bounds, ncon, y_exp, 
                store, optimize, num_retries, value_limits):
    """
    Executes a retry loop strategy for multi-objective optimization. This
    process includes generating random weights, scaling them to match
    constraints, optimizing the given function, and storing the results
    if they meet specific criteria. Additionally, it may generate and
    save plots of results for visualization.

    Args:
        pid (int): Process ID to identify and manage the random state
            specific to the process.
        rgs (List[np.random.Generator]): List of random number generators
            associated with each process.
        fun (Callable): The function to be optimized in the retry loop.
        weight_bounds (Bounds): The lower and upper bounds for the weights
            used for scaling in the optimization process.
        ncon (int): The number of constraints to account for during
            the optimization process.
        y_exp (float): Exponent used to adjust the scaling of weights
            in the optimization process.
        store (Store): Instance of a class responsible for managing
            optimization results and storing data.
        optimize (Callable): Optimization function that evaluates
            the objective or constraints based on inputs.
        num_retries (int): The number of retries allowed for achieving
            a successful optimization result.
        value_limits (Optional[List[float]]): Optional limits
            enforced on the objective values from the optimization results.
            If None, no objective value constraints are applied.
    """
    store.create_xs_view()
    lower = store.lower
    wlb = np.array(weight_bounds.lb)
    wub = np.array(weight_bounds.ub)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):    
        while store.get_runs_compare_incr(num_retries):      
            try:       
                rg = rgs[pid]
                w = rg.uniform(size=len(wub))          
                w /= _avg_exp(w, y_exp) # correct scaling
                w = wlb + w * (wub - wlb)
                wrapper = mo_wrapper(fun, w, ncon, y_exp)  
                x, y, evals = optimize(wrapper.eval, Bounds(store.lower, store.upper), None, 
                                         [rg.uniform(0.05, 0.1)]*len(lower), rg, store)
                objs = wrapper.mo_eval(x) # retrieve the objective values
                if value_limits is None or all([objs[i] < value_limits[i] for i in range(len(w))]):
                    store.add_result(y, x, evals, np.inf)   
                    if not store.plot_name is None:
                        name = store.plot_name + "_moretry_" + str(store.get_count_evals())
                        xs = np.array(store.get_xs())
                        ys = np.array([fun(x) for x in xs])
                        np.savez_compressed(name, xs=xs, ys=ys) 
                        plot(name, ncon, xs, ys)
            except Exception as ex:
                print(str(ex))
            
def pareto(xs: np.ndarray, ys: np.ndarray):
    """
    Filter and sort elements based on Pareto efficiency.

    This function calculates the Pareto front for the provided arrays, `xs`
    and `ys`. The Pareto front contains elements that are not dominated
    by any other element in the dataset. The function returns the filtered
    and sorted elements that belong to the Pareto front sorted by the
    values in the associated `ys` array.

    Args:
        xs: A NumPy array representing the input data values. The
            `xs` array contains the attributes aligned with `ys`.
        ys: A NumPy array indicating the criteria or objectives that
            the Pareto efficiency is calculated upon.

    Returns:
        A tuple containing two NumPy arrays:
        - A filtered and sorted array of elements from `xs`
          corresponding to the Pareto front.
        - A filtered and sorted array of elements from `ys`
          corresponding to the Pareto front.
    """
    par = _pareto(ys)
    xp = xs[par]
    yp = ys[par]
    ya = np.argsort(yp.T[0])
    return xp[ya], yp[ya]
     
class mo_wrapper(object):
    """
    Wraps a multi-objective evaluation function with additional functionality.

    This class encapsulates a multi-objective evaluation function, integrating
    weights for objectives and constraints, and provides methods to evaluate the
    function with weight-driven adjustments and constraints handling. It is
    intended for use in scenarios requiring objective evaluations with flexible
    weighting and constraint violations.

    Attributes:
        fun (Callable): The multi-objective evaluation function to be wrapped and
            processed.
        weights (list[float]): List of weights, where the first set applies to
            objectives, and the remainder applies to constraints.
        ny (int): Total number of weights, corresponding to the number of
            objectives and constraints combined.
        nobj (int): The number of objective components in the evaluation.
        ncon (int): The number of constraint components in the evaluation.
        y_exp (int): The exponential factor for averaging the weighted evaluation
            results.
    """
   
    def __init__(self, fun, weights, ncon, y_exp=2):
        """
        Initializes an object with specified function, weights, constraints, and an optional exponent parameter.

        Args:
            fun: Callable. The function to be evaluated.
            weights: List[float]. The weights associated with the objectives.
            ncon: int. The number of constraints.
            y_exp: int, optional. The exponent for the calculations, default is 2.
        """
        self.fun = fun
        self.weights = weights 
        self.ny = len(weights)
        self.nobj = self.ny - ncon
        self.ncon = ncon
        self.y_exp = y_exp

    def eval(self, x):
        """
        Evaluates the provided function and computes a weighted result. If there are
        violations in the constraints, their contribution to the result is added.

        Args:
            x: Array-like, input to the function to be evaluated.

        Returns:
            float: Weighted result of the evaluation including contributions from
            constraints, if any.
        """
        y = self.fun(np.array(x))
        weighted = _avg_exp(self.weights*y, self.y_exp)
        if self.ncon > 0: # check constraint violations
            violations = np.fromiter((i for i in range(self.nobj, self.ny) if y[i] > 0), dtype=int)
            if len(violations) > 0:
                weighted += sum(self.weights[violations])     
        return weighted
            
    def mo_eval(self, x):
        """
        Evaluates a given function on the input after converting it to a NumPy array.

        Args:
            x: The input value to be evaluated. Should be compatible with NumPy array
               operations.

        Returns:
            The result of the function evaluation after processing the input.
        """
        return self.fun(np.array(x))
        
def minimize_plot(name: str, 
                  optimizer: Optimizer, 
                  fun: Callable[[ArrayLike], float], 
                  bounds: Bounds, 
                  weight_bounds, 
                  ncon: Optional[int] = 0, 
                  value_limits: Optional[ArrayLike] = None, 
                  num_retries: Optional[int] = 1024, 
                  exp: Optional[float] = 2.0, 
                  workers: Optional[int] = mp.cpu_count(),
                  statistic_num = 0, plot_name = None):
    """
    Minimizes a given function using a specified optimizer and plots the results.

    This function performs optimization on a target function within the specified
    bounds and additional constraints. The results of the optimization, including
    inputs and outputs, are saved to a compressed file, and a plot is generated
    to visualize the optimization results.

    Args:
        name (str): The base name for the output files and plots.
        optimizer (Optimizer): The optimization algorithm to be used.
        fun (Callable[[ArrayLike], float]): The target function to minimize.
        bounds (Bounds): The bounds within which the optimization is performed.
        weight_bounds: Constraints on weight values for optimization.
        ncon (Optional[int]): The number of constraints in optimization. Defaults
            to 0.
        value_limits (Optional[ArrayLike]): Limits for input values in the
            optimization process. Defaults to None.
        num_retries (Optional[int]): Number of retries for optimization in case
            of failure or randomness. Defaults to 1024.
        exp (Optional[float]): Exponent for value weighting during optimization.
            Defaults to 2.0.
        workers (Optional[int]): Number of parallel workers used during
            optimization. Defaults to the number of CPU cores available.
        statistic_num: Statistical parameter to control optimization behavior.
        plot_name: Custom name for the output plot.

    """
    time0 = time.perf_counter() # optimization start time
    name += '_' + optimizer.name
    logger.info('optimize ' + name) 
    xs, ys = minimize(fun, bounds, weight_bounds, ncon,
             value_exp = exp,
             value_limits = value_limits,
             num_retries = num_retries,              
             optimizer = optimizer,
             workers = workers,
             statistic_num = statistic_num, plot_name = plot_name)
    logger.info(name + ' time ' + str(dtime(time0))) 
    np.savez_compressed(name, xs=xs, ys=ys) 
    plot(name, ncon, xs, ys)
    
def plot(name, ncon, xs, ys, eps = 1E-2, all=True, interp=False, plot3d=False):
    """
    Plots the feasible solution space and Pareto front for a given set of data points.

    This function processes data points and constraints to determine the feasible
    region and subsequently calculates the Pareto-optimal front. It can also
    generate visualizations of all solutions and the Pareto front if required.

    Args:
        name (str): The name for the visualization files.
        ncon (int): The number of constraint functions.
        xs (array-like): A collection of decision variable values.
        ys (array-like): A collection of corresponding objective function values.
        eps (float): Tolerance value for feasibility conditions. Default is 1E-2.
        all (bool): Flag to indicate if plotting all solutions is required.
            Default is True.
        interp (bool): Flag to determine if interpolation is applied to the
            Pareto front plot. Default is False.
        plot3d (bool): Flag indicating whether to create a 3D Pareto front plot.
            Default is False.

    Raises:
        Exception: If an error occurs during computation or plotting.
    """
    try:  
        if ncon > 0: # select feasible
            ycon = np.array([np.maximum(y[-ncon:], 0) for y in ys])  
            con = np.sum(ycon, axis=1)
            nobj = len(ys[0]) - ncon
            feasible = np.fromiter((i for i in range(len(ys)) if con[i] < eps), dtype=int)
            if len(feasible) > 0:
                xs, ys = xs[feasible], np.array([y[:nobj] for y in ys[feasible]])
            else:
                print("no feasible")
                return
        if all:
            retry.plot(ys, 'all_' + name + '.png', interp=False)
        xs, ys = pareto(xs, ys)
        for x, y in zip(xs, ys):
            print(str(list(y)) + ' ' + str([round(xi,5) for xi in x]))
        retry.plot(ys, 'front_' + name + '.png', interp=interp, plot3d=plot3d)
    except Exception as ex:
        print(str(ex))

def adv_minimize_plot(name: str, 
                      optimizer: Optimizer, 
                      fun: Callable[[ArrayLike], float], 
                      bounds: Optional[Bounds],
                      value_limit: Optional[float] = np.inf, 
                      num_retries: Optional[int] = 1024, 
                      statistic_num: Optional[int] = 0):
    """
    Minimizes a given function using the provided optimizer, retries for better results,
    plots the outcomes, and saves the results.

    This function performs a smart optimization by leveraging retry strategies
    to achieve better outcomes. It evaluates the function over all inputs after
    optimization, generates convergence plots, and computes a Pareto front, saving
    the associated data and visuals.

    Args:
        name (str): The base name for the output files generated during the
            optimization process.
        optimizer (Optimizer): The optimization algorithm to be used for minimizing
            the function.
        fun (Callable[[ArrayLike], float]): The objective function to minimize.
            It should accept an input array-like variable and return a float score.
        bounds (Optional[Bounds]): Bounds for the optimization process. Defines
            the variable limits for the optimization.
        value_limit (Optional[float]): A threshold to terminate the optimization
            early when the function's value is below this limit. Default is infinity.
        num_retries (Optional[int]): The number of retries for optimization to improve
            results when the process fails or is suboptimal. Default is 1024.
        statistic_num (Optional[int]): A statistic counter for internal use during
            optimization or retries. Default is 0.
    """
    time0 = time.perf_counter() # optimization start time
    name += '_smart_' + optimizer.name
    logger.info('smart optimize ' + name) 
    store = advretry.Store(lambda x:fun(x)[0], bounds, capacity=5000,
                           num_retries=num_retries, statistic_num = statistic_num) 
    advretry.retry(store, optimizer.minimize, value_limit)
    xs = np.array(store.get_xs())
    ys = np.fromiter((fun(x) for x in xs), dtype=float)
    retry.plot(ys, '_all_' + name + '.png', interp=False)
    np.savez_compressed(name , xs=xs, ys=ys)
    xs, front = pareto(xs, ys)
    logger.info(name+ ' time ' + str(dtime(time0))) 
    retry.plot(front, '_front_' + name + '.png')

def _avg_exp(y, y_exp):
    """
    Calculates the generalized mean (also known as the power mean) of a list of numbers.

    This function computes a weighted generalized mean of a list of numbers `y` using
    the specified exponent `y_exp`. The calculation involves summing the power of each
    element in the list to the given exponent and then applying the inverse of the same
    exponent to the resulting sum.

    Args:
        y: List or iterable of numbers for which the generalized mean will be calculated.
        y_exp: Exponent parameter to control the type of mean (e.g., arithmetic, geometric).

    Returns:
        Weighted generalized mean computed from the input list based on the provided exponent.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = sum([y[i]**y_exp for i in range(len(y))])**(1.0/y_exp)
    return weighted

def _pareto_values(ys):
    """
    Determines the Pareto-optimal solutions from a given set of objective values.

    The function processes a given 2D array of objective values and identifies
    the Pareto-optimal points among them. Pareto-optimal solutions are those that
    cannot be dominated by any other solutions in all objectives.

    Args:
        ys (numpy.ndarray): A 2D array where each row represents a solution, and
            each column corresponds to an objective.

    Returns:
        numpy.ndarray: A 2D array containing only the Pareto-optimal solutions
            from the input.
    """
    ys = ys[ys.sum(1).argsort()[::-1]]
    undominated = np.ones(ys.shape[0], dtype=bool)
    for i in range(ys.shape[0]):
        n = ys.shape[0]
        if i >= n:
            break
        undominated[i+1:n] = (ys[i+1:] >= ys[i]).any(1) 
        ys = ys[undominated[:n]]
    return ys

def _pareto(ys):
    """
    Identifies the Pareto-optimal points from the given set of points.

    Pareto-optimal points are those for which no other point dominates them. A point
    dominates another if it is less than or equal to the other point in all dimensions
    and strictly less in at least one dimension. This function removes dominated points
    from the dataset iteratively.

    Args:
        ys (np.ndarray): A 2D array where each row represents a point in a multidimensional
            space, and each column corresponds to a dimension.

    Returns:
        np.ndarray: An array of indices representing the Pareto-optimal points in the
            provided dataset.
    """
    pareto = np.arange(ys.shape[0])
    index = 0  # Next index to search for
    while index < len(ys):
        mask = np.any(ys < ys[index], axis=1)
        mask[index] = True
        pareto = pareto[mask]  # Remove dominated points
        ys = ys[mask]
        index = np.sum(mask[:index])+1
    return pareto
    
