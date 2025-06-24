# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - advretry.py

 Description:
  - This module implements an advanced retry mechanism for optimization tasks
    using the Fast CMA-ES algorithm. It provides functionality for parallel
    evaluations, statistical tracking, and result persistence.
  - The retry mechanism allows for multiple attempts to optimize a function
    while managing resources efficiently. It supports parallel processing and
    statistical analysis of the optimization process.

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

import time
import os
import math
import threadpoolctl
import _pickle as cPickle
import bz2
import ctypes as ct
import numpy as np
from numpy.linalg import norm
import random
import multiprocessing as mp
from multiprocessing import Process
from numpy.random import Generator, PCG64DXSM, SeedSequence
from scipy.optimize import OptimizeResult, Bounds
from loguru import logger
from fcmaes.retry import _convertBounds, plot, Shared2d
from fcmaes.optimizer import Optimizer, dtime, fitting, de_cma

from typing import Optional, Callable, List, Tuple
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def minimize(fun: Callable[[ArrayLike], float], 
             bounds: Bounds,
             value_limit: Optional[float] = np.inf,
             num_retries: Optional[int] = 5000,
             workers: Optional[int] = mp.cpu_count(),
             popsize: Optional[int] = 31,
             min_evaluations: Optional[int] = 1500,
             max_eval_fac: Optional[int] = None,
             check_interval: Optional[int] = 100,
             capacity: Optional[int] = 500,
             stop_fitness: Optional[float] = -np.inf,
             optimizer: Optional[Optimizer] = None,
             statistic_num: Optional[int] = 0,
             datafile: Optional[str]  = None
             ) -> OptimizeResult:
    """
    Minimizes an objective function using a specified optimizer with options for retrying
    and parallel evaluations, storing intermediate and final results.

    Args:
        fun (Callable[[ArrayLike], float]): The objective function to be minimized.
            It should accept a single argument as an array-like structure and return
            a float value representing the function evaluation.
        bounds (Bounds): The bounds of the search space for the optimization problem.
        value_limit (Optional[float]): The optional threshold for the objective function value.
            If exceeded, the optimization process will terminate.
        num_retries (Optional[int]): The number of retries allowed for optimization attempts.
        workers (Optional[int]): The number of workers available for parallel computation.
        popsize (Optional[int]): The size of the population in the optimization algorithm.
        min_evaluations (Optional[int]): The minimum number of function evaluations to perform
            before considering termination.
        max_eval_fac (Optional[int]): The maximum number of evaluation factors allowed.
        check_interval (Optional[int]): The interval at which evaluations are checked during
            retries.
        capacity (Optional[int]): The capacity of the storage to hold records and data
            during optimization.
        stop_fitness (Optional[float]): The stopping criteria for fitness. Optimization halts
            if this value is achieved or surpassed.
        optimizer (Optional[Optimizer]): The optimization algorithm to be used. If None is
            provided, a default optimizer is created.
        statistic_num (Optional[int]): The number of statistical records to maintain if required.
        datafile (Optional[str]): The path to the file used to store or load intermediate
            optimization data.

    Returns:
        OptimizeResult: The result of the optimization process, which includes details like
            the best-found solution, its fitness value, and related metadata about the
            optimization process.
    """

    if optimizer is None:
        optimizer = de_cma(min_evaluations, popsize, stop_fitness)     
    if max_eval_fac is None:
        max_eval_fac = int(min(50, 1 + num_retries // check_interval))
    store = Store(fun, bounds, max_eval_fac, check_interval, capacity, num_retries, 
                  statistic_num, datafile)
    if not datafile is None:
        try:
            store.load(datafile)
        except:
            pass
    return retry(store, optimizer.minimize, value_limit, workers, stop_fitness)

def retry(store: Store, 
          optimize: Callable, 
          value_limit:Optional[float] = np.inf, 
          workers=mp.cpu_count(), 
          stop_fitness = -np.inf) -> OptimizeResult:
    """
    Retries the optimization process using multiple worker processes and random
    number generators. This function parallelizes the optimization task, applies
    a stopping criterion based on the provided fitness value, and selects the best
    result from the optimization attempts.

    Args:
        store (Store): Stores results of optimization during the process.
        optimize (Callable): The optimization function to be applied.
        value_limit (Optional[float]): The upper limit for the function value
            considered in optimization. Defaults to positive infinity (np.inf).
        workers (int): The number of parallel workers to allocate for the process.
            Defaults to the total number of CPU cores available.
        stop_fitness (float): The stopping fitness criterion for the optimization
            process. The optimization stops when this value is reached. Defaults
            to negative infinity (-np.inf).

    Returns:
        OptimizeResult: The result of the optimization process containing the best
        solution found (x), the fitness of this solution (fun), the number of
        function evaluations (nfev), and a success flag (success).
    """
    sg = SeedSequence()
    rgs = [Generator(PCG64DXSM(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
            args=(pid, rgs, store, optimize, value_limit, stop_fitness)) for pid in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    store.sort()
    store.dump()
    return OptimizeResult(x=store.get_x_best(), fun=store.get_y_best(), 
                          nfev=store.get_count_evals(), success=True)

def minimize_plot(name: str, 
                  optimizer: Optimizer, 
                  fun: Callable[[ArrayLike], float], 
                  bounds: Bounds, 
                  value_limit: Optional[float] = np.inf, 
                  plot_limit: Optional[float] = np.inf, 
                  num_retries: Optional[int] = 1024, 
                  workers: Optional[int] = mp.cpu_count(), 
                  stop_fitness: Optional[float] = -np.inf, 
                  statistic_num: Optional[int] = 5000) -> OptimizeResult:
    """
    Minimizes an objective function using a specified optimizer and plots the optimization
    progress. The function performs multiple retries, applies value and plot limits,
    and saves the optimization improvements for further analysis.

    Args:
        name (str): The base name for the optimization run, used in logs and
            output file names.
        optimizer (Optimizer): The optimization algorithm instance to be used.
        fun (Callable[[ArrayLike], float]): The objective function to minimize.
        bounds (Bounds): Bounds for the input space of the optimization.
        value_limit (Optional[float]): The threshold for the function value beyond
            which the results are not considered. Defaults to infinity.
        plot_limit (Optional[float]): The limit for the maximum function value to
            include in the plot. Defaults to infinity.
        num_retries (Optional[int]): The number of optimization retries allowed.
            Defaults to 1024.
        workers (Optional[int]): The number of parallel workers to use during
            optimization. Defaults to the number of CPU cores available.
        stop_fitness (Optional[float]): The fitness threshold to stop the
            optimization early if reached. Defaults to negative infinity.
        statistic_num (Optional[int]): The number of stored samples for statistical
            analysis during optimization. Defaults to 5000.

    Returns:
        OptimizeResult: The results of the optimization, which include details on
        the best solution found, its fitness, and other metrics.
    """
    time0 = time.perf_counter() # optimization start time
    name += '_' + optimizer.name
    logger.info('optimize ' + name)       
    store = Store(fun, bounds, capacity = 500, statistic_num = statistic_num, 
                  num_retries=num_retries)
    ret = retry(store, optimizer.minimize, value_limit, workers, stop_fitness)
    impr = store.get_improvements()
    np.savez_compressed(name, ys=impr)
    filtered = np.array([imp for imp in impr if imp[1] < plot_limit])
    if len(filtered) > 0: impr = filtered
    logger.info(name + ' time ' + str(dtime(time0))) 
    plot(impr, 'progress_aret.' + name + '.png', label = name, 
         xlabel = 'time in sec', ylabel = r'$f$')
    return ret
 
class Store(object):
    """
    Manages the storing, evaluation, and tracking of data in an optimization problem.

    The class enables handling function evaluations and implements tools for statistics
    tracking, result persistence, and multiprocessing compatibility. It facilitates managing
    optimization tasks with specific bounds, ensuring efficient computational resource usage
    and tracking statistical progress over iterations.

    Attributes:
        fun (Callable[[ArrayLike], float]): The fitness function to be optimized.
        lower (ArrayLike): The lower bounds for the optimization problem.
        upper (ArrayLike): The upper bounds for the optimization problem.
        delta (ArrayLike): The difference between upper and lower bounds.
        capacity (int): Maximum storage capacity for evaluated results.
        num_retries (int): Maximum number of retries during the optimization process.
        eval_fac_incr (float): Increment factor for evaluation adjustments.
        max_eval_fac (int): Maximum evaluation factor throughout the retries.
        check_interval (int): Interval for sorting the evaluation store.
        dim (int): Dimension of the optimization problem based on bounds.
        t0 (float): Timestamp indicating the start of evaluations (used for timing statistics).
        statistic_num (int): Number of statistical points maintained for tracking.
        datafile (Optional[str]): Path to file for saving and loading data.
    """
         
    def __init__(self, 
                 fun: Callable[[ArrayLike], float], # fitness function
                 bounds: Bounds, # bounds of the objective function arguments
                 max_eval_fac: Optional[int] = None, # maximal number of evaluations factor
                 check_interval: Optional[int] = 100, # sort evaluation store after check_interval iterations
                 capacity: Optional[int] = 500, # capacity of the evaluation store
                 num_retries: Optional[int] = None,
                 statistic_num: Optional[int] = 0,
                 datafile: Optional[str] = None
               ):
        """
        Initializes an instance of the class with parameters and attributes required for
        managing function optimization within specified bounds using shared multiprocessing
        resources. Sets up the evaluation store, statistic tracking, and random generator.

        Args:
            fun: A fitness function to be optimized. It takes an array-like input and returns
                a float value.
            bounds: Bounds of the objective function arguments defined as an instance of the
                Bounds class.
            max_eval_fac: Optional; Maximum number of evaluations factor. Defaults to None. If
                None, it is calculated based on `num_retries` and `check_interval`.
            check_interval: Optional; Number of iterations after which the evaluation store is
                sorted. Defaults to 100.
            capacity: Optional; Capacity of the evaluation store. Defaults to 500.
            num_retries: Optional; Maximum number of retries allowed. If None, it is calculated
                based on `max_eval_fac` and `check_interval`. Defaults to None.
            statistic_num: Optional; Number of statistics points to be maintained for tracking.
                Set to 0 to disable statistics. Defaults to 0.
            datafile: Optional; Path to a file for saving/loading relevant data. Defaults to None.
        """
        self.fun = fun
        self.lower, self.upper = _convertBounds(bounds)
        self.delta = self.upper - self.lower      
        self.capacity = capacity
        if max_eval_fac is None:
            if num_retries is None:
                max_eval_fac = 50
            else:
                max_eval_fac = int(min(50, 1 + num_retries // check_interval))
        if num_retries == None:
            num_retries = max_eval_fac * check_interval
        self.num_retries = num_retries
        # increment eval_fac so that max_eval_fac is reached at last retry
        self.eval_fac_incr = max_eval_fac / (num_retries/check_interval)
        self.max_eval_fac = max_eval_fac
        self.check_interval = check_interval       
        self.dim = len(self.lower)
        self.t0 = time.perf_counter()
        self.statistic_num = statistic_num
        self.datafile = datafile
        self.rg = random.Random()
        #self.rg = Generator(PCG64DXSM()))
        #self.rg = Generator(PCG64DXSM(random.randint(0, 2**63 - 1)))
    
        #shared between processes
        self.add_mutex = mp.Lock()    
        self.check_mutex = mp.Lock()                     
        self.xs = mp.RawArray(ct.c_double, capacity * self.dim)
        self.ys = mp.RawArray(ct.c_double, capacity)                  
        self.eval_fac = mp.RawValue(ct.c_double, 1)
        self.count_evals = mp.RawValue(ct.c_long, 0)   
        self.count_runs = mp.RawValue(ct.c_int, 0) 
        self.num_stored = mp.RawValue(ct.c_int, 0)  
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.worst_y = mp.RawValue(ct.c_double, np.inf)  
        self.best_x = mp.RawArray(ct.c_double, self.dim)
 
        if statistic_num > 0:  # enable statistics                          
            self.statistic_num = statistic_num
            self.time = mp.RawArray(ct.c_double, self.statistic_num)
            self.val = mp.RawArray(ct.c_double, self.statistic_num)
            self.si = mp.RawValue(ct.c_int, 0)
            self.sevals = mp.RawValue(ct.c_long, 0)
            self.bval = mp.RawValue(ct.c_double, np.inf)

    # register improvement - time and value
    def wrapper(self, x: ArrayLike) -> float:
        """
        Wrapper function to evaluate a given function, update statistics, and log the
        results. It is typically used in optimization procedures to monitor evaluations
        and track progress over iterations.

        Args:
            x (ArrayLike): The input array to evaluate the function.

        Returns:
            float: The evaluated function output for the given input array.
        """
        y = self.fun(x)
        self.sevals.value += 1
        if y < self.bval.value:
            self.bval.value = y
            si = self.si.value
            if si < self.statistic_num - 1:
                self.si.value = si + 1
            self.time[si] = dtime(self.t0)
            self.val[si] = y  
            logger.info(str(self.time[si]) + ' '  + 
                      str(self.sevals.value) + ' ' + 
                      str(y) + ' ' + 
                      str(list(x)))
        return y
                    
    # persist store
    def save(self, name: str):
        """
        Saves the current data of the instance to a compressed file using bz2 and cPickle modules.

        The method compresses and serializes the data obtained from `get_data` and saves
        it to a file with the specified name appended by the '.pbz2' extension.

        Args:
            name (str): The desired name for the output file, excluding the extension.
        """
        with bz2.BZ2File(name + '.pbz2', 'w') as f:
            cPickle.dump(self.get_data(), f)

    def load(self, name: str):
        """Loads and processes data from a compressed file.

        This method allows loading data from a file compressed with BZ2 and serialized
        with cPickle. After loading, the data is processed or assigned using the
        `set_data` method of the instance.

        Args:
            name (str): The name of the file (without extension) to load data from.
        """
        data = cPickle.load(bz2.BZ2File(name + '.pbz2', 'rb'))
        self.set_data(data)
  
    def get_data(self) -> List:
        """
        Retrieves and aggregates data from various attributes and methods.

        This method gathers data by calling several other methods and accessing a specific
        attribute. It consolidates these pieces of data into a single list and returns it.

        Returns:
            List: A list containing data from `get_xs()`, `get_ys()`, `get_x_best()`,
            `get_y_best()` methods, and the `num_stored.value` attribute.
        """
        data = []
        data.append(self.get_xs())
        data.append(self.get_ys())
        data.append(self.get_x_best())
        data.append(self.get_y_best())
        data.append(self.num_stored.value)
        return data
        
    def set_data(self, data: ArrayLike):
        """
        Sets the data for an internal storage structure and processes it.

        Args:
            data (ArrayLike): A multi-dimensional array-like structure containing:
                - data[0]: The x-coordinates.
                - data[1]: The y-coordinates.
                - data[2]: A subset of best x-coordinates.
                - data[3]: The best y-coordinate.
                - data[4]: The total count of stored elements.

        """
        xs = data[0]
        ys = data[1]
        for i in range(len(ys)):
            self.replace(i, ys[i], xs[i])
        self.best_x[:] = data[2][:]
        self.best_y.value = data[3]
        self.num_stored.value = data[4]
        self.sort()
               
    def get_improvements(self) -> np.ndarray:
        """
        Calculates and returns an array of improvements based on stored time and value data.

        The method processes the time and value attributes up to the index defined by `si.value`
        and combines them into a structured NumPy array. It allows extracting the corresponding
        values and improvements over the determined slice of the data.

        Returns:
            np.ndarray: A NumPy array containing pairs of time and value up to `si.value`.
        """
        return np.array(list(zip(self.time[:self.si.value], self.val[:self.si.value])))
 
    # get num best values at evenly distributed times
    def get_statistics(self, num: int) -> List:
        """
        Calculates and returns a list of statistics determined by evenly dividing the provided
        time series data into a specified number of segments.

        This function processes time series and value arrays to calculate specific statistics
        based on the distribution of data over a fixed number of intervals. It uses the time
        array to sample the values array at designated points, ensuring that the result represents
        data distributed across the defined segments.

        Args:
            num (int): The number of segments into which the time series data will be divided.

        Returns:
            List: A list containing the computed statistics based on the segmented intervals of
            the time series data.

        Raises:
            IndexError: If the number of segments exceeds the number of available data points.
        """
        ts = self.time[:self.si.value]
        vs = self.val[:self.si.value]
        mt = ts[-1]
        dt = 0.9999999 * mt / num
        stats = []
        ti = 0
        val = vs[0]
        for i in range(num):
            while ts[ti] < (i+1) * dt:
                ti += 1
                val = vs[ti]
            stats.append(val)
        return stats
                                    
    def eval_num(self, max_evals: int) -> int:
        """
        Calculates the evaluation number based on a multiplier and maximum evaluations.

        This method computes the product of a multiplier (`eval_fac.value`) and the
        provided maximum evaluations (`max_evals`). The result is cast to an integer
        and returned.

        Args:
            max_evals (int): The maximum number of evaluations.

        Returns:
            int: The calculated evaluation number.
        """
        return int(self.eval_fac.value * max_evals)
                                               
    def limits(self) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the limits and other related calculations used for determining bounds
        during operations like crossover and mutation. The function calculates adjusted
        bounds, scaled deviations, and other factors based on the input variables, making
        use of random factors and mutex locks for thread safety.

        Args:
            None

        Returns:
            Tuple[float, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            A tuple containing the following elements:
              - A float value representing the initial target point y0.
              - A NumPy array representing the point x1 selected for crossover.
              - A NumPy array representing the calculated lower bounds for crossover.
              - A NumPy array representing the calculated upper bounds for crossover.
              - A NumPy array representing the scaled standard deviations for crossover.

        Raises:
            None
        """
        diff_fac = self.rg.uniform(0.5, 1.0)
        lim_fac =  self.rg.uniform(2.0, 4.0) * diff_fac
        with self.add_mutex:
            i, j = self.crossover()
            if i < 0:
                return math.inf, None, None, None, None
            x0 = np.asarray(self.get_x(i))
            x1 = np.asarray(self.get_x(j))
            y0 = self.get_y(i)
             
        deltax = np.abs(x1 - x0)
        delta_bound = np.maximum(0.0001, lim_fac * deltax)
        lower = np.maximum(self.lower, x0 - delta_bound)
        upper = np.minimum(self.upper, x0 + delta_bound)
        sdev = np.clip(diff_fac * deltax / self.delta, 0.001, 0.5)        
        return y0, x1, lower, upper, sdev
                 
    def distance(self, xprev: np.ndarray, x: np.ndarray) -> float:
        """
        Calculates the normalized distance between two points in a Euclidean space.

        The distance is normalized based on the difference between two points, scaled
        by a pre-defined parameter `delta`, and divided by the square root of the
        number of dimensions (`dim`).

        Args:
            xprev (np.ndarray): A numpy array representing the previous point.
            x (np.ndarray): A numpy array representing the current point.

        Returns:
            float: The normalized distance between `xprev` and `x`.
        """
        return norm((x - xprev) / self.delta) / math.sqrt(self.dim)
        
    def replace(self, i: int, y: float, x: np.ndarray):
        """
        Replaces the y and x values at a specified index.

        This method updates the y and x values in the object's internal data structure at the index
        provided by the user.

        Args:
            i (int): Index at which to set the new y and x values.
            y (float): The new value for y to be set at the specified index.
            x (np.ndarray): The new value for x to be set at the specified index.
        """
        self.set_y(i, y)
        self.set_x(i, x)
 
    def crossover(self) -> Tuple[int,int]: # Choose two good entries for recombination
        """
        Selects two good entries for recombination based on a probabilistic threshold.

        This method attempts to choose two distinct indices from a pool based on a
        random limit derived from the current number of stored entries. The method
        performs several attempts to identify these indices under the defined constraints.
        If successful, it returns the selected indices; otherwise, it returns default
        values indicating no valid selection.

        Returns:
            Tuple[int, int]: A tuple containing two indices selected for recombination.
            If the selection fails, returns (-1, -1).
        """
        n = self.num_stored.value
        if n < 2:
            return -1, -1
        lim = self.rg.uniform(min(0.1*n, 1), 0.2*n)/n
        for _ in range(100):
            i1 = -1
            i2 = -1
            for j in range(n):
                if self.rg.random() < lim:
                    if i1 < 0:
                        i1 = j
                    else:
                        i2 = j
                        return i1, i2
        return -1, -1
            
    def sort(self) -> int:
        """
        Sorts and updates stored data based on the given criteria.

        The method sorts the stored data `ys` in ascending order of their values. It ensures
        diversity by selecting specific data points that meet the distance threshold from the
        two most recently added elements. It then updates the stored data to retain only the
        best 90% of the sorted elements up to the storage capacity.

        Returns:
            int: The updated number of stored elements.

        Raises:
            ValueError: If the number of stored elements is less than 2.
        """
        ns = self.num_stored.value
        if ns < 2:
            return

        ys = np.asarray(self.ys[:ns])
        yi = ys.argsort()

        ys2 = []
        xs2 = []
        for i in range(ns):
            y = ys[yi[i]]
            x = np.asarray(self.get_x(yi[i]))  # preserve diversity
            if np.all([self.distance(xp, x) > 0.15 for xp in xs2[-2:]]): 
                ys2.append(y)
                xs2.append(x)

        ns = min(len(ys2),int(0.9*self.capacity)) # keep 90% best 
        for i in range(ns):
            self.replace(i, ys2[i], xs2[i])
        self.num_stored.value = ns     
        self.worst_y.value = self.get_y(ns-1)
        return ns

    def add_result(self, y: float, x: np.ndarray, evals: int, limit: Optional[float] = np.inf):
        """
        Adds a result to the current optimization process, updating the best result if
        necessary and storing the new result in the internal data structure.

        This method is designed to handle updates in a thread-safe manner. It increments
        the number of evaluations, checks if the new result is within the given limit,
        and updates the best result as well as the storage if the criteria are met. If the
        storage reaches its capacity, it triggers a sorting operation to maintain order.

        Args:
            y (float): The result value to be added.
            x (np.ndarray): The array representing input parameters corresponding to the result.
            evals (int): The number of evaluations associated with this result.
            limit (Optional[float]): The threshold value. Results with a value greater than
                this limit are ignored. Defaults to infinity.
        """
        with self.add_mutex:
            self.count_evals.value += evals
            if y < limit:
                if y < self.best_y.value:
                    self.best_y.value = y
                    self.best_x[:] = x[:]
                    self.dump()
                    if not self.datafile is None:
                        self.save(self.datafile)

                if self.num_stored.value >= self.capacity - 1:
                    self.sort()
                ns = self.num_stored.value
                self.replace(ns, y, x)
                self.num_stored.value = ns + 1
      
    def get_x_best(self) -> np.ndarray:
        """
        Returns a copy of the best solutions stored in the internal state.

        This method retrieves the best solution(s) found, which are stored
        internally, and returns a copy of the data to avoid unintentional
        modifications to the original data.

        Returns:
            np.ndarray: A numpy array containing the best solution(s).
        """
        return np.array(self.best_x[:])

    def get_x(self, pid) -> np.ndarray:
        """
        Retrieves a segment of the `xs` array corresponding to the provided `pid`.

        Args:
            pid (int): Index used to calculate the segment.

        Returns:
            np.ndarray: The segment of the `xs` array.
        """
        return self.xs[pid*self.dim:(pid+1)*self.dim]

    def get_xs(self)-> np.ndarray:
        """
        Builds and returns an array of x values stored in the object.

        This method iterates over the number of stored x values and retrieves each
        value using the `get_x` method. The retrieved values are then compiled into
        a NumPy array, which is returned.

        Returns:
            np.ndarray: A NumPy array containing the retrieved x values.
        """
        return np.array([self.get_x(i) for i in range(self.num_stored.value)])

    def get_y(self, pid: int) -> float:
        """
        Fetches a value for a given ID from a dictionary of float values.

        This method retrieves a corresponding float value from the `ys` dictionary
        based on the provided integer ID. The dictionary `ys` maps integer IDs to
        float values. The function returns the float value associated with the
        supplied `pid`.

        Args:
            pid (int): The unique identifier used to fetch a value from the `ys`
                dictionary.

        Returns:
            float: The value corresponding to the provided `pid` from the `ys`
                dictionary.
        """
        return self.ys[pid]

    def get_ys(self) -> np.ndarray:
        """
        Gets the stored y-values up to the specified count.

        This method retrieves the portion of the y-values list that corresponds to
        the count determined by `num_stored.value`. The result is returned as a
        NumPy array.

        Returns:
            np.ndarray: A NumPy array containing the stored y-values up to the specified count.
        """
        return np.array(self.ys[:self.num_stored.value])

    def get_y_best(self) -> float:
        """
        Returns the best value of y stored in the `best_y` attribute.

        The function accesses the `best_y` attribute and retrieves its value
        as a float.

        Returns:
            float: The best value of y.
        """
        return self.best_y.value

    def get_count_evals(self) -> int:
        """
        Retrieves the value of the evaluation count.

        This method accesses the `count_evals` attribute and returns its integer
        value. It is primarily intended to report the current count of evaluations
        stored in the `count_evals` attribute.

        Returns:
            int: The current evaluation count stored in `count_evals`.
        """
        return self.count_evals.value
  
    def get_count_runs(self) -> int:
        """
        Retrieves the count of runs.

        This method returns the value of the `count_runs` attribute, representing
        the total count of runs made.

        Returns:
            int: The value of the `count_runs` attribute.
        """
        return self.count_runs.value

    def set_x(self, pid, xs):
        """
        Sets a subset of the `xs` list for a specific process ID.

        This function updates a section of the `xs` attribute corresponding to a
        particular process ID (`pid`) based on the provided input values.

        Args:
            pid: Process ID whose section in `xs` is to be updated.
            xs: List of values to update in the specific section of `xs` for the given
                process ID.
        """
        self.xs[pid*self.dim:(pid+1)*self.dim] = xs[:]

    def set_y(self, pid, y):
        """
        Sets the value of y for a given pid within the 'ys' mapping.

        Modifies the associated value of y in the dictionary 'ys' for the
        provided process identifier (pid). This method stores or updates
        the y value tied to a specific pid.

        Args:
            pid: Identifier for the process whose y value is being set.
            y: The value to associate with the given pid in the 'ys' mapping.
        """
        self.ys[pid] = y

    def get_runs_compare_incr(self, limit: float) -> bool:
        """
        Compares the current count of runs against a specified limit and increments internal
        counters accordingly if the limit is not exceeded.

        Args:
            limit (float): The upper threshold to compare against the current count of runs.

        Returns:
            bool: True if the current count is less than the limit and the increment operation was
            performed. False otherwise.
        """
        with self.add_mutex:
            if self.count_runs.value < limit:
                self.count_runs.value += 1
                if self.count_runs.value % self.check_interval == self.check_interval-1:
                    if self.eval_fac.value < self.max_eval_fac:
                        self.eval_fac.value += self.eval_fac_incr
                    self.sort()                
                return True
            else:
                return False 

    def dump(self):
        """
        Logs a summary of the execution metrics and current status.

        This method collects relevant metrics about the execution process and generates a
        formatted log message to provide insights into the current state. Metrics such as
        evaluation counts, best and worst outcomes, and a snapshot of the best solutions are
        included in the message.

        Args:
            None

        Returns:
            None
        """
        Ys = self.get_ys()
        vals = []
        for i in range(min(20, len(Ys))):
            vals.append(round(Ys[i],2))     
        dt = dtime(self.t0)+.000001            
        message = '{0} {1} {2} {3} {4:.6f} {5:.2f} {6} {7} {8!s} {9!s}'.format(
            dt, int(self.count_evals.value / dt), self.count_runs.value, self.count_evals.value, 
            self.best_y.value, self.worst_y.value, self.num_stored.value, int(self.eval_fac.value), 
            vals, self.best_x[:])
        logger.info(message)
   
def _retry_loop(pid, rgs, store, optimize, value_limit, stop_fitness = -np.inf):
    """
    Retries a loop for optimization until stopping criteria are met.

    The function performs optimization processes in a loop, sampling solutions and evaluating their
    fitness to improve a defined objective within bounds. It also incorporates handling parallel
    executions for optimization using private random generator and thread limits.

    Args:
        pid (int): The process or thread identifier for parallel computation.
        rgs (List[RandomState]): A list of random generator states that control the stochastic
            behavior of each process or thread.
        store (Store): The shared data store, containing optimization inherent details like lower
            and upper bounds, number of retries, best fitness achieved, and statistical configurations.
        optimize (Callable): The optimization function responsible for processing sample solutions
            and measuring fitness outcomes.
        value_limit (float): The maximum value allowed for a solution, restricting unacceptable
            outliers in the results.
        stop_fitness (float): The fitness threshold at which the optimization ceases if exceeded
            by best achieved fitness. Defaults to negative infinity.
    """
    fun = store.wrapper if store.statistic_num > 0 else store.fun
    #with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
    while store.get_runs_compare_incr(store.num_retries) and store.best_y.value > stop_fitness:
        if _crossover(fun, store, optimize, rgs[pid]):
            continue
        try:
            rg = rgs[pid]
            dim = len(store.lower)
            sol, y, evals = optimize(fun, Bounds(store.lower, store.upper), None,
                                     [rg.uniform(0.05, 0.1)]*dim, rg, store)
            store.add_result(y, sol, evals, value_limit)
        except Exception as ex:
            continue
 
def _crossover(fun, store, optimize, rg):
    """
    Performs a crossover operation to optimize a function.

    This function attempts to optimize a given function using an optimization
    strategy by utilizing the provided parameters and probabilistic logic to decide
    whether crossover should be performed. It uses the specified random generator
    and optimization function, and updates the solution store upon successful
    optimization completion.

    Args:
        fun: The objective function to be optimized.
        store: An object that stores limits, results, and other related data.
        optimize: A callable function used to perform the optimization.
        rg: A random generator for probabilistic decisions and randomness during
            the optimization process.

    Returns:
        bool: True if the optimization process completes successfully; False otherwise.
    """
    if rg.uniform(0,1) < 0.5:
        return False
    y0, guess, lower, upper, sdev = store.limits()
    if guess is None:
        return False
    guess = fitting(guess, lower, upper) # take X from lower
    try:       
        sol, y, evals = optimize(fun, Bounds(lower, upper), guess, sdev, rg, store)
        store.add_result(y, sol, evals, y0) # limit to y0  
    except:
        return False   
    return True
