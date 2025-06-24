# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - retry.py

 Description:
  - This module implements a retry mechanism for optimization problems
  using parallel optimization. It allows multiple attempts to find a
  solution to a given optimization problem, leveraging different
  optimization strategies such as differential evolution and CMA-ES.
  - The retry mechanism is designed to handle large-scale optimization
  problems efficiently by distributing the workload across multiple
  processes.
  - The module provides a `minimize` function that accepts an objective
  function, bounds, and various parameters to control the optimization
  process.

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
import math
import os
import sys
import threadpoolctl
import ctypes as ct
from scipy import interpolate
import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize import OptimizeResult, Bounds
import multiprocessing as mp
from multiprocessing import Process
from fcmaes.optimizer import de_cma, dtime, Optimizer
from fcmaes.evaluator import is_debug_active, is_trace_active
from loguru import logger
from typing import Optional, Callable, List
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def minimize(fun: Callable[[ArrayLike], float], 
             bounds: Bounds, 
             value_limit: Optional[float] = np.inf,
             num_retries: Optional[int] = 1024,
             workers: Optional[int] = mp.cpu_count(),
             popsize: Optional[int] = 31, 
             max_evaluations: Optional[int] = 50000, 
             capacity: Optional[int] = 500,
             stop_fitness: Optional[float] = -np.inf,
             optimizer: Optional[Optimizer] = None,
             statistic_num: Optional[int] = 0,
             ) -> OptimizeResult:
    """
    Minimizes a given objective function using the specified optimizer and stores the progress.

    This function attempts to find the minimum value of a given function within the bounds
    provided. It incorporates retry logic to handle multiple optimization attempts in case
    of failure or suboptimal results. The optimization process can also operate in a multiprocess
    environment with a configurable number of workers. Additionally, progress data is stored
    to analyze statistical results throughout the optimization.

    Args:
        fun: The objective function to minimize. The function must accept an array-like input
            and return a scalar float as the function value.
        bounds: Bounds for the input variables of the objective function. Defines the
            acceptable search space during optimization.
        value_limit: Upper limit for the acceptable function value. Defaults to infinity.
            After reaching this limit, retries are triggered if configured.
        num_retries: Number of times the optimization process should be retried upon failure
            or suboptimal results. Defaults to 1024.
        workers: The number of parallel workers to use for optimization. Defaults to the
            number of CPU cores available.
        popsize: Population size for the optimization process. Determines the number
            of potential solutions considered at each iteration.
        max_evaluations: The maximum number of evaluations allowed in the optimization
            process. This is a hard limit on computational effort.
        capacity: The storage capacity for retaining past evaluation results or progress
            data. Used for analyzing statistics or performance.
        stop_fitness: The target fitness value for stopping the optimization. When a result
            meets or exceeds this fitness value, the process terminates. Defaults to
            negative infinity.
        optimizer: An optimization algorithm instance to drive the optimization process.
            If not provided, a default optimizer is created based on `de_cma`.
        statistic_num: The statistical metric number to associate with the optimization
            data. Used for tracking or analyzing statistical trends.

    Returns:
        OptimizeResult: The result of the optimization process, including details about
        the optimal solution, number of function evaluations, and success status.
    """

    if optimizer is None:
        optimizer = de_cma(max_evaluations, popsize, stop_fitness)        
    store = Store(fun, bounds, capacity = capacity, statistic_num = statistic_num)
    return retry(store, optimizer.minimize, num_retries, value_limit, workers, stop_fitness)

def retry(store: Store, 
          optimize: Callable, 
          num_retries: int, 
          value_limit: Optional[float] = np.inf, 
          workers: Optional[int] = mp.cpu_count(), 
          stop_fitness: Optional[float] = -np.inf) -> OptimizeResult:
    """
    Retries optimization multiple times using the specified number of worker processes. Each worker
    conducts optimization in parallel and contributes to the shared results store. After all workers
    complete their tasks, the stored results are sorted, persisted, and the best result is returned.

    Args:
        store (Store): A shared result store object that collects optimization results from all workers.
        optimize (Callable): A callable function that performs the optimization for a single run.
        num_retries (int): The number of retries/iterations each worker should perform.
        value_limit (Optional[float]): The upper limit for objective values to consider valid. Defaults
            to positive infinity.
        workers (Optional[int]): The number of worker processes to spawn for parallel optimization.
            Defaults to the number of CPU cores available.
        stop_fitness (Optional[float]): The fitness value at which the optimization can early stop.
            Defaults to negative infinity.

    Returns:
        OptimizeResult: An object containing the best solution found, its objective value, number of
            function evaluations, and a success status.
    """
    sg = SeedSequence()
    rgs = [Generator(PCG64DXSM(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
            args=(pid, rgs, store, optimize, num_retries, value_limit, stop_fitness)) for pid in range(workers)]
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
    Minimizes an optimization problem and plots the improvement process while storing results.

    Args:
        name: A string serving as the descriptive name or title for the optimization process.
        optimizer: An instance of the Optimizer class used for carrying out the optimization.
        fun: A callable that represents the objective function to minimize, which takes
             an array-like input and returns a float value.
        bounds: An instance of the Bounds class specifying the valid range for the optimization.
        value_limit: Optional float indicating the upper limit for significant improvements.
                     Defaults to positive infinity.
        plot_limit: Optional float representing the limit for including improvements in the plot.
                    Defaults to positive infinity.
        num_retries: Optional integer specifying the maximum number of retries for the optimization
                     process. Defaults to 1024.
        workers: Optional integer indicating the number of multiprocessing workers to run
                 the optimization. Defaults to the number of available CPU cores.
        stop_fitness: Optional float defining the fitness value at which to stop the optimization
                      early if achieved. Defaults to negative infinity.
        statistic_num: Optional integer indicating the number of most recent statistics
                       to retain in memory. Defaults to 5000.

    Returns:
        OptimizeResult: The result object generated by the optimizer containing information
                        on the solution, including the best parameters found and associated
                        fitness value.
    """
    time0 = time.perf_counter() # optimization start time
    name += '_' + optimizer.name
    logger.info('optimize ' + name)       
    store = Store(fun, bounds, capacity = 500, statistic_num = statistic_num)
    ret = retry(store, optimizer.minimize, num_retries, value_limit, workers, stop_fitness)
    impr = store.get_improvements()
    np.savez_compressed(name, ys=impr)
    for _ in range(10):
        filtered = np.array([imp for imp in impr if imp[1] < plot_limit])
        if len(filtered) > 0: 
            impr = filtered
            break
        else:
            plot_limit *= 3
    logger.info(name + ' time ' + str(dtime(time0))) 
    plot(impr, 'progress_ret.' + name + '.png', label = name, 
         xlabel = 'time in sec', ylabel = r'$f$')
    return ret

def plot(front: ArrayLike, fname: str, interp: Optional[bool] = True, 
         label: Optional[str] = r'$\chi$', 
         xlabel: Optional[str] = r'$f_1$', ylabel:Optional[str] = r'$f_2$', 
         zlabel: Optional[str] = r'$f_3$', plot3d: Optional[bool] = False, 
         s = 1, dpi=300):
    """
    Plots a given front using either 2D or 3D visualization, with options for
    interpolation and customization of the appearance and labels. If the input
    front has one or more objectives, it chooses the appropriate plotting
    method based on dimensionality, adds interpolated lines if specified, and
    saves the resulting figure to a file.

    Args:
        front: Array-like structure representing the input data points. Each
            row corresponds to a point, and each column represents a dimension
            (objective value).
        fname: str. Path and name of the file where the plot will be saved.
        interp: Optional[bool]. Enables interpolation to smooth the 2D plot
            lines. Default is True.
        label: Optional[str]. Label for the data points on the plot. Default is
            r'$\chi$'.
        xlabel: Optional[str]. Label for the x-axis. Default is r'$f_1$'.
        ylabel: Optional[str]. Label for the y-axis. Default is r'$f_2$'.
        zlabel: Optional[str]. Label for the z-axis. Used only for 3D plots.
            Default is r'$f_3$'.
        plot3d: Optional[bool]. If True and the front has 3 dimensions, generates
            a 3D scatter plot. Default is False.
        s: Plot marker size for scatter points. Default is 1.
        dpi: Resolution of the saved plot in dots per inch. Default is 300.
    """
    if len(front[0]) == 3 and plot3d:
        plot3(front, fname, label, xlabel, ylabel, zlabel)
        return
    if len(front[0]) >= 3:
        for i in range(1, len(front[0])):
            plot(front.T[np.array([0,i])].T, str(i) + '_' + fname, 
                 interp=interp, ylabel = r'$f_{0}$'.format(i+1))     
        return   
    if len(front[0]) == 1:
        ys = np.array(list(zip(range(100), [front[0][0]]*100)))
        plot(ys, str(1) + '_' + fname, 
                 interp=interp, xlabel = '', ylabel = r'$f_{0}$'.format(1))     
        return      
    import matplotlib.pyplot as pl
    fig, ax = pl.subplots(1, 1)
    x = front[:, 0]; y = front[:, 1]
    if interp and len(x) > 2:
        xa = np.argsort(x)
        xs = x[xa]; ys = y[xa]
        x = []; y = []
        for i in range(len(xs)): # filter equal x values
            if i == 0 or xs[i] > xs[i-1] + 1E-5:
                x.append(xs[i]); y.append(ys[i])
        tck = interpolate.InterpolatedUnivariateSpline(x,y,k=1)
        x = np.linspace(min(x),max(x),1000)
        y = [tck(xi) for xi in x]
    ax.scatter(x, y, label=label, s=s)
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.savefig(fname, dpi=dpi)
    pl.close('all')

def plot3(front: ArrayLike, fname: str, label: Optional[str] =r'$\chi$', 
         xlabel: Optional[str] = r'$f_1$', ylabel: Optional[str] = r'$f_2$', 
         zlabel: Optional[str] = r'$f_3$'):
    """
    Creates and saves a 3D scatter plot from given data.

    This function generates a 3D scatter plot with labeled axes using the data
    provided in a 2D array-like object. The plot is saved to the specified file
    with high resolution. The labels for each axis and the plot can be customized.

    Args:
        front (ArrayLike): A 2D array-like object containing the data points to
            plot. It should have exactly three columns corresponding to the
            x, y, and z coordinates.
        fname (str): The file name where the generated plot should be saved.
        label (Optional[str]): The label for the scatter plot. Defaults to r'$\chi$'.
        xlabel (Optional[str]): The label for the x-axis. Defaults to r'$f_1$'.
        ylabel (Optional[str]): The label for the y-axis. Defaults to r'$f_2$'.
        zlabel (Optional[str]): The label for the z-axis. Defaults to r'$f_3$'.
    """
    import matplotlib.pyplot as pl
    fig = pl.figure()
    ax = fig.add_subplot(projection='3d')
    x = front[:, 0]; y = front[:, 1]; z = front[:, 2]
    ax.scatter(x, y, z, label=label, s=1)
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()
    #pl.show()
    fig.savefig(fname, dpi=300)
    pl.close('all')


dtype_map = {
    'int32': ct.c_int32,
    'int64': ct.c_int64,
    'float32': ct.c_float,
    'float64': ct.c_double,
}
    
class Shared2d():
    """A 2D shared memory array wrapper.

    This class allows sharing of a 2D NumPy array between processes using
    shared memory. It provides methods to set and retrieve the data, as well
    as to update specific rows of the array.

    Attributes:
        rows (int): Number of rows in the 2D array.
        cols (int): Number of columns in the 2D array.
        dtype: Data type of the elements in the 2D array.
    """
    def __init__(self, xs):
        """
        Initializes an instance with given data, extracting shape and data type, and setting up a shared memory array.

        Args:
            xs: numpy.ndarray
                The input array used to initialize the instance.

        """
        self.rows, self.cols = xs.shape
        self.dtype = xs.dtype
        self.ra = mp.RawArray(dtype_map[str(xs.dtype)], self.rows*self.cols)       
        self.set(xs)
        
    def set_i(self, i, x):
        """
        Updates a specific row in the object's view with new data.

        This method modifies the specified row of the object's view by replacing it
        with the provided data.

        Args:
            i (int): Index of the row to update within the view.
            x: New data to replace the current row content. The type should match
               the requirements of the view.
        """
        self.view()[i, :] = x
    
    def view(self):
        """
        Converts the binary data buffer to a NumPy array and reshapes it according to
        the specified rows and columns.

        Returns:
            numpy.ndarray: A reshaped NumPy array created from the binary buffer.
        """
        return np.frombuffer(self.ra, dtype=self.dtype).reshape((self.rows, self.cols))
        
    def set(self, xs):
        """
        Sets the values of the object by copying the given array to its data view.

        Args:
            xs: The array-like object whose values are to be copied.

        """
        np.copyto(self.view(), xs)
 
class Store(object):
    """
    Manages and tracks the optimization process by storing evaluation results, maintaining statistics,
    and providing utility functions for analyzing improvements.

    This class is designed for optimization tasks, allowing the storage of evaluated points,
    their objective function values, and statistical tracking of improvements. It supports
    multi-processed environments using shared memory for data storage.

    Attributes:
        fun (Callable[[ArrayLike], float]): The fitness function to be optimized.
        lower (ArrayLike): The lower bounds of the objective function arguments.
        upper (ArrayLike): The upper bounds of the objective function arguments.
        capacity (int): Maximum capacity of the evaluation store.
        check_interval (int): Number of iterations after which the evaluation memory is sorted.
        dim (int): Dimensionality of the problem space.
        xs (Shared2d): Shared 2D array for storing the evaluated points in the search space.
        ys (multiprocessing.RawArray): Shared array for storing the corresponding function values.
        count_evals (multiprocessing.RawValue): Counter for the total number of evaluations performed.
        count_runs (multiprocessing.RawValue): Counter for the number of optimization runs.
        num_stored (multiprocessing.RawValue): Counter for the number of currently stored evaluations.
        count_stat_runs (multiprocessing.RawValue): Counter for the number of recorded statistics.
        t0 (float): Reference time for tracking durations.
        mean (multiprocessing.RawValue): Running mean of function values in the store.
        qmean (multiprocessing.RawValue): Running squared mean difference for standard deviation calculation.
        best_y (multiprocessing.RawValue): Best objective function value found so far.
        best_x (multiprocessing.RawArray): Coordinates of the point corresponding to the best objective function value.
        statistic_num (int): Number of statistics points to record.
        time (multiprocessing.RawArray): Array for storing timestamps of improvements.
        val (multiprocessing.RawArray): Array for storing values of improvements.
        si (multiprocessing.RawValue): Index for the next statistic to record.
        sevals (multiprocessing.RawValue): Counter for total evaluations stored for statistics.
        bval (multiprocessing.RawValue): Current best value stored for statistics tracking.
    """
       
    def __init__(self, 
                 fun: Callable[[ArrayLike], float], # fitness function
                 bounds: Bounds, # bounds of the objective function arguments
                 check_interval: Optional[int] = 10, # sort evaluation memory after check_interval iterations
                 capacity: Optional[int] = 500, # capacity of the evaluation store
                 statistic_num: Optional[int] = 0
                ):
        """
        Initializes an instance to manage shared memory for evaluating fitness functions
        with capabilities for storing results, maintaining statistics, and handling bounds
        in a multiprocessing environment.

        Args:
            fun (Callable[[ArrayLike], float]): The fitness function to evaluate.
            bounds (Bounds): The bounds of the objective function arguments.
            check_interval (Optional[int]): The interval (in iterations) to sort the
                evaluation memory. Default is 10.
            capacity (Optional[int]): The maximum capacity of the evaluation store.
                Default is 500.
            statistic_num (Optional[int]): The number of statistics to maintain. If greater
                than 0, statistics tracking is enabled. Default is 0.
        """
        self.fun = fun
        self.lower, self.upper = _convertBounds(bounds)
        self.capacity = capacity
        self.check_interval = check_interval
        self.dim = len(self.lower)
        self.delta = []
        for k in range(self.dim):
            self.delta.append(self.upper[k] - self.lower[k])
        
        #shared between processes
        self.add_mutex = mp.Lock()    
        self.xs = Shared2d(np.empty((self.capacity, self.dim), dtype = np.float64))
        self.create_xs_view()
        self.ys = mp.RawArray(ct.c_double, self.capacity)  
        self.count_evals = mp.RawValue(ct.c_long, 0)   
        self.count_runs = mp.RawValue(ct.c_int, 0) 
        self.num_stored = mp.RawValue(ct.c_int, 0)   
        self.count_stat_runs = mp.RawValue(ct.c_int, 0)  
        self.t0 = time.perf_counter()
        self.mean = mp.RawValue(ct.c_double, 0) 
        self.qmean = mp.RawValue(ct.c_double, 0) 
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.best_x = mp.RawArray(ct.c_double, self.dim)
        self.statistic_num = statistic_num
        # statistics   
        self.statistic_num = statistic_num                         
        if statistic_num > 0:  # enable statistics                          
            self.time = mp.RawArray(ct.c_double, self.statistic_num)
            self.val = mp.RawArray(ct.c_double, self.statistic_num)
            self.si = mp.RawValue(ct.c_int, 0)
            self.sevals = mp.RawValue(ct.c_long, 0)
            self.bval = mp.RawValue(ct.c_double, np.inf)

    # register improvement - time and value
    def wrapper(self, x: ArrayLike):
        """
        Evaluates a given function with input data, tracks evaluation statistics, and updates the
        best observed value if applicable.

        Args:
            x (ArrayLike): Input data for the function evaluation.

        Returns:
            float: The function output for the given input.
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
            logger.trace(
                f'{self.time[si]} {self.sevals.value} {self.sevals.value/(1E-9 + self.time[si]):.1f} {y} {list(x)}'
            )
        return y
         
    def get_improvements(self):
        """
        Generates an array of improvements based on time and value up to a specific index.

        This method combines the time and value attributes into a NumPy array of tuples,
        containing the data up to the si attribute's value. The method assumes that the
        attributes `time` and `val` are iterable and contain corresponding elements. The
        method is useful for analyzing the progression or trend present within the provided
        values.

        Returns:
            numpy.ndarray: A NumPy array of tuples where each tuple is composed of elements
            from time and val attributes up to the index indicated by si.

        Raises:
            TypeError: If `time` or `val` attributes are not iterable or do not contain
            corresponding elements.
            ValueError: If `si.value` exceeds the lengths of `time` or `val` attributes.

        """
        return np.array(list(zip(self.time[:self.si.value], self.val[:self.si.value])))
        
    # get num best values at evenly distributed times
    def get_statistics(self, num: int) -> List:
        """
        Calculates statistics for the provided number of samples by segmenting
        and mapping values based on time intervals.

        This method segments a range of time into equal intervals based on
        the given number of samples. It then maps each interval to the
        corresponding value at the closest timestamp within that interval.

        Args:
            num (int): The number of intervals or samples for which statistics
                are calculated.

        Returns:
            List: A list of values corresponding to the calculated statistics
                for each interval.
        """
        ts = self.time[:self.si.value]
        ys = self.val[:self.si.value]
        mt = ts[-1]
        dt = 0.9999999 * mt / num
        conv = []
        ti = 0
        val = ys[0]
        for i in range(num):
            while ts[ti] < (i+1) * dt:
                ti += 1
                val = ys[ti]
            conv.append(val)
        return conv
    
    def eval_num(self, max_evals: int) -> int:
        """
        Returns the number of evaluations specified.

        This method accepts the maximum number of evaluations as an integer input
        and simply returns it unchanged. It is commonly used to track or specify
        the number of steps or iterations allowed in a process.

        Args:
            max_evals (int): The maximum number of evaluations to be returned.

        Returns:
            int: The provided maximum evaluations.
        """
        return max_evals
                                             
    def replace(self, i: int, y: float, xs: ArrayLike):
        """
        Replaces the values at a specified index with new values.

        This method updates the y-value and x-values at the specified index by
        calling internal methods.

        Args:
            i (int): Index at which the replacement occurs.
            y (float): New y-value to set at the specified index.
            xs (ArrayLike): New x-values to set at the specified index.
        """
        self.set_y(i, y)
        self.set_x(i, xs)
             
    def sort(self) -> int: # sort all entries to make room for new ones, determine best and worst
        """
        Sort entries to retain the best ones, thereby optimizing storage for new entries.

        This method sorts the stored entries based on their associated values, retains the top
        90% (as determined by the capacity), and updates the internal data structures to reflect
        the new ordering and count of retained entries. It subsequently returns the updated
        number of stored entries.

        Returns:
            int: The number of stored entries after sorting and truncating to the top 90%.

        """
        ns = self.num_stored.value
        ys = np.asarray(self.ys[:ns])
        yi = ys.argsort()
        numStored = min(ns, int(0.9*self.capacity)) # keep 90% best 
        self.xs_view[:numStored] = self.xs_view[yi][:numStored]
        self.ys[:numStored] = ys[yi][:numStored]
        self.num_stored.value = numStored  
        return numStored        
            
    def add_result(self, y: float, x: ArrayLike, evals: int, limit=np.inf):
        """
        Adds a result to the data structure, updating statistical values and recording the
        given data point. It updates attributes like the best value (`best_y`) and associated
        point (`best_x`), checks storage capacity, and computes statistical measures.

        Args:
            y (float): The new value to be added.
            x (ArrayLike): The associated data point corresponding to the value `y`.
            evals (int): Number of evaluations associated with the newly added data.
            limit (float, optional): The maximum cap for `y` values to be considered. Default
                is `np.inf`.

        Raises:
            ValueError: If the input data is inconsistent or violates preconditions of the
                data structure. (Error not directly visible in provided code)
        """
        with self.add_mutex:
            self.incr_count_evals(evals)
            if y < limit:  
                self.count_stat_runs.value += 1
                if y < self.best_y.value:
                    self.best_y.value = y
                    self.best_x[:] = x[:]
                    self.dump()
                if self.num_stored.value >= self.capacity-1:
                    self.sort()
                cnt = self.count_stat_runs.value
                diff = min(1E20, y - self.mean.value) # avoid overflow
                self.qmean.value += (cnt - 1)/ cnt * diff*diff ;
                self.mean.value += diff / cnt
                ns = self.num_stored.value
                self.num_stored.value = ns + 1
                self.xs_view[self.num_stored.value, :] = x
                self.ys[self.num_stored.value] = y
                if is_debug_active():
                    dt = dtime(self.t0)  
                    message = '{0} {1} {2} {3} {4:.6f} {5:.6f} {6:.2f} {7:.2f}'.format(
                        dt, int(self.count_evals.value / dt), self.count_runs.value, self.count_evals.value, \
                        y, self.best_y.value, self.get_y_mean(), self.get_y_standard_dev())
                    logger.debug(message)

    def get_x_best(self) -> np.ndarray:
        """
        Retrieves a copy of the best_x attribute as a numpy array.

        Returns:
            np.ndarray: A copy of the current best_x attribute, converted to a numpy array.
        """
        return np.array(self.best_x[:])
    
    def create_xs_view(self): # needs to be called in the target process
        """
        Creates and initializes a view for the 'xs' object.

        This method generates a view from the 'xs' object and assigns it to the
        `xs_view` attribute. It is essential to call this method in the target
        process to properly initialize the `xs_view`.

        Raises:
            AttributeError: If 'xs' does not have a 'view()' method or is not
                initialized.

        """
        self.xs_view = self.xs.view()
 
    def get_xs(self) -> np.ndarray:
        """
        Returns a view of the stored elements up to the count specified by `num_stored`.

        Retrieves a NumPy array with the elements stored up to a set boundary, defined by
        `num_stored.value`. This method utilizes the array's view functionality for efficient
        data access without additional memory allocation.

        Returns:
            np.ndarray: A NumPy array view with elements up to the specified count.
        """
        return self.xs.view()[:self.num_stored.value]
        
    def get_y(self, pid: int) -> float:
        """
        Retrieves the y-coordinate value corresponding to a given particle ID.

        Args:
            pid (int): The particle ID whose y-coordinate is to be retrieved.

        Returns:
            float: The y-coordinate value associated with the specified particle ID.
        """
        return self.ys[pid]

    def get_y_best(self) -> float:
        """
        Returns the best y-value.

        This method retrieves the best y-value currently stored in the `best_y`
        attribute. The value is returned as a floating-point number.

        Returns:
            float: The best y-value stored in the `best_y` attribute.
        """
        return self.best_y.value
    
    def get_ys(self) -> np.ndarray:
        """
        Gets the stored y-values as a NumPy array.

        This method retrieves the stored y-values up to the current number of
        stored values and returns them as a NumPy array.

        Returns:
            np.ndarray: A NumPy array containing the y-values.
        """
        return np.array(self.ys[:self.num_stored.value])
             
    def get_y_mean(self) -> float:
        """
        Calculates and returns the mean value along the Y-axis.

        This method retrieves the precomputed mean value for the Y-axis
        from the `mean` attribute and returns it.

        Returns:
            float: The mean value along the Y-axis.
        """
        return self.mean.value

    def get_y_standard_dev(self) -> float:
        """
        Calculates the standard deviation based on the collected statistics.

        This method calculates the standard deviation for the y-values from
        statistical data. If no statistical runs have been recorded, it
        returns 0 to avoid division by zero.

        Returns:
            float: The standard deviation of the y-values from the statistical
            data. Returns 0 if there are no recorded statistical runs.

        Raises:
            None
        """
        cnt = self.count_stat_runs.value
        return 0 if cnt <= 0 else math.sqrt(self.qmean.value / cnt)

    def get_count_evals(self) -> int:
        """
        Retrieves the current evaluation count value.

        This method returns the integer value of the current evaluation count, allowing
        users to access the internal evaluation counter's value for further processing
        or inspection.

        Returns:
            int: The current count of evaluations.
        """
        return self.count_evals.value
 
    def get_count_runs(self) -> int:
        """
        Retrieves the current value of `count_runs`, representing the count of runs.

        Returns:
            int: The current count of runs.
        """
        return self.count_runs.value

    def get_runs_compare_incr(self, limit: float):
        """
        Compares the current run count with the specified limit and increments the
        count if it is below the given limit.

        Args:
            limit (float): The maximum limit up to which the run count can be incremented.

        Returns:
            bool: True if the run count was incremented, False otherwise.
        """
        with self.add_mutex:
            if self.count_runs.value < limit:
                self.count_runs.value += 1
                return True
            else:
                return False 
       
    def incr_count_evals(self, evals):
        """
        Increases the count of evaluations and performs sorting based on a condition.

        This method increments the evaluation counter (`count_evals`) by the specified
        value and performs a sort operation if the current number of runs (`count_runs`)
        is one less than a multiple of the check interval (`check_interval`).

        Args:
            evals (int): The number of evaluations to add to the current count.
        """
        if self.count_runs.value % self.check_interval == self.check_interval-1:
            self.sort()
        self.count_evals.value += evals
            
    def dump(self):
        """
        Dumps debugging information if debugging is active.

        This method obtains and processes data related to evaluations, computed
        values (Ys), and other statistics, then logs a formatted message for
        debugging purposes.

        Args:
            None

        Raises:
            None
        """
        if not is_debug_active():
            return
        Ys = self.get_ys()
        vals = []
        for i in range(min(20, len(Ys))):
            vals.append(round(Ys[i],4))     
        dt = dtime(self.t0)                   
        message = '{0} {1} {2} {3} {4:.6f} {5:.2f} {6:.2f} {7!s} {8!s}'.format(
            dt, int(self.count_evals.value / dt), self.count_runs.value, self.count_evals.value, \
                self.best_y.value, self.get_y_mean(), self.get_y_standard_dev(), vals, self.best_x[:])
        logger.debug(message)

        
def _retry_loop(pid, rgs, store, optimize, num_retries, value_limit, stop_fitness = -np.inf):
    """
    Executes a retry loop for optimization, managing retries and applying constraints
    based on fitness value and evaluation limits.

    Args:
        pid (int): Process ID or thread identifier for parallel processing.
        rgs (list): List of random generators, corresponding to each process or thread.
        store (object): A storage or result management object handling bounds,
            results, statistics, and fitness evaluations.
        optimize (callable): Optimization function to be executed during the loop.
        num_retries (int): Maximum number of retries allowed during the optimization process.
        value_limit (float): Constraint limit for the solution value to validate results.
        stop_fitness (float, optional): Fitness threshold that, once achieved, stops
            further processing. Defaults to negative infinity.
    """
    store.create_xs_view()
    fun = store.wrapper if store.statistic_num > 0 else store.fun    
    lower = store.lower
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        while store.get_runs_compare_incr(num_retries) and store.best_y.value > stop_fitness:      
            try:       
                rg = rgs[pid]
                sol, y, evals = optimize(fun, Bounds(store.lower, store.upper), None, 
                                         [rg.uniform(0.05, 0.1)]*len(lower), rg, store)
                store.add_result(y, sol, evals, value_limit)   
            except Exception as ex:
                print(str(ex))

def _convertBounds(bounds):
    """
    Convert bounds to a pair of arrays representing lower and upper limits.

    This function processes the given `bounds` and converts them into two arrays
    that represent the lower and upper bounds, respectively. The `bounds` argument
    must either be an instance of the `Bounds` class or a sequence of real
    valued (min, max) pairs for each variable.

    Raises:
        ValueError: If `bounds` is None.
        ValueError: If `bounds` is not an instance of `Bounds` or a valid sequence
            of (min, max) pairs.
        ValueError: If size or values within the `bounds` are not finite real
            values.

    Args:
        bounds: The bounds to process, either as an instance of the `Bounds`
            class or a sequence of real valued (min, max) pairs for each variable.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays. The first
            array represents the lower limits, and the second represents the
            upper limits.
    """
    if bounds is None:
        raise ValueError('bounds need to be defined')
    if isinstance(bounds, Bounds):
        limits = np.array(new_bounds_to_old(bounds.lb,
                                                 bounds.ub,
                                                 len(bounds.lb)),
                               dtype=float).T
    else:
        limits = np.array(bounds, dtype='float').T
    if (np.size(limits, 0) != 2 or not
            np.all(np.isfinite(limits))):
        raise ValueError('bounds should be a sequence containing '
                         'real valued (min, max) pairs for each value'
                         ' in x')
    return limits[0], limits[1]
