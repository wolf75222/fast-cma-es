# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - multiretry.py

 Description:
  - Parallel optimization retry of a list of problems.

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
import _pickle as cPickle
import bz2
import multiprocessing as mp
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.optimizer import de_cma, eprint, Optimizer
from fcmaes import advretry

from fcmaes.evaluator import is_debug_active
from loguru import logger
from typing import Optional, Callable, Tuple, List
from numpy.typing import ArrayLike

def minimize(problems: ArrayLike, 
             ids: Optional[ArrayLike] = None, 
             retries_inc: Optional[int] = min(256, 8*mp.cpu_count()), 
             num_retries: Optional[int] = 10000,
             keep: Optional[float] = 0.7, 
             optimizer: Optional[Optimizer] = de_cma(1500), 
             datafile = None) -> List:

    """
    Minimizes the given set of problems by iteratively applying the optimizer and
    removing the worst-performing solutions until only one remains. Supports
    retries and optional data file usage for saving and loading intermediate
    results.

    Minimization of a list of optimization problems by first applying parallel retry
    to filter the best ones and then applying coordinated retry to evaluate these further.
    Can replace mixed integer optimization if the integer variables are narrowly bound.
    In this case all combinations of these integer values can be enumerated to generate a
    list of problem instances each representing one combination. See for instance
    https://www.esa.int/gsp/ACT/projects/gtop/tandem where there is a problem instance for each
    planet sequence.

    Args:
        problems (ArrayLike): List or array-like structure containing the problems
            to be minimized.
        ids (Optional[ArrayLike]): List or array-like structure containing
            IDs for the problems. Defaults to None, in which case IDs will
            be auto-generated as strings in ascending order starting from 1.
        retries_inc (Optional[int]): Number of initial retries increment. Defaults
            to min(256, 8 times the number of CPU cores).
        num_retries (Optional[int]): Maximum number of retries for each problem.
            Defaults to 10000.
        keep (Optional[float]): Fraction of best-performing problems to retain in
            each iteration. Defaults to 0.7.
        optimizer (Optional[Optimizer]): Optimizer object used to minimize the
            problems. Defaults to de_cma(1500).
        datafile: Optional parameter representing the file path to load and save
            intermediate results. Defaults to None.

    Returns:
        List: Sorted list of all problem statistics after successful
            minimization.
    """

    solver = multiretry()
    n = len(problems)
        
    for i in range(n):    
        id = str(i+1) if ids is None else ids[i]   
        solver.add(problem_stats(problems[i], id, i, retries_inc, num_retries))
    
    if not datafile is None:
        solver.load(datafile)
        
    while solver.size() > 1:    
        solver.retry(optimizer)
        to_remove = int(round((1.0 - keep) * solver.size()))
        if to_remove == 0 and keep < 1.0:
            to_remove = 1
        solver.remove_worst(to_remove)
        solver.dump()
        if not datafile is None:
            solver.save(datafile)
            
    idx = solver.values_all().argsort()
    return list(np.asarray(solver.all_stats)[idx])
        
class problem_stats:
    """Represents statistics and operational parameters for a specific problem.

    This class encapsulates the problem's parameters, optimization store, retry mechanism,
    and manages retries for optimization processes. It is designed to handle optimization
    problems with adjustable retry increments and constraints.

    Attributes:
        prob: The problem instance containing the problem's functions and bounds.
        name (str): The name of the problem.
        fun: The function representing the objective of the problem.
        retries_inc (int): The number of retries incrementally added on each retry operation.
        value (float): The best value obtained during optimization.
        id: The identifier for the problem instance.
        index: The index associated with the problem instance.
    """
    def __init__(self, prob, id, index, retries_inc = 64, num_retries = 10000):
        """
        Initializes an instance of the class with the given problem, identifier, index,
        optional retries increment value, and number of retries.

        Args:
            prob: An object representing the problem to solve.
            id: An identifier associated with the instance.
            index: An integer specifying the index of the instance.
            retries_inc: Optional integer specifying the increment for retries (default 64).
            num_retries: Optional integer specifying the total number of retries (default 10000).
        """
        self.store = advretry.Store(prob.fun, prob.bounds, num_retries=num_retries)
        self.prob = prob
        self.name = prob.name
        self.fun = prob.fun
        self.retries_inc = retries_inc
        self.value = 0
        self.id = id
        self.index = index
        self.ret = None
        self.store.num_retries = self.retries_inc

    def retry(self, optimizer):
        """
        Retries the optimization process and updates the relevant attributes in the store.

        This method increments the retry count stored in the store. It performs an optimization
        retry using the provided `optimizer` and updates the best obtained value in the
        optimization process.

        Args:
            optimizer: An optimization object used to minimize the function. It must provide
                a `minimize` method to execute the optimization operations.
        """
        self.store.num_retries += self.retries_inc
        self.ret = advretry.retry(self.store, optimizer.minimize)
        self.value = self.store.get_y_best()
 
class multiretry:
    """
    A class for managing and retrying task statistics with extended functionalities like removing
    worst-performing tasks, saving and loading data, and retrieving metrics.

    The class is designed to facilitate the management of task-related statistics, allowing
    operations such as retrying tasks with an optimizer, sorting tasks based on performance,
    and persisting data for external use. It also supports evaluating and managing both primary
    and aggregated statistics.

    Attributes:
        problem_stats (list): A list of statistics related to problems currently being managed.
        all_stats (list): A comprehensive list of all statistics including completed and
            ongoing problems.
    """
    def __init__(self):
        """
        Represents a container designed to manage problem statistics and aggregate data.

        Attributes:
            problem_stats (list): A list designed to hold statistics related specifically to problems.
            all_stats (list): A list intended to store aggregated or general statistics.
        """
        self.problem_stats = []
        self.all_stats = []
    
    def add(self, stats):
        """
        Adds the provided statistics to the problem and all statistics lists.

        Args:
            stats: The statistics to be added to the lists.
        """
        self.problem_stats.append(stats)
        self.all_stats.append(stats)
    
    def retry(self, optimizer):
        """
        Retries optimization for each problem in the problem statistics list.

        This method iterates through the list of problem statistics and attempts to
        retry the optimization process using the given optimizer.

        Args:
            optimizer: The optimizer instance to use for retrying the problems.
        """
        for ps in self.problem_stats:
            if is_debug_active():
                logger.debug("problem " + ps.prob.name + ' ' + str(ps.id))
            ps.retry(optimizer)
    
    def values(self):
        """
        Generates a NumPy array from the values of the `problem_stats` attribute.

        Returns:
            numpy.ndarray: A NumPy array containing the values of `problem_stats`,
            converted to float type.
        """
        return np.fromiter((ps.value for ps in self.problem_stats), dtype=float)
     
    def remove_worst(self, n = 1):
        """
        Removes the worst-performing items from the current problem statistics list.

        This method removes items from the problem statistics list based on their
        performance metrics. The items are sorted in ascending order first,
        and the specified number of worst-performing items (based on the input value)
        are removed from the end of the list.

        Args:
            n (int, optional): The number of worst-performing items to remove. Defaults
                to 1.

        """
        idx = self.values().argsort()
        self.problem_stats = list(np.asarray(self.problem_stats)[idx])
        for _ in range(n):
            self.problem_stats.pop(-1)

    def size(self):
        """
        Calculates and returns the size of the problem statistics.

        This function determines the size based on the number of entries
        in the `problem_stats` attribute.

        Returns:
            int: The number of entries in `problem_stats`.
        """
        return len(self.problem_stats)
                    
    def dump(self):
        """
        Dumps the problem statistics for debugging purposes.

        This method is intended for use in debugging scenarios, where it logs the
        problem statistics contained in the object. The logging only occurs if
        debugging is active.

        Args:
            None

        Raises:
            None

        Returns:
            None
        """
        if is_debug_active():
            for i in range(self.size()):
                ps = self.problem_stats[i]
                logger.debug(str(ps.id) + ' ' + str(ps.value))
                
    def dump_all(self):
        """
        Dumps all statistics in a sorted manner when debug mode is active.

        This method sorts the internal `all_stats` attribute based on the indices
        of the `values_all` method's results. It then logs each statistic's `id` and
        `value` using the debug logger.

        Raises:
            None
        """
        if is_debug_active():
            idx = self.values_all().argsort()
            self.all_stats = list(np.asarray(self.all_stats)[idx])
            for i in range(len(self.all_stats)):
                ps = self.all_stats[i]
                logger.debug(str(ps.id) + ' ' + str(ps.value))

    def values_all(self):
        """
        Gets the values of all stats as a numpy array.

        This method iterates over all stats and extracts their values, returning them
        as a numpy array of type float.

        Returns:
            numpy.ndarray: A numpy array containing the float values of all stats.
        """
        return np.fromiter((ps.value for ps in self.all_stats), dtype=float)
 
    def result(self):
        """
        Sorts statistics based on values and returns a list of optimization results.

        This method processes the data contained in the `all_stats` attribute by sorting
        it using the `values_all` method. It extracts optimization-related information,
        such as the best solution, function value, number of function evaluations, and
        success status, for each statistics entry.

        Returns:
            List[List]: A list containing sublists, where each sublist consists of a
                problem instance and its corresponding optimization outcome
                encapsulated in an `OptimizeResult` object.
        """
        idx = self.values_all().argsort()
        self.all_stats = list(np.asarray(self.all_stats)[idx])
        ret = []
        for i in range(len(self.all_stats)):
            problem = self.all_stats[i].prob
            store = self.all_stats[i].store
            ret.append([problem, 
                        OptimizeResult(x=store.get_x_best(), fun=store.get_y_best(), 
                          nfev=store.get_count_evals(), success=True)])
            
    # persist all stats
    def save(self, name):
        """
        Saves the data returned by the `get_data` method into a compressed bz2 file
        with the specified name.

        The data is serialized using `cPickle` and saved in a `bz2` compressed format.
        If an error occurs during this process, an error message is printed.

        Args:
            name (str): The base name of the file where the data will be saved.
                The `.pbz2` extension will be appended automatically.
        """
        try:
            with bz2.BZ2File(name + '.pbz2', 'w') as f: 
                cPickle.dump(self.get_data(), f)
        except Exception as ex:
            eprint('error writing data file ' + name + '.pbz2 ' + str(ex))

    def load(self, name):
        """
        Loads data from a compressed pickle (.pbz2) file and sets it to the current instance.

        This function attempts to load the serialized data from the specified file name, decompressing it
        if necessary, and assigns the loaded data to the current instance using the `set_data` method. If
        an error occurs during the file reading or deserialization process, an error message is printed
        with details of the exception.

        Args:
            name (str): Name of the file (without the extension) from which the data is to be loaded.

        Raises:
            Exception: If there is an error during file reading or data deserialization.
        """
        try:
            data = cPickle.load(bz2.BZ2File(name + '.pbz2', 'rb'))
            self.set_data(data)
        except Exception as ex:
            eprint('error reading data file ' + name + '.pbz2 ' + str(ex))
  
    def get_data(self):
        """
        Fetches and aggregates data from all stats objects.

        This method iterates over the `all_stats` collection, retrieves the
        data from each `store` property of the stats objects, and appends
        it to the resulting list.

        Returns:
            list: A list containing the aggregated data from all stats objects.
        """
        data = []
        for stats in self.all_stats:            
            data.append(stats.store.get_data())
        return data
        
    def set_data(self, data):
        """
        Sets the provided `data` to the corresponding store objects in `all_stats`.

        This method iterates over the `data` list and assigns each element of `data`
        to the store of the corresponding index in the `all_stats` list.

        Args:
            data (list): A list where each element corresponds to data for the store of
                the same position in the `all_stats` list.
        """
        for i in range(len(data)):
            self.all_stats[i].store.set_data(data[i])
        
    
