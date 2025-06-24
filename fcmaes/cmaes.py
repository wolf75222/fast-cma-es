# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - cmaes.py

 Description:
  - This module implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    for optimization tasks. It provides a framework for minimizing scalar functions
    using evolutionary strategies, with support for parallel execution and adaptive
    stopping criteria.
  - The implementation is based on the original CMA-ES algorithm and has been optimized
    for performance using NumPy and SciPy libraries.


 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] http://cma.gforge.inria.fr/cmaes.m
  - [3] https://www.researchgate.net/publication/227050324_The_CMA_Evolution_Strategy_A_Comparing_Review

 Documentation:
  -


=============================================================================
"""

import sys
import os
import math
import numpy as np
from time import time
import ctypes as ct
import multiprocessing as mp
from scipy import linalg
from scipy.optimize import OptimizeResult, Bounds
from numpy.random import PCG64DXSM, Generator
from fcmaes.evaluator import Evaluator, serial, _check_bounds, _fitness, is_debug_active

from loguru import logger
from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float], 
             bounds: Optional[Bounds] = None, 
             x0: Optional[ArrayLike] = None,
             input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3,
             popsize: Optional[int] = 31,
             max_evaluations: Optional[int]  = 100000,
             max_iterations: Optional[int]  = 100000,
             workers: Optional[int]  = 1,
             accuracy: Optional[float] = 1.0,
             stop_fitness: Optional[float] = -np.inf,
             is_terminate: Optional[Callable[[ArrayLike, float], bool]]  = None,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             runid: Optional[int] = 0,
             normalize: Optional[bool] = True,
             update_gap: Optional[int] = None) -> OptimizeResult:
    """
    Minimizes a scalar function using the Covariance Matrix Adaptation Evolution
    Strategy (CMA-ES) algorithm.

    This function is designed to handle optimization tasks for non-linear, non-convex,
    and possibly noisy objective functions. It supports parallel execution via workers,
    normalization of the input search space, and adaptive stopping criteria based on given
    conditions.

    Args:
        fun (Callable[[ArrayLike], float]): The objective function to be minimized.
            It should take an input array and return a scalar value.
        bounds (Optional[Bounds]): The lower and upper bounds for the search space.
            If None, the search space is considered unbounded.
        x0 (Optional[ArrayLike]): Initial solution guess. If None, an initial guess
            will be generated randomly.
        input_sigma (Optional[Union[float, ArrayLike, Callable]]): The initial
            standard deviation for the sampling. Can be float, array, or callable.
        popsize (Optional[int]): The population size for the CMA-ES algorithm.
            Defaults to 31.
        max_evaluations (Optional[int]): Maximum number of function evaluations
            allowed. Defaults to 100,000.
        max_iterations (Optional[int]): Maximum number of iterations allowed
            for the algorithm. Defaults to 100,000.
        workers (Optional[int]): Number of parallel processes to be used. If set
            to 1 or less, the algorithm will run in serial mode. Defaults to 1.
        accuracy (Optional[float]): The accuracy tolerance used to adjust the
            optimization stopping criteria. Defaults to 1.0.
        stop_fitness (Optional[float]): Objective function value at which the
            optimization process is terminated if reached. Defaults to -infinity.
        is_terminate (Optional[Callable[[ArrayLike, float], bool]]): Custom
            termination condition provided as a callable. Defaults to None.
        rg (Optional[Generator]): The random number generator to be used. Defaults
            to `Generator(PCG64DXSM())`.
        runid (Optional[int]): Identifier for the optimization run. Defaults to 0.
        normalize (Optional[bool]): Indicates if the search space should be normalized.
            Defaults to True.
        update_gap (Optional[int]): Interval for delayed updates in the algorithm.
            If None, updates are not delayed.

    Returns:
        OptimizeResult: Object containing the results of the optimization. Includes
            the optimized solution (`x`), the minimized function value (`fun`),
            number of function evaluations (`nfev`), number of iterations (`nit`),
            algorithm exit status (`status`), and success flag (`success`).

    """
  
    if workers is None or workers <= 1:
        fun = serial(fun)        
    cmaes = Cmaes(bounds, x0, 
                      input_sigma, popsize, 
                      max_evaluations, max_iterations, 
                      accuracy, stop_fitness, 
                      is_terminate, rg, np.random.randn, runid, normalize, 
                      update_gap, fun)        
    if workers and workers > 1:
        x, val, evals, iterations, stop = cmaes.do_optimize_delayed_update(fun, workers=workers)
    else:      
        x, val, evals, iterations, stop = cmaes.doOptimize()
    return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, 
                          success=True)

class Cmaes(object):
    """
    Optimization solver implementing the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    The CMA-ES algorithm is an evolutionary strategy for solving complex optimization problems. This
    class is used to configure and execute the"""
    
    def __init__(self, bounds: Optional[Bounds] = None,
                        x0: Optional[ArrayLike] = None,
                        input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3,
                        popsize: Optional[int] = 31,
                        max_evaluations: Optional[int] = 100000,
                        max_iterations: Optional[int] = 100000,
                        accuracy: Optional[int] = 1.0,
                        stop_fitness: Optional[float] = -np.inf,
                        is_terminate: Optional[bool] = None,
                        rg: Optional[Generator] = Generator(PCG64DXSM()), # used if x0 is undefined
                        randn: Optional[Callable] = np.random.randn, # used for random offspring 
                        runid: Optional[int] = 0,
                        normalize: Optional[bool] = True,
                        update_gap: Optional[int] = None,
                        fun: Optional[Callable[[ArrayLike], float]] = None
                        ):
        """
        Initializes the optimizer with user-defined or default parameters.

        This constructor sets up the configuration for the optimization process. Users can customize the
        initial state, population size, termination criteria, and other critical parameters to tune the
        algorithm's behavior. The initialization process also ensures that random generation settings are
        appropriately defined or defaulted.

        Args:
            bounds (Optional[Bounds]): Specifies the boundaries within which the optimization will occur.
            x0 (Optional[ArrayLike]): Initial guess or starting point for optimization.
            input_sigma (Optional[Union[float, ArrayLike, Callable]]): Initial standard deviation of the
                distribution used in the optimization process, with a default value of 0.3.
            popsize (Optional[int]): Population size for the evolutionary algorithm; default is 31.
            max_evaluations (Optional[int]): Maximum number of function evaluations allowed during the
                optimization process; default is 100000.
            max_iterations (Optional[int]): Maximum number of optimization iterations; default is 100000.
            accuracy (Optional[int]): Desired accuracy or precision level for the optimization process, with
                a default value of 1.0.
            stop_fitness (Optional[float]): Fitness value at which the optimization terminates early. The
                default is negative infinity (-np.inf), implying no fitness-based stopping criterion.
            is_terminate (Optional[bool]): Flag indicating whether early termination criteria are defined.
            rg (Optional[Generator]): A generator object for random number generation, used if x0 is
                undefined; the default is Generator(PCG64DXSM()).
            randn (Optional[Callable]): Function for generating random offspring during the optimization
                process; default is `np.random.randn`.
            runid (Optional[int]): Identifier for the optimization run, useful for tracking multiple runs;
                default is 0.
            normalize (Optional[bool]): Flag indicating whether inputs should be normalized during the
                process; default is True.
            update_gap (Optional[int]): Frequency of updating certain parameters or values in the
                optimization process.
            fun (Optional[Callable[[ArrayLike], float]]): Objective function to be minimized or maximized.
                This function should accept an array-like input and return a float value representing
                the objective value.
        """

    # runid used in is_terminate callback to identify a specific run at different iteration
        self.runid = runid
    # bounds and guess
        lower, upper, guess = _check_bounds(bounds, x0, rg)   
        self.fitfun = _fitness(fun, lower, upper, normalize)
    # initial guess for the arguments of the fitness function
        self.guess = self.fitfun.encode(guess)
    # random generators    
        self.rg = rg # used if x0 is undefined
        self.randn = randn # used for random offspring 
    # accuracy = 1.0 is default, > 1.0 reduces accuracy
        self.accuracy = accuracy
    # callback to check if to terminate
        self.is_terminate = is_terminate
    # Number of objective variables/problem dimension
        self.dim = guess.size
    #     Population size, offspring number. The primary strategy parameter to play
    #     with, which can be increased from its default value. Increasing the
    #     population size improves global search properties in exchange to speed.
    #     Speed decreases, as a rule, at most linearly with increasing population
    #     size. It is advisable to begin with the default small population size.
        if popsize:
            self.popsize = popsize #population size
        else:
            self.popsize = 4 + int(3. * math.log(self.dim))
    #     Individual sigma values - initial search volume. input_sigma determines
    #     the initial coordinate wise standard deviations for the search. Setting
    #     SIGMA one third of the initial search region is appropriate.   
        if callable(input_sigma):
            input_sigma=input_sigma()
        if isinstance(input_sigma, list):
            self.insigma = np.asarray(input_sigma)
        elif np.isscalar(input_sigma):
            self.insigma = np.full(self.dim, input_sigma)    
        else:
            self.insigma = input_sigma
    # Overall standard deviation - search volume.
        self.sigma = max(self.insigma)
    # termination criteria
    # Maximal number of iterations allowed.
        self.max_evaluations = max_evaluations
        self.max_iterations = max_iterations
    # Limit for fitness value.
        self.stop_fitness = stop_fitness
    # Stop if x-changes larger stopTolUpX.
        self.stopTolUpX = 1e3 * self.sigma
    # Stop if x-change smaller stopTolX.
        self.stopTolX = 1e-11 * self.sigma * accuracy
    # Stop if fun-changes smaller stopTolFun.
        self.stopTolFun = 1e-12 * accuracy
    # Stop if back fun-changes smaller stopTolHistFun.
        self.stopTolHistFun = 1e-13 * accuracy
    # selection strategy parameters
    # Number of parents/points for recombination.
        self.mu = int(self.popsize/2)
    # timing / global best value    
        if is_debug_active():
            self.best_y = mp.RawValue(ct.c_double, 1E99)
            self.n_evals = mp.RawValue(ct.c_long, 0)
            self.time_0 = time()
            
    # Array for weighted recombination.    
        self.weights = (np.log(np.arange(1, self.mu+1, 1)) * -1) + math.log(self.mu + 0.5)
        sumw = np.sum(self.weights)
        sumwq = np.einsum('i,i->', self.weights, self.weights)
        self.weights *= 1./sumw        
    # Variance-effectiveness of sum w_i x_i.
        self.mueff = sumw * sumw / sumwq 

    # dynamic strategy parameters and constants
    # Cumulation constant.
        self.cc = (4. + self.mueff / self.dim) / (self.dim + 4. + 2. * self.mueff / self.dim)
    # Cumulation constant for step-size.
        self.cs = (self.mueff + 2.) / (self.dim + self.mueff + 3.)
    # Damping for step-size.    
        self.damps = (1. + 2. * max(0., math.sqrt((self.mueff - 1.) / (self.dim + 1.)) - 1.)) * \
            max(0.3, 1. - self.dim / (1e-6 + min(self.max_iterations, self.max_evaluations/self.popsize))) + self.cs
    # Learning rate for rank-one update.
        self.ccov1 = 2. / ((self.dim + 1.3) * (self.dim + 1.3) + self.mueff)
    # Learning rate for rank-mu update'
        self.ccovmu = min(1. - self.ccov1, 2. * (self.mueff - 2. + 1. / self.mueff) \
                / ((self.dim + 2.) * (self.dim + 2.) + self.mueff))
    # Expectation of ||N(0,I)|| == norm(randn(N,1)).
        self.chiN = math.sqrt(self.dim) * (1. - 1. / (4. * self.dim) + 1 / (21. * self.dim * self.dim))
        self.ccov1Sep = min(1., self.ccov1 * (self.dim + 1.5) / 3.)
        self.ccovmuSep = min(1. - self.ccov1, self.ccovmu * (self.dim + 1.5) / 3.)        
    # lazy covariance update gap
        self.lazy_update_gap = 1. / (self.ccov1 + self.ccovmu + 1e-23) / self.dim / 10 \
                                    if update_gap is None else update_gap

    # CMA internal values - updated each generation
    # Objective variables.
        self.xmean = self.guess
    # Evolution path.
        self.pc = np.zeros(self.dim)
    # Evolution path for sigma.
        self.ps = np.zeros(self.dim)
    # Norm of ps, stored for efficiency.
        self.normps = math.sqrt(self.ps @ self.ps)
    # Coordinate system.
        self.B = np.eye(self.dim)        
    # Diagonal of sqrt(D), stored for efficiency.
        self.diagD = self.insigma / self.sigma
        self.diagC = self.diagD * self.diagD
    # B*D, stored for efficiency.
        self.BD = self.B * self.diagD
    # Covariance matrix.
        self.C = self.B @ (np.diag(np.ones(self.dim)) @ self.B)
    # Number of iterations already performed.
        self.iterations = 0
    # Size of history queue of best values.
        self.historySize = 10 + int(3. * 10. * self.dim / popsize)    
        
        self.iterations = 0
        self.last_update = 0
        self.stop = 0
        self.best_value = sys.float_info.max
        self.best_x = None    
        # History queue of best values.
        self.fitness_history = np.full(self.historySize, sys.float_info.max)
        self.fitness_history[0] = self.best_value    
        self.arz = None
        self.fitness = None

    def ask(self) -> np.array:
        """
        Generates a NumPy array by decoding elements in the current data structure.

        The method processes elements stored in the internal structure, applies a decoding
        mechanism via the fitfun's decode method, and produces an array containing the results.

        Args:
            self: Instance of the class containing necessary properties and methods to execute
                the decoding logic.

        Returns:
            np.array: A NumPy array containing the decoded elements.
        """

        self.newArgs()
        return np.array([self.fitfun.decode(x) for x in self.arx])
 
    def tell(self, 
             ys: np.ndarray, 
             xs: Optional[np.ndarray] = None) -> int:
        """
        Processes the input function values and updates the internal state of the CMA-ES algorithm.

        Args:
            ys (np.ndarray): An array of objective function values corresponding to the solutions.
            xs (Optional[np.ndarray]): An optional array of candidate solutions. If not provided,
                must have `arz` defined or `ask` must have been called prior.

        Returns:
            int: The stop condition value indicating the status of the CMA-ES run.

        Raises:
            ValueError: If `xs` is None and `arz` is not defined or `ask` has not been previously
                called.
        """

        if xs is None:
            if self.arz is None:
                raise ValueError('either call ask before or define xs')
        else:
            self.arx = np.array([self.fitfun.encode(x) for x in xs])
            try:
                self.arz = (linalg.inv(self.BD) @ \
                            ((self.arx - self.xmean).transpose() / self.sigma)).transpose()   
            except Exception:
                if self.arz is None: 
                    self.arz = self.randn(self.popsize, self.dim)
        self.fitness = np.asarray(ys)
        self.iterations += 1
        self.updateCMA()
        self.arz = None
        return self.stop
    
    def population(self) -> np.array:
        """
        Decodes the given array of representations into their respective solutions.

        This method utilizes the decoding function defined by the `fitfun` attribute to
        convert the input array `arx` into a population of solutions. The returned value
        is the decoded representation of the input data.

        Returns:
            np.array: Decoded population solutions.
        """
        return self.fitfun.decode(self.arx)

    def result(self) -> OptimizeResult:
        """
        Returns the optimization result containing the details of the optimization process.

        The result is an instance of `OptimizeResult`, which summarizes the outcome
        of the optimization, including the final optimum, function values, and
        additional side information.

        Args:
            self: The instance of the class containing the optimization process data.

        Returns:
            OptimizeResult: A dataclass containing the following attributes:
                - x: The best solution found during the optimization process.
                - fun: The value of the function at the best solution.
                - nfev: The number of function evaluations performed.
                - nit: The total number of iterations performed.
                - status: The status code indicating the stopping condition.
                - success: A boolean value indicating if the optimization was
                  successful.

        """
        return OptimizeResult(x=self.best_x, fun=self.best_value,
                              nfev=self.fitfun.evaluation_counter, 
                              nit=self.iterations, status=self.stop, success=True)
        
    def ask_one(self) -> np.array:
        """
        Generates a single decoded solution vector based on the current distribution.

        This method utilizes the generated random numbers and transformation matrices to
        produce a candidate solution vector that adheres to feasibility constraints, while
        decoding it through the associated objective function.

        Args:
            self: Instance of the class containing current state for generating a
                feasible and decoded solution vector.

        Returns:
            np.array: A decoded feasible solution vector.
        """
        arz = self.randn(self.dim) 
        delta = (self.BD @ arz.transpose()) * self.sigma
        arx = self.fitfun.closestFeasible(self.xmean + delta.transpose())  
        return self.fitfun.decode(arx)

    def tell_one(self,
                 y: float, 
                 x: np.array) -> int:
        """
        Processes the given fitness value and solution vector, performs updates
        to internal data structures, and logs progress during optimization.

        This function evaluates and integrates a solution (x) and its corresponding
        fitness value (y), updating the CMA-ES optimization state when the required
        population size is reached.

        Args:
            y: A float representing the fitness value of the solution.
            x: A numpy array containing the solution.

        Returns:
            An integer indicating whether the optimization process should stop.
        """

        if self.fitness is None or not type(self.fitness) is list:
            self.arx = []
            self.fitness = []
        self.fitness.append(y)
        self.arx.append(x)
        if len(self.fitness) >= self.popsize:
            self.fitness = np.asarray(self.fitness)
            self.arx = np.array([self.fitfun.encode(x) for x in self.arx])
            try:
                self.arz = (linalg.inv(self.BD) @ \
                            ((self.arx - self.xmean).transpose() / self.sigma)).transpose()   
            except Exception:
                if self.arz is None: 
                    self.arz = self.randn(self.popsize, self.dim)
            self.iterations += 1
            self.updateCMA()
            self.arz = None
            self.arx = []
            self.fitness = []
        
        if is_debug_active():
            self.n_evals.value += 1
            if y < self.best_y.value or self.n_evals.value % 1000 == 999:           
                if y < self.best_y.value: self.best_y.value = y
                t = time() - self.time_0
                c = self.n_evals.value
                message = '"c/t={0:.2f} c={1:d} t={2:.2f} y={3:.5f} yb={4:.5f} x={5!s}'.format(
                    c/t, c, t, y, self.best_y.value, x)
                logger.debug(message)
        return self.stop                
           
    def newArgs(self):
        """
        Generates a new population of candidate solutions for optimization.

        The method produces random offspring solutions based on the current mean and the
        covariance matrix of the search distribution. These offspring are then adjusted
        to ensure they are closest to feasible solutions within the search space.

        Args:
            self: The instance of the class invoking this method.
        """
        # generate random offspring
        self.arz = self.randn(self.popsize, self.dim)    
        delta = (self.BD @ self.arz.transpose()) * self.sigma
        self.arx = self.fitfun.closestFeasible(self.xmean + delta.transpose())  
    
    def do_optimize_delayed_update(self, fun, max_evals=None, workers=mp.cpu_count()):
        """
        Optimizes a given function asynchronously with delayed updates.

        This method performs optimization by evaluating the function in parallel.
        It uses a worker-based approach where several evaluations of `fun` are
        performed simultaneously to explore the parameter space. The results
        are processed iteratively to determine the best solution based on the given
        optimization criteria.

        Args:
            self: Represents the instance of the class in which this function is
                defined.
            fun: Callable optimization objective function to be minimized or maximized.
            max_evals: Optional; Maximum number of evaluations to perform. If not
                provided, the default value of `self.max_evaluations` is used.
            workers: Optional; The number of parallel workers to use for evaluating
                the objective function. Defaults to the number of CPU cores.

        Returns:
            tuple: A tuple containing:
                - best_x: The best solution found for the input function.
                - best_value: The best function value found.
                - evals: The total number of evaluations performed.
                - iterations: The number of iterations completed during optimization.
                - stop: The stop condition or code signaling the termination reason.
        """
        if not max_evals is None:
            self.max_evaluations =  max_evals
        evaluator = Evaluator(fun)
        evaluator.start(workers)
        evals_x = {}
        self.evals = 0;
        for _ in range(workers): # fill queue
            x = self.ask_one()
            evaluator.pipe[0].send((self.evals, x))
            evals_x[self.evals] = x # store x
            self.evals += 1
            
        while True: # read from pipe, tell es and create new x
            evals, y = evaluator.pipe[0].recv()
            
            x = evals_x[evals] # retrieve evaluated x
            del evals_x[evals]
            stop = self.tell_one(y, x) # tell evaluated x
            if stop != 0 or self.evals >= self.max_evaluations:
                break # shutdown worker if stop criteria met
            
            x = self.ask_one() # create new x
            evaluator.pipe[0].send((self.evals, x))       
            evals_x[self.evals] = x  # store x
            self.evals += 1            
        evaluator.stop()
        return self.best_x, self.best_value, evals, self.iterations, self.stop 
         
    def doOptimize(self):
        """
        Performs an optimization process within a generation loop until termination
        conditions are met. This method repeatedly generates candidate solutions,
        evaluates them using the objective function, updates the optimization state,
        and checks for stopping criteria.

        Args:
            self:

        Returns:
            tuple: A tuple containing the following elements:
                - best_x: The best solution found during the optimization process.
                - best_value: The objective function value of the best solution.
                - fitfun.evaluation_counter: Total number of evaluations performed.
                - iterations: Total number of iterations executed.
                - stop: Stop flag indicating the reason for termination.
        """
        # -------------------- Generation Loop --------------------------------
        while True:
            if self.iterations > self.max_iterations:
                break
            self.iterations += 1
            if self.fitfun.evaluation_counter > self.max_evaluations:
                break
            xs = self.ask()
            ys = self.fitfun.values(xs)
            self.tell(ys, xs)
            if self.stop != 0:
                break
        return self.best_x, self.best_value, self.fitfun.evaluation_counter, self.iterations, self.stop 
        
    def updateCMA(self):
        """
        Updates the evolutionary process using the Covariance Matrix Adaptation (CMA)
        strategy. This method handles the main operational logic for adaptive step-size
        control, selection, recombination, fitness evaluation, constraint checking, and
        diverse termination criteria during the optimization.

        Args:
            self: Instance of the class to which this method belongs.

        Returns:
            int: Returns -1 if the fitness values contain NaN or infinite entries;
            otherwise, returns nothing.
        """
        # Stop for Nan / infinite fitness values
        if np.isfinite(self.fitness).sum() < self.popsize:
            return -1
        # Sort by fitness and compute weighted mean into xmean
        arindex = self.fitness.argsort()        
        best_fitness = self.fitness[arindex[0]]
        worstFitness = self.fitness[arindex[-1]]                        
        if self.best_value > best_fitness:
            self.best_value = best_fitness
            self.best_x = self.fitfun.decode(self.arx[arindex[0]])    
            if best_fitness < self.stop_fitness:
                self.stop = 1
                return 

        # Calculate new xmean, this is selection and recombination
        xold = self.xmean # for speed up of Eq. (2) and (3)
        bestIndex = arindex[:self.mu] 
        bestArx = self.arx[bestIndex]
        self.xmean = np.transpose(bestArx) @ self.weights
        bestArz = self.arz[bestIndex]
        zmean = np.transpose(bestArz) @ self.weights
        hsig = self.updateEvolutionPaths(zmean, xold)            
        # Adapt step size sigma - Eq. (5)
        self.sigma *= math.exp(min(1.0, (self.normps / self.chiN - 1.) * self.cs / self.damps))            
        
        if self.iterations >= self.last_update + self.lazy_update_gap:
            self.last_update = self.iterations
            negccov = self.updateCovariance(hsig, bestArx, self.arz, arindex, xold)
            self.updateBD(negccov)                        
            # handle termination criteria
            sqrtDiagC = np.sqrt(np.abs(self.diagC))
            pcCol = self.pc
            for i in range(self.dim):
                if self.sigma * max(abs(pcCol[i]), sqrtDiagC[i]) > self.stopTolX:
                    break
                if i == self.dim - 1:
                    self.stop = 2
            if self.stop != 0:
                return            
            for i in range(self.dim):
                if self.sigma * sqrtDiagC[i] > self.stopTolUpX:
                    self.stop = 3
                    break
            if self.stop != 0:
                return 
        history_best = min(self.fitness_history)
        history_worst = max(self.fitness_history)
        if self.iterations > 2 and max(history_worst, worstFitness) - min(history_best, best_fitness) < self.stopTolFun:
            self.stop = 4
            return 
        if self.iterations > self.fitness_history.size and history_worst - history_best < self.stopTolHistFun:
            self.stop = 5
            return 
        # condition number of the covariance matrix exceeds 1e14
        if min(self.diagD) != 0 and \
                max(self.diagD) / min(self.diagD) > 1e7 * 1.0 / math.sqrt(self.accuracy):
            self.stop = 6
            return 
        # call callback
        if (not self.is_terminate is None) and \
                       self.is_terminate(self.runid, self.iterations, self.best_value):
            self.stop = 7
            return 
        # Adjust step size in case of equal function values (flat fitness)
        if self.best_value == self.fitness[arindex[int(0.1 + self.popsize / 4.)]]:
            self.sigma *= math.exp(0.2 + self.cs / self.damps)
        if self.iterations > 2 and max(history_worst, best_fitness) - min(history_best, best_fitness) == 0:
            self.sigma *= math.exp(0.2 + self.cs / self.damps)
        # store best in history
        self.fitness_history[1:] = self.fitness_history[:-1]
        self.fitness_history[0] = best_fitness
        return       
    
    def updateEvolutionPaths(self, zmean, xold):
        """
        Updates the evolution paths required for generating new samples in
        the covariance matrix adaptation evolution strategy (CMA-ES). This
        method modifies the evolution paths `ps` and `pc` based on the
        weighted mean of new samples (`zmean`) and the previous solution
        (`xold`). Additionally, it computes a flag (`hsig`) indicating
        whether the evolution path `ps` satisfies specific conditions,
        which is used to adapt the covariance matrix of the algorithm.

        Args:
            zmean: Weighted mean of new samples in the search space.
            xold: Previous solution in the search space.

        Returns:
            bool: A flag indicating whether the evolution path `ps` satisfies
            the computed conditions.
        """

        self.ps = self.ps * (1. - self.cs) + \
            ((self.B @ zmean) * math.sqrt(self.cs * (2. - self.cs) * self.mueff))
        self.normps = math.sqrt(self.ps @ self.ps) 
        hsig = self.normps / math.sqrt(1. - math.pow(1. - self.cs, 2. * self.iterations)) / \
            self.chiN < 1.4 + 2. / (self.dim + 1.)
        self.pc *= 1. - self.cc        
        if hsig:
            self.pc += (self.xmean - xold) * (math.sqrt(self.cc * (2. - self.cc) * self.mueff) / self.sigma)
        return hsig
    
    def updateCovariance(self, hsig, bestArx, arz, arindex, xold):
        """Updates the covariance matrix and returns the negative covariance scaling factor.

        This method adjusts the covariance matrix based on the current best solution, the
        distribution of previous samples, and other adaptations. It incorporates mechanisms
        to adaptively handle both positive and negative covariance contributions and ensures
        that minimal residual variance is maintained for better exploration capabilities.

        Args:
            self: Instance of the class calling this method.
            hsig (bool): Indicates the success of the covariance step size adaptation.
            bestArx (numpy.ndarray): Array containing the best solutions in the current iteration.
            arz (numpy.ndarray): Array of normalized random vectors representing candidate solutions.
            arindex (numpy.ndarray): Array containing sorted indices corresponding to ranked solutions.
            xold (numpy.ndarray): Centroid of the previous population.

        Returns:
            float: The negative covariance scaling factor (negccov), which indicates the adaptation
            strength for negative covariance components.
        """
      
        negccov = 0.
        if self.ccov1 + self.ccovmu > 0:
            arpos = (bestArx - xold) * (1. / self.sigma) # mu difference vectors
            pc2d = self.pc[:, np.newaxis] 
            roneu = (pc2d @ np.transpose(pc2d)) * self.ccov1
            # minor correction if hsig==false
            oldFac = 0 if hsig else self.ccov1 * self.cc * (2. - self.cc)
            oldFac += 1. - self.ccov1 - self.ccovmu
            # Adapt covariance matrix C active CMA
            negccov = (1. - self.ccovmu) * 0.25 * self.mueff \
                    / (math.pow(self.dim + 2., 1.5) + 2. * self.mueff)
            negminresidualvariance = 0.66
            # keep at least 0.66 in all directions, small popsize are most critical
            negalphaold = 0.5 # where to make up for the variance loss,
            # prepare vectors, compute negative updating matrix Cneg
            arReverseIndex = arindex[::-1]
            arzneg = arz[arReverseIndex[:self.mu]]
            arnorms = np.sqrt(np.einsum('ij->i', arzneg * arzneg))
            idxnorms = arnorms.argsort()
            arnormsSorted = arnorms[idxnorms]
            idxReverse = idxnorms[::-1]
            arnormsReverse = arnorms[idxReverse]
            arnorms = arnormsReverse / arnormsSorted
            arnormsInv = np.empty(arnorms.size)
            arnormsInv[idxnorms] = arnorms
            # check and set learning rate negccov
            negcovMax = (1. - negminresidualvariance) / ((arnormsInv*arnormsInv) @ self.weights)
            if negccov > negcovMax:
                negccov = negcovMax
            arzneg = np.transpose(arzneg) * arnormsInv
            artmp = self.BD @ arzneg
            Cneg = artmp @ (np.diagflat(self.weights) @ np.transpose(artmp))
            oldFac += negalphaold * negccov
            C = (self.C * oldFac) + roneu + \
                np.transpose(arpos * (self.ccovmu + (1. - negalphaold) * negccov)) @ \
                    np.transpose(self.weights * np.transpose(arpos))
            self.C = C - Cneg*negccov
        return negccov
    
    def updateBD(self, negccov):
        """
        Updates the internal state of the covariance matrix and its derived attributes, such as eigenvalues, eigenvectors,
        and transformations for sampling scaled isotropic Gaussian distributions. The method ensures numerical stability
        by handling edge cases like non-positive eigenvalues or large differences between eigenvalues.

        Args:
            self: Instance of the class.
            negccov: The covariance matrix adjustment parameter used for updating.
        """
 
        if self.ccov1 + self.ccovmu + negccov > 0 and \
                self.iterations % (1. / (self.ccov1 + self.ccovmu + negccov) / self.dim / 10.) < 1.:
            # to achieve O(N^2) enforce symmetry to prevent complex numbers
            self.C = np.triu(self.C, 0) + np.transpose(np.triu(self.C, 1))
            
            # diagD defines the scaling
            eigen_values, eigen_vectors = linalg.eigh(self.C)

            idx = eigen_values.argsort()[::-1]   
            self.diagD = eigen_values[idx]
            self.B = eigen_vectors[:,idx]

            # Coordinate system
            if min(self.diagD) <= 0:    
                self.diagD = np.maximum(self.diagD, 0)
                tfac = max(self.diagD) / 1e14
                self.C += np.eye(self.dim) * tfac
                self.diagD += np.ones(self.dim) * tfac

            if max(self.diagD) > 1e14 * min(self.diagD):
                tfac = max(self.diagD) / 1e14 - min(self.diagD)
                self.C += np.eye(self.dim) * tfac
                self.diagD += np.ones(self.dim) * tfac
                
            self.diagC = np.diag(self.C)
            self.diagD = np.sqrt(self.diagD) # diagD contains standard deviations now
            
            self.BD = self.B * self.diagD # O(n^2)
                        