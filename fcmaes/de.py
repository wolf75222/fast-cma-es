# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - de.py

 Description:
  - Numpy based implementation of Differential Evolution using the DE/best/1 strategy.
    Derived from its C++ counterpart [2].

    Uses three deviations from the standard DE algorithm:
    a) temporal locality introduced in [3] to improve convergence speed.
    b) reinitialization of individuals based on their age.
    c) oscillating CR/F parameters.

    You may keep parameters F and Cr at their defaults since this implementation works well with the given settings for most problems,
    since the algorithm oscillates between different F and Cr settings.

    The filter parameter is inspired by "Surrogate-based Optimisation for a Hospital Simulation"
    [4] where a machine learning classifier is used to
    filter candidate solutions for DE. A filter object needs to provide function add(x, y) to enable learning and
    a predicate is_improve(x, x_old, y_old) used to decide if function evaluation of x is worth the effort.

    The ints parameter is a boolean array indicating which parameters are discrete integer values. This
    parameter was introduced after observing non optimal results for the ESP2 benchmark problem:
    [5]
    If defined it causes a "special treatment" for discrete variables: They are rounded to the next integer value and
    there is an additional mutation to avoid getting stuck at local minima. This behavior is specified by the internal
    function _modifier which can be overwritten by providing the optional modifier argument. If modifier is defined,
    ints is ignored.

    Use the C++ implementation combined with parallel retry instead for objective functions which are fast to evaluate.
    For expensive objective functions (e.g. machine learning parameter optimization) use the workers
    parameter to parallelize objective function evaluation. This causes delayed population update.
    It is usually preferrable if popsize > workers and workers = mp.cpu_count() to improve CPU utilization.


 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/deoptimizer.cpp
  - [3] https://www.researchgate.net/publication/309179699_Differential_evolution_for_protein_folding_optimization_based_on_a_three-dimensional_AB_off-lattice_model
  - [4] https://dl.acm.org/doi/10.1145/3449726.3463283
  - [5] https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/DockerCFDBenchmark.py

 Documentation:
  -


=============================================================================
"""


import numpy as np
import math, sys
from time import time
import ctypes as ct
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.evaluator import Evaluator, is_debug_active
import multiprocessing as mp
from collections import deque
from loguru import logger
from typing import Optional, Callable, Tuple, Union
from numpy.typing import ArrayLike

def minimize(fun: Callable[[ArrayLike], float], 
             dim: Optional[int] = None,
             bounds: Optional[Bounds] = None,
             popsize: Optional[int] = 31,
             max_evaluations: Optional[int] = 100000,
             workers: Optional[int] = None,
             stop_fitness: Optional[float] = -np.inf,
             keep: Optional[int] = 200,
             f: Optional[float] = 0.5,
             cr: Optional[float] = 0.9,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             filter = None,
             ints: Optional[ArrayLike] = None,
             min_mutate: Optional[float] = 0.1,
             max_mutate: Optional[float] = 0.5,
             modifier: Optional[Callable] = None) -> OptimizeResult:
    """
    Minimize an objective function using the Differential Evolution (DE) algorithm.

    This function optimizes a given objective function using the DE algorithm, which is
    a population-based metaheuristic optimization technique. It iteratively improves a
    population of potential solutions based on mutation, crossover, and selection
    strategies until a stopping criterion is met. The user can define various parameters
    to customize the behavior of the optimization process.

    Args:
        fun: The objective function to be minimized, which takes a single argument
            (a candidate solution) and returns a scalar value representing its fitness.
        dim: Optional, number of dimensions of the input solution vector. If not
            provided, it will be inferred from `bounds`.
        bounds: Optional, bounds for each dimension of the input solution represented
            as a sequence of (min, max) tuples. It defines the permissible search space.
        popsize: Optional, population size for the DE algorithm. Determines the number
            of candidate solutions in each iteration. Default is 31.
        max_evaluations: Optional, maximum number of fitness evaluations allowed for
            the optimization process. Default is 100000.
        workers: Optional, number of parallel workers for evaluations. If set to more
            than 1, evaluations will be performed in parallel. If None or 1, sequential
            evaluation will be used. Default is None.
        stop_fitness: Optional, fitness value at which optimization will stop early.
            If a solution with fitness below this value is found, the optimization
            terminates. Default is -infinity.
        keep: Optional, defines how many solutions to retain for rebounded sampling.
            Default is 200.
        f: Optional, scale factor used to control mutation in the DE algorithm. Default
            is 0.5.
        cr: Optional, crossover rate used to control recombination in the DE algorithm.
            Default is 0.9.
        rg: Optional, random number generator used for stochastic components in the DE
            algorithm. Default is Generator(PCG64DXSM()).
        filter: Optional, a function to filter and validate solutions generated during
            the optimization process.
        ints: Optional, an array-like structure specifying indices of dimensions that
            should be treated as integers.
        min_mutate: Optional, minimum mutation factor for the DE mutation strategy.
            Default is 0.1.
        max_mutate: Optional, maximum mutation factor for the DE mutation strategy.
            Default is 0.5.
        modifier: Optional, a custom function to modify the behavior of the DE
            algorithm during optimization.

    Returns:
        OptimizeResult: An object containing the optimization results:
            - x: The best solution found.
            - fun: The objective function value corresponding to `x`.
            - nfev: The total number of function evaluations performed.
            - nit: The number of iterations completed.
            - status: An integer indicating why the optimization stopped. A positive
                value represents success.
            - success: A boolean indicating whether the optimization was successful.

    Raises:
        Exception: If any fatal error occurs during optimization, the function will
            return an OptimizeResult object with `success` set to False.
    """

    
    de = DE(dim, bounds, popsize, stop_fitness, keep, f, cr, rg, filter, ints,  
            min_mutate, max_mutate, modifier)
    try:
        if workers and workers > 1:
            x, val, evals, iterations, stop = de.do_optimize_delayed_update(fun, max_evaluations, workers)
        else:      
            x, val, evals, iterations, stop = de.do_optimize(fun, max_evaluations)
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, 
                              success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)  

class DE(object):
    """A Differential Evolution (DE) optimization algorithm.

    This class implements a stochastic optimization algorithm called Differential
    Evolution. It is designed for solving optimization problems over continuous
    domains. The algorithm generates a population of candidate solutions and
    iteratively improves upon them using operations inspired by natural evolution.

    Attributes:
        dim (int): Dimensionality of the problem space.
        lower (np.ndarray): Lower bounds for the problem space.
        upper (np.ndarray): Upper bounds for the problem space.
        popsize (int): Size of the population.
        stop_fitness (float): Early stopping criterion based on fitness value.
        keep (int): Number of iterations to preserve local improvements.
        rg (Generator): Random number generator for sampling operations.
        F0 (float): Differential weight.
        Cr0 (float): Crossover probability.
        stop (int): Termination condition flag.
        iterations (int): Number of completed iterations.
        evals (int): Number of function evaluations performed.
        p (int): Current index for population members.
        improves (collections.deque): Queue of improvement candidates observed in
            recent iterations.
        filter (Optional): Optional filter for evaluating candidate solutions.
        ints (np.ndarray): Array indicating integer-constrained dimensions.
        min_mutate (float): Minimum mutation factor.
        max_mutate (float): Maximum mutation factor.
        best_x (np.ndarray): Best solution vector found so far.
        best_value (float): Fitness value of the best solution.
        best_i (int): Index of the best solution in the population.
        pop_iter (np.ndarray): Array storing the iteration at which each member
            was last updated.
        x (np.ndarray): Current population of candidate solutions.
        x0 (np.ndarray): Previous population of candidate solutions.
        y (np.ndarray): Fitness values of the current population.
    """
    def __init__(self,
                dim: int,
                bounds: Bounds,  
                popsize: Optional[int] = 31, 
                stop_fitness: Optional[float] = -np.inf, 
                keep: Optional[int] = 200, 
                F: Optional[float] = 0.5, 
                Cr: Optional[float] = 0.9, 
                rg: Optional[Generator] = Generator(PCG64DXSM()),
                filter: Optional = None,
                ints: Optional[ArrayLike] = None,
                min_mutate: Optional[float] = 0.1,
                max_mutate: Optional[float] = 0.5, 
                modifier: Optional[Callable] = None):
        """
        Initializes the optimizer with the specified parameters and configurations. This class is
        designed to manage and operate on populations of candidate solutions using specified rules
        and modifiers. It supports constraints like integer variables and bounds, and provides
        features such as mutation settings, filtering, and random number generation control.

        Args:
            dim (int): Dimensionality of the problem (number of variables to be optimized).
            bounds (Bounds): An object specifying the lower and upper bounds for the variables.
            popsize (Optional[int]): Size of the population to be evolved. Default is 31.
            stop_fitness (Optional[float]): Stopping criterion based on fitness value. Default is -inf.
            keep (Optional[int]): Maximum number of generations to be stored for tracking. Default is 200.
            F (Optional[float]): Differential weight for mutation. Default is 0.5.
            Cr (Optional[float]): Crossover probability. Default is 0.9.
            rg (Optional[Generator]): Random number generator for stochastic operations. Default
                is `Generator(PCG64DXSM())`.
            filter (Optional): Custom filter to apply on solutions during evolution. Default is None.
            ints (Optional[ArrayLike]): Indices of integer variables in the solution. Default is None.
            min_mutate (Optional[float]): Minimum mutation value used for specific mutation rules.
                Default is 0.1.
            max_mutate (Optional[float]): Maximum mutation value applied during evolution.
                Default is 0.5.
            modifier (Optional[Callable]): A custom variable modifier function applied during evolution.
                If None and `ints` is not None, a default integer modifier is used. Default is None.
        """
        self.dim, self.lower, self.upper = _check_bounds(bounds, dim)
        if popsize is None:
            popsize = 31
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.keep = keep 
        self.rg = rg
        self.F0 = F
        self.Cr0 = Cr
        self.stop = 0
        self.iterations = 0
        self.evals = 0
        self.p = 0
        self.improves = deque()
        self.filter = filter
        self.ints = np.array(ints)
        self.min_mutate = min_mutate
        self.max_mutate = max_mutate 
        # use default variable modifier for int variables if modifier is None
        if modifier is None and not ints is None:
            self.lower = self.lower.astype(float)
            self.upper = self.upper.astype(float)
            self.modifier = self._modifier
        else:
            self.modifier = modifier
        self._init()                     
        if is_debug_active():
            self.best_y = mp.RawValue(ct.c_double, 1E99)
            self.n_evals = mp.RawValue(ct.c_long, 0)
            self.time_0 = time()
     
    def ask(self) -> np.ndarray:
        """
        Generates and returns the next set of samples to be evaluated based on the evolutionary strategy.

        The method ensures that all individuals in the population either utilize previously identified
        improvements or are generated as new samples. If improvements are queued, they are integrated
        into the population. Otherwise, new samples are generated for population members.

        Returns:
            np.ndarray: An array containing the generated or improved samples for the entire population.
        """
        
        xs = [None] * self.popsize
        for _ in range(self.popsize):
            if self.improves:
                p, x = self.improves[0]
                if xs[p] is None:
                    xs[p] = x
                    self.improves.popleft()
                else:
                    break
            else:
                break
        for p in range(self.popsize):
            if xs[p] is None:
                _, _, xs[p] = self._next_x(p)      
        self.asked = xs      
        return xs
 
    def tell(self, 
             ys:ArrayLike, 
             xs:Optional[ArrayLike] = None) -> int:

        """
        Tells the algorithm about the function evaluations (objective values) for the proposed
        solutions and updates the state accordingly.

        This method is used to inform the algorithm of the results of one or more objective
        function evaluations. This information is essential for the algorithm to proceed
        with the optimization process and adjust its internal model or state.

        Args:
            ys (ArrayLike): Array of function evaluation results (objective values)
                corresponding to the proposed solutions.
            xs (Optional[ArrayLike]): Array of proposed solutions for which the objective
                values (ys) have been evaluated. If not provided, the method will use
                solutions that were previously "asked" by the algorithm.

        Returns:
            int: An integer indicating the stop condition of the optimization process.
        """

        if xs is None:
            xs = self.asked
        self.evals += len(ys)
        for p in range(len(ys)):
            self.tell_one(p, ys[p], xs[p])
        return self.stop

    def population(self) -> np.ndarray:
        """
        Retrieves the population data stored in the class.

        The method returns the population data, which is stored in the object's state. It is expected
        to be a NumPy array representing specific data related to the population.

        Returns:
            np.ndarray: The population data stored in the object.
        """
        return self.x

    def result(self) -> OptimizeResult:
        """
        Returns an optimization result object containing the final outcome of the
        optimization process.

        This method encapsulates the key output data in an `OptimizeResult` object,
        providing details about the best solution found during the optimization,
        the function value at that solution, the total number of function evaluations,
        the number of iterations performed, and the status of the optimizer.

        Returns:
            OptimizeResult: An object containing the results of the optimization process,
            including details such as the optimal solution (`x`), the objective function
            value at that solution (`fun`), total function evaluations (`nfev`),
            number of iterations (`nit`), the stopping status (`status`), and a success
            flag (`success`).
        """
        return OptimizeResult(x=self.best_x, fun=self.best_value,
                              nfev=self.iterations*self.popsize, 
                              nit=self.iterations, status=self.stop, success=True)
    
    def ask_one(self) -> Tuple[int, np.ndarray]:
        """
        Generates the next candidate from the population queue or computes a new one.

        This function retrieves the next candidate solution for evaluation. If pre-generated
        candidates are available in the queue (`self.improves`), it dequeues one. Otherwise,
        it computes a new candidate based on the current population index.

        Returns:
            Tuple[int, np.ndarray]: A tuple where the first element is the population index
            of the candidate, and the second element is the candidate solution in the form
            of a numpy array.

        Raises:
            No explicit exceptions are raised by this method.
        """
        
        if self.improves:
            p, x = self.improves.popleft()
        else:
            p = self.p
            _, _, x = self._next_x(p)
            self.p = (self.p + 1) % self.popsize
        return p, x

    def tell_one(self, p: int, y:float , x:ArrayLike) -> int:
        """
        Process a single individual's evaluation update within a population-based optimization
        algorithm. Updates the values of the provided population member based on the fitness
        of the given evaluation and decides whether to replace it with a new candidate. Tracks
        improvements and adjusts population state as necessary.

        Args:
            p (int): Index of the population member to update.
            y (float): Evaluated fitness value of the candidate solution.
            x (ArrayLike): Candidate solution corresponding to the evaluated fitness.

        Returns:
            int: Indicator for whether the optimization should stop (1 for stop, 0 for continue).
        """

        if not self.filter is None:
            self.filter.add(x, y)
        
        if (self.y[p] > y):
            # temporal locality
            if self.iterations > 1:
                self.improves.append((p, self._next_improve(self.x[self.best_i], x, self.x0[p])))     
            self.x0[p] = self.x[p]
            self.x[p] = x
            self.y[p] = y
            if self.y[self.best_i] > y:
                self.best_i = p
                if self.best_value > y:
                    self.best_x = x
                    self.best_value = y
                    if self.stop_fitness > y:
                        self.stop = 1
            self.pop_iter[p] = self.iterations
        else:
            if self.rg.uniform(0, self.keep) < self.iterations - self.pop_iter[p]:
                self.x[p] = self._sample()
                self.y[p] = np.inf
        
        if is_debug_active():
            self.n_evals.value += 1
            if y < self.best_y.value or self.n_evals.value % 1000 == 999:           
                if y < self.best_y.value: self.best_y.value = y
                t = time() - self.time_0 + 1E-9
                c = self.n_evals.value
                message = '"c/t={0:.2f} c={1:d} t={2:.2f} y={3:.5f} yb={4:.5f} x={5!s}'.format(
                    c/t, c, t, y, self.best_y.value, x)
                logger.debug(message)

        return self.stop 

    def _init(self):
        """
        Initializes the optimization population and sets up initial values.

        This method initializes the population matrix and related attributes for an
        optimization problem. It samples the initial values for the population, assigns
        the initial objective values to infinity, and determines the placeholder for
        the best solution and its corresponding value.

        Attributes:
            x (ndarray): The population matrix of shape (popsize, dim), where each row
                represents an individual solution in the population.
            x0 (ndarray): A matrix of the initial population, identical to `x` at this
                stage.
            y (ndarray): Array of objective values for each individual in the
                population.
            best_x (ndarray): Best solution found so far, initialized with the first
                individual of the population.
            best_value (float): Objective value of the best solution found so far,
                initialized to infinity.
            best_i (int): Index of the current best solution in the population,
                initialized to 0.
            pop_iter (ndarray): Array to track the iteration or generation count for
                each individual in the population.
        """
        self.x = np.zeros((self.popsize, self.dim))
        self.x0 = np.zeros((self.popsize, self.dim))
        self.y = np.empty(self.popsize)
        for i in range(self.popsize):
            self.x[i] = self.x0[i] = self._sample()
            self.y[i] = np.inf
        self.best_x = self.x[0]
        self.best_value = np.inf
        self.best_i = 0
        self.pop_iter = np.zeros(self.popsize)
       
    def apply_fun(self, x, x_old, y_old):
        """
        Applies the function to the input or returns a default fallback based on a filter.

        This method evaluates the given function `fun` for the input `x`. If a `filter`
        is present, it checks the `filter` criteria to determine if the evaluation
        should proceed. If the criteria are met, it applies the function and adds
        the result to the filter, otherwise it returns a predefined fallback value
        (1E99). If no `filter` exists, the function is always evaluated.

        Args:
            x: The current input to the function `fun` to be evaluated.
            x_old: The previous input value in the process for comparison in the
                filter.
            y_old: The previous output value in the process for comparison in the
                filter.

        Returns:
            The result of evaluating the function `fun(x)` if no filter exists or the
            filter criteria are fulfilled. Otherwise, returns the fallback value
            1E99.
        """
        if self.filter is None:
            self.evals += 1
            return self.fun(x)
        else:
            if self.filter.is_improve(x, x_old, y_old):
                self.evals += 1
                y = self.fun(x)
                self.filter.add(x, y)
                return y
            else:    
                return 1E99
       
    def do_optimize(self, fun, max_evals):
        """
        Optimizes a given objective function using a population-based strategy. The optimization process is based on
        iterative improvements of candidate solutions to minimize the objective function value.

        Args:
            fun: Callable
                The objective function to be minimized. Must accept input vectors resembling the solution
                representation.
            max_evals: int
                The maximum number of function evaluations allowed during the optimization process.

        Returns:
            Tuple
                A tuple containing the following:
                - best_x: Array or list representing the best solution found during the optimization.
                - best_value: float, the value of the objective function for the best solution.
                - evals: int, the total number of function evaluations performed.
                - iterations: int, the total number of iterations completed during the optimization process.
                - stop: int, an indicator explaining the stop condition (e.g., termination criteria met).
        """
        self.fun = fun
        self.max_evals = max_evals    
        self.iterations = 0
        self.evals = 0
        while self.evals < self.max_evals:
            for p in range(self.popsize):
                xb, xi, x = self._next_x(p)
                y = self.apply_fun(x, xi, self.y[p])
                if y < self.y[p]:
                    # temporal locality
                    if self.iterations > 1:
                        x2 = self._next_improve(xb, x, xi)
                        y2 = self.apply_fun(x2, x, y)
                        if y2 < y:
                            y = y2
                            x = x2
                    self.x[p] = x
                    self.y[p] = y
                    self.pop_iter[p] = self.iterations
                    if y < self.y[self.best_i]:
                        self.best_i = p;
                        if y < self.best_value:
                            self.best_value = y;
                            self.best_x = x;
                            if self.stop_fitness > y:
                                self.stop = 1
                else:
                    # reinitialize individual
                    if self.rg.uniform(0, self.keep) < self.iterations - self.pop_iter[p]:
                        self.x[p] = self._sample()
                        self.y[p] = np.inf
                if self.evals >= self.max_evals:
                    break

        return self.best_x, self.best_value, self.evals, self.iterations, self.stop

    def do_optimize_delayed_update(self, fun, max_evals, workers=mp.cpu_count()):
        """
        Performs a delayed update optimization process using multiple workers for
        parallel evaluation. The method initializes workers, evaluates population
        solutions, and determines the best solution through iterated queries
        and evaluations.

        Args:
            fun: Callable function to be optimized.
            max_evals: Maximum number of function evaluations to perform.
            workers: Integer specifying the number of workers for parallel
                execution. Defaults to the number of available CPU cores.

        Returns:
            A tuple containing:
                - best_x: The best solution found during the optimization process.
                - best_value: The objective function value corresponding to the
                  best solution.
                - evals: The total number of evaluations performed.
                - iterations: The number of optimization iterations completed.
                - stop: An integer representing the stopping criterion met.
        """
        self.fun = fun
        self.max_evals = max_evals    
        evaluator = Evaluator(self.fun)
        evaluator.start(workers)
        evals_x = {}
        self.iterations = 0
        self.evals = 0
        self.p = 0
        self.improves = deque()
        for _ in range(workers): # fill queue with initial population
            p, x = self.ask_one()
            evaluator.pipe[0].send((self.evals, x))
            evals_x[self.evals] = p, x # store x
            self.evals += 1
            
        while True: # read from pipe, tell de and create new x
            evals, y = evaluator.pipe[0].recv()            
            p, x = evals_x[evals] # retrieve evaluated x
            del evals_x[evals]
            self.tell_one(p, y, x) # tell evaluated x
            if self.stop != 0 or self.evals >= self.max_evals:
                break # shutdown worker if stop criteria met
            
            for _ in range(workers):
                p, x = self.ask_one() # create new x          
                if self.filter is None or \
                    self.filter.is_improve(x, self.x[p], self.y[p]):
                        break
            evaluator.pipe[0].send((self.evals, x))       
            evals_x[self.evals] = p, x  # store x
            self.evals += 1
            
        evaluator.stop()
        return self.best_x, self.best_value, self.evals, self.iterations, self.stop
       
    def _next_x(self, p):
        """
        Determines the next vector in an evolutionary computation process.

        This method calculates a new candidate vector based on differential
        evolution principles. If a predefined number of iterations has been
        reached and the process is in even iterations, certain parameters
        (Cr and F) are halved. Random indices are selected to generate a
        mutation vector, ensuring the indices do not overlap with the
        current or best vector. A trial vector is then computed using a
        crossover operation.

        Args:
            p: int
                Index of the current vector in the population.

        Returns:
            tuple:
                Contains the best vector (xb), the current vector (xp),
                and the newly generated vector (x).
        """
        if p == 0:
            self.iterations += 1
            self.Cr = 0.5*self.Cr0 if self.iterations % 2 == 0 else self.Cr0
            self.F = 0.5*self.F0 if self.iterations % 2 == 0 else self.F0
        while True:
            r1, r2 = self.rg.integers(0, self.popsize, 2)
            if r1 != p and r1 != self.best_i and r1 != r2 \
                    and r2 != p and r2 != self.best_i:
                break
        xp = self.x[p]
        xb = self.x[self.best_i]
        x1 = self.x[r1]
        x2 = self.x[r2]
        x = self._feasible(xb + self.F * (x1 - x2))
        r = self.rg.integers(0, self.dim)
        tr = np.array(
            [i != r and self.rg.random() > self.Cr for i in range(self.dim)])    
        x[tr] = xp[tr]  
        if not self.modifier is None:
            x = self.modifier(x)
        return xb, xp, x

    def _next_improve(self, xb, x, xi):
        """
        Compute the next feasible point to improve optimization.

        This method adjusts the provided point to ensure it meets feasibility
        criteria and optionally applies a modifier to the point before returning.

        Args:
            xb: The base point for the improvement computation.
            x: The current point to be adjusted.
            xi: The initial guess or reference point for adjustment.

        Returns:
            The next feasible point after adjustment and optional modification.
        """
        x = self._feasible(xb + ((x - xi) * 0.5))
        if not self.modifier is None:
            x = self.modifier(x)
        return x
            
    def _sample(self):
        """
        Generates a sample from the specified distribution.

        If the `upper` attribute is not set, a random sample is drawn from a normal
        distribution. Otherwise, a sample is generated from a uniform distribution
        bounded by the `lower` and `upper` attributes. Optionally, a modifier function
        can be applied to the generated value before returning it.

        Returns:
            float: A random sample from the specified distribution. If a modifier
            function is applied, the modified value is returned.
        """
        if self.upper is None:
            return self.rg.normal()
        else:
            x = self.rg.uniform(self.lower, self.upper)
            if not self.modifier is None:
                x = self.modifier(x)
            return x
    
    def _feasible(self, x):
        """
        Checks the feasibility of a value with respect to specified bounds and adjusts it
        if necessary by clipping it to the range defined by the lower and upper bounds.

        Args:
            x: The value to check and potentially adjust.

        Returns:
            The original value if it lies within the bounds or a clipped value adjusted
            to satisfy the bounds if it falls outside the range.
        """
        if self.upper is None:
            return x
        else:
            return np.clip(x, self.lower, self.upper)
    
    # default modifier for integer variables
    def _modifier(self, x):
        """
        Modifies integer elements of the input array based on mutation probability.

        This method mutates specific elements of the input array, which are
        specified by the attribute `ints`. Mutation is governed by random values
        generated within the range defined by `min_mutate` and `max_mutate`.
        The mutated values lie within the bounds specified by `lower` and `upper`.

        Args:
            x (np.ndarray): Input array containing elements to mutate.

        Returns:
            np.ndarray: Mutated version of the input array.
        """
        x_ints = x[self.ints]
        n_ints = len(self.ints)
        lb = self.lower[self.ints]
        ub = self.upper[self.ints]
        to_mutate = self.rg.uniform(self.min_mutate, self.max_mutate)
        # mututate some integer variables
        x[self.ints] = np.array([x if self.rg.random() > to_mutate/n_ints else 
                           int(self.rg.uniform(lb[i], ub[i]))
                           for i, x in enumerate(x_ints)])
        return x   
                        
def _check_bounds(bounds, dim):
    """
    Validates and processes the input bounds and dimensions for subsequent operations.

    This function ensures that either `bounds` or `dim` is provided and determines the valid
    dimensionality and boundary arrays for further processing. If `bounds` is None, the
    function uses the provided `dim`. Otherwise, it extracts and converts the lower and
    upper bounds arrays from `bounds`.

    Args:
        bounds: User-provided object containing boundary attributes (`lb` and `ub`)
            for dimensional constraints. Can be None if `dim` is specified.
        dim: An integer defining the dimensionality, used only if `bounds` is None.

    Returns:
        Tuple comprising:
            - An integer representing the number of dimensions (from `bounds` or `dim`).
            - A numpy array of lower boundary values (`bounds.lb`) if provided; otherwise None.
            - A numpy array of upper boundary values (`bounds.ub`) if provided; otherwise None.

    Raises:
        ValueError: If both `bounds` and `dim` are None.
    """
    if bounds is None and dim is None:
        raise ValueError('either dim or bounds need to be defined')
    if bounds is None:
        return dim, None, None
    else:
        return len(bounds.ub), np.asarray(bounds.lb), np.asarray(bounds.ub)
    

