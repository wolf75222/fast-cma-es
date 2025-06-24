# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - mode.py

 Description:
  - Numpy based implementation of multi objective
    Differential Evolution using either the DE/rand/1 strategy
    or a NSGA-II like population update (parameter 'nsga_update=True)'.
    Then it works similar to NSGA-II.

    Supports parallel fitness function evaluation.

    Features enhanced multiple constraint ranking [2]
    improving its performance in handling constraints for engineering design optimization.

    Enables the comparison of DE and NSGA-II population update mechanism with everything else
    kept completely identical.

    Requires python 3.5 or higher.

    Uses the following deviation from the standard DE algorithm:
    a) oscillating CR/F parameters.

    You may keep parameters F and CR at their defaults since this implementation works well with the given settings for most problems,
    since the algorithm oscillates between different F and CR settings.

    For expensive objective functions (e.g. machine learning parameter optimization) use the workers
    parameter to parallelize objective function evaluation. The workers parameter is limited by the
    population size.

    The ints parameter is a boolean array indicating which parameters are discrete integer values. This
    parameter was introduced after observing non optimal DE-results for the ESP2 benchmark problem:
    [3]
    If defined it causes a "special treatment" for discrete variables: They are rounded to the next integer value and
    there is an additional mutation to avoid getting stuck at local minima. This behavior is specified by the internal
    function _modifier which can be overwritten by providing the optional modifier argument. If modifier is defined,
    ints is ignored.

    See [4] for a detailed description.

 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] https://www.jstage.jst.go.jp/article/tjpnsec/11/2/11_18/_article/-char/en/
  - [3] https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/DockerCFDBenchmark.py
  - [4] https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/MODE.adoc

 Documentation:
  -


=============================================================================
"""
from __future__ import annotations

import numpy as np
import os, sys, time
import ctypes as ct
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds

from fcmaes.evaluator import Evaluator, parallel_mo
from fcmaes import moretry
import multiprocessing as mp
from fcmaes.optimizer import dtime
from fcmaes.retry import Shared2d
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
             modifier: Callable = None,
             min_mutate: Optional[float] = 0.1,
             max_mutate: Optional[float] = 0.5,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             store: Optional[store] = None) -> Tuple[np.ndarray, np.ndarray]:

    """
    Minimizes a multi-objective function subject to constraints and bounds using evolutionary
    algorithms. The function supports parallel computation and allows various configurations
    to modify the optimization process.

    Args:
        mofun (Callable[[ArrayLike], ArrayLike]): A callable representing the multi-objective
            function to minimize. It must take an array-like input and return an array-like
            output.
        nobj (int): The number of objective functions.
        ncon (int): The number of constraints.
        bounds (Bounds): The bounds for the decision variables.
        guess (Optional[np.ndarray]): Initial guess for the population. If not provided,
            a random guess will be generated.
        popsize (Optional[int]): The size of the population for the evolutionary algorithm.
        max_evaluations (Optional[int]): The maximum number of allowable function evaluations
            during the optimization process.
        workers (Optional[int]): The number of workers to use for parallel computation. A
            value of 1 indicates serial computation.
        f (Optional[float]): Differential evolution scale factor.
        cr (Optional[float]): Differential evolution crossover probability.
        pro_c (Optional[float]): Crossover probability for simulated binary crossover (SBX).
        dis_c (Optional[float]): Distribution index for simulated binary crossover (SBX).
        pro_m (Optional[float]): Mutation probability for polynomial mutation.
        dis_m (Optional[float]): Distribution index for polynomial mutation.
        nsga_update (Optional[bool]): Flag to enable or disable NSGA-II style updates.
        pareto_update (Optional[int]): The frequency of Pareto front updates.
        ints (Optional[ArrayLike]): Specific indices of decision variables that are treated
            as integers.
        modifier (Callable): A callable that modifies the process of variable updates. This
            can be used to apply additional constraints or actions during the optimization
            process.
        min_mutate (Optional[float]): Minimum mutation factor for decision variables.
        max_mutate (Optional[float]): Maximum mutation factor for decision variables.
        rg (Optional[Generator]): An instance of random number generator to ensure repeatability.
        store (Optional[store]): An optional storage object to store results at the end of
            optimization.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the optimal decision variables
            (x) and the corresponding objective function values (y). If optimization fails,
            returns (None, None).
    """

    try:   
        mode = MODE(nobj, ncon, bounds, popsize,
            f, cr, pro_c, dis_c, pro_m, dis_m, nsga_update, pareto_update, rg, ints, min_mutate, max_mutate, modifier)
        mode.set_guess(guess, mofun, rg)
        if workers <= 1:
            x, y, = mode.minimize_ser(mofun, max_evaluations)
        else:
            x, y = mode.minimize_par(mofun, max_evaluations, workers)
        if not store is None:
            store.add_results(x, y)
        return x, y
    except Exception as ex:
        print(str(ex))  
        return None, None  

    
class store():

    """
    Handles the management of shared data storage and processing for multi-objective optimization.

    This class facilitates the storage and processing of results for multi-objective optimization
    problems using shared memory objects to support multiprocessing. It includes functionality for
    adding new results, managing capacity, and extracting Pareto fronts.

    Attributes:
        dim (int): Dimensionality of the input data.
        nobj (int): Number of objectives for the optimization problem.
        capacity (int): Maximum capacity of the storage.
    """
    
    def __init__(self, dim, nobj, capacity = mp.cpu_count()*512):
        """
        Initializes an instance of the class with provided dimensions and objects.

        The constructor sets up shared memory buffers for storing data (`xs` and `ys`)
        and initializes synchronization mechanisms and counters to effectively handle
        data storage and retrieval.

        Args:
            dim: Number of dimensions for each data point.
            nobj: Number of objectives for each data point.
            capacity: Total initial storage capacity. Defaults to the number of CPU
                cores multiplied by 512.

        Attributes:
            dim (int): Number of dimensions for each data point.
            nobj (int): Number of objectives for each data point.
            capacity (int): Total initial storage capacity.
        """
        self.dim = dim
        self.nobj = nobj
        self.capacity = capacity
        self.add_mutex = mp.Lock()    
        self.xs = Shared2d(np.empty((self.capacity, self.dim), dtype = np.float64))
        self.ys = Shared2d(np.empty((self.capacity, self.nobj), dtype = np.float64)) 
        self.create_views()
        self.num_stored = mp.RawValue(ct.c_int, 0) 
        self.num_added = mp.RawValue(ct.c_int, 0) 

    def create_views(self): # needs to be called in the target process
        """
        Creates views for the class's xs and ys attributes.

        This method generates views for the xs and ys attributes, which should
        be initialized prior to calling this function. It ensures the views are
        references to the existing data rather than independent copies.

        Args:
            None

        Returns:
            None
        """
        self.xs_view = self.xs.view()
        self.ys_view = self.ys.view()

    def get_xs(self) -> np.ndarray:
        """
        Returns a view of the xs ndarray.

        This method provides a view of the internal numpy array `xs`, allowing
        users to access the data without creating a copy.

        Returns:
            np.ndarray: A view of the `xs` numpy array.
        """
        return self.xs.view()

    def get_ys(self) -> np.ndarray:
        """
        Returns a view of the `ys` attribute.

        This method provides a view of the `ys` NumPy array, allowing the caller to
        access the same data without creating a copy.

        Returns:
            np.ndarray: A view of the `ys` NumPy array.
        """
        return self.ys.view()

    def add_result(self, x, y):
        """
        Adds a result to the storage, ensuring that the storage capacity is not exceeded.
        If the storage reaches its capacity, it performs truncation and stores the new values accordingly.

        Args:
            x: The input data point to be added.
            y: The corresponding result or label associated with `x`.
        """
        with self.add_mutex:
            self.num_added.value += 1
            if self.num_stored.value >= self.capacity: 
                self.get_front(update=True)
                if self.num_stored.value >= self.capacity: 
                    n = int(self.num_stored.value/2)
                    self.xs_view[:n] = self.xs_view[:2*n:2] 
                    self.ys_view[:n] = self.ys_view[:2*n:2]                                
                    self.num_stored.value = n
            i = self.num_stored.value              
            self.xs_view[i] = x
            self.ys_view[i] = y[:self.nobj]
            self.num_stored.value = i + 1

    def add_results(self, xs, ys):
        """
        Adds the content of the provided `xs` and `ys` to the storage, while managing the storage
        capacity and ensuring proper synchronization.

        If the storage exceeds 90% of its capacity during the operation, the addition process will
        terminate early to avoid overfilling. The addition process is thread-safe due to `add_mutex`.

        Args:
            xs: A sequence containing the elements to be added to the storage.
            ys: A sequence of sequences, where each sub-sequence represents corresponding
                entries to `xs` and is truncated to the first `nobj` elements during storage.
        """
        with self.add_mutex:
            self.num_added.value += 1
            i = self.num_stored.value
            for j in range(len(xs)):
                if i < self.capacity:                
                    self.xs_view[i] = xs[j] 
                    self.ys_view[i] = ys[j][:self.nobj]
                    i += 1
                else:
                    self.get_front(update=True)
                    i = self.num_stored.value
                    if i > 0.9*self.capacity: # give up
                        return
            self.num_stored.value = i
                      
    def get_front(self, update=False):
        """
        Retrieves the Pareto front from stored values of `xs_view` and `ys_view`. Optionally, updates the
        values in-place if `update` is set to True.

        Args:
            update (bool): Determines whether the current views and stored count are updated with the
                Pareto front values.

        Returns:
            tuple: Contains two elements `(xf, yf)` where:
                `xf` - The x-coordinates of the Pareto front.
                `yf` - The y-coordinates of the Pareto front.
        """
        stored = self.num_stored.value
        xs = self.xs_view[:stored]
        ys = self.ys_view[:stored]
        xf, yf = moretry.pareto(xs, ys)
        if update:
            n = len(yf)
            self.xs_view[:n] = xf 
            self.ys_view[:n] = yf                                
            self.num_stored.value = n
        return xf, yf

    def get_content(self):
        """
        Retrieves and returns the stored content.

        This method accesses the current amount of stored data and retrieves
        the corresponding portions of `xs_view` and `ys_view` based on
        the stored count.

        Returns:
            tuple: A tuple containing two arrays or lists. The first element
            is the portion of `xs_view` up to the stored count, and the second
            element is the portion of `ys_view` up to the stored count.
        """
        stored = self.num_stored.value
        return self.xs_view[:stored], self.ys_view[:stored]

class MODE(object):
    """
    Multi-Objective Differential Evolution (MODE) optimization algorithm.

    This class implements a multi-objective optimization algorithm based on
    the Differential Evolution (DE) framework. It is designed to handle
    optimization problems with both multiple objectives and constraints. The
    algorithm aims to generate a set of Pareto-optimal solutions through
    evolutionary strategies while supporting parallel and serial execution for
    fitness evaluations.

    Attributes:
        nobj (int): Number of objectives to optimize.
        ncon (int): Number of constraints.
        dim (int): Dimensionality of the search space, derived from the bounds.
        lower (np.ndarray): Lower bounds for the search space dimensions.
        upper (np.ndarray): Upper bounds for the search space dimensions.
        popsize (int): Population size used for the optimization process.
        rg (numpy.random.Generator): Random number generator for reproducibility.
        F (float): Control parameter for differential weight in DE strategy.
        Cr (float): Crossover probability for DE strategy.
        pro_c (float): Probability for simulated binary crossover (SBX).
        dis_c (float): Distribution index for SBX.
        pro_m (float): Probability for polynomial mutation.
        dis_m (float): Distribution index for polynomial mutation.
        nsga_update (bool): Indicates whether NSGA-II update is enabled.
        pareto_update (int): Determines pareto update intensity; allows more elite selection if greater.
        min_mutate (float): Minimum possible mutation factor.
        max_mutate (float): Maximum possible mutation factor.
        modifier (Callable): A function to modify variables, allowing for domain-specific constraints.
    """
    def __init__(self, 
                nobj: int,
                ncon: int, 
                bounds: Bounds,
                popsize: Optional[int] = 64, 
                F: Optional[float] = 0.5, 
                Cr: Optional[float] = 0.9, 
                pro_c: Optional[float] = 0.5,
                dis_c: Optional[float] = 15.0,
                pro_m: Optional[float] = 0.9,
                dis_m: Optional[float] = 20.0,
                nsga_update: Optional[bool] = True,
                pareto_update: Optional[int] = 0,
                rg: Optional[Generator] = Generator(PCG64DXSM()),
                ints: Optional[ArrayLike] = None,
                min_mutate: Optional[float] = 0.1,
                max_mutate: Optional[float] = 0.5,   
                modifier: Callable = None):
        """
        Initializes the class with the given parameters and sets up the necessary attributes
        for the optimization process. This constructor validates the input bounds, adjusts
        the population size for NSGA updates, and sets default values for various optimization
        and mutation parameters.

        Args:
            nobj (int): The number of objectives for the optimization problem.
            ncon (int): The number of constraints for the optimization problem.
            bounds (Bounds): The lower and upper bounds for the decision variables.
            popsize (Optional[int]): The size of the population for the optimization process.
                Default is 64. Adjusted to be even if nsga_update is True.
            F (Optional[float]): Mutation factor for DE-based updates. Default is 0.5.
            Cr (Optional[float]): Crossover probability for DE-based updates. Default is 0.9.
            pro_c (Optional[float]): Crossover probability in SBX (Simulated Binary
                Crossover). Default is 0.5.
            dis_c (Optional[float]): Distribution index for SBX. Default is 15.0.
            pro_m (Optional[float]): Mutation probability. Default is 0.9.
            dis_m (Optional[float]): Distribution index for mutation (polynomial). Default
                is 20.0.
            nsga_update (Optional[bool]): Flag to enable NSGA (Non-dominated Sorting Genetic
                Algorithm) update. Default is True.
            pareto_update (Optional[int]): Criterion for updating Pareto front. Default is 0.
            rg (Optional[Generator]): Random number generator for the optimization process.
                Default is Generator(PCG64DXSM()).
            ints (Optional[ArrayLike]): Indices of integer decision variables. Used to
                support mixed-integer optimization. Automatically disabled if nsga_update
                is True.
            min_mutate (Optional[float]): Minimum mutation rate for the decision variables.
                Default is 0.1.
            max_mutate (Optional[float]): Maximum mutation rate for the decision variables.
                Default is 0.5.
            modifier (Callable): Function to modify integer decision variables. If not
                provided, a default modifier is used when integer variables exist.
        """
        self.nobj = nobj
        self.ncon = ncon
        self.dim, self.lower, self.upper = _check_bounds(bounds, None)
        if popsize is None:
            popsize = 64
        if popsize % 2 == 1 and nsga_update: # nsga update requires even popsize
            popsize += 1
        self.popsize = popsize
        self.rg = rg
        self.F0 = F
        self.Cr0 = Cr
        self.pro_c = pro_c
        self.dis_c = dis_c
        self.pro_m = pro_m
        self.dis_m = dis_m    
        self.nsga_update = nsga_update
        self.pareto_update = pareto_update
        self.stop = 0
        self.iterations = 0
        self.evals = 0
        self.mutex = mp.Lock()
        self.p = 0
        # nsga update doesn't support mixed integer
        self.ints = None if (ints is None or nsga_update) else np.array(ints)
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
               
    def set_guess(self, guess, mofun, rg = None):
        """
        Sets the initial guess values for an optimization or search algorithm.

        This function allows setting up initial guesses for the population values
        and their corresponding evaluations. Users can provide guesses as either
        a NumPy array of candidate solutions or a tuple containing candidate
        solutions and their objective function evaluations. Additionally, the
        function provides flexibility in defining a random number generator for
        sampling the guesses.

        Args:
            guess: A NumPy array containing candidate solution points, or a tuple
                (guess, ys), where `guess` is an array of candidate solutions, and
                `ys` contains objective evaluations corresponding to these solutions.
            mofun: A callable that computes the objective value for a given candidate
                solution. This function is applied to `guess` if `guess` is a
                NumPy array.
            rg: An optional random number generator (`numpy.random.Generator`)
                instance. If not specified, a default generator is created using
                PCG64DXSM.

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
        Generates the next batch of candidate solutions and updates internal population.

        Generates new solutions for the evolution strategy by iterating through the
        given population size, updating the extended population array with new values.
        Returns the array of newly generated solutions after the update.

        Returns:
            np.ndarray: The newly generated candidate solutions array.
        """
        for p in range(self.popsize):
            self.x[p + self.popsize] = self._next_x(p)
        return self.x[self.popsize:]
                
    def tell(self, ys: np.ndarray, xs: Optional[np.ndarray] = None):
        """
        Updates the internal population data with the given `ys` values, and optionally,
        with the corresponding `xs` values, if provided. This method is used to record
        new evaluations for individuals in the population and subsequently triggers
        population update.

        Args:
            ys (np.ndarray): Array containing the fitness values of the individuals to
                be updated in the population.
            xs (Optional[np.ndarray], optional): Array containing the decision variables
                of individuals to be updated in the population. Defaults to None.
        """
        if not xs is None:
            for p in range(self.popsize):
                self.x[p + self.popsize] = xs[p]
        for p in range(self.popsize):
            self.y[p + self.popsize] = ys[p]
        self.pop_update()
                         
    def _init(self):
        """
        Initializes the population matrices, decision variables, and objective values
        for a multi-objective optimization problem. It also initializes auxiliary
        variables used in optimization algorithms.

        Attributes:
            x: numpy.ndarray. A matrix containing decision variables for the
                population. The shape is `(2 * popsize, dim)`.
            y: numpy.ndarray. A matrix for storing objective and constraint
                evaluations. The shape is `(2 * popsize, nobj + ncon)`.
            vx: numpy.ndarray. A copy of the initial `x` matrix, used for
                intermediate calculations during optimization.
            vp: int. An auxiliary variable initialized to 0 for optimization purposes.
            ycon: None or type(related attribute if any). A placeholder for constraint-related
                computations, initialized as None.
            eps: int. An auxiliary variable for numerical stability or algorithmic
                adjustments, initialized to 0.
        """
        self.x = np.empty((2*self.popsize, self.dim))
        self.y = np.empty((2*self.popsize, self.nobj + self.ncon))
        for i in range(self.popsize):
            self.x[i] = self._sample()
            self.y[i] = np.array([1E99]*(self.nobj + self.ncon))
        self.vx = self.x.copy()
        self.vp = 0
        self.ycon = None
        self.eps = 0

    def minimize_ser(self, 
                     fun: Callable[[ArrayLike], ArrayLike], 
                     max_evaluations: Optional[int] = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Minimizes a given function using a custom optimization approach until the maximum number of evaluations
        is reached. The function utilizes a population-based strategy for optimization, where potential solutions
        are evaluated, and the results are used to iteratively improve the population.

        Args:
            fun: Callable function to be minimized. It takes an array-like input and returns an array-like output.
            max_evaluations: Optional maximum number of function evaluations to perform. Defaults to 100000.

        Returns:
            A tuple consisting of the final population of solutions (xs) and their corresponding evaluated
            results (ys).
        """
        evals = 0
        while evals < max_evaluations:
            xs = self.ask()
            ys = np.array([fun(x) for x in xs])
            self.tell(ys)
            evals += self.popsize
        return xs, ys

        
    def minimize_par(self, 
                     fun: Callable[[ArrayLike], ArrayLike], 
                     max_evaluations: Optional[int] = 100000, 
                     workers: Optional[int] = mp.cpu_count()) -> Tuple[np.ndarray, np.ndarray]:
        """
        Minimizes the given multi-objective function in parallel, using a specified number
        of workers and an evaluation limit. This method leverages a parallel computation
        framework to evaluate the function across multiple populations simultaneously,
        improving efficiency for computationally expensive optimization tasks.

        Args:
            fun (Callable[[ArrayLike], ArrayLike]): A function to be minimized. It takes
                an array-like input and returns an array-like output.
            max_evaluations (Optional[int]): The maximum number of function evaluations
                to perform. Defaults to 100,000 if not specified.
            workers (Optional[int]): The number of parallel workers to use for function
                evaluation. Defaults to the number of CPU cores available.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the final population of
                candidate solutions (xs) and the corresponding function evaluations (ys).
        """
        fit = parallel_mo(fun, self.nobj + self.ncon, workers)
        evals = 0
        while evals < max_evaluations:
            xs = self.ask()
            ys = fit(xs)
            self.tell(ys)
            evals += self.popsize
        fit.stop()
        return xs, ys
                                    
    def pop_update(self):
        """
        Updates the population by performing non-dominated sorting, crowding distance sorting,
        and variation for optimization purposes. This function presorts objective values if
        only one objective exists, reallocates population members based on domination level,
        and sorts for crowding when necessary. It also applies variation operators to update
        the population diversity.

        Raises:
            ValueError: If the total population size exceeds the specified `popsize`.

        Args:
            None

        Returns:
            None
        """
        y0 = self.y
        x0 = self.x
        if self.nobj == 1:
            yi = np.flip(np.argsort(self.y[:,0]))
            y0 = self.y[yi]
            x0 = self.x[yi]    
        domination, self.ycon, self.eps = pareto_domination(y0, self.nobj, self.ncon, self.ycon, self.eps)
        x = []
        y = []
        maxdom = int(max(domination))
        for dom in range(maxdom, -1, -1):
            domlevel = [p for p in range(len(domination)) if domination[p] == dom]
            if len(domlevel) == 0:
                continue
            if len(x) + len(domlevel) <= self.popsize:
                # whole level fits
                x = [*x, *x0[domlevel]]
                y = [*y, *y0[domlevel]]
            else: # sort for crowding
                nx = x0[domlevel]
                ny = y0[domlevel]    
                si = [0]
                if len(ny) > 1:                
                    cd = crowd_dist(ny)
                    si = np.flip(np.argsort(cd))
                for p in si:
                    if len(x) >= self.popsize:
                        break
                    x.append(nx[p])
                    y.append(ny[p])
                break # we have filled popsize members                                
        self.x[:self.popsize] = x[:self.popsize]
        self.y[:self.popsize] = y[:self.popsize]
        if self.nsga_update:
            self.vx = variation(self.x[:self.popsize], self.lower, self.upper, self.rg, 
                pro_c = self.pro_c, dis_c = self.dis_c, pro_m = self.pro_m, dis_m = self.dis_m) 
       
    def _next_x(self, p):
        """
        Determines the next candidate solution vector based on the current population
        and the DE (Differential Evolution) strategy, including NSGA-II or standard
        DE/pareto/1 strategies.

        This function applies specific update strategies to generate a feasible vector
        that follows the optimization constraints and the mutation/crossover
        mechanism. Depending on the selected strategy, it either takes from an elite
        subset or uses various sampling methods from the entire population.
        Differentiation is performed using a weighted mutation process on other
        population members.

        Args:
            p (int): Index of the current individual in the population that is being
                processed.

        Returns:
            numpy.ndarray: A feasible vector based on the mutation and crossover
            operations, bounded within the defined lower and upper limits.
        """
        if self.nsga_update: # use NSGA-II update strategy.
            x = self.vx[self.vp]
            self.vp = (self.vp + 1) % self.popsize # only use the elite
            return x
        # use standard DE/pareto/1 strategy.
        if p == 0: # switch FR / CR every generation
            self.iterations += 1
            self.Cr = 0.5*self.Cr0 if self.iterations % 2 == 0 else self.Cr0
            self.F = 0.5*self.F0 if self.iterations % 2 == 0 else self.F0
        while True:
            if self.pareto_update > 0: # sample elite solutions
                r1, r2 = self.rg.integers(0, self.popsize, 2)
                rb = int(self.popsize * (self.rg.random() ** (1.0 + self.pareto_update)))
            else:
                # sample from whole population
                r1, r2, rb = self.rg.integers(0, self.popsize, 3)
            if r1 != p and r1 != rb and r1 != r2 and r2 != rb \
                and r2 != p and rb != p:
                break
        xp = self.x[p]
        xb = self.x[rb]
        x1 = self.x[r1]
        x2 = self.x[r2]
        x = self._feasible(xb + self.F * (x1 - x2))
        r = self.rg.integers(0, self.dim)
        tr = np.array(
            [i != r and self.rg.random() > self.Cr for i in range(self.dim)])    
        x[tr] = xp[tr]    
        if not self.modifier is None:
            x = self.modifier(x)   
        return x.clip(self.lower, self.upper)
    
    def _sample(self):
        """
        Generates a sampled value based on specific distribution boundaries.

        If the `upper` attribute is not defined, a normal distribution value is
        sampled. Otherwise, a uniform distribution value is sampled between
        `lower` and `upper` attributes.

        Returns:
            float: A sampled value from the appropriate distribution.
        """
        if self.upper is None:
            return self.rg.normal()
        else:
            return self.rg.uniform(self.lower, self.upper)
    
    def _feasible(self, x):
        """
        Ensures that the input value x is constrained within the specified bounds,
        if bounds are defined. If an upper bound is not specified, the original
        value is returned.

        Args:
            x: The input value to be checked or clipped.

        Returns:
            The input value x, clipped between the lower and upper bounds if
            bounds are defined, or the original value if no bounds are present.
        """
        if self.upper is None:
            return x
        else:
            return np.clip(x, self.lower, self.upper)
        
    # default modifier for integer variables
    def _modifier(self, x):
        """
        Modifies the given input array by mutating certain integer variables based on
        a specified mutation probability.

        Args:
            x (np.ndarray): The input array to be modified.

        Returns:
            np.ndarray: The modified array after applying integer mutations.
        """
        x_ints = x[self.ints]
        n_ints = len(self.ints)
        lb = self.lower[self.ints]
        ub = self.upper[self.ints]
        to_mutate = self.rg.uniform(self.min_mutate, self.max_mutate)
        # mututate some integer variables
        x_ints = np.array([x if self.rg.random() > to_mutate/n_ints else 
                           int(self.rg.uniform(lb[i], ub[i]))
                           for i, x in enumerate(x_ints)])
        return x   
    
    def _is_dominated(self, y, p):
        """
        Determines whether a given solution vector is dominated by another solution
        vector in a multi-objective optimization context.

        Args:
            y: A sequence representing the solution vector to be checked.
            p: An integer index identifying the row number of the comparison vector
                in the multi-objective dataset.

        Returns:
            bool: True if the solution vector `y` is dominated by the vector located
            at index `p` in the dataset, otherwise False.
        """
        return np.all(np.fromiter((y[i] >= self.y[p, i] for i in range(len(y))), dtype=bool))

                    
def _check_bounds(bounds, dim):
    """
    Validates and processes the provided bounds and dimension information to ensure that
    either valid bounds or dimension are available. Converts bounds to Numpy arrays if they
    are provided, and determines the dimensionality based on the input.

    Args:
        bounds: Object that defines lower and upper bounds. If provided, its 'lb' and 'ub'
            attributes are processed.
        dim: An integer representing the dimensionality of the data or the length of bounds.

    Returns:
        A tuple containing:
            - int: The dimensionality derived from bounds or directly from the `dim`.
            - numpy.ndarray or None: Lower bounds as a NumPy array, or None if not provided.
            - numpy.ndarray or None: Upper bounds as a NumPy array, or None if not provided.

    Raises:
        ValueError: If both `dim` and `bounds` are None.
    """
    if bounds is None and dim is None:
        raise ValueError('either dim or bounds need to be defined')
    if bounds is None:
        return dim, None, None
    else:
        return len(bounds.ub), np.asarray(bounds.lb), np.asarray(bounds.ub)

def _filter(x, y):
    """
    Filters and sorts input arrays based on specific conditions.

    This function processes two input arrays `x` and `y`. It first determines the
    maximum value along the first axis of `y`, sorts both arrays based on this
    criterion, and then applies additional filtering conditions on `y` to exclude
    specific rows.

    Args:
        x: The input array to be processed and filtered.
        y: A 2D array whose maximum values along the first axis determine the
            sorting order, and which is filtered according to additional
            conditions.

    Returns:
        A tuple containing:
        - The filtered and sorted version of the input array `x`.
        - The filtered and sorted version of the array `y`.
    """
    ym = np.amax(y,axis=1)
    sorted = np.argsort(ym)
    x = x[sorted]
    y = y[sorted]
    y = np.array([yi for yi in y if yi[0] < 1E99])
    x = np.array(x[:len(y)])
    return x,y

def objranks(objs):
    """
    Computes the rank of objects based on their sum of ranks across all columns.

    The function calculates the rank of each object in each column, then sums these
    ranks across the columns to produce a single rank value for each object.

    Args:
        objs (numpy.ndarray): A 2-dimensional numpy array where each row represents
            an object, and each column represents a category or feature.

    Returns:
        numpy.ndarray: A 1-dimensional numpy array containing the computed rank
            for each object based on the sum of its ranks across all columns.
    """
    ci = objs.argsort(axis=0)
    rank = np.empty_like(ci)
    ar = np.arange(objs.shape[0])
    for i in range(objs.shape[1]): 
        rank[ci[:,i], i] = ar 
    rank = np.sum(rank, axis=1)
    return rank

def ranks(cons, feasible, eps):
    """
    Computes the ranks of constraint values, adjusted by their feasibility and scaled by the
    fraction of violations for each constraint.

    Args:
        cons (numpy.ndarray): A 2D array of constraint values where rows represent different
            samples and columns represent different constraints.
        feasible (numpy.ndarray): A 1D boolean array indicating whether each row (sample) in
            `cons` is feasible (True) or not (False).
        eps (float): A scalar threshold value used to determine the violation of constraints.

    Returns:
        numpy.ndarray: A 1D array that contains the computed rank for each sample in `cons`.
    """
    ci = cons.argsort(axis=0)
    rank = np.empty_like(ci)
    ar = np.arange(cons.shape[0])
    for i in range(cons.shape[1]): 
        rank[ci[:,i], i] = ar
    rank[feasible] = 0
    alpha = np.sum(np.greater(cons, eps), axis=1) / cons.shape[1] # violations
    alpha = np.tile(alpha, (cons.shape[1],1)).T
    rank = rank*alpha
    rank = np.sum(rank, axis=1)
    return rank

def get_valid(xs, ys, nobj):
    """
    Filters the input data based on a validity condition and returns the valid elements.

    Args:
        xs: Array-like structure containing input data.
        ys: Array-like structure, typically associated with `xs`, used for the validity check.
        nobj: Integer specifying the number of columns to skip in `ys` for the validity check.

    Returns:
        Tuple containing:
            - The filtered elements of `xs` that satisfy the validity condition.
            - The filtered elements of `ys` that align with the valid `xs`.

    """
    valid = (ys.T[nobj:].T <= 0).all(axis=1)
    return xs[valid], ys[valid]

def pareto_sort(x0, y0, nobj, ncon):
    """
    Sorts solutions based on Pareto dominance and crowding distance.

    This function sorts a given set of solutions into levels based on Pareto
    dominance. Within each level, solutions are further sorted by crowding
    distance to ensure diversity. The input arrays `x0` and `y0` are reordered
    accordingly, and the resulting sorted arrays are returned.

    Args:
        x0: Array-like structure representing the original decision variables of
            the solutions.
        y0: Array-like structure representing the objectives of the solutions.
        nobj: Integer specifying the number of objectives for each solution.
        ncon: Integer specifying the number of constraints for each solution.

    Returns:
        Tuple of two NumPy arrays:
            - The reordered `x0` array based on Pareto dominance and crowding
              distance.
            - The reordered `y0` array based on Pareto dominance and crowding
              distance.
    """
    domination, _, _ = pareto_domination(y0, nobj, ncon)
    x = []
    y = []
    maxdom = int(max(domination))
    for dom in range(maxdom, -1, -1):
        domlevel = [p for p in range(len(domination)) if domination[p] == dom]
        if len(domlevel) == 0:
            continue
        nx = x0[domlevel]
        ny = y0[domlevel]    
        si = [0]
        if len(ny) > 1:                
            cd = crowd_dist(ny)
            si = np.flip(np.argsort(cd))
        for p in si:
            x.append(nx[p])
            y.append(ny[p])                             
    return np.array(x), np.array(y)

def pareto_domination(ys, nobj, ncon, last_ycon = None, last_eps = 0):
    """
    Determines Pareto domination levels for a set of solutions considering objectives
    and constraint violations. Divides solutions into feasible and infeasible sets,
    calculates their dominance levels based on objectives and constraints, and
    returns the relevant rankings.

    Args:
        ys (list or np.ndarray): A list or array of solutions where each solution
            contains concatenated values of objectives and constraints.
        nobj (int): Number of objectives in each solution.
        ncon (int): Number of constraints in each solution.
        last_ycon (np.ndarray, optional): Array representing constraint violations used
            in the previous iteration for adjusting tolerance to small violations.
            Defaults to None.
        last_eps (float, optional): Tolerance level for constraint violations in the
            last iteration. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of Pareto domination levels for all solutions.
            - np.ndarray: Array of adjusted constraint violations for each solution.
            - float: Adjusted tolerance for constraint violations.

    Raises:
        None
    """
    if ncon == 0:
        return pareto_levels(ys), None, 0
    else:
        eps = 0 # adjust tolerance to small constraint violations
        if not last_ycon is None and np.amax(last_ycon) < 1E90:
            eps = 0.5*(last_eps + 0.5*np.mean(last_ycon, axis=0))
            if np.amax(eps) < 1E-8: # ignore small eps
                eps = 0
        
        yobj = np.array([y[:nobj] for y in ys])
        ycon = np.array([np.maximum(y[-ncon:], 0) for y in ys])  
        popn = len(ys)              
        feasible = np.less_equal(ycon, eps).all(axis=1)
        
        csum = ranks(ycon, feasible, eps)
        if sum(feasible) > 0:
            csum += objranks(yobj)
        
        ci = np.argsort(csum)
        domination = np.zeros(popn)
        # first pareto front of feasible solutions
        cy = np.fromiter((i for i in ci if feasible[i]), dtype=int)
        if len(cy) > 0:
            ypar = pareto_levels(yobj[cy])
            domination[cy] = ypar        

        # then constraint violations   
        ci = np.fromiter((i for i in ci if not feasible[i]), dtype=int) 
        if len(ci) > 0:    
            cdom = np.arange(len(ci), 0, -1)
            domination[ci] += cdom
            if len(cy) > 0: # priorize feasible solutions
                domination[cy] += len(ci) + 1
                
        return domination, ycon, eps
 
def pareto_levels(ys):
    """
    Determines the Pareto levels of a given set of points.

    This function identifies the domination levels of a set of points, where the
    domination level of a point represents the number of other points dominating
    it. Domination is determined based on whether one point strictly dominates
    another in all dimensions.

    Args:
        ys (numpy.ndarray): A 2D array where each row represents a point in
            a multi-dimensional space, and domination is evaluated across
            dimensions.

    Returns:
        numpy.ndarray: A 1D array where each index corresponds to the domination
            level of the respective point in the input array.
    """
    popn = len(ys)
    pareto = np.arange(popn)
    index = 0  # Next index to search for
    domination = np.zeros(popn)
    while index < len(ys):
        mask = np.any(ys < ys[index], axis=1)
        mask[index] = True
        pareto = pareto[mask]  # Remove dominated points
        domination[pareto] += 1
        ys = ys[mask]
        index = np.sum(mask[:index])+1
    return domination

def crowd_dist(y): # crowd distance for 1st objective
    """
    Calculates the crowding distance for the first objective in a multi-objective optimization problem.

    This function determines for each solution how dense the neighborhood is based on the distances
    to its nearest neighbors in the sorted objective space. It assigns a numerical value to each
    solution reflecting this crowding distance.

    Args:
        y (List[List[float]]): A list of solutions, where each solution is represented as a list
            of objective values. This function specifically uses the first objective value for
            the crowding distance calculation.

    Returns:
        numpy.ndarray: An array of crowding distances for each solution in the input.
    """
    n = len(y)
    y0 = np.fromiter((yi[0] for yi in y), dtype=float)
    si = np.argsort(y0) # sort 1st objective
    y0_s = y0[si] # sorted
    d = y0_s[1:n] - y0_s[0:n-1] # neighbor distance
    if max(d) == 0:
        return np.zeros(n)
    dsum = np.zeros(n)
    dsum += np.array(list(d) + [0]) # distance to left
    dsum += np.array([0] + list(d)) # distance to right
    dsum[0] = 1E99 # keep borders
    dsum[-1] = 1E99
    ds = np.empty(n)
    ds[si] = dsum # inverse order
    return ds

# derived from https://github.com/ChengHust/NSGA-II/blob/master/GLOBAL.py
def variation(pop, lower, upper, rg, pro_c = 1, dis_c = 20, pro_m = 1, dis_m = 20):
    """
    Applies genetic variation operations, including simulated binary crossover (SBX) and polynomial
    mutation, on a population to produce offspring. The method modifies the population based on crossover
    and mutation probabilities, as well as distribution control parameters.

    Args:
        pop: numpy.ndarray
            Input population array, where each row represents an individual and each column represents
            a design variable.
        lower: numpy.ndarray
            Lower bounds for each design variable.
        upper: numpy.ndarray
            Upper bounds for each design variable.
        rg: numpy.random.Generator
            Random number generator instance.
        pro_c: float, optional
            Probability of performing crossover, default is 1.
        dis_c: float, optional
            Distribution index for crossover, default is 20.
        pro_m: float, optional
            Probability of performing mutation, default is 1.
        dis_m: float, optional
            Distribution index for mutation, default is 20.

    Returns:
        numpy.ndarray
            Modified population (offspring) after applying crossover and mutation operations.
    """
    dis_c *= 0.5 + 0.5*rg.random() # vary spread factors randomly 
    dis_m *= 0.5 + 0.5*rg.random() 
    pop = pop[:(len(pop) // 2) * 2][:]
    (n, d) = np.shape(pop)
    parent_1 = pop[:n // 2, :]
    parent_2 = pop[n // 2:, :]
    beta = np.zeros((n // 2, d))
    mu = rg.random((n // 2, d))
    beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (dis_c + 1))
    beta[mu > 0.5] = np.power(2 * mu[mu > 0.5], -1 / (dis_c + 1))
    beta = beta * ((-1)** rg.integers(2, size=(n // 2, d)))
    beta[rg.random((n // 2, d)) < 0.5] = 1
    if pro_c < 1.0:
        beta[np.tile(rg.random((n // 2, 1)) > pro_c, (1, d))] = 1
    parent_mean = (parent_1 + parent_2) * 0.5
    parent_diff = (parent_1 - parent_2) * 0.5
    offspring = np.vstack((parent_mean + beta * parent_diff, parent_mean - beta * parent_diff))
    site = rg.random((n, d)) < pro_m / d
    mu = rg.random((n, d))
    temp = site & (mu <= 0.5)
    lower, upper = np.tile(lower, (n, 1)), np.tile(upper, (n, 1))
    norm = (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp]) * \
                           (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(1. - norm, dis_m + 1.),
                                     1. / (dis_m + 1)) - 1.)
    temp = site & (mu > 0.5)
    norm = (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp]) * \
                           (1. - np.power(
                               2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(1. - norm, dis_m + 1.),
                               1. / (dis_m + 1.)))
    offspring = np.clip(offspring, lower, upper)
    return offspring

def feasible(xs, ys, ncon, eps = 1E-2):
    """
    Determines feasible solutions based on constraints and filters the input arrays
    `xs` and `ys` accordingly. It evaluates constraint violations and ensures that
    violations are below the threshold (`eps`) for feasibility.

    Args:
        xs (np.ndarray): Input array of solutions.
        ys (np.ndarray): Array of objective and constraint values corresponding
            to the solutions in `xs`.
        ncon (int): Number of constraints. Constraints are assumed to be the
            last `ncon` columns of the `ys` array.
        eps (float, optional): Feasibility threshold. Defaults to 1E-2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered solutions (`xs`) and their
        corresponding objective values (`ys`), with infeasible solutions removed.
    """
    if ncon > 0: # select feasible
        ycon = np.array([np.maximum(y[-ncon:], 0) for y in ys])  
        con = np.sum(ycon, axis=1)
        nobj = len(ys[0]) - ncon
        feasible = np.fromiter((i for i in range(len(ys)) if con[i] < eps), dtype=int)
        if len(feasible) > 0:
            xs, ys = xs[feasible], np.array([y[:nobj] for y in ys[feasible]])
        else:
            print("no feasible")
    return xs, ys

def is_feasible(y, nobj, eps = 1E-2):
    """
    Determines if a given solution is feasible based on constraints and tolerance.

    This function evaluates feasibility by checking whether the sum of violated
    constraints (if any) is less than the provided tolerance value. If there
    are no constraints, the solution is automatically considered feasible.

    Args:
        y: List or array-like containing values of both objectives and constraints.
        nobj: Integer specifying the number of objectives in the array 'y'. The
            rest are treated as constraints.
        eps: Float representing the tolerance level for determining feasibility.
            Defaults to 1E-2.

    Returns:
        bool: True if the solution is feasible (i.e., sum of violated constraints
        is below the specified tolerance), otherwise False.
    """
    ncon = len(y) - nobj
    if ncon == 0:
        return True
    else:
        c = np.sum(np.maximum(y[-ncon:], 0))
        return c < eps

class wrapper(object):
    """
    A wrapper class to manage function calls with additional features such as tracking progress,
    logging results, and storing or plotting outcomes.

    This class allows function evaluation with multiprocess support, tracks the number of
    evaluations, records the best observed outcomes, logs key metrics at specified intervals,
    and interfaces with optional storage or plotting mechanisms.

    Attributes:
        fun (Callable[[ArrayLike], ArrayLike]): The function to be wrapped and called.
        nobj (int): Number of objectives for the optimization or function evaluation.
        evals (mp.RawValue): A shared counter tracking the number of function evaluations.
        t0 (float): The time the wrapper instance was initialized, for logging purposes.
        best_y (mp.RawArray): Array to store the best observed outcomes for each objective.
        store (Optional[store]): Optional storage object to store function results.
        interval (Optional[int]): Specifies how often to log or perform certain actions,
            in terms of number of function evaluations. Default is 100000.
        plot (Optional[bool]): If True, enables plotting after storing results. Default is False.
        name (Optional[str]): A name used for file save operations when the store is provided.
        lock (mp.Lock): A multiprocessing lock to manage access to shared resources safely.
    """
   
    def __init__(self, 
                 fun: Callable[[ArrayLike], ArrayLike], 
                 nobj: int, 
                 store: Optional[store] = None, 
                 interval: Optional[int] = 100000, 
                 plot: Optional[bool] = False, 
                 name: Optional[str] = None):
        """
        Initializes an object for handling and tracking function evaluations, storing results,
        and optionally plotting performance data.

        Args:
            fun (Callable[[ArrayLike], ArrayLike]): The objective function to evaluate.
            nobj (int): The number of objectives to optimize.
            store (Optional[store]): The storage mechanism for results. Defaults to None.
            interval (Optional[int]): The interval at which logs or updates occur.
                Defaults to 100000.
            plot (Optional[bool]): Whether to enable plotting of performance data.
                Defaults to False.
            name (Optional[str]): A custom name for the object. Defaults to None.
        """
        self.fun = fun
        self.nobj = nobj
        self.evals = mp.RawValue(ct.c_long, 0)
        self.t0 = time.perf_counter()
        self.best_y = mp.RawArray(ct.c_double, nobj)  
        for i in range(nobj):
            self.best_y[i] = sys.float_info.max
        self.store = store
        self.interval = interval
        self.plot = plot
        self.name = name
        self.lock = mp.Lock()
    
    def __call__(self, x: ArrayLike) -> np.ndarray:
        """
        Executes the callable object with the input data, applies the function to evaluate the data,
        updates internal state related to evaluations and results, and performs logging, saving,
        and plotting based on specific conditions.

        Args:
            x (ArrayLike): Input data to evaluate using the callable object's function.

        Returns:
            np.ndarray: Evaluated output from the function if successful, otherwise None.
        """
        try:
            y = self.fun(x)
            with self.lock:
                self.evals.value += 1
            if not self.store is None and is_feasible(y, self.nobj):
                self.store.create_views()
                self.store.add_result(x, y[:self.nobj])
            improve = False
            for i in range(self.nobj):
                if y[i] < self.best_y[i]:
                    improve = True 
                    self.best_y[i] = y[i] 
            #improve = improve# and self.evals.value > 10000
            if self.evals.value % self.interval == 0 or improve:
                constr = np.maximum(y[self.nobj:], 0)
                logger.info( 
                    f'{dtime(self.t0)} {self.evals.value} {self.evals.value/(1E-9 + dtime(self.t0)):.1f} {self.best_y[:]} {list(constr)} {list(x)}'      
                )
                if (not self.store is None) and (not self.name is None):
                    try:
                        xs, ys = self.store.get_front()
                        num = self.store.num_stored.value
                        name = self.name + '_' + str(num)
                        np.savez_compressed(name, xs=xs, ys=ys)
                        if self.plot:
                            moretry.plot(name, 0, xs, ys, all=False)
                    except Exception as ex:
                        print(str(ex))                                                
            return y
        except Exception as ex:
            print(str(ex))  
            return None  
 
def minimize_plot(name: str, 
                 fun: Callable[[ArrayLike], ArrayLike], 
                 nobj: int,
                 ncon: int, 
                 bounds: Bounds,
                 popsize: Optional[int] = 64, 
                 max_evaluations: Optional[int] = 100000, 
                 nsga_update: Optional[bool] = True,
                 pareto_update: Optional[int] = 0,
                 ints: Optional[ArrayLike] = None,
                 workers: Optional[int] = mp.cpu_count()) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function minimizes a multi-objective function using evolutionary algorithms
    and generates a plot showing the optimization results. The optimization data is
    also saved to a compressed .npz file.

    Args:
        name (str): Name identifier for the optimization process. It is used
            to name the resulting data and plots.
        fun (Callable[[ArrayLike], ArrayLike]): The objective function to be
            minimized. It should take an array-like input and return an array-like
            output of the objectives.
        nobj (int): The number of objectives to be optimized.
        ncon (int): The number of constraints for the optimization problem.
        bounds (Bounds): Bounds object specifying the variable lower and
            upper bounds for the problem.
        popsize (Optional[int]): The population size for the evolutionary
            algorithm. Defaults to 64.
        max_evaluations (Optional[int]): The maximum number of function evaluations
            allowed. Defaults to 100000.
        nsga_update (Optional[bool]): A boolean flag indicating whether to use
            NSGA-II-like updates for the optimization process. Defaults to True.
        pareto_update (Optional[int]): An integer parameter specifying the update
            strategy for Pareto optimization when nsga_update is False. Defaults to 0.
        ints (Optional[ArrayLike]): An array-like object indicating which decision
            variables are to be treated as integers. Defaults to None.
        workers (Optional[int]): Number of worker processes to use for
            parallelization. Defaults to the number of available CPU cores.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            A tuple containing:
            - xs: A NumPy array representing the decision variable values of the
              solutions.
            - ys: A NumPy array representing the objective values of the solutions.
    """
    name += '_mode_' + str(popsize) + '_' + \
                ('nsga_update' if nsga_update else ('de_update_' + str(pareto_update)))
    logger.info('optimize ' + name) 
    xs, ys = minimize(fun, nobj, ncon, bounds, popsize = popsize, max_evaluations = max_evaluations,
                   nsga_update = nsga_update, pareto_update = pareto_update, workers=workers, ints=ints)
    np.savez_compressed(name, xs=xs, ys=ys)
    moretry.plot(name, ncon, xs, ys)
    return xs, ys
