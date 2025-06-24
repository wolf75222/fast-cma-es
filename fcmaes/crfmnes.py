# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - crfmnes.py

 Description:
  - Numpy based implementation of Fast Moving Natural Evolution Strategy
  for High-Dimensional Problems (CR-FM-NES), see [2].
  - Derived from [1] and [3].


 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] https://arxiv.org/abs/2201.11422
  - [3] https://github.com/nomuramasahir0/crfmnes

 Documentation:
  -


=============================================================================
"""
import math
import numpy as np
import os
from scipy.optimize import OptimizeResult, Bounds
from numpy.random import PCG64DXSM, Generator
from fcmaes.evaluator import _get_bounds, _fitness, serial, parallel

from typing import Optional, Callable, Union, Dict
from numpy.typing import ArrayLike



# evaluation value of the infeasible solution
INFEASIBLE = np.inf

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float],
             bounds: Optional[Bounds] = None,
             x0: Optional[ArrayLike] = None,
             input_sigma: Optional[float] = 0.3,
             popsize: Optional[int] = 32,
             max_evaluations: Optional[int] = 100000,
             workers: Optional[int] = None,
             stop_fitness: Optional[float] = -np.inf,
             is_terminate: Optional[Callable[[ArrayLike, float], bool]] = None,
             rg: Optional[Generator] = Generator(PCG64DXSM()),
             runid: Optional[int] = 0,
             normalize: Optional[bool] = False,
             options: Optional[Dict] = {}
             ) -> OptimizeResult:
    """
    Minimizes an objective function using the CR-FM-NES algorithm. This method is suitable for
    optimization problems in continuous domains where solutions are represented by real-valued vectors.
    It supports various features like boundary constraints, parallel evaluations, fitness normalization,
    and custom termination conditions.

    Args:
        fun (Callable[[ArrayLike], float]): The objective function to be minimized. It should accept
            a single input representing a candidate solution and return a scalar fitness value.
        bounds (Optional[Bounds]): The boundary constraints for the optimization problem. It defines
            the feasible region for the search.
        x0 (Optional[ArrayLike]): An optional initial guess for the optimization. If not provided,
            the algorithm will randomly initialize a starting point.
        input_sigma (Optional[float]): The initial standard deviation for the search distribution.
            Defaults to 0.3.
        popsize (Optional[int]): The population size for the evolutionary algorithm. Defaults to 32.
        max_evaluations (Optional[int]): The maximum number of allowable function evaluations
            before termination. Defaults to 100000.
        workers (Optional[int]): The number of parallel workers to use for fitness evaluation. If not
            specified, defaults to sequential execution.
        stop_fitness (Optional[float]): The target fitness value that will terminate the algorithm
            if reached. Defaults to negative infinity.
        is_terminate (Optional[Callable[[ArrayLike, float], bool]]): A custom termination function.
            If provided, it should return a boolean indicating whether optimization should stop.
        rg (Optional[Generator]): A random number generator for reproducibility. Defaults to
            `Generator(PCG64DXSM())`.
        runid (Optional[int]): An identifier for the optimization run, for tracking purposes. Default
            is 0.
        normalize (Optional[bool]): Flag indicating whether to perform fitness normalization. Defaults
            to False.
        options (Optional[Dict]): A dictionary containing additional configuration parameters for
            the optimizer.

    Returns:
        OptimizeResult: A data structure containing the results of the optimization. Attributes include
        the optimal solution found, its fitness value, the number of function evaluations and iterations,
        as well as termination status and success flag.
    """

    cr = CRFMNES(None, bounds, x0, input_sigma, popsize, 
                 max_evaluations, stop_fitness, is_terminate, runid, normalize, options, rg, workers, fun)
    
    cr.optimize()

    return OptimizeResult(x=cr.f.decode(cr.x_best), fun=cr.f_best, nfev=cr.no_of_evals, 
                          nit=cr.g, status=cr.stop, 
                          success=True)

class CRFMNES:
    """
    CRFMNES is an implementation of the Covariance Rank-based Fast Mutation Evolution Strategy
    (CR-FM-NES), an optimization algorithm for solving problems with or without constraints.

    This class is designed for evolutionary optimization processes, providing features for handling
    constraints, managing population rankings, and utilizing various evolutionary strategies for
    adaptive parameter tuning. It internally manages individual solution vectors, constraints,
    population size, and custom fitness evaluation functions.

    Attributes:
        dim (int): Number of dimensions in the optimization problem.
        sigma (float): Mutation strength for adaptive parameter control.
        popsize (int): Number of individuals in the population.
        max_evaluations (int): Maximum number of function evaluations allowed before termination.
        stop_fitness (float): Target value for the optimization. Process terminates if reached.
        is_terminate (bool or None): Flags whether the optimization should terminate prematurely.
        rg (Generator): Random number generator instance for reproducibility and randomness.
        runid (int): Unique identifier for distinguishing optimization runs.
        constraint (List[List[float]]): Bounds for variables. Used for constraint handling.
        v (np.ndarray): Evolution vector for controlling the search directions.
        D (np.ndarray): Scaling factor for mutation parameters.
        penalty_coef (float): Coefficient for penalizing constraint violation.
        use_constraint_violation (bool): Indicates if constraint violations should influence optimization.
        w_rank_hat (np.ndarray): Rank-based weights before normalization.
        w_rank (np.ndarray): Normalized rank weights used for recombination.
        mueff (float): Variance effective selection mass for mutation and update dynamics.
        cs (float): Cumulation step-size adaptation parameter.
        cc (float): Cumulation rank-one strategy parameter.
        c1_cma (float): Rank-one covariance matrix update learning rate.
        chiN (float): Expected length for normal distribution under CMA-ES geometry.
        pc (np.ndarray): Evolution path for cumulative rank-one strategy updates.
        ps (np.ndarray): Evolution path for cumulative step-size adaptation.
        h_inv (float): Inverse of an internal constant, utilized for certain computations.
        eta_m (float): Learning rate for model vector updates.
        eta_move_sigma (float): Learning rate for standard mutation updates.
        g (int): Current generation or iteration count of the optimization process.
        no_of_evals (int): Total number of function evaluations performed so far.
        iteration (int): Internal tracking of algorithm iterations.
        stop (int): Indicates the status of termination (-1: error, 0: not stopped, >0: stopped).
        idxp (np.ndarray): Indices for positive directions during symmetry computation.
        idxm (np.ndarray): Indices for negative directions during symmetry computation.
        z (np.ndarray): Mutant vectors prior to rescaling, used in evolution processes.
        f_best (float): Best fitness value observed in the optimization process.
        x_best (np.ndarray): Solution vector corresponding to the best observed fitness.
    """
    def __init__(self, 
                dim = None, 
                bounds: Optional[Bounds] = None, 
                x0: Optional[ArrayLike] = None,
                input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3, 
                popsize: Optional[int] = 32,  
                max_evaluations: Optional[int] = 100000, 
                stop_fitness: Optional[float] = -np.inf, 
                is_terminate: Optional[bool] = None, 
                runid: Optional[int] = 0, 
                normalize: Optional[bool] = False,
                options: Optional[Dict] = {}, 
                rg: Optional[Generator] = Generator(PCG64DXSM()), 
                workers: Optional[int] = None, 
                fun: Optional[Callable[[ArrayLike], float]] = lambda x: 0):
        """
        Initializes the class with parameters for optimization. This method allows for the configuration of various
        optimization parameters, including the dimensional bounds, population size, sigma value, stopping criteria,
        and many others. It ensures consistency across parameters, initializes the fitness function, sets up constraints,
        and prepares internal attributes for optimization. The class is tailored for scenarios where optimization must
        be conducted within specified bounds using a population-based approach.

        Args:
            dim (Optional[int]): Dimensionality of the problem. If not specified, it is inferred from bounds or x0.
            bounds (Optional[Bounds]): Bounds for the optimization problem. Each dimension can have distinct bounds.
            x0 (Optional[ArrayLike]): Initial guess for optimization. If not provided, derived using bounds and dimensions.
            input_sigma (Optional[Union[float, ArrayLike, Callable]]): Sigma parameter that controls the search spread.
                Default is 0.3.
            popsize (Optional[int]): Population size. Should be an even number. If odd, it will be automatically adjusted.
            max_evaluations (Optional[int]): Maximum evaluations allowed for the optimization. Default is 100,000.
            stop_fitness (Optional[float]): Target fitness threshold for stopping optimization. Default is -infinity.
            is_terminate (Optional[bool]): Whether termination is allowed based on certain conditions. Default is None.
            runid (Optional[int]): Run ID to differentiate optimization runs. Default is 0.
            normalize (Optional[bool]): Whether to normalize inputs. Default is False.
            options (Optional[Dict]): Additional options for customization, such as constraints or penalties.
            rg (Optional[Generator]): Random number generator for initializing population. Default uses PCG64DXSM.
            workers (Optional[int]): Number of workers for parallel evaluations. Default is None, meaning serial execution.
            fun (Optional[Callable[[ArrayLike], float]]): Optimization function to be evaluated. Default is a zero function.
        """
        if popsize is None:
            popsize = 32         
        if popsize % 2 == 1: # requires even popsize
            popsize += 1
        if dim is None:
            if not x0 is None: dim = len(x0)
            else: 
                if not bounds is None: dim = len(bounds.lb)
        lower, upper, guess = _get_bounds(dim, bounds, x0, rg) 
        self.fun = serial(fun) if (workers is None or workers <= 1) else parallel(fun, workers)  
        self.f = _fitness(self.fun, lower, upper, normalize)       
        if options is None:
            options = {}
        if not lower is None:
            options['constraint'] = [ [lower[i], upper[i]] for i in range(dim)]   
        self.constraint = options.get('constraint', [[-np.inf, np.inf] for _ in range(dim)])
        if 'seed' in options.keys():
            np.random.seed(options['seed'])
        sigma = input_sigma
        if not np.isscalar(sigma):
            sigma = np.mean(sigma)         
        self.m = np.array([self.f.encode(guess)]).T

        self.dim = dim
        self.sigma = sigma
        self.popsize = popsize
              
        self.max_evaluations = max_evaluations
        self.stop_fitness = stop_fitness
        self.is_terminate = is_terminate
        self.rg = rg
        self.runid = runid

        self.v = options.get('v', self.rg.normal(0,1,(dim, 1)) / np.sqrt(dim))
        
        self.D = np.ones([dim, 1])
        self.penalty_coef = options.get('penalty_coef', 1e5)
        self.use_constraint_violation = options.get('use_constraint_violation', True)

        self.w_rank_hat = (np.log(self.popsize / 2 + 1) - np.log(np.arange(1, self.popsize + 1))).reshape(self.popsize, 1)
        self.w_rank_hat[np.where(self.w_rank_hat < 0)] = 0
        self.w_rank = self.w_rank_hat / sum(self.w_rank_hat) - (1. / self.popsize)
        self.mueff = 1 / ((self.w_rank + (1 / self.popsize)).T @ (self.w_rank + (1 / self.popsize)))[0][0]
        self.cs = (self.mueff + 2.) / (self.dim + self.mueff + 5.)
        self.cc = (4. + self.mueff / self.dim) / (self.dim + 4. + 2. * self.mueff / self.dim)
        self.c1_cma = 2. / (math.pow(self.dim + 1.3, 2) + self.mueff)
        # initialization
        self.chiN = np.sqrt(self.dim) * (1. - 1. / (4. * self.dim) + 1. / (21. * self.dim * self.dim))
        self.pc = np.zeros([self.dim, 1])
        self.ps = np.zeros([self.dim, 1])
        # distance weight parameter
        self.h_inv = get_h_inv(self.dim)
        self.alpha_dist = lambda lambF: self.h_inv * min(1., math.sqrt(self.popsize / self.dim)) * math.sqrt(
            lambF / self.popsize)
        self.w_dist_hat = lambda z, lambF: exp(self.alpha_dist(lambF) * np.linalg.norm(z))
        # learning rate
        self.eta_m = 1.0
        self.eta_move_sigma = 1.
        self.eta_stag_sigma = lambda lambF: math.tanh((0.024 * lambF + 0.7 * self.dim + 20.) / (self.dim + 12.))
        self.eta_conv_sigma = lambda lambF: 2. * math.tanh((0.025 * lambF + 0.75 * self.dim + 10.) / (self.dim + 4.))
        self.c1 = lambda lambF: self.c1_cma * (self.dim - 5) / 6 * (lambF / self.popsize)
        self.eta_B = lambda lambF: np.tanh((min(0.02 * lambF, 3 * np.log(self.dim)) + 5) / (0.23 * self.dim + 25))

        self.g = 0
        self.no_of_evals = 0
        self.iteration = 0
        self.stop = 0

        self.idxp = np.arange(self.popsize / 2, dtype=int)
        self.idxm = np.arange(self.popsize / 2, self.popsize, dtype=int)
        self.z = np.zeros([self.dim, self.popsize])

        self.f_best = float('inf')
        self.x_best = np.empty(self.dim)

    def __del__(self):
        """
        Handles the cleanup process upon object deletion.

        This method ensures proper resource management by stopping the execution
        of the `parallel` object, if applicable, when the instance is about to
        be destroyed.

        Raises:
            AttributeError: If `self.fun` does not have a `stop` method and is an instance of `parallel`.
        """
        if isinstance(self.fun, parallel):
            self.fun.stop()
        
    def calc_violations(self, x):
        """
        Calculates the constraint violations for a population.

        This function evaluates how much each solution in the population violates the given
        constraints. Violations are penalized by a predefined penalty coefficient.

        Args:
            x: numpy.ndarray. A 2D array representing the population. Each column represents
                a single solution, and each row represents a variable in the solution.

        Returns:
            numpy.ndarray: A 1D array containing the total calculated violations for each
                solution in the population.
        """
        violations = np.zeros(self.popsize)
        for i in range(self.popsize):
            for j in range(self.dim):
                violations[i] += (- min(0, x[j][i] - self.constraint[j][0]) + max(0, x[j][i] - self.constraint[j][1])) * self.penalty_coef
        return violations

    def optimize(self) -> int:
        """
        Executes an optimization process by iteratively evaluating, decoding, and improving
        solutions until termination criteria are met.

        This method utilizes a generation loop to perform the optimization. It stops either
        when a defined evaluation limit is reached, or due to an external stop signal,
        or when execution encounters an unrecoverable exception.

        Raises:
            Exception: If an error occurs during the execution of the optimization process.

        Returns:
            int: The status of the optimization process upon termination. A value of -1
            indicates an error during execution; other stop conditions may yield different
            results.
        """
        # -------------------- Generation Loop --------------------------------
        while True:
            if self.no_of_evals > self.max_evaluations:
                break
            if self.stop != 0:
                break
            try:
                x = self.ask()
                y = self.f.values(self.f.decode(self.f.closestFeasible(x)))
                self.tell(y)
                if self.stop != 0:
                    break 
            except Exception as ex:
                self.stop = -1
                break

    def ask(self) -> np.ndarray:
        """
        Generate and return the next set of candidate solution vectors for optimization.
        This method computes a population of new solution vectors and applies certain
        transformations based on the algorithm's state variables.

        Returns:
            np.ndarray: A 2D array where each row represents a candidate solution vector.
        """
        d = self.dim
        popsize = self.popsize
        zhalf = self.rg.normal(0,1,(d, int(popsize / 2)))  # dim x popsize/2
        self.z[:, self.idxp] = zhalf
        self.z[:, self.idxm] = -zhalf
        self.normv = np.linalg.norm(self.v)
        self.normv2 = self.normv ** 2
        self.vbar = self.v / self.normv
        self.y = self.z + ((np.sqrt(1 + self.normv2) - 1) * (self.vbar @ (self.vbar.T @ self.z)))
        self.x = self.m + (self.sigma * self.y) * self.D
        return self.x.T

    def tell(self, evals_no_sort: np.ndarray) -> int:
        """
        Provides the functionality to update and optimize based on the given evaluations
        with sorting, constraints handling, and adaptive parameters for step size and
        evolution paths.

        Args:
            evals_no_sort (np.ndarray): The evaluation scores for the current population
                before sorting.

        Returns:
            int: The stop condition value, indicating the termination status of the
                optimization process.
        """
        violations = np.zeros(self.popsize)
        if self.use_constraint_violation:
            violations = self.calc_violations(self.x)
            sorted_indices = sort_indices_by(evals_no_sort + violations, self.z)
        else:
            sorted_indices = sort_indices_by(evals_no_sort, self.z)
        best_eval_id = sorted_indices[0]
        f_best = evals_no_sort[best_eval_id]
        x_best = self.x[:, best_eval_id]
        self.z = self.z[:, sorted_indices]
        y = self.y[:, sorted_indices]
        x = self.x[:, sorted_indices]

        self.no_of_evals += self.popsize
        self.g += 1
 
        if f_best < self.f_best:
            self.f_best = f_best
            self.x_best = x_best           
            # print(self.no_of_evals, self.g, self.f_best)

        # This operation assumes that if the solution is infeasible, infinity comes in as input.
        lambF = np.sum(evals_no_sort < np.finfo(float).max)

        # evolution path p_sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2. - self.cs) * self.mueff) * (self.z @ self.w_rank)
        ps_norm = np.linalg.norm(self.ps)
        # distance weight
        f1 =  self.h_inv * min(1., math.sqrt(self.popsize / self.dim)) * math.sqrt(lambF / self.popsize)        
        w_tmp = self.w_rank_hat * np.exp(np.linalg.norm(self.z, axis = 0) * f1).reshape((self.popsize,1))
        weights_dist = w_tmp / sum(w_tmp) - 1. / self.popsize
        # switching weights and learning rate
        weights = weights_dist if ps_norm >= self.chiN else self.w_rank
        eta_sigma = self.eta_move_sigma if ps_norm >= self.chiN else self.eta_stag_sigma(
            lambF) if ps_norm >= 0.1 * self.chiN else self.eta_conv_sigma(lambF)
        # update pc, m
        wxm = (x - self.m) @ weights
        self.pc = (1. - self.cc) * self.pc + np.sqrt(self.cc * (2. - self.cc) * self.mueff) * wxm / self.sigma
        self.m += self.eta_m * wxm
        # calculate s, t
        # step1
        normv4 = self.normv2 ** 2
        exY = np.append(y, self.pc / self.D, axis=1)  # dim x popsize+1
        yy = exY * exY  # dim x popsize+1
        ip_yvbar = self.vbar.T @ exY
        yvbar = exY * self.vbar  # dim x popsize+1. exYのそれぞれの列にvbarがかかる
        gammav = 1. + self.normv2
        vbarbar = self.vbar * self.vbar
        alphavd = min(1, np.sqrt(normv4 + (2 * gammav - np.sqrt(gammav)) / np.max(vbarbar)) / (2 + self.normv2))  # scalar
        
        t = exY * ip_yvbar - self.vbar * (ip_yvbar ** 2 + gammav) / 2  # dim x popsize+1
        b = -(1 - alphavd ** 2) * normv4 / gammav + 2 * alphavd ** 2
        H = np.ones([self.dim, 1]) * 2 - (b + 2 * alphavd ** 2) * vbarbar  # dim x 1
        invH = H ** (-1)
        s_step1 = yy - self.normv2 / gammav * (yvbar * ip_yvbar) - np.ones([self.dim, self.popsize + 1])  # dim x popsize+1
        ip_vbart = self.vbar.T @ t  # 1 x popsize+1
 
        s_step2 = s_step1 - alphavd / gammav * ((2 + self.normv2) * (t * self.vbar) - self.normv2 * vbarbar @ ip_vbart)  # dim x popsize+1
        invHvbarbar = invH * vbarbar
        ip_s_step2invHvbarbar = invHvbarbar.T @ s_step2  # 1 x popsize+1
        
        div = 1 + b * vbarbar.T @ invHvbarbar
        if np.amin(abs(div)) == 0:
            return -1
        
        s = (s_step2 * invH) - b / div * invHvbarbar @ ip_s_step2invHvbarbar  # dim x popsize+1
        ip_svbarbar = vbarbar.T @ s  # 1 x popsize+1
        t = t - alphavd * ((2 + self.normv2) * (s * self.vbar) - self.vbar @ ip_svbarbar)  # dim x popsize+1
        # update v, D
        exw = np.append(self.eta_B(lambF) * weights, np.array([self.c1(lambF)]).reshape(1, 1),
                        axis=0)  # popsize+1 x 1
        self.v = self.v + (t @ exw) / self.normv
        self.D = self.D + (s @ exw) * self.D
        # calculate detA
        if np.amin(self.D) < 0:
            return -1

        nthrootdetA = exp(np.sum(np.log(self.D)) / self.dim + np.log(1 + (self.v.T @ self.v)[0][0]) / (2 * self.dim))
         
        self.D = self.D / nthrootdetA
        
        # update sigma
        G_s = np.sum((self.z * self.z - np.ones([self.dim, self.popsize])) @ weights) / self.dim
        self.sigma = self.sigma * exp(eta_sigma / 2 * G_s)
        return self.stop

    def population(self) -> np.ndarray:
        """
        Retrieves the current population.

        Returns the current state of the population as a NumPy array.

        Returns:
            np.ndarray: The current population as a NumPy array.
        """
        return self.x

    def result(self) -> OptimizeResult:
        """
        Returns the optimization result as an instance of `OptimizeResult`.

        The function compiles the results of the optimization process into an
        `OptimizeResult` object, providing the best solution found, its associated
        function value, the number of function evaluations performed, the number of iterations,
        the stopping status, and the success flag.

        Returns:
            OptimizeResult: The result of the optimization containing the following fields:
                - x: The best solution found (self.x_best).
                - fun: The value of the function at the best solution (self.f_best).
                - nfev: The total number of function evaluations (self.no_of_evals).
                - nit: The number of iterations performed (self.g).
                - status: The stopping status of the optimization (self.stop).
                - success: A boolean indicating whether the optimization was successful.
        """
        return OptimizeResult(x=self.x_best, fun=self.f_best, nfev=self.no_of_evals,
                              nit=self.g, status=self.stop, success=True)
        
def exp(a):
    """
    Calculates the exponential of a given number, with a cap to avoid overflow.

    The function computes the exponential value using the `math.exp` function. It ensures
    that the input does not exceed 100 to prevent overflow.

    Args:
        a: A float or integer representing the input value for which to calculate the exponential.

    Returns:
        The exponential value of the input `a`, respecting a maximum cap of 100 for the input.
    """
    return math.exp(min(100, a)) # avoid overflow

def get_h_inv(dim):
    """
    Computes the inverse of a specific mathematical function using the Newton-Raphson
    method.

    The function uses an iterative method to compute the value of h_inv such that
    a given mathematical equation is satisfied. The iteration stops when the
    function's value is sufficiently close to zero, adhering to a tolerance of `1e-10`.

    Args:
        dim (float): A dimension parameter that affects the behavior of the function.

    Returns:
        float: The calculated value of h_inv that satisfies the given equation.
    """
    f = lambda a, b: ((1. + a * a) * exp(a * a / 2.) / 0.24) - 10. - dim
    f_prime = lambda a: (1. / 0.24) * a * exp(a * a / 2.) * (3. + a * a)
    h_inv = 1.0
    while abs(f(h_inv, dim)) > 1e-10:
        h_inv = h_inv - 0.5 * (f(h_inv, dim) / f_prime(h_inv))
    return h_inv

def sort_indices_by(evals, z):
    """
    Sorts indices based on evaluated values, `evals`, while prioritizing feasible
    over infeasible solutions. Infeasible solutions are further sorted by their
    Euclidean distance.

    Args:
        evals: List or array of values (e.g., objective function evaluations)
            with INFEASIBLE denoting infeasibility.
        z: 2D array-like object, typically representing decision variables or
            characteristics of solutions.

    Returns:
        numpy.ndarray: Array of indices sorted based on feasibility and
                       additional criteria for infeasible solutions.
    """
    lam = len(evals)
    evals = np.array(evals)
    sorted_indices = np.argsort(evals)
    sorted_evals = evals[sorted_indices]
    no_of_feasible_solutions = np.where(sorted_evals != INFEASIBLE)[0].size
    if no_of_feasible_solutions != lam:
        infeasible_z = z[:, np.where(evals == INFEASIBLE)[0]]
        distances = np.sum(infeasible_z ** 2, axis=0)
        infeasible_indices = sorted_indices[no_of_feasible_solutions:]
        indices_sorted_by_distance = np.argsort(distances)
        sorted_indices[no_of_feasible_solutions:] = infeasible_indices[indices_sorted_by_distance]
    return sorted_indices
