# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - diversifier.py

 Description:
  - Numpy based implementation of an diversifying wrapper / parallel retry mechanism.

    Uses the archive from CVT MAP-Elites [2] to maintain a set of diverse solutions
    and generalizes ideas from CMA-ME [3]
    to other wrapped algorithms.

    Both the parallel retry and the archive based modification of the fitness
    function enhance the diversification of the optimization result.
    The resulting archive may be stored and can be used to continue the
    optimization later.

    Requires a QD-fitness function returning both an fitness value and a
    behavior vector used to determine the corresponding archive niche using
    Voronoi tesselation.

    Returns an archive of niche-elites containing also for each niche statistics
    about the associated solutions.


 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] https://arxiv.org/abs/1610.05729
  - [3] https://arxiv.org/pdf/1912.02400.pdf

 Documentation:
  -


=============================================================================
"""
from __future__ import annotations


import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence
from multiprocessing import Process
from scipy.optimize import Bounds
from fcmaes.optimizer import dtime, de_cma, Optimizer
import multiprocessing as mp
import ctypes as ct
from time import perf_counter
from fcmaes.mapelites import Archive, update_archive, rng
from fcmaes import advretry
from fcmaes.evaluator import is_debug_active
from loguru import logger
import threadpoolctl

from typing import Optional, Callable, Tuple, Dict
from numpy.typing import ArrayLike

def minimize(qd_fitness: Callable[[ArrayLike], Tuple[float, np.ndarray]], 
            bounds: Bounds,
            qd_bounds: Bounds,
            niche_num: Optional[int] = 10000,
            samples_per_niche: Optional[int] = 20,
            max_evals: Optional[int] = None,
            workers: Optional[int] = mp.cpu_count(),
            archive: Optional[Archive] = None,
            opt_params: Optional[Dict] = {},
            use_stats: Optional[bool] = False,            
            ) -> Archive:

    """
    Performs parallel optimization to minimize the fitness function with a focus on
    quality-diversity. The function initializes an archive that stores optimal
    inputs and their fitness values, organizes the records into defined niches,
    and carries out optimization using parallel processing.

    Args:
        qd_fitness: Callable fitness function that takes an input array and returns
            a tuple containing a scalar objective value and an array of descriptor
            values.
        bounds: Bounds object defining the lower and upper bounds of the search
            space.
        qd_bounds: Bounds object defining the lower and upper bounds for the
            quality-diversity archive.
        niche_num: Optional; The number of niches to partition the archive.
            Default is 10000.
        samples_per_niche: Optional; The number of samples to initialize in each
            niche. Default is 20.
        max_evals: Optional; The maximum number of function evaluations. If not
            provided, the default is workers * 50000.
        workers: Optional; The number of parallel processes to use. Default is the
            number of CPU cores available.
        archive: Optional; An existing archive object to initialize the optimization
            process. If not provided, a new archive is created.
        opt_params: Optional; A dictionary of parameters used by the optimization
            function. Default is an empty dictionary.
        use_stats: Optional; A flag to determine whether to track statistical
            properties within the archive. Default is False.

    Returns:
        Archive: An archive object containing the optimized results, including
            fitness values and descriptors.
    """

    if max_evals is None:
        max_evals = workers*50000
    dim = len(bounds.lb)
    if archive is None: 
        archive = Archive(dim, qd_bounds, niche_num, use_stats)
        archive.init_niches(samples_per_niche)
        # initialize archive with random values
        archive.xs_view[:] = rng.uniform(bounds.lb, bounds.ub, (niche_num, dim))       
    t0 = perf_counter()   
    qd_fitness.archive = archive # attach archive for logging     
    minimize_parallel_(archive, qd_fitness, bounds, workers, opt_params, max_evals)
    if is_debug_active():
        ys = np.sort(archive.get_ys())[:min(100, archive.capacity)] # best fitness values
        logger.debug(f'best {min(ys):.3f} worst {max(ys):.3f} ' + 
                 f'mean {np.mean(ys):.3f} stdev {np.std(ys):.3f} time {dtime(t0)} s')
    return archive

def apply_advretry(fitness: Callable[[ArrayLike], float], 
                   qd_fitness: Callable[[ArrayLike], Tuple[float, np.ndarray]], 
                   bounds: Bounds, 
                   archive: Archive, 
                   optimizer: Optional[Optimizer] = None, 
                   num_retries: Optional[int] = 1000, 
                   workers: Optional[int] = mp.cpu_count(),
                   max_eval_fac: Optional[float] = 5.0,
                   xs: Optional[np.ndarray] = None,
                   ys: Optional[np.ndarray] = None,
                   x_conv: Callable[[ArrayLike], ArrayLike] = None):

    """
    Applies an advanced retry mechanism to optimize solutions for a given fitness function
    and update an archive with optimized results.

    This function manages an iterative process where previously computed solutions from
    an archive are refined and optimized using a specified optimizer. Advanced retry logic
    is utilized to ensure the process effectively minimizes the given fitness function
    within the defined bounds and constraints.

    Args:
        fitness: Callable that evaluates the fitness of a solution. Must return a
            floating-point fitness score for a given input.
        qd_fitness: Callable that evaluates the quality-diversity of a solution. Returns
            a tuple where the first item is the fitness value (float) and the second item is
            a feature descriptor (e.g., NumPy array).
        bounds: Boundary constraints for the optimization process.
        archive: Archive object that stores and manages solutions.
        optimizer: Optional optimizer to be used for the minimization process. If none,
            defaults to a DE-CMA optimizer with 1500 iterations.
        num_retries: Optional integer setting the number of retries allowed in
            advanced retry logic. Defaults to 1000.
        workers: Optional integer specifying the number of workers for parallelization.
            Defaults to the number of CPUs available.
        max_eval_fac: Optional floating-point factor setting the maximum allowed
            fitness function evaluations per retry. Defaults to 5.0.
        xs: Optional NumPy array containing a set of previously obtained solutions.
            If none, solutions are derived from the archive's current entries.
        ys: Optional NumPy array containing fitness values corresponding to `xs`.
            If none, fitness values are derived from the archive's entries.
        x_conv: Optional callable to transform or convert solutions (`xs`) before
            evaluating their quality-diversity fitness. If none, no transformation is applied.
    """

    if optimizer is None:
        optimizer = de_cma(1500)
    # generate advretry store
    store = advretry.Store(fitness, bounds, num_retries=num_retries, 
                           max_eval_fac=max_eval_fac)  
                         
    # select only occupied entries
    if xs is None:
        ys = archive.get_ys()    
        valid = (ys < np.inf)
        ys = ys[valid]
        xs = archive.xs_view[valid]
    t0 = perf_counter() 
    # transfer to advretry store
    for i in range(len(ys)):
        store.add_result(ys[i], xs[i], 1)
    # perform parallel retry
    advretry.retry(store, optimizer.minimize, workers=workers)
    # transfer back to archive 
    xs = store.xs_view
    if not x_conv is None:
        xs = [x_conv(x) for x in xs]
    yds = [qd_fitness(x) for x in xs]
    descs = np.array([yd[1] for yd in yds])
    ys = np.array([yd[0] for yd in yds])
    niches = archive.index_of_niches(descs)
    for i in range(len(ys)):
        archive.set(niches[i], (ys[i], descs[i]), xs[i])
    archive.argsort()
    if is_debug_active():
        ys = np.sort(archive.get_ys())[:min(100, archive.capacity)] # best fitness values
        logger.debug(f'best {min(ys):.3f} worst {max(ys):.3f} ' + 
                 f'mean {np.mean(ys):.3f} stdev {np.std(ys):.3f} time {dtime(t0)} s')    

def minimize_parallel_(archive, fitness, bounds, workers, opt_params, max_evals):
    """
    Minimizes a fitness function in parallel using multiple workers.

    This function utilizes multiprocessing to divide the workload of minimizing
    a fitness function across several worker processes. Each worker operates
    independently using its own random number generator, and the results are
    combined to find the minimum value.

    Args:
        archive: Object or structure used to store and manage state or intermediate
            results during the optimization process. Details of its structure or
            behavior depend on the implementation.
        fitness: Callable that evaluates the fitness or cost function to be
            minimized. Accepts input variables and returns a numerical value
            representing the cost/fitness.
        bounds: Defines the boundaries within which the optimization process is
            allowed to search. Typically, this can be a list of tuples specifying
            the lower and upper bounds for each dimension.
        workers: Integer specifying the number of parallel workers (processes) to
            execute the fitness function evaluation and optimization tasks.
        opt_params: Parameters or configuration values required for the optimization
            method. Includes details specific to the optimization algorithm being
            employed.
        max_evals: Integer representing the maximum number of fitness function
            evaluations allowed across all worker processes in total.

    """
    sg = SeedSequence()
    rgs = [Generator(PCG64DXSM(s)) for s in sg.spawn(workers)]
    evals = mp.RawValue(ct.c_long, 0)
    proc=[Process(target=run_minimize_,
            args=(archive, fitness, bounds, rgs[p],
                  opt_params, p, workers, evals, max_evals)) for p in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
                    
def run_minimize_(archive, fitness, bounds, rg, opt_params, p, workers, evals, max_evals):
    """
    Executes the optimization processes involving MAP-Elites or solvers in a sequential or
    multiple configuration depending on the provided optimization parameters. The method adapts
    to the given `opt_params`, evaluating various solvers until the maximum number of allowed
    evaluations (`max_evals`) is reached.

    Args:
        archive: Archive object used for storing solutions and managing population niches.
        fitness: A callable fitness function used to evaluate solutions.
        bounds: Boundary constraints associated with the problem.
        rg: Random generator for deterministic random processes.
        opt_params: Optimization parameters, which can be in the form of a dictionary, list,
            tuple, or NumPy array.
        p: Integer indicating the number of processing resources available.
        workers: Integer specifying the number of worker threads or processes allocated for
            solver execution.
        evals: A shared counter object tracking the current total number of evaluations performed.
        max_evals: Maximum allowed number of evaluations over the optimization processes.

    """
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        if isinstance(opt_params, (list, tuple, np.ndarray)):
            default_workers = int(workers/2) if len(opt_params) > 1 else workers
            for params in opt_params: # call MAP-Elites
                if 'elites' == params.get('solver'):
                    elites_workers = params.get('workers', default_workers) 
                    if p < elites_workers:
                        run_map_elites_(archive, fitness, bounds, rg, evals, max_evals, params)
                        return
        while evals.value < max_evals: # call solvers in loop     
            best_x = None
            if isinstance(opt_params, (list, tuple, np.ndarray)):
                for params in opt_params: # call in sequence
                    if 'elites' == params.get('solver'):
                        continue # ignore in loop
                    if best_x is None:
                        # selecting a niche elite is no improvement over random x0
                        x0 = None#, _, _ = archive.random_xs_one(select_n, rg)
                        best_x = minimize_(archive, fitness, bounds, rg, evals, max_evals, params, 
                                           x0 = x0)
                    else:
                        best_x = minimize_(archive, fitness, bounds, rg, evals, max_evals, params, x0 = best_x)
            else:        
                minimize_(archive, fitness, bounds, rg, evals, max_evals, opt_params) 

from fcmaes.mapelites import variation_,  iso_dd_
                
def run_map_elites_(archive, fitness, bounds, rg, evals, max_evals, opt_params = {}):
    """
    Executes the MAP-Elites algorithm for a given archive and fitness function.

    The function iteratively generates a population of individuals, applies variation
    operators on them, evaluates their fitness, and updates the archive with the newly
    generated individuals. It can utilize either simulated binary crossover (SBX) or isolation
    distribution crossover (ISO/DD) depending on the specified optimization parameters.
    Additionally, the function includes boundaries for the search space and adjusts the
    archive to maintain its capacity.

    Args:
        archive (object): The data structure representing the archive of solutions.
        fitness (callable): The fitness function to evaluate each solution.
        bounds (object): The search space bounds containing lower (`lb`) and upper (`ub`)
            limits for variables.
        rg (numpy.random.Generator): Random number generator for stochastic operations.
        evals (object): Object to track the number of evaluations performed.
        max_evals (int): Maximum number of evaluations allowed for the algorithm.
        opt_params (dict, optional): A dictionary of optional parameters for optimization.
            Includes:
            - popsize (int): Population size for the algorithm (default: 32).
            - use_sbx (bool): Whether to use simulated binary crossover (default: True).
            - dis_c (float): Distribution index for simulated binary crossover (SBX) (default: 20).
            - dis_m (float): Distribution index for mutation (default: 20).
            - iso_sigma (float): Standard deviation for isotropic distribution (default: 0.01).
            - line_sigma (float): Standard deviation for line distribution (default: 0.2).
    """
    popsize = opt_params.get('popsize', 32)  
    use_sbx = opt_params.get('use_sbx', True)     
    dis_c = opt_params.get('dis_c', 20)   
    dis_m = opt_params.get('dis_m', 20)  
    iso_sigma = opt_params.get('iso_sigma', 0.01)
    line_sigma = opt_params.get('line_sigma', 0.2)
    select_n = archive.capacity
    while evals.value < max_evals:              
        if use_sbx:
            pop = archive.random_xs(select_n, popsize, rg)
            xs = variation_(pop, bounds.lb, bounds.ub, rg, dis_c, dis_m)
        else:
            x1 = archive.random_xs(select_n, popsize, rg)
            x2 = archive.random_xs(select_n, popsize, rg)
            xs = iso_dd_(x1, x2, bounds.lb, bounds.ub, rg, iso_sigma, line_sigma)    
        yds = [fitness(x) for x in xs]
        evals.value += popsize
        descs = np.array([yd[1] for yd in yds])
        niches = archive.index_of_niches(descs)
        for i in range(len(yds)):
            archive.set(niches[i], yds[i], xs[i])
        archive.argsort()   
        select_n = archive.get_occupied()  

def minimize_(archive, fitness, bounds, rg, evals, max_evals, opt_params, x0 = None):
    """
    Minimizes a given objective function using an evolutionary algorithm or the B.I.T.E. solver.

    The function determines the solver type based on the provided optimization parameters
    and runs the optimization process accordingly. It continuously updates an archive
    of candidate solutions and evaluates their fitness while adhering to the specified
    evaluation and iteration limits. A stopping condition is also applied based on lack
    of improvement.

    Args:
        archive: Archive object for storing solution candidates and their respective
            fitness evaluations.
        fitness: Callable representing the fitness function or objective
            function to be minimized.
        bounds: Bounds or constraints for the solution search space.
        rg: Random number generator to ensure reproducibility.
        evals: A mutable object, typically an integer, tracking the number of
            evaluations performed.
        max_evals: Integer specifying the maximum number of fitness evaluations to allow.
        opt_params: Dictionary containing optimization parameters, such as solver type
            and stopping criteria.
        x0: Optional initial guess or starting point for the optimization process.

    Returns:
        The best found solution, represented as a real-valued array,
        that optimizes the provided fitness function.
    """
    if 'BITE_CPP' == opt_params.get('solver'):
        return run_bite_(archive, fitness, bounds, rg, evals, max_evals, opt_params, x0 = None)
    else:
        es = get_solver_(bounds, opt_params, rg, x0) 
        stall_criterion = opt_params.get('stall_criterion', 20)
        max_evals_iter = opt_params.get('max_evals', 50000)
        max_iters = int(max_evals_iter/es.popsize)
        old_ys = None
        last_improve = 0
        best_x = None
        best_y = np.inf
        for iter in range(max_iters):
            xs = es.ask()
            ys, real_ys = update_archive(archive, xs, fitness)
            evals.value += es.popsize
            # update best real fitness
            yi = np.argmin(real_ys)
            ybest = real_ys[yi] 
            if ybest < best_y:
                best_y = ybest
                best_x = xs[yi]
            if not old_ys is None:
                if (np.sort(ys) < old_ys).any():
                    last_improve = iter          
            if last_improve + stall_criterion < iter:
                break
            stop = es.tell(ys)
            if stop != 0 or evals.value >= max_evals:
                break 
            old_ys = np.sort(ys)
        return best_x # real best solution

from fcmaes import cmaes, cmaescpp, crfmnescpp, pgpecpp, decpp, crfmnes, de, bitecpp

def run_bite_(archive, fitness, bounds, rg, evals, max_evals, opt_params, x0 = None):
    """
    Runs the BiteOpt algorithm to optimize a given fitness function.

    The function utilizes the BiteOpt implementation from bitecpp
    to minimize the provided fitness function over the given bounds and constraints.
    It supports dynamic updates to the solution archive and stops execution
    based on a defined maximum evaluation limit or optimization parameters.

    Args:
        archive: An archive to keep track of the solution space explored during the optimization process.
        fitness: A callable that takes an input, evaluates it, and returns a fitness value.
        bounds: The variable bounds for the optimization problem.
        rg: A random generator instance to ensure reproducibility in the optimization process.
        evals: An object containing a mutable integer used to track the number of evaluations performed.
        max_evals: An integer defining the maximum number of evaluations before the optimization halts.
        opt_params: A dictionary containing optimization parameters such as 'max_evals' and 'stall_criterion'.
        x0: Optional starting point for the optimization. Defaults to None.

    Returns:
        The optimized solution vector obtained from BiteOpt.
    """
    # BiteOpt doesn't support ask/tell, so we have to "patch" fitness. Note that Voronoi 
    # tesselation is more expensive if called for single behavior vectors and not for batches. 
    
    def fit(x: Callable[[ArrayLike], float]):
        """
        Evaluates a given function with constraints on a maximum number of evaluations.

        This function checks if the given number of evaluations exceeds the permitted
        maximum before proceeding. If the threshold is not surpassed, it updates the
        archive with the computed fitness values and returns the fitness of the evaluated
        input.

        Args:
            x: A callable function that takes an ArrayLike input and returns a float
                representing the fitness value.

        Returns:
            float: The computed fitness value of the input `x`. If the maximum allowed
            evaluations are reached, returns infinity.

        Raises:
            ValueError: If the input function (or its return value) does not align with
            the expected structure or type definitions during processing.
        """
        if evals.value >= max_evals:
            return np.inf
        evals.value += 1
        ys, _ = update_archive(archive, [x], fitness)
        return ys[0]
    
    max_evals_iter = opt_params.get('max_evals', 50000)       
    stall_criterion = opt_params.get('stall_criterion', 20)   
    #popsize = opt_params.get('popsize', 0) 
    ret = bitecpp.minimize(fit, bounds, x0 = x0, M = 1, 
                           stall_criterion = stall_criterion,
                           max_evaluations = max_evals_iter, rg = rg)
    return ret.x   

def get_solver_(bounds, opt_params, rg, x0 = None):
    """
    Selects and initializes the appropriate optimization solver based on the specified
    parameters. The solver is chosen from a set of predefined options, and it is configured
    with the given dimensions, bounds, mean, population size, and other solver-specific
    parameters.

    Args:
        bounds: Object representing the bounds for the optimization problem. It provides
            attributes like lower bounds (`lb`) and upper bounds (`ub`).
        opt_params: Dictionary containing optional solver parameters, such as:
            - 'popsize': Population size
            - 'sigma': Step size
            - 'mean': Initial mean position
            - 'solver': Name of the solver to use, e.g., 'CMA', 'CMA_CPP', etc.
        rg: Random number generator for initializing values within the specified bounds
            or for stochastic components of the solver.
        x0: Optional starting position for the optimization. Overrides the `mean` parameter
            if provided.

    Returns:
        The initialized optimization solver object if the specified solver name is valid.
        Returns `None` if an invalid solver name is provided.
    """
    dim = len(bounds.lb)
    popsize = opt_params.get('popsize', 31) 
    #sigma = opt_params.get('sigma',rg.uniform(0.03, 0.3)**2)
    sigma = opt_params.get('sigma',rg.uniform(0.1, 0.5)**2)
    #sigma = opt_params.get('sigma',rg.uniform(0.2, 0.5)**2)
    #sigma = opt_params.get('sigma',rg.uniform(0.1, 0.5))
    mean = opt_params.get('mean', rg.uniform(bounds.lb, bounds.ub)) \
                if x0 is None else x0
    name = opt_params.get('solver', 'CMA_CPP')
    if name == 'CMA':
        return cmaes.Cmaes(bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'CMA_CPP':
        return cmaescpp.ACMA_C(dim, bounds, x0 = mean, #stop_hist = 0,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'CRMFNES':
        return crfmnes.CRFMNES(dim, bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'CRMFNES_CPP':
        return crfmnescpp.CRFMNES_C(dim, bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'DE':
        return de.DE(dim, bounds, popsize = popsize, rg = rg)
    elif name == 'DE_CPP':
        return decpp.DE_C(dim, bounds, popsize = popsize, rg = rg)
    elif name == 'PGPE':
        return pgpecpp.PGPE_C(dim, bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    else:
        print ("invalid solver")
        return None
            
