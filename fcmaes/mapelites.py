# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - mapelites.py

 Description:
  - Numpy based implementation of CVT MAP-Elites including CMA-ES emitter and CMA-ES drilldown. 

    See [2] and [3]
    
    MAP-Elites implementations differ in the following details:
    
    1) Initialisation of the behavior space:
    
    a) Generated from some solution distribution by applying the fitness function to determine their behavior.
    b) Generated from uniform samples of the behavior space. 
    
    We use b) because random solutions may cover only parts of the behavior space. Some parts may only be reachable 
    by optimization. Another reason: Fitness computations may be expensive. Therefore we don't compute fitness
    values for the initial solution population.     
    
    2) Initialization of the niches: 
    
    a) Generated from some solution distribution.
    b) Generated from uniform samples of the solution space. These solutions are never evaluated but serve as
    initial population for SBX or Iso+LineDD. Their associated fitness value is set to math.inf (infinity).
    
    We use b) because this way we: 
    - Avoid computing fitness values for the initial population.
    - Enhance the diversity of initial solutions emitted by SBX or Iso+LineDD.
    
    3) Iso+LineDD [4] is implemented but doesn't work well with extremely ragged solution
    landscapes. Therefore SBX+mutation is the default setting.
    
    4) SBX (Simulated binary crossover) is taken from mode.py and simplified. It is combined with mutation.
    Both spread factors - for crossover and mutation - are randomized for each application. 
    
    5) Candidates for CMA-ES are sampled with a bias to better niches. As for SBX only a subset of the archive is
    used, the worst niches are ignored. 
    
    6) There is a CMA-ES drill down for specific niches - in this mode all solutions outside the niche
    are rejected. Restricted solution box bounds are used derived from statistics maintained by the archive
    during the addition of new solution candidates. 
    
    7) The QD-archive uses shared memory to reduce inter-process communication overhead.


 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] https://arxiv.org/abs/1610.05729
  - [3] https://arxiv.org/pdf/1912.02400.pdf
  - [4] https://arxiv.org/pdf/1804.03906

 Documentation:
  -


=============================================================================
"""
from __future__ import annotations


import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence
from multiprocessing import Process
import multiprocessing as mp
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from scipy.optimize import Bounds
from pathlib import Path
from fcmaes.retry import Shared2d
from fcmaes.optimizer import dtime
from fcmaes import cmaescpp
from numpy.random import default_rng
import ctypes as ct
from time import perf_counter
import threadpoolctl
from numba import njit
from fcmaes.evaluator import is_debug_active
from loguru import logger

from typing import Optional, Callable, Tuple, Dict
from numpy.typing import ArrayLike

rng = default_rng()

def optimize_map_elites(qd_fitness: Callable[[ArrayLike], Tuple[float, np.ndarray]], 
                        bounds: Bounds, 
                        qd_bounds: Bounds, 
                        niche_num: Optional[int] = 4000, 
                        samples_per_niche: Optional[int] = 20, 
                        workers: Optional[int] = mp.cpu_count(), 
                        iterations: Optional[int] = 100, 
                        archive: Optional[Archive] = None, 
                        me_params: Optional[Dict] = {}, 
                        cma_params: Optional[Dict] = {}, 
                        use_stats: Optional[bool] = False,
                        ) -> Archive:

    """
    Optimizes a map-elites evolutionary algorithm to find diverse and high-performing solutions 
    by distributing samples into niches, iteratively evolving them, and maintaining an archive 
    to store the found solutions.

    This function leverages a map-elites approach to optimize solutions across a given feature 
    space or behavioral space defined by the `qd_bounds` parameter. The objective of the optimization 
    is to populate an archive with solutions that exhibit high fitness while ensuring diversity 
    in behavior across the space. The archive is continuously updated and sorted during the iterations 
    to reflect the best solutions in each niche. The performance is monitored throughout the 
    optimization process.

    Args:
        qd_fitness (Callable[[ArrayLike], Tuple[float, np.ndarray]]): The quantitative fitness 
            function used to evaluate the solutions. It must return a tuple containing the fitness 
            score and the corresponding feature descriptor/behavior representation.
        bounds (Bounds): Represents the variable bounds for the optimization space.
        qd_bounds (Bounds): Defines the range of the quality-diversity feature/behavioral space.
        niche_num (Optional[int]): The total number of niches in the archive. Determines how 
            the solutions will be distributed across the feature space. Defaults to 4000.
        samples_per_niche (Optional[int]): The number of samples generated per niche during 
            initialization. Defaults to 20.
        workers (Optional[int]): Number of CPU workers/threads to parallelize the process. 
            Defaults to the number of CPUs available.
        iterations (Optional[int]): The total number of iterations to evolve the archive. 
            Defaults to 100.
        archive (Optional[Archive]): Pre-existing archive object for storing solutions. If `None`, 
            a new archive is initialized. Defaults to None.
        me_params (Optional[Dict]): Additional hyperparameters specific to the map-elites algorithm. 
            Defaults to an empty dictionary.
        cma_params (Optional[Dict]): Hyperparameters specific to the CMA-ES algorithm used 
            for evolutionary strategies. Defaults to an empty dictionary.
        use_stats (Optional[bool]): Flag to determine whether statistical data should be 
            collected and maintained in the archive. Defaults to False.

    Returns:
        Archive: The final archive containing diverse high-performing solutions distributed 
            across the quality-diversity space.
    """

    dim = len(bounds.lb) 
    if archive is None: 
        archive = Archive(dim, qd_bounds, niche_num, use_stats)
        archive.init_niches(samples_per_niche)
        # initialize archive with random values
        self.xs.view()[:] = rng.uniform(bounds.lb, bounds.ub, (niche_num, dim))
    t0 = perf_counter() 
    qd_fitness.archive = archive # attach archive for logging  
    for iter in range(iterations):
        archive.argsort() # sort archive to select the best_n
        optimize_map_elites_(archive, qd_fitness, bounds, workers,
                    me_params, cma_params)
        if is_debug_active():
            ys = np.sort(archive.get_ys())[:100] # best 100 fitness values
            logger.debug(f'best 100 iter {iter} best {min(ys):.3f} worst {max(ys):.3f} ' + 
                     f'mean {np.mean(ys):.3f} stdev {np.std(ys):.3f} time {dtime(t0)} s')
    return archive

def empty_archive(dim: int, 
                  qd_bounds: Bounds, 
                  niche_num: int, 
                  samples_per_niche: int, 
                  use_stats: Optional[bool] = False) -> Archive:

    """
    Creates and initializes an archive for quality diversity (QD) experiments.

    This function creates an Archive object with the specified dimensions,
    bounding box, number of niches, and optionally enables statistics collection.
    It also initializes the niches in the archive with a specified number 
    of samples per niche.

    Args:
        dim (int): Dimensionality of the search/grid space for the archive.
        qd_bounds (Bounds): Bounds specifying the range for niche space coordinates.
        niche_num (int): Number of niches to be created in the archive.
        samples_per_niche (int): Number of samples to initialize in each niche.
        use_stats (Optional[bool], optional): If True, statistics collection will be
            enabled in the archive. Defaults to False.

    Returns:
        Archive: The initialized archive object for QD experiments.

    """

    archive = Archive(dim, qd_bounds, niche_num, use_stats)
    archive.init_niches(samples_per_niche)
    return archive

def set_KDTree(archive: Archive,
                        centers:Optional[np.ndarray] = None, 
                        niche_num: Optional[int]  = None, 
                        qd_bounds: Optional[Bounds] = None, 
                        samples_per_niche: Optional[int] = 100):

    """
    Sets up a KDTree for the given archive using specified or default parameters.

    This function initializes a KDTree structure with the given or generated 
    centers, enabling efficient spatial searches within the archive. If the centers 
    are not provided, they will be generated using the `get_centers_` function 
    based on the number of niches, dimensionality of the bounds, and the number 
    of samples per niche. This KDTree is essential for spatial partitioning 
    and querying in the archive.

    Args:
        archive (Archive): The archive object for which the KDTree will be set up.
        centers (Optional[np.ndarray]): Coordinates of the centers to construct 
            the KDTree. If `None`, they will be generated automatically.
        niche_num (Optional[int]): The number of niches used to determine the 
            grid of centers if `centers` is not provided.
        qd_bounds (Optional[Bounds]): The bounds within which the niches lie.
        samples_per_niche (Optional[int]): The number of sample centers per niche 
            to generate when `centers` is not provided. Defaults to 100.
    """

    if centers is None: # cache centers 
        centers = get_centers_(niche_num, len(qd_bounds.lb), samples_per_niche)
    archive.kdt = KDTree(centers, leaf_size=30, metric='euclidean')  
    archive.centers = centers         

def load_archive(name: str, 
                 bounds: Bounds, 
                 qd_bounds: Bounds, 
                 niche_num: Optional[int] = 10000,
                 use_stats: Optional[bool] = False, 
                 ) -> Archive:

    """
    Loads an archive based on the provided parameters.

    This function creates and configures an Archive instance using the specified
    parameters. It initializes the Archive with bounds, QD bounds, niche number,
    and optional usage of statistics. After initialization, it loads the archive
    data from the provided file name.

    Args:
        name (str): The file name from which to load the archive.
        bounds (Bounds): The lower and upper bounds for the archive's dimensionality.
        qd_bounds (Bounds): The bounds specific to the QD optimization process.
        niche_num (Optional[int]): The number of niches in the archive. Defaults to 10000.
        use_stats (Optional[bool]): Flag to indicate whether statistics should be used.
            Defaults to False.

    Returns:
        Archive: An Archive instance loaded with the specified configurations and data.
    """
     
    dim = len(bounds.lb)
    archive = Archive(dim, qd_bounds, niche_num, name, use_stats)
    archive.load(name)
    return archive

def optimize_map_elites_(archive, fitness, bounds, workers,
                         me_params, cma_params):
    """
    Optimize a solution using Map-Elites algorithm via multi-processing.

    This function orchestrates the execution of the Map-Elites optimization
    algorithm over multiple processes. Each process operates independently with its
    own random generator to explore the solution space. It uses CMA-ES to update
    solutions iteratively. The provided parameters configure the optimization
    behavior, such as the solution archive, fitness evaluation function, search
    bounds, and process settings specific to Map-Elites and CMA-ES.

    Args:
        archive: A data structure to store the solutions found during the 
            optimization process and their corresponding evaluation. 
        fitness: A callable function that evaluates the quality of a solution. 
        bounds: A sequence of tuples defining the lower and upper bounds for
            each decision variable.
        workers: An integer specifying the number of parallel processes
            to run during optimization.
        me_params: A dictionary or configuration object with parameters
            for the Map-Elites algorithm.
        cma_params: A dictionary or configuration object with parameters
            for the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    """
    sg = SeedSequence()
    rgs = [Generator(PCG64DXSM(s)) for s in sg.spawn(workers)]
    proc=[Process(target=run_map_elites_,
            args=(archive, fitness, bounds, rgs[p],
                  me_params, cma_params)) for p in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
          
def run_map_elites_(archive, fitness, bounds, rg, 
                    me_params, cma_params):
    """
    Runs the MAP-Elites algorithm with optional SBX (Simulated Binary Crossover) or 
    iso+line variation, followed by CMA-ES optimization. The algorithm iteratively 
    updates the archive with new solutions, balancing exploration and exploitation 
    over the specified number of generations.

    Args:
        archive: An archive object for storing solutions, descriptors, and their 
            fitness values, supporting niche identification and updating mechanisms.
        fitness: Callable that computes the fitness and descriptor of a given solution.
        bounds: Bounds object holding the lower and upper bounds for the solutions 
            during variation and optimization.
        rg: A random generator instance used for stochastic operations within the 
            algorithm.
        me_params: Dictionary containing tunable MAP-Elites parameters with keys 
            such as 'generations', 'chunk_size', 'use_sbx', 'dis_c', 'dis_m', 
            'iso_sigma', and 'line_sigma'.
        cma_params: Dictionary containing CMA-ES-specific parameters such as 
            'cma_generations'.

    """
    generations = me_params.get('generations', 10) 
    chunk_size = me_params.get('chunk_size', 20)   
    use_sbx = me_params.get('use_sbx', True)     
    dis_c = me_params.get('dis_c', 20)   
    dis_m = me_params.get('dis_m', 20)  
    iso_sigma = me_params.get('iso_sigma', 0.02)
    line_sigma = me_params.get('line_sigma', 0.2)
    cma_generations = cma_params.get('cma_generations', 20)
    select_n = archive.capacity
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        for _ in range(generations):                
            if use_sbx:
                pop = archive.random_xs(select_n, chunk_size, rg)
                xs = variation_(pop, bounds.lb, bounds.ub, rg, dis_c, dis_m)
            else:
                x1 = archive.random_xs(select_n, chunk_size, rg)
                x2 = archive.random_xs(select_n, chunk_size, rg)
                xs = iso_dd_(x1, x2, bounds.lb, bounds.ub, rg, iso_sigma, line_sigma)    
            yds = [fitness(x) for x in xs]
            descs = np.array([yd[1] for yd in yds])
            niches = archive.index_of_niches(descs)
            for i in range(len(yds)):
                archive.set(niches[i], yds[i], xs[i]) 
            archive.argsort()   
            select_n = archive.get_occupied()            
    
        for _ in range(cma_generations):                
            optimize_cma_(archive, fitness, bounds, rg, cma_params)    

def optimize_cma_(archive, fitness, bounds, rg, cma_params):
    """
    Optimizes a solution using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    The function utilizes the process of evolutionary optimization based on objective fitness,
    adaptive step size, and population-based approach. It leverages a provided archive to 
    introduce variations in the space of potential solutions.

    Args:
        archive: An instance of a solution archive storing previously evaluated solutions and
            assisting in generating new candidate solutions.
        fitness: A callable fitness function to evaluate the quality of solutions.
        bounds: The bounds or constraints of the solution space to limit candidate solutions.
        rg: A random generator instance used to introduce stochasticity in optimization steps.
        cma_params: A dictionary containing configuration parameters for the CMA-ES optimizer,
            such as population size, sigma (step size), maximum iterations, and other hyperparameters.

    Raises:
        ValueError: If parameters in cma_params are invalid (e.g., negative population size).
    """
    select_n = cma_params.get('best_n', 100)
    x0, y, iter = archive.random_xs_one(select_n, rg)
    sigma = cma_params.get('sigma',rg.uniform(0.03, 0.3)**2)
    popsize = cma_params.get('popsize', 31) 
    es = cmaescpp.ACMA_C(archive.dim, bounds, x0 = x0,  
                         popsize = popsize, input_sigma = sigma, rg = rg)
    maxiters = cma_params.get('maxiters', 100)
    stall_criterion = cma_params.get('stall_criterion', 5)
    old_ys = None
    last_improve = 0
    for iter in range(maxiters):
        xs = es.ask()
        improvement, ys = update_archive(archive, xs, fitness)
        if iter > 0:
            if (np.sort(ys) < old_ys).any():
                last_improve = iter          
        if last_improve + stall_criterion < iter:
            # no improvement
            break
        if es.tell(improvement) != 0:
            break 
        old_ys = np.sort(ys)

@np.errstate(invalid='ignore')        
def update_archive(archive: Archive, xs: np.ndarray, 
                   fitness: Optional[Callable[[ArrayLike], Tuple[float, np.ndarray]]] = None,
                   yds: Optional[ArrayLike] = None):
    """
    Evaluates a population, updates the archive with new solutions, and determines rankings.

    This function processes a set of solutions and their associated descriptions or fitness
    evaluations. The solutions are compared against an archive to determine and store better
    solutions. It prioritizes updates to empty niches in the archive and normalizes improvements
    for comparisons. The function finally returns the improvement values and the real fitness
    values for the solutions.

    Args:
        archive (Archive): The archive object where solutions and corresponding fitness data are stored.
        xs (np.ndarray): A set of solutions to be evaluated and compared against the archive.
        fitness (Optional[Callable[[ArrayLike], Tuple[float, np.ndarray]]]): 
            A callable function that evaluates a solution and returns its fitness value and description.
        yds (Optional[ArrayLike]): Precomputed fitness values and descriptions, if available.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Improvement values compared to archive elite solutions.
            - The real fitness values evaluated for the given solutions.

    Raises:
        ValueError: If `ys` or `yds` contain invalid or mismatched content.
    """
    # evaluate population, update archive and determine ranking
    popsize = len(xs)
    if yds is None: 
        yds = [fitness(x) for x in xs]
    descs = np.array([yd[1] for yd in yds])
    niches = archive.index_of_niches(descs)
    # real values
    ys = np.fromiter((yd[0] for yd in yds), dtype=float)
    oldys = np.fromiter((archive.get_y(niches[i]) for i in range(popsize)), dtype=float)
    improvement = ys - oldys
    neg = np.argwhere(improvement < 0)
    if len(neg) > 0:
        neg = neg.reshape((len(neg)))
        # update archive for all real improvements
        for i in neg:
            archive.set(niches[i], yds[i], xs[i])
        # prioritize empty niches
        empty = (improvement == -np.inf) # these need to be sorted according to fitness
        occupied = np.logical_not(empty)
        min_valid = np.amin(improvement[occupied]) if sum(occupied) > 0 else 0
        norm_ys = ys[empty] - np.amax(ys) - 1E-9
        improvement[empty] = min_valid + norm_ys
    # return both improvement compared to archive elites  and real fitness
    return improvement, ys

@njit()
def get_grid_indices(ds, capacity, lb, ub):
    """
    Computes and returns grid indices for a dataset based on the provided grid
    capacity, lower, and upper bounds. Each point in the dataset is normalized
    and then converted into a grid index using the specified parameters.

    Args:
        ds (np.ndarray): Input dataset (2D numpy array) where each row is a point
            in a multi-dimensional space.
        capacity (int): Total number of grid cells (capacity of the grid).
        lb (np.ndarray): Lower bounds of the dataset in each dimension.
        ub (np.ndarray): Upper bounds of the dataset in each dimension.

    Returns:
        np.ndarray: An array of grid indices, where each index corresponds to a
        point in the input dataset.

    """
    rdim = int(capacity ** (1/ds.shape[1]) + 0.5)
    ds_norm = (ds - lb) / (ub - lb)
    indices = np.empty(len(ds), dtype=np.int32)   
    for i, d in enumerate(ds_norm):
        index = 0
        f = 1
        for di in d:
            index += f * int(rdim*di)
            f *= rdim
        indices[i] = max(0, min(capacity-1, int(index)))    
    return indices
        
class Archive(object):
    """Handles the storage, organization, and management of solutions in a 
    multi-dimensional archive for Quality-Diversity optimization.

    The Archive class allows for storing solutions with associated features 
    and fitness values, clustering them into niches, and updating the archive 
    based on improvements to the corresponding niches. The class also 
    provides methods to retrieve statistical data on stored solutions as 
    well as mechanisms for saving/loading the archive state.

    Attributes:
        dim (int): Dimensionality of the solution vectors.
        qd_dim (int): Dimensionality of the quality-diversity descriptors.
        qd_bounds (Bounds): Bounds for the quality-diversity descriptors.
        desc_lb (ndarray): Lower bounds for the descriptors.
        desc_scale (ndarray): Scale of the descriptor bounds, calculated as 
            the difference between upper and lower bounds.
        capacity (int): Maximum capacity of the archive in terms of number 
            of solutions that can be stored.
        name (Optional[str]): Optional name for the archive.
        use_stats (bool): Flag to determine if solution statistics such as 
            mean, standard deviation, min, max, etc., should be collected.
    """
       
    def __init__(self, 
                 dim: int,
                 qd_bounds: Bounds,
                 capacity: int,    
                 name: Optional[str] = "",
                 use_stats = False             
                ):
        """
        Initializes an instance of the class with specified parameters.

        Args:
            dim (int): Dimensionality of the data to be handled.
            qd_bounds (Bounds): An object representing the lower and upper bounds 
                for the QD space.
            capacity (int): Maximum capacity or size to store elements.
            name (Optional[str]): Optional string name identifier for the instance. 
                Defaults to an empty string.
            use_stats (bool): Indicates whether to use statistics in computations or 
                operations. Defaults to False.
        """
        self.dim = dim
        self.qd_dim = len(qd_bounds.lb)
        self.qd_bounds = Bounds(np.array(qd_bounds.lb), np.array(qd_bounds.ub))
        self.desc_lb = self.qd_bounds.lb
        self.desc_scale = self.qd_bounds.ub - self.qd_bounds.lb
        self.capacity = capacity
        self.name = name
        self.cs = None
        self.lock = mp.Lock()
        self.use_stats = use_stats
        self.reset()
    
    def reset(self):
        """
        Resets the internal state of the object to its initial configuration by reinitializing its shared
        data structures and clearing any stored values. This allows the object to reuse storage and 
        resources effectively.

        Attributes:
            xs (Shared2d): A 2D shared array initialized with an empty array of shape `(capacity, dim)`
                and data type `np.float64`. Represents the primary 2D data storage.
            ds (Shared2d): A 2D shared array initialized with an empty array of shape `(capacity, qd_dim)`
                and data type `np.float64`. Represents supplementary 2D data storage.
            ys (mp.RawArray): A 1D shared array initialized to hold float values, representing 
                additional numeric data.
            counts (mp.RawArray): A 1D shared array initialized to hold integer long values, used 
                to maintain counts for each capacity slot.
            occupied (mp.RawValue): A shared value for tracking the count of currently occupied 
                slots in the storage.
            stats (mp.RawArray): An optional shared array for maintaining statistical records. 
                Its size depends on the combination of `capacity`, `dim`, and the `use_stats` flag.
                When `use_stats` is False, it is initialized with size `0`.

        Raises:
            ValueError: Raised if any invalid or unexpected conditions arise during the reset operation,
                such as capacity mismatch, invalid dimensions, or unavailable memory for shared arrays.
        """
        self.xs = Shared2d(np.empty((self.capacity, self.dim), dtype = np.float64))
        self.ds = Shared2d(np.empty((self.capacity, self.qd_dim), dtype = np.float64))
        self.create_views()
        self.ys = mp.RawArray(ct.c_double, self.capacity)
        self.counts = mp.RawArray(ct.c_long, self.capacity) # count
        self.occupied = mp.RawValue(ct.c_long, 0)
        self.stats = mp.RawArray(ct.c_double, self.capacity * self.dim * 4 if self.use_stats else 0)
        for i in range(self.capacity):
            self.counts[i] = 0
            self.set_y(i, np.inf)  
            self.ds_view[i] = np.full(self.qd_dim, np.inf)
            if self.stats:
                self.set_stat(i, 0, np.zeros(self.dim)) # mean
                self.set_stat(i, 1, np.zeros(self.dim)) # qmean
                self.set_stat(i, 2, np.full(self.dim, np.inf)) # min
                self.set_stat(i, 3, np.full(self.dim, -np.inf)) # max
         
    def init_niches(self, samples_per_niche: int = 10):
        """
        Initializes niches for clustering based on the specified sampling method and parameters.

        This method configures the clustering mode (CVT-clustering or grid-clustering)
        based on the `samples_per_niche` parameter. It also sets up necessary resources,
        such as a KDTree and shared memory for centroids, when CVT-clustering is selected.

        Args:
            samples_per_niche (int, optional): Defines the sampling strategy for niches.
                If greater than 0, CVT-clustering is used; otherwise, grid-clustering is used.

        """
        # If samples_per_niche > 0 cvt-clustering is used, else grid-clustering is used.   
        self.cvt_clustering = samples_per_niche > 0
        if self.cvt_clustering:
            set_KDTree(self, None, self.capacity, self.qd_bounds, samples_per_niche)
            self.cs = mp.RawArray(ct.c_double, self.capacity * self.qd_dim)
            self.set_cs(self.centers)
    
    def get_occupied_data(self):
        """
        Retrieves the data points marked as "occupied".

        This method processes the 'ys' array (retrieved by the `get_ys` method) to identify
        elements corresponding to occupied data points. Occupied data points are determined
        by checking where the values in 'ys' are less than positive infinity. Once identified,
        it returns the `ys`, `ds_view`, and `xs_view` data points corresponding to the
        occupied entries.

        Returns:
            tuple: A tuple containing three arrays:
                - The 'ys' values for occupied data points.
                - The corresponding 'ds_view' values for occupied data points.
                - The corresponding 'xs_view' values for occupied data points.
        """
        ys = self.get_ys()
        occupied = (ys < np.inf)
        return ys[occupied], self.ds_view[occupied], self.xs_view[occupied]        
   
    def join(self, archive: Archive):
        """
        Joins data from the given archive into the current archive by retrieving and processing
        occupied data, assigning it to corresponding niches, and sorting the archive.

        Args:
            archive (Archive): An archive containing data to be joined with the current archive.
        """
        ys, ds, xs = archive.get_occupied_data()
        niches = archive.index_of_niches(ds)
        yds = np.array([(y, d) for y, d in zip(ys, ds)]) 
        for i in range(len(ys)):
            archive.set(niches[i], yds[i], xs[i]) 
        archive.argsort()   

    def fname(self, name):
        """
        Generates a formatted string using name, capacity, dimension, and quality-dimension
        values for a specific architecture.

        Args:
            name: The name string that will be included in the formatted output.

        Returns:
            str: A formatted string in the format 'arch.{name}.{self.capacity}.{self.dim}.
                {self.qd_dim}'.
        """
        return f'arch.{name}.{self.capacity}.{self.dim}.{self.qd_dim}'
           
    def save(self, name: str):
        """
        Saves the processed data and corresponding metadata in a compressed NPZ format.

        The method stores the data views such as xs (features), ds (distances), ys (labels), 
        and other optional attributes like clustering and statistics in a compressed file. 
        The filename of the saved file is dynamically generated based on the provided name.

        Args:
            name: Name used to generate the filename for the saved data.
        """ 
        np.savez_compressed(self.fname(name), 
                            xs=self.xs_view, 
                            ds=self.ds_view, 
                            ys=self.get_ys(), 
                            cs=self.get_cs() if self.cvt_clustering else np.empty(0),
                            stats=self.get_stats(),
                            counts=self.get_counts()
                            )

    def load(self, name: str):
        """
        Loads data from a specified file into the current object, initializing or 
        updating its properties and attributes. This method reads numpy data files 
        and adjusts internal structures, such as clustering state, dimensions, capacity, 
        and associated statistics based on the data provided.

        Args:
            name (str): The base name of the file from which to load data.

        Raises:
            FileNotFoundError: If the file specified by the name parameter does not exist.
            ValueError: If the data in the file is inconsistent or incompatible with 
                        the current object's structure.
        """   
        self.cs = mp.RawArray(ct.c_double, self.capacity * self.qd_dim)
        with np.load(self.fname(name) + '.npz') as data:
            self.cvt_clustering = len(data['cs']) > 0
            xs = data['xs']
            ds = data['ds']
            self.xs.view()[:] = xs
            self.ds.view()[:] = ds
            self.set_ys(data['ys'])
            if self.cvt_clustering:
                self.set_cs(data['cs'])
            self.counts[:] = data['counts']
            stats = data['stats']
            if len(stats) == len(self.stats):
                self.set_stats(stats)
        self.occupied.value = np.count_nonzero(self.get_ys() < np.inf)
        self.dim = xs.shape[1]
        self.qd_dim = ds.shape[1]
        self.capacity = xs.shape[0]
        if self.cvt_clustering:
            set_KDTree(self, self.get_cs(), None, None, None)
      
    def index_of_niches(self, ds):
        """
        Determines the indices of the niches for the given data samples (ds) based on the 
        clustering method used (either k-means clusters or grid-based clustering).

        Args:
            ds: Input data samples for which the niche indices are to be determined.

        Returns:
            ndarray: Niche indices of the input data samples.
        """
        if hasattr(self, "kdt"): # use k-means clusters
            return self.kdt.query(self.encode_d(ds), k=1, sort_results=False)[1].T[0] 
        else: # use grid based clustering
            return get_grid_indices(ds, self.capacity, self.qd_bounds.lb, self.qd_bounds.ub)
        
    def in_niche_filter(self, 
                        fit: Callable[[ArrayLike], float], 
                        index: int):
        """
        Filters an element based on its fitness and niche index.

        This function evaluates whether an element, represented by its fitness value and niche index, satisfies the 
        condition required for inclusion in a niche. The filtering mechanism is defined externally and passed in 
        as a callable.

        Args:
            fit (Callable[[ArrayLike], float]): A callable function that takes an array-like object and calculates 
                a fitness value for evaluation.
            index (int): The index of the element within its group to be filtered.

        Returns:
            bool: True if the element satisfies the niche filter condition, False otherwise.
        """
        return in_niche_filter(fit, index, self.index_of_niches)
                                                               
    def set(self, 
            i: int, 
            yd: np.ndarray, 
            x: np.ndarray):
        """
        Updates statistical data and modifies the internal data store with the given 
        values if certain conditions are met (e.g., improvement in objective value `y`).

        Args:
            i (int): The index representing the position in the internal data storage 
                to update.
            yd (np.ndarray): A numpy array containing the objective value `y` and 
                additional associated data `d` as elements.
            x (np.ndarray): A numpy array representing the new input data associated 
                with the index `i`.

        """ 
        self.update_stats(i, x)
        y, d = yd
        # register improvement
        yold = self.get_y(i)
        if y < yold:
            if yold == np.inf: # not yet occupied
                self.occupied.value += 1
            self.set_y(i, y)
            self.xs_view[i] = x
            self.ds_view[i] = d
    
    def update_stats(self, 
                     i: int, 
                     x: np.ndarray):
        """
        Updates the statistical information for the given index and array input. Tracks 
        counts, mean, quadratic mean (qmean), minimum, and maximum values for the 
        specified index.

        Args:
            i (int): Index of the element whose statistics need to be updated.
            x (numpy.ndarray): Input array used to update the statistics.
        """
        with self.lock:
            self.counts[i] += 1
        count = self.counts[i]
        if self.use_stats:  
            mean = self.get_x_mean(i)
            diff = x - mean      
            self.set_stat(i, 0, mean + diff * (1./count)) # mean
            self.set_stat(i, 1, self.get_stat(i, 1) + np.multiply(diff,diff) * ((count-1)/count)) # qmean                  
            self.set_stat(i, 2, np.minimum(x, self.get_stat(i, 2))) # min
            self.set_stat(i, 3, np.maximum(x, self.get_stat(i, 3))) # max
 
    def get_occupied(self) -> int:
        """
        Retrieves the current number of occupied spaces from the associated value.

        Returns:
            int: The number of occupied spaces.
        """
        return self.occupied.value
    
    def get_count(self, i: int) -> int:
        """
        Get the count for a specific index.

        This method retrieves the count value from the `counts` collection based on the 
        provided index.

        Args:
            i (int): The index of the element whose count is to be retrieved.

        Returns:
            int: The count corresponding to the given index.
        """
        return self.counts[i]

    def get_counts(self) -> np.ndarray:
        """
        Retrieves a copy of the `counts` as a NumPy array.

        This method returns a deep copy of the `counts` list, converted to a NumPy
        array. It ensures that the original data remains unaltered while providing
        a convenient format for numerical operations.

        Returns:
            np.ndarray: A NumPy array containing the copied elements of `counts`.
        """
        return np.array(self.counts[:])
 
    def get_x_mean(self, i: int) -> np.ndarray:
        """
        Calculates and retrieves the mean value for a specific index of data.

        This function calls another internal method to obtain the required statistic
        associated with the given index and statistic type (mean in this case).

        Args:
            i (int): The index of the data for which the mean value will be computed.

        Returns:
            np.ndarray: The calculated mean value for the specified index.
        """
        return self.get_stat(i, 0)

    def get_x_stdev(self, i: int) -> np.ndarray:
        """
        Calculates the standard deviation of x for a given identifier.

        This method computes the standard deviation of the variable 'x', based on the
        statistics collected for a specific identifier. If no data is available for
        the identifier, it returns a zero array with the appropriate dimension.

        Args:
            i (int): Identifier used to retrieve the specific statistics.

        Returns:
            np.ndarray: A numpy array representing the standard deviation of x, with
            the same dimension as the data.
        """
        count = self.get_count(i)
        if count == 0:
            return np.zeros(self.dim)
        else:
            qmean = np.array(self.get_stat(i, 1))
            return np.sqrt(qmean * (1./count))

    def get_x_min(self, i: int) -> np.ndarray:
        """
        Gets the minimum x-coordinate value from a statistical data set.

        This function retrieves the minimum value of the x-coordinate from the
        dataset for a specified index and returns it as a NumPy array.

        Args:
            i (int): The index of the dataset for which the minimum x-coordinate
                value is retrieved.

        Returns:
            np.ndarray: The minimum x-coordinate value in the dataset corresponding
                to the provided index.
        """
        return self.get_stat(i, 2)

    def get_x_max(self, i: int) -> np.ndarray:
        """
        Returns the maximum value of the 'x' statistic for the specified index.

        This method retrieves the maximum value of the 'x' statistic by utilizing
        the `get_stat` method with specific parameters.

        Args:
            i (int): The index for which the 'x' statistic's maximum value is to
                be retrieved.

        Returns:
            np.ndarray: The maximum value of the 'x' statistic for the specified
                index.
        """
        return self.get_stat(i, 3)

    def create_views(self): # needs to be called in the target process
        """
        Creates views for the attributes within the target process.

        This method initializes views for specific attributes to enhance
        performance and allow efficient data manipulation in the target
        process. The views are created using the `view()` method.

        Raises:
            AttributeError: If either `xs` or `ds` is not properly initialized
            or does not support the `view()` method.
        """
        self.xs_view = self.xs.view()
        self.ds_view = self.ds.view()
           
    def encode_d(self, d):
        """
        Encodes the given value using the specified scale and lower bound.

        This method calculates a normalized value by subtracting the lower
        bound (self.desc_lb) from the input and then dividing by the scale
        factor (self.desc_scale).

        Args:
            d: A numerical value to be encoded.

        Returns:
            float: The encoded value after applying the normalization formula.
        """
        return (d - self.desc_lb) / self.desc_scale
    
    def decode_d(self, d):
        """
        Decodes a given value `d` using the object's scaling factor and lower bound.

        This method applies a transformation to the input value `d` based on the
        attributes `desc_scale` and `desc_lb` of the object. It linearly scales
        and offsets the input to produce the decoded value.

        Args:
            d: Input value to decode.

        Returns:
            The decoded value as a float.
        """
        return (d * self.desc_scale) + self.desc_lb

    def get_xs(self) -> np.ndarray:
        """
        Gets a view of the `xs` attribute.

        This method returns a view of the `xs` attribute, allowing access to
        the data without copying it. The returned view reflects changes made
        to the original data.

        Returns:
            np.ndarray: A view of the `xs` attribute.
        """
        return self.xs.view()

    def get_ds(self) -> np.ndarray:
        """
        Returns a view of the `ds` attribute.

        This method provides a view of the data stored in the `ds` attribute. It
        enables access to the data without modifying the original structure.

        Returns:
            np.ndarray: A view of the `ds` attribute as a NumPy array.
        """
        return self.ds.view()
    
    def get_y(self, i: int) -> float:
        """
        Retrieves the y-coordinate value at a given index from the list of y-values.

        Args:
            i (int): The index of the y-value to retrieve.

        Returns:
            float: The y-coordinate value at the specified index.
        """
        return self.ys[i]
        
    def get_ys(self) -> np.ndarray:
        """
        Returns a copy of the `ys` array.

        This method provides access to a copy of the `ys` attribute, ensuring
        that the original data remains unaltered.

        Returns:
            np.ndarray: A numpy array representing a copy of the `ys` attribute.
        """
        return np.array(self.ys[:])
    
    def get_qd_score(self) -> float:
        """
        Calculates the QD (Quality-Diversity) score based on the provided array of values.

        The method processes an array of values (`ys`) to compute a QD score using either the sum of reciprocal
        values for positive values or the sum of negated negative values. The computed score is dependent on
        whether all the values in the array are positive or some are negative.

        Returns:
            float: The calculated QD score.
        """
        ys = self.get_ys()
        occupied = (ys != np.inf)
        ys = ys[occupied]
        if len(ys) == 0:
            return 0
        min_y = np.amin(ys)
        if min_y > 0: # if all y > 0 use sum of reciprocal
            return np.sum(np.reciprocal(ys, where = ys!=0)) 
        else: # else use only the negative ones
            neg = (ys < 0)
            ys = ys[neg]
            return np.sum(-ys) 
         
    def set_y(self, i: int, y: float):
        """
        Updates the `ys` attribute at the specified index with a new value.

        Args:
            i (int): The index to update in the `ys` attribute.
            y (float): The new value to set at the specified index.
        """
        self.ys[i] = y

    def set_ys(self, ys: ArrayLike):
        """
        Sets multiple y-values for the object by iterating over the provided array-like collection.

        Args:
            ys (ArrayLike): An array-like collection of y-values to be set.

        """
        for i in range(len(ys)):
            self.set_y(i, ys[i])
            
    def get_c(self, i: int) -> float:
        """
        Retrieves a slice of the `cs` list for the given index.

        Args:
            i (int): Index to slice the `cs` list.

        Returns:
            float: The extracted portion of the `cs` list corresponding to the
            specified index, scaled by `qd_dim`.
        """
        return self.cs[i*self.qd_dim:(i+1)*self.qd_dim]
        
    def get_cs(self) -> np.ndarray:
        """
        Generates an array of results produced by the `get_c` method for a range
        of indices.

        The `get_cs` method computes an array containing the output of the `get_c`
        method for a sequence of indices up to a specified capacity. Each index is
        processed individually to construct the final array.

        Returns:
            np.ndarray: A NumPy array comprising results obtained from the `get_c`
            method called for indices ranging from 0 to the object's capacity.
        """
        return np.array([self.get_c(i) for i in range(self.capacity)])

    def get_cs_decoded(self) -> np.ndarray:
        """
        Returns decoded data using the decode_d method.

        This method retrieves data by iterating over a range equivalent to the
        capacity attribute, applies the get_c method to each index, and returns
        the results as a NumPy array. The resulting array is then passed to the
        decode_d method to decode and return the final data.

        Returns:
            np.ndarray: Decoded data obtained from processing the capacity range.
        """
        return self.decode_d(np.array([self.get_c(i) for i in range(self.capacity)]))
           
    def set_c(self, i: int, c: float):
        """Sets the values of a specific segment of the 'cs' attribute.

        This method updates a specific segment of the 'cs' attribute based on the
        provided index and float values. The index determines the segment of
        'cs' to be updated, while the float input defines the new values to
        assign to the segment.

        Args:
            i (int): Index to determine the specific segment of 'cs' to update.
            c (float): New float value that will replace the content of the
                specified segment in 'cs'.

        """
        self.cs[i*self.qd_dim:(i+1)*self.qd_dim] = c[:]

    def set_cs(self, cs: ArrayLike):
        """
        Sets the values from the provided array-like object to the internal structure by calling
        `set_c` method for each element of the array-like object.

        Args:
            cs (ArrayLike): An array-like object containing elements to be set using `set_c` method.
        """
        for i in range(len(cs)):
            self.set_c(i, cs[i])

    def get_stat(self, i: int, j: int) -> float:
        """
        Retrieves a specific statistical value for given indices i and j. The method
        calculates an index based on the input values and extracts the corresponding
        dimension slice from the stats attribute.

        Args:
            i (int): Represents the first index to calculate the position.
            j (int): Represents the second index to calculate the position.

        Returns:
            float: A specific statistical value derived from the stats attribute.
        """
        p = 4*i+j
        return self.stats[p*self.dim:(p+1)*self.dim]

    def get_stats(self) ->  np.ndarray:
        """
        Returns a copy of the stats data as a NumPy array.

        This method provides access to the current statistical data by returning a
        NumPy array that contains a copy of the internal stats data.

        Returns:
            np.ndarray: A NumPy array containing a copy of the stats data.
        """
        return np.array(self.stats[:])
           
    def set_stat(self, i: int, j: int, stat: ArrayLike):
        """
        Sets a specific portion of the stats array based on the given indices and data.

        This function updates the stats array by calculating an index position derived from
        the given indices `i` and `j`. The data in `stat` is then inserted into the stats array
        at a position determined by the indices and the object's dimension property.

        Args:
            i (int): Row index used to determine the position in the stats array.
            j (int): Column index used to determine the position in the stats array.
            stat (ArrayLike): Array of statistical data to update the stats array.

        """
        p = 4*i+j
        self.stats[p*self.dim:(p+1)*self.dim] = stat[:]

    def set_stats(self, stats: ArrayLike):
        """
        Sets the statistics for the object.

        This method updates the internal statistics of the object with the values
        provided in the input array. The values in the input array are assigned
        directly to the object's statistics.

        Args:
            stats (ArrayLike): The array-like object containing the new statistics to
                be set. It must be compatible with the internal statistics structure
                of the object.
        """
        self.stats[:] = stats[:]

    def random_xs(self, best_n: int, chunk_size: int, rg: Generator) -> np.ndarray:
        """
        Generates a random selection of indices and returns corresponding elements from the view.

        This function utilizes a random number generator to produce a set of indices based on
        the provided parameters and returns the corresponding elements from an internal view.

        Args:
            best_n (int): The upper limit for the random integer selection range.
            chunk_size (int): The number of random integers to select.
            rg (Generator): A random number generator instance to produce the selection.

        Returns:
            np.ndarray: The array of selected elements corresponding to the randomly generated indices.
        """
        selection = rg.integers(0, best_n, chunk_size)
        if best_n < self.capacity: 
            selection = np.fromiter((self.si[i] for i in selection), dtype=int)
        return self.xs_view[selection]
    
    def random_xs_one(self, best_n: int, rg: Generator) -> Tuple[np.ndarray, float, int]:
        """
        Selects and retrieves random elements based on the given range and size.

        This function utilizes a random number generator to select an index from a range
        up to `best_n` and fetch the corresponding elements.

        Args:
            best_n: The upper limit (exclusive) for selecting a random index, provided
                as an integer.
            rg: An instance of a random number generator implementing the `Generator`
                interface, used for generating random numbers.

        Returns:
            A tuple containing:
                - The randomly selected x-coordinate as a numpy array.
                - The corresponding y-coordinate as a float.
                - The randomly chosen index as an integer.
        """
        i = int(rg.random()*best_n)
        return self.get_x(i), self.get_y(i),  i
        
    def argsort(self) -> np.ndarray:
        """
        Sorts the indices of the array returned by `get_ys()` in ascending order.

        This method sorts the elements of the array obtained by calling `get_ys()` and
        returns the indices that would sort this array in ascending order. The sorted
        indices are also stored in the `si` attribute for future use.

        Returns:
            np.ndarray: The indices that would sort the array in ascending order.
        """
        self.si = np.argsort(self.get_ys())
        return self.si
            
    def dump(self, n: Optional[int] = None):
        """
        Prints a sorted dump of data points up to the specified limit.

        This method retrieves a list of data points (`ys`), sorts them in ascending
        order, and prints the indices, values, additional details, and associated
        features for the top `n` data points. If `n` is not specified, it defaults to
        the object's capacity.

        Args:
            n: Optional[int], optional. The number of data points to print. Defaults to
                the object's capacity.
        """
        if n is None:
            n = self.capacity
        ys = self.get_ys()
        si = np.argsort(ys)
        for i in range(n):
            print(si[i], ys[si[i]], self.get_d(si[i]), self.get_x(si[i]))    
    
    def info(self) -> str:
        """
        Generates a formatted string containing key metrics.

        This method calculates and returns a string with the following information:
        occupied cells, the quality diversity (QD) score, the lowest value from
        a collection of data ('ys'), and the total count of items. The output is
        formatted as a single string.

        Returns:
            str: A formatted string that includes occupied cells, QD score rounded
            to three decimal places, the minimum value from 'ys' rounded to three
            decimal places, and the total count of items.
        """
        occ = self.get_occupied()
        score = self.get_qd_score()
        best_y = np.amin(self.get_ys())
        count = np.sum(self.get_counts())
        return f'{occ} {score:.3f} {best_y:.3f} {count}'

class wrapper(object):
    """A callable class designed to fit and evaluate a function with given inputs.

    This class acts as a wrapper that integrates a user-defined fitness function. It manages
    the evaluation count, best fitness values, and optional logging and saving mechanisms for
    performance monitoring. It is particularly useful for optimization tasks that leverage
    multi-processing for concurrent evaluations.

    Attributes:
        fit (Callable[[ArrayLike], Tuple[float, np.ndarray]]): The user-defined function
            that computes the fitness value and its associated descriptors for a given input.
        evals (multiprocessing.RawValue): A counter for the number of evaluations performed.
       """

    def __init__(self, 
                 fit:Callable[[ArrayLike], Tuple[float, np.ndarray]], 
                 qd_dim: int, 
                 interval: Optional[int] = 1000000,
                 save_interval: Optional[int] = 1E20):
        """
        Initializes a new instance of the class.

        This constructor sets up the main attributes required for the class, including
        the fitness function, dimensionality of the optimization problem, evaluation
        intervals, and other related parameters.

        Args:
            fit (Callable[[ArrayLike], Tuple[float, np.ndarray]]): Fitness function used
                to evaluate solutions. It should take an input of type ArrayLike and
                return a tuple containing a float and an np.ndarray.
            qd_dim (int): Dimensionality of the search/solution space to be optimized.
            interval (Optional[int]): Interval for certain operations during optimization.
                Defaults to 1,000,000 if not specified.
            save_interval (Optional[int]): Interval for saving checkpoints. Defaults to
                an extremely high value (1E20) if not specified.
        """
        self.fit = fit
        self.evals = mp.RawValue(ct.c_int, 0) 
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.t0 = perf_counter()
        self.qd_dim = qd_dim
        self.interval = interval
        self.save_interval = save_interval
        self.lock = mp.Lock()
        
    def __call__(self, x: ArrayLike):
        """
        Handles the evaluation of input data, including logging and archiving results, while maintaining
        the state of evaluations, the best observed outcome, and potential exceptions during execution.

        Args:
            x (ArrayLike): Input data array to be evaluated.

        Returns:
            tuple: A tuple containing:
                - A float value representing the evaluation result or `np.inf` if the result is invalid.
                - A numpy array representing the descriptor or a zero array if the result is invalid.

        Raises:
            Exception: Logs the exception details if execution fails.
        """
        try:
            if np.isnan(x).any():
                return np.inf, np.zeros(self.qd_dim)
            with self.lock:
                self.evals.value += 1
            log = self.evals.value % self.interval == 0
            save = self.evals.value % self.save_interval == 0
            y, desc = self.fit(x)
            if np.isnan(y) or np.isnan(desc).any():
                return np.inf, np.zeros(self.qd_dim)
            y0 = y if np.isscalar(y) else sum(y)
            if y0 < self.best_y.value:
                self.best_y.value = y0
                log = True 
            if log:
                archinfo = self.archive.info() if hasattr(self, 'archive') else ''
                logger.info(
                    f'{dtime(self.t0)} {archinfo} {self.evals.value:.0f} {self.evals.value/(1E-9 + dtime(self.t0)):.1f} {self.best_y.value:.3f} {list(x)}'
                )            
            if save and hasattr(self, 'archive'):
                self.archive.save(f'{self.evals.value}')
            return y, desc
        except Exception as ex:
            print(str(ex))  
            return np.inf, np.zeros(self.qd_dim)
        
class in_niche_filter(object):
    """Filters and evaluates data based on niche and fitness criteria.

    This class determines whether a given data point belongs to a specific niche
    and evaluates it based on a fitness function. If the data point does not belong
    to the specified niche, it assigns an infinite fitness value to indicate that
    the point is not relevant to the niche being evaluated.

    Attributes:
        fit (Callable[[ArrayLike], Tuple[float, np.ndarray]]): A fitness evaluation
            function that takes an input of type ArrayLike and returns a tuple with a
            fitness value and a descriptor.
        index (int): Index representing the specific niche to evaluate against.
        index_of_niches (Callable[[ArrayLike], np.ndarray]): A function that determines
            the niche index for a given input based on its descriptor.
    """
    
    def __init__(self, 
                 fit:Callable[[ArrayLike], Tuple[float, np.ndarray]], 
                 index: int, 
                 index_of_niches: Callable[[ArrayLike], np.ndarray]):
        """
        Initializes an instance of a class to manage niche-based functionality and fitness evaluation.

        This constructor sets up the necessary components to evaluate fitness and handle niche-related
        computations.

        Args:
            fit (Callable[[ArrayLike], Tuple[float, np.ndarray]]): A function that takes an array-like
                input and returns a tuple containing a float representing the fitness value and a
                numpy.ndarray for additional fitness-related data.
            index (int): An integer representing the index or identifier for this instance.
            index_of_niches (Callable[[ArrayLike], np.ndarray]): A function that takes an array-like
                input and returns a numpy.ndarray corresponding to the indices of niches associated
                with the input.
        """
        self.fit = fit
        self.index_of_niches = index_of_niches
        self.index = index

    def __call__(self, x: ArrayLike) -> float:
        """
        Evaluates the input by fitting it and checking against a specific index. Returns the
        computed result if the index matches; otherwise, returns infinity.

        Args:
            x (ArrayLike): Input array to be evaluated.

        Returns:
            float: Computed result `y` if the index matches; otherwise, infinity (np.inf).
        """
        y, desc = self.fit(x)
        if self.index_of_niches([desc])[0] == self.index:
            return y
        else:
            return np.inf

def variation_(pop, lower, upper, rg, dis_c = 20, dis_m = 20):
    """
    Generates offspring using a variation operator for a given population according to the simulated
    binary crossover (SBX) and polynomial mutation mechanisms. This method scales both the spread factors
    of crossover (`dis_c`) and mutation (`dis_m`) randomly within their range to introduce stochasticity.

    Args:
        pop (np.ndarray): 2D array representing the population, where each row corresponds to an individual and
                          each column to a decision variable.
        lower (np.ndarray): 1D array specifying the lower bound of each decision variable. Its length must match
                            the number of columns in the population.
        upper (np.ndarray): 1D array specifying the upper bound of each decision variable. Its length must match
                            the number of columns in the population.
        rg (np.random.Generator): Random generator instance for stochastic operations.
        dis_c (float, optional): Distribution index for crossover operation, controlling spread of solutions
                                 during SBX. Defaults to 20.
        dis_m (float, optional): Distribution index for mutation operation, controlling spread of mutations.
                                 Defaults to 20.

    Returns:
        np.ndarray: The newly generated offspring population, with the same shape as the input `pop`.
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
    parent_mean = (parent_1 + parent_2) / 2
    parent_diff = (parent_1 - parent_2) / 2
    offspring = np.vstack((parent_mean + beta * parent_diff, parent_mean - beta * parent_diff))
    site = rg.random((n, d)) < 1.0 / d
    mu = rg.random((n, d))
    temp = site & (mu <= 0.5)
    lower, upper = np.tile(lower, (n, 1)), np.tile(upper, (n, 1))
    norm = (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp]) * \
                           (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(np.abs(1. - norm), dis_m + 1.),
                                     1. / (dis_m + 1)) - 1.)
    temp = site & (mu > 0.5)
    norm = (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp]) * \
                           (1. - np.power(
                               2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(np.abs(1. - norm), dis_m + 1.),
                               1. / (dis_m + 1.)))
    return np.clip(offspring, lower, upper)

def iso_dd_(x1, x2, lower, upper, rg, iso_sigma = 0.01, line_sigma = 0.2):
    """
    Generates isotropic deviation data within specified bounds.

    This function creates a variation of the input data `x1` influenced by
    Gaussian noise parameters for isotropic and line deviations. The result
    is clipped to remain within the range defined by the `lower` and `upper`
    bounds.

    Args:
        x1: The primary input array for generating isotropic deviation data.
        x2: The secondary input array influencing line deviation calculation.
        lower: The lower bound for clipping the resulting data.
        upper: The upper bound for clipping the resulting data.
        rg: Random generator object used for generating Gaussian noise.
        iso_sigma: Standard deviation of the isotropic Gaussian noise. Defaults
            to 0.01.
        line_sigma: Standard deviation of the line Gaussian noise. Defaults to
            0.2.

    Returns:
        ndarray: The processed array with isotropic and line deviations
        applied and bounded by the given limits.

    """
    a = rg.normal(0, iso_sigma, x1.shape) 
    b = rg.normal(0, line_sigma, x2.shape) 
    z = x1 + a + np.multiply(b, (x1 - x2))
    return np.clip(z, lower, upper)

def get_centers_(niche_num, dim, samples_per_niche):
    """
    Determines and returns the centers of niches in a defined space using k-means clustering.
    If the data is cached, it loads the cached centers for faster access. Otherwise, it computes
    the centers by applying k-means clustering on randomly generated samples and caches the
    result for future use.

    Args:
        niche_num (int): Number of niches or clusters to determine centers for.
        dim (int): Dimensionality of the space where the niches are located.
        samples_per_niche (int): Number of samples used per niche to compute the centers.

    Returns:
        numpy.ndarray: An array representing the determined centers of the niches.
    """
    p = Path('voronoi_cache')
    p.mkdir(exist_ok=True)
    fname = f'centers_{niche_num}_{dim}_{samples_per_niche}.npz'
    files = p.glob(fname)
    for file in files: # if cached just load
        with np.load(file) as data:
            return data['cs']
    else:
        descs = rng.uniform(0, 1, (niche_num*samples_per_niche, dim))
        # Applies KMeans to the random samples determine the centers of each niche."""
        k_means = KMeans(init='k-means++', n_clusters=niche_num, n_init=1, verbose=1)
        k_means.fit(descs)
        centers = k_means.cluster_centers_
        np.savez_compressed(f'voronoi_cache/centers_{niche_num}_{dim}_{samples_per_niche}', cs=centers)
        return centers
