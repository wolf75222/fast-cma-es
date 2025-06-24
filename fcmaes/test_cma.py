# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - test_cma.py

 Description:
  - This file contains unit tests for the Fast CMA-ES library.

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
import multiprocessing as mp
import numpy as np
from scipy.optimize import OptimizeResult
from fcmaes.testfun import Wrapper, Rosen, Rastrigin, Eggholder
from fcmaes import cmaes, de, decpp, cmaescpp, retry, advretry
from fcmaes.optimizer import de_cma_py

def almost_equal(X1, X2, eps = 1E-5):
    """
    Determines if two numerical values or sets of numerical values are approximately
    equal within a specified tolerance.

    This function compares two scalar values or lists of scalar values and checks if
    they are approximately equivalent, considering a given threshold for acceptable
    deviation. The comparison is performed element-wise for lists.

    Args:
        X1: A scalar value or a list of scalar values to compare.
        X2: Another scalar value or a list of scalar values to compare.
        eps: A small positive float specifying the tolerance for comparison. Default
            is 1E-5.

    Returns:
        bool: True if the values are approximately equal within the specified
        tolerance; False otherwise.
    """
    if np.isscalar(X1):
        X1 = [X1]
        X2 = [X2]
    if len(X1) != len(X2):
        return False
    for i in range(len(X1)):
        a = X1[i]
        b = X2[i]
        if abs(a) < eps or abs(b) < eps:
            if abs(a - b) > eps:
                return False
        else:
            if abs(a / b - 1 > eps):
                return False
    return True

def test_rastrigin_python():
    """
    Tests the Rastrigin function optimization using the CMA-ES algorithm.

    This function performs a series of tests to ensure that the CMA-ES
    optimization algorithm correctly minimizes the Rastrigin function,
    a standard test function in optimization. The function evaluates
    whether the optimization reaches the target goal within constraints
    (using bounds, standard deviations, and maximum number of evaluations),
    and whether the returned results conform to expected outputs.

    Args:
        None

    Raises:
        AssertionError: If any of the assertions fail, indicating that
            the optimization did not meet the expected conditions or outcomes.
    """
    popsize = 100
    dim = 3
    testfun = Rastrigin(dim)
    sdevs = [1.0]*dim
    max_eval = 100000

    limit = 0.0001   
    # stochastic optimization may fail the first time
    for _ in range(5):
        # use a wrapper to monitor function evaluations
        wrapper = Wrapper(testfun.fun, dim)
        ret = cmaes.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                       max_evaluations = max_eval, popsize=popsize)
        if limit > ret.fun:
            break
    
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval / popsize + 2 > ret.nit) # too much iterations
    assert(ret.status == 4) # wrong cma termination code
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_python():
    """
    Tests the Rosenbrock function optimization using the CMA-ES algorithm.

    This test evaluates the performance of the CMA-ES optimization algorithm on
    the Rosenbrock function with specific settings for dimensionality, population
    size, standard deviations, and maximum function evaluations. It checks if the
    optimization target is reached within the given constraints and validates the
    result consistency of the optimization process.

    Raises:
        AssertionError: If the optimization does not achieve the expected results
            including:
            - Optimization target (minimum function value) not reached.
            - Excessive function evaluations during optimization.
            - Excessive iterations performed.
            - Mismatched function call count from the optimization algorithm versus
              the internal wrapper.
            - Incorrect best solution vector or value returned.
    """
    popsize = 31
    dim = 5
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 100000
    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = cmaes.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                       max_evaluations = max_eval, popsize=popsize)
        if limit > ret.fun:
            break
    
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval / popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_ask_tell():
    """
    Tests the ask-tell interface of the CMA-ES algorithm on the Rosenbrock
    function over multiple iterations and asserts the optimization results.

    The function initializes a CMA-ES instance with the specified
    hyperparameters and evaluates its performance over a predefined
    number of iterations and maximum evaluations. The results of the
    optimization process are validated against threshold values to
    ensure that the CMA-ES implementation satisfies expected performance
    criteria.

    Raises:
        AssertionError: If the optimization target is not achieved within
            the defined constraints of function evaluations or iterations.
    """
    popsize = 31
    dim = 5
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 100000   
    limit = 0.00001 
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        es = cmaes.Cmaes(testfun.bounds,
                popsize = popsize, input_sigma = sdevs)       
        iters = max_eval // popsize
        for j in range(iters):
            xs = es.ask()
            ys = [wrapper.eval(x) for x in xs]
            stop = es.tell(ys)
            if stop != 0:
                break 
        ret = OptimizeResult(x=es.best_x, fun=es.best_value, 
                             nfev=wrapper.get_count(), 
                             nit=es.iterations, status=es.stop)
        if limit > ret.fun:
            break
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval / popsize + 2 > ret.nit) # too much iterations
#     assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
#     assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_cpp():
    """
    Runs a test on a Rosenbrock function optimization using CMA-ES (Covariance Matrix Adaptation
    Evolution Strategy) implemented in C++ and Python wrappers. This test validates the optimization
    procedure by asserting constraints on the optimization target, number of function evaluations,
    and results consistency.

    The function initializes test parameters including population size, dimensions, standard deviations,
    and maximum evaluations. It then runs the optimization multiple times, stopping early if a defined
    limit on the objective function is achieved. Assertions verify the results of the optimization process.

    Raises:
        AssertionError: If the optimization result does not meet the defined constraints on the
            objective function value, number of function evaluations, best solution `x`, or best
            objective function value `y`.
    """
    popsize = 31
    dim = 5
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 100000
    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = cmaescpp.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                   max_evaluations = max_eval, popsize=popsize)
        if limit > ret.fun:
            break

    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls 
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_parallel():
    """
    Tests the optimization algorithm's ability to minimize Rosenbrock's function
    using a parallelized CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    implementation with a preset population size. Verifies the optimization
    process meets expected limits for function calls, iterations, and optimization
    accuracy.

    Raises:
        AssertionError: If the optimization target is not reached within the
            specified limit, the number of function calls exceeds the maximum
            evaluations, the number of iterations exceeds the expected bound,
            or the resulting best X and Y do not match the expected values within
            the given accuracy.
    """
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 10000
    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = cmaes.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = popsize)
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(almost_equal(ret.fun, wrapper.get_best_y(), eps=1E-1)) # wrong best y returned

def test_rosen_cpp_parallel():
    """
    Tests the parallel implementation of the Rosenbrock function optimization using a
    C++ CMA-ES (Covariance Matrix Adaptation Evolution Strategy) wrapper.

    This function evaluates the optimization process using the Rosenbrock function.
    The optimization parameters include the population size, dimensionality, initial
    standard deviations, maximum evaluations, and function bounds. The function tests
    whether the optimization converges within the limits, ensures the function calls
    and iterations do not exceed expected numbers, and validates the returned best
    solution and function value.

    Raises:
        AssertionError: If any of the assertions for convergence, function calls,
        iterations, or optimized solution validations fail.
    """
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    sdevs = [1.0]*dim
    max_eval = 10000
    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = cmaescpp.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = popsize)
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(almost_equal(ret.fun, wrapper.get_best_y(), eps = 1E-1)) # wrong best y returned

def test_rosen_de():
    """
    Tests the Rosenbrock function optimization using differential evolution.

    This function evaluates the performance of the Differential Evolution (DE)
    optimization algorithm for minimizing the Rosenbrock function, a common
    benchmark test for optimization routines. It ensures that the target
    optimization and other performance metrics (e.g., function calls, iterations)
    were achieved successfully.

    Raises:
        AssertionError: If the optimization target is not reached within the given
            constraints of maximum evaluations, iterations, or if there are
            discrepancies in the optimization results, such as mismatch in the
            best found solution or the number of function calls.
    """
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    max_eval = 10000    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = de.minimize(wrapper.eval, dim, testfun.bounds,
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = None)
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + 2*popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_rosen_de_parallel():
    """
    Tests the Differential Evolution (DE) algorithm's functionality and behavior using the
    Rosenbrock function within a parallel environment. The method sets up parameters,
    executes the DE optimization, and validates several conditions to ensure the correctness
    and performance of the optimization process.

    Raises:
        AssertionError: If the optimization target is not reached.
        AssertionError: If the number of function evaluations exceeds the defined limit.
        AssertionError: If the number of iterations exceeds the expected maximum.
        AssertionError: If the returned number of function evaluations does not match
            the wrapper's recorded count.
        AssertionError: If the incorrect best Y value is returned by the optimization.
    """
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    max_eval = 10000    
    limit = 0.01   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = de.minimize(wrapper.eval, dim, testfun.bounds,
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = popsize)
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
#     assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(almost_equal(ret.fun, wrapper.get_best_y(), eps = 1E-1)) # wrong best y returned

def test_rosen_ask_tell_de():
    """
    Tests the Differential Evolution optimization process for the Rosenbrock
    function using an ask-and-tell interface. The test validates convergence
    performance, ensures the optimization adheres to provided constraints,
    and checks correctness of the achieved results against defined limits.

    Args:
        None

    Raises:
        AssertionError: If the optimization target function value is not less
        than the specified limit.
        AssertionError: If the number of function evaluations surpasses the
        allowed maximum evaluations plus the buffer (2 * population size).
        AssertionError: If the number of iterations exceeds the allowed maximum
        iterations plus a buffer.
        AssertionError: If the optimized function value returned does not
        closely approximate the expected best value with the given tolerance.

    """
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    max_eval = 10000  
    limit = 0.00001 
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        es = de.DE(dim, testfun.bounds, popsize = popsize)       
        iters = max_eval // popsize
        for j in range(iters):
            xs = es.ask()
            ys = [wrapper.eval(x) for x in xs]
            stop = es.tell(ys, xs)
            if stop != 0:
                break 
        ret = OptimizeResult(x=es.best_x, fun=es.best_value, 
                             nfev=wrapper.get_count(), 
                             nit=es.iterations, status=es.stop)
        if limit > ret.fun:
            break
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + 2*popsize >= ret.nfev) # too much function calls
    assert(max_eval / popsize + 2 > ret.nit) # too much iterations
#     assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(almost_equal(ret.fun, wrapper.get_best_y(), eps = 1E-1)) # wrong best y returned

def test_rosen_decpp():
    """
    Test the Rosenbrock optimization function using the Differential Evolution with Population Adaptation
    (DECPP) algorithm. This test evaluates various aspects of the optimization's outcome, including
    convergence to the optimal result, the number of function evaluations, iterations, and consistency
    of return values.

    Variables:
        popsize (int): Sample size of the population used in DECPP optimization.
        dim (int): Dimensionality of the optimization problem. For the Rosenbrock function, this specifies
            the input size.
        testfun (Rosen): Instance of the Rosen class to define the objective function and bounds.
        max_eval (int): Maximum number of function evaluations allowed for the optimization.
        limit (float): Threshold for the optimization objective value to consider a successful result.
    """
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    max_eval = 10000    
    limit = 0.00001   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = decpp.minimize(wrapper.eval, dim, testfun.bounds,
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = None)
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + 2*popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(almost_equal(ret.fun, wrapper.get_best_y())) # wrong best y returned

def test_rosen_decpp_parallel():
    """
    Tests the `decpp.minimize` function applied to optimize the Rosenbrock
    function in parallel with specific conditions. Verifies correctness
    based on optimization target, number of function evaluations,
    iterations, and final results for optimization.

    Args:
        None

    Raises:
        AssertionError: If optimization target is not reached.
        AssertionError: If the number of function calls exceeds the expected
            maximum.
        AssertionError: If the number of iterations surpasses the expected
            maximum.
        AssertionError: If the best found `x` does not match the expected
            value within a given tolerance.
        AssertionError: If the best found `y` does not match the expected
            value within a given tolerance.
    """
    popsize = 8
    dim = 2
    testfun = Rosen(dim)
    max_eval = 10000    
    limit = 0.01   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = decpp.minimize(wrapper.eval, dim, testfun.bounds,
                       max_evaluations = max_eval, 
                       popsize=popsize, workers = popsize)
        if limit > ret.fun:
            break
       
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(max_eval // popsize + 2 > ret.nit) # too much iterations
    #assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x(), eps = 1E-2)) # wrong best X returned
    assert(almost_equal(ret.fun, wrapper.get_best_y(), eps = 1E-2)) # wrong best y returned

def test_eggholder_python():
    """
    Tests the Eggholder function minimization using the CMA-ES optimization algorithm.

    This function evaluates the performance of the CMA-ES optimizer in minimizing the
    Eggholder function, a complex and multimodal test function, over a specified number
    of dimensions and population size. The test ensures that optimization targets are
    reached within the defined function evaluation limits. It also verifies various
    aspects of the optimization process, including the number of function evaluations and
    the final results.

    Args:
        None

    Raises:
        AssertionError: Raised if the optimization target is not achieved, if the number
            of function evaluations exceeds the limit, or if the returned results are
            inconsistent with the wrapper's records.
    """
    popsize = 1000
    dim = 2
    testfun = Eggholder()
    # use a wrapper to monitor function evaluations
    sdevs = [1.0]*dim
    max_eval = 100000
    
    limit = -800   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)        
        ret = cmaes.minimize(wrapper.eval, testfun.bounds, input_sigma = sdevs, 
                       max_evaluations = max_eval, popsize=popsize)
        if limit > ret.fun:
            break
   
    assert(limit > ret.fun) # optimization target not reached
    assert(max_eval + popsize >= ret.nfev) # too much function calls
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_eggholder_retry():
    """
    Tests the retry optimization functionality for the Eggholder function.

    This test evaluates the retry-based optimization implementation by invoking
    the minimize method with the Eggholder function as the target to check if it
    achieves desired optimization thresholds. The test asserts that the
    optimization target is reached, the correct number of function evaluations
    are performed, and the best x and y values returned align with the best
    encountered during evaluation.

    Args:
        None

    Raises:
        AssertionError: If the optimization target is not reached, if the
            number of function evaluations does not match, if the best x values
            obtained mismatch, or if the best y values obtained mismatch.
    """
    dim = 2
    testfun = Eggholder()

    limit = -956   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = retry.minimize(wrapper.eval, testfun.bounds, 
                             num_retries=100)
        if limit > ret.fun:
            break

    assert(limit > ret.fun) # optimization target not reached
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

from fcmaes.optimizer import de_cma

def test_eggholder_advanced_retry():
    """
    Tests the advanced retry mechanism of the Eggholder function optimization process.

    The function defines a two-dimensional Eggholder function and ensures that optimization
    performs effectively within a certain threshold. It validates the following aspects:
    - The optimization result successfully satisfies the objective function target.
    - The number of function evaluations matches the expected count.
    - The optimal solution X and corresponding Y are correctly identified.

    Raises:
        AssertionError: If the optimization target is not reached, function evaluations do not match,
            or if the best X and Y values are incorrect.
    """
    dim = 2
    testfun = Eggholder()
    
    limit = -956   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = advretry.minimize(wrapper.eval, testfun.bounds, 
                                num_retries=96)
        if limit > ret.fun:
            break
        
    assert(limit > ret.fun) # optimization target not reached
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(almost_equal(ret.fun, wrapper.get_best_y())) # wrong best y returned

def test_eggholder_retry_python():
    """
    Tests the Eggholder function optimization with retry mechanism.

    This function performs optimization of the Eggholder function across 5 attempts,
    utilizing a retry logic to reinitialize and retry multiple times on failure. The
    test ensures that the optimization goal is met within the given constraints
    and verifies the correctness of various outcomes, including the function call
    count, the best identified arguments (`x`), and the best function evaluation
    (`y`).

    Raises:
        AssertionError: If the optimization target is not reached within the specified
            constraints, if the number of function evaluations (`nfev`) is incorrect,
            if the best arguments (`x`) returned are incorrect, or if the best function
            evaluation (`y`) returned is incorrect.
    """
    dim = 2
    testfun = Eggholder()
    
    optimizer = de_cma_py(10000)
    limit = -956   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)

        ret = retry.minimize(wrapper.eval, testfun.bounds, 
                             num_retries=32, optimizer = optimizer)
        if limit > ret.fun:
            break

    assert(limit > ret.fun) # optimization target not reached
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(ret.fun == wrapper.get_best_y()) # wrong best y returned

def test_eggholder_advanced_retry_python():
    """
    Tests the advanced retry mechanism for optimizing the Eggholder function.

    This test evaluates the functionality of the advanced retry mechanism combined
    with the differential evolution CMA optimizer (`de_cma_py`) on the Eggholder
    benchmark function. It verifies several key performance metrics such as the
    correctness of the returned optimal value, the number of function evaluations,
    and the retrieved best solution coordinates.

    Raises:
        AssertionError: If the limit is not surpassed by the optimizer after
            retries or if returned values (e.g., optimal solution, function
            evaluations, best coordinate) do not align with the expected results.
    """
    dim = 2
    testfun = Eggholder()

    optimizer = de_cma_py(10000)
    limit = -956   
    for _ in range(5):
        wrapper = Wrapper(testfun.fun, dim)
        ret = advretry.minimize(wrapper.eval, testfun.bounds, 
                                num_retries=32, optimizer = optimizer)
        if limit > ret.fun:
            break

    assert(limit > ret.fun) # optimization target not reached
    assert(ret.nfev == wrapper.get_count()) # wrong number of function calls returned
    assert(almost_equal(ret.x, wrapper.get_best_x())) # wrong best X returned
    assert(almost_equal(ret.fun, wrapper.get_best_y())) # wrong best y returned

#test_rosen_decpp_parallel()
 
