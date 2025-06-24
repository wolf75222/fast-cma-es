# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - testfun.py

 Description:
  - This module provides a set of test functions for optimization
    problems, including Rosenbrock, Rastrigin, Cigar, Sphere, and others.

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

import sys
import math
import numpy as np
import ctypes as ct
import multiprocessing as mp
from scipy.optimize import Bounds

class Wrapper(object):
    """Wrapper for parallel computation with shared state.

    This class acts as a wrapper to provide thread/process-safe management
    of shared variables during multiprocessing operations. It enables
    evaluation of a function and maintains the best result (minimum value
    of function output) along with a counter of evaluations.

    Attributes:
        func (Callable): The function to be evaluated.
        best_x (mp.RawArray): Shared memory array to store the inputs
            corresponding to the best function output.
        best_y (mp.RawValue): Shared memory value to store the best function
            output observed.
        count (mp.RawValue): Shared memory counter to track the number
            of function evaluations.
    """
   
    def __init__(self, fun, dim):
        """
        Initializes the instance with function to optimize and dimension of the problem.

        Args:
            fun: Callable function to optimize.
            dim: int. The dimensionality of the optimization problem.

        Attributes:
            func: Callable. The function to optimize.
            statMutex: multiprocessing.Lock. A lock to ensure thread safety.
            best_x: multiprocessing.Array. Shared array storing the best optimization
                solution found so far.
            best_y: multiprocessing.Value. Shared value storing the best objective
                value found so far.
            count: multiprocessing.Value. Shared value counting the number of function
                evaluations.
        """
        self.func = fun
        self.statMutex = mp.Lock()    
        self.best_x = mp.RawArray(ct.c_double, dim)
        self.best_y = mp.RawValue(ct.c_double, sys.float_info.max) 
        self.count = mp.RawValue(ct.c_int, 0) 
    
    def eval(self, x):
        """
        Evaluates a given input `x` using the function `func`, updates the best observed
        value and corresponding input if applicable, and increments the evaluation
        counter.

        Args:
            x: Input data to be evaluated by the function `func`.

        Returns:
            The result of evaluating `x` using the function `func`.
        """
        y = self.func(x)
        with self.statMutex:
            if y < self.best_y.value:
                self.best_y.value = y
                self.best_x[:] = x[:]
                #print(str(self.count.value) + " " + str(y) + " " + str(x))
            self.count.value += 1
        return y
    
    def get_best_x(self):
        """
        Returns a copy of the best_x attribute.

        This method provides a copy of the `best_x` attribute to ensure the original
        data remains unaltered and secure from unintended modifications.

        Returns:
            list: A copy of the best_x attribute.
        """
        return self.best_x[:]

    def get_best_y(self):
        """
        Retrieves the value of the best_y attribute.

        This method returns the current value of the `best_y` attribute, which is likely
        used as a part of an optimization process or to represent the best-known value
        achieved during a computation.

        Returns:
            Any: The value of the `best_y` attribute.
        """
        return self.best_y.value
    
    def get_count(self):
        """
        Fetches the current value of the count.

        The method retrieves the current value associated with the `count` attribute. The returned
        value represents the current count stored within the object.

        Returns:
            int: The current value of the count.
        """
        return self.count.value
    
class Rosen(object):
    """
    Represents the Rosen class, which encapsulates details and functionality
    specific to Rosen operations.

    This class is designed to initialize and handle Rosen-related computations
    and configurations. It sets up the initial parameters based on given
    dimensions to facilitate further processes.

    Attributes:
        dim (int): The dimensionality for initializing the Rosen computation.
    """
    def __init__(self, dim):
        """
        Initializes an instance of the class with a given dimensionality, setting up
        specific bounds and objective function.

        Args:
            dim: The dimensionality of the problem. Specifies the size of bounds and
                other configurations.
        """
        _testfun.__init__(self, 'rosen', _rosen, [-5]*dim, [5]*dim)

class Elli(object):
    """Represents an Elli object.

    This class is used to initialize and manage Elli objects with specified
    dimensionality. The purpose of the Elli object is primarily related to its
    usage with a range of defined lower and upper bounds.

    Attributes:
        dim (int): The dimensionality of the Elli object.
    """
    def __init__(self, dim):
        """
        Initializes an instance of the class and sets up the function parameters.

        Args:
            dim: The dimensionality of the input space for the function.
        """
        _testfun.__init__(self, 'elli', _elli, [-5]*dim, [5]*dim)

class Cigar(object):
    
    def __init__(self, dim):    
        _testfun.__init__(self, 'cigar', _cigar, [-5]*dim, [5]*dim)

class Sphere(object):
    """
    Represents a Sphere object.

    The Sphere object serves as a conceptual representation for a mathematical
    sphere within a certain dimension. It defines its boundaries and provides
    appropriate initialization for creating a sphere.

    Attributes:
        dim (int): The dimensionality of the sphere, specifying how many
            dimensions the sphere spans.
    """
    def __init__(self, dim):
        """
        Initializes an instance with given dimensional constraints for a 'sphere' type function.

        Args:
            dim (int): Dimensionality of the space to be initialized.
        """
        _testfun.__init__(self, 'sphere', _sphere, [-5]*dim, [5]*dim)
  
class Rastrigin(object):
    """
    Represents the Rastrigin function commonly used in optimization problems.

    The Rastrigin function is a non-convex function used as a performance test
    problem for optimization algorithms. It is a typical example of non-linear
    multimodal functions and is highly multimodal, meaning it has many local
    minima. This class initializes the Rastrigin function with specified
    dimensionality and its domain. It inherits from a base function class.

    Attributes:
        name (str): The name of the optimization function.
        domain (list of float): The lower and upper bounds for each dimension
            of the input domain.
    """
    def __init__(self, dim):
        """
        Initializes the given object configuration for the "rastrigin" function.

        Args:
            dim (int): Dimensionality of the function, defining the number of variables
                the function considers during evaluation.
        """
        _testfun.__init__(self, 'rastrigin', _rastrigin, [-5.12]*dim, [5.12]*dim)

class Eggholder(object):
    """
    Represents the Eggholder optimization function.

    The Eggholder function is a mathematical function commonly used for testing
    optimization algorithms. It is characterized by a complex surface with
    several local minima and a global minimum. The function is typically used
    within the specified domain and is considered challenging due to its rugged
    nature and numerous local optima.

    Attributes:
        name (str): Name of the function, set to 'eggholder'.
        function (callable): The mathematical representation of the Eggholder
            function.
        lower_bounds (List[int]): Lower bounds for the function's domain, set
            to [-512, -512].
        upper_bounds (List[int]): Upper bounds for the function's domain, set
            to [512, 512].
    """
    def __init__(self):
        """
        Initializes an object with specific function parameters and bounds.

        This constructor calls the parent class initializer and sets up the
        parameters for a specific optimization function, including its name,
        function reference, and upper/lower bounds.

        Args:
            None

        Returns:
            None
        """
        _testfun.__init__(self, 'eggholder', _eggholder, [-512]*2, [512]*2)

class RastriginMean(object):
    """
    Represents a Rastrigin mean test function.

    This class defines a Rastrigin mean function, typically used as a benchmark
    function in optimization problems. The aim of this function is to evaluate
    and analyze the performance of optimization algorithms. It initializes the
    function with a given dimension and a specific parameter `n`.

    Attributes:
        dim (int): The dimension of the function space.
        n (int): A specific parameter that influences the Rastrigin mean function.
    """
    def __init__(self, dim, n):
        """
        Initializes an instance of the Rastrigin mean test function.

        Args:
            dim: Dimensionality of the input vector.
            n: Number of components used in calculating the mean.
        """
        fun = lambda x: _rastrigin_mean(x, n)
        _testfun.__init__(self, 'rastrigin_mean', fun, [-5.12]*dim, [5.12]*dim)

class _testfun(object):
    """
    Represents a test function, its name, functional implementation, and bounds.

    This class encapsulates information about a test function, including its
    name, the function itself, its variable bounds, and its wrapper for
    managing execution.

    Attributes:
        name (str): Name of the test function.
        fun (Callable): The functional implementation of the test function.
        bounds (Bounds): Boundaries within which the function operates.
        wrapper (Wrapper): Wrapper object for managing function execution and
            variable handling.
    """
    def __init__(self, name, fun, lower, upper):
        """
        Initializes an instance of the class with a name, a function, and bounds for
        the variables.

        Args:
            name: str. The name associated with the instance.
            fun: Callable. The main function provided as input.
            lower: Sequence[float]. The lower bounds for the function's input variables.
            upper: Sequence[float]. The upper bounds for the function's input variables.
        """
        self.name = name 
        self.fun = fun
        self.bounds = Bounds(lower, upper)
        self.wrapper = Wrapper(self.fun, len(lower))   
    
def _rosen(xs, alpha=1e2):
    """
    Computes the Rosenbrock function for the given input.

    The Rosenbrock function is a non-convex mathematical function often used
    as a performance test problem for optimization algorithms. This function
    takes a list of numerical inputs and evaluates the Rosenbrock polynomial
    for those inputs. It also supports handling scalars and will return outputs
    as either a scalar or a list depending on the input.

    Args:
        xs: A numerical iterable or scalar. The input values for which the
            Rosenbrock function is to be computed. If the first element of `xs`
            is scalar, it is treated as a single value.
        alpha: An optional numerical value. Represents the parameter to scale
            the quadratic term in the Rosenbrock function. Defaults to 1e2.

    Returns:
        float or list: The evaluated Rosenbrock function result. If the input
        is a scalar or a single list, it returns a scalar result. If the input
        is a list of multiple numerical iterables, it returns a list of computed
        results for each input iterable.
    """
    xs = [xs] if np.isscalar(xs[0]) else xs 
    xs = np.asarray(xs)
    f = [sum(alpha * (x[:-1]**2 - x[1:])**2 + (1. - x[:-1])**2) for x in xs]
    return f if len(f) > 1 else f[0]  # 1-element-list into scalar

def _rastrigin(x):
    """
    Computes the Rastrigin function value for a given input vector.

    The Rastrigin function is commonly used as a non-convex test function for
    optimization algorithms. It is highly multimodal, meaning it has many local
    minima, which makes it a challenging benchmark for optimization techniques.

    Args:
        x (array-like): A vector of input values on which the Rastrigin function
            is evaluated. Must be convertible to a NumPy array.

    Returns:
        float: The computed value of the Rastrigin function for the input vector.
    """
    dim = len(x)
    x = np.asarray(x)
    return 10.0*dim + sum(x*x - 10.0*np.cos(2.0*math.pi*x))

def _cigar(x):
    """
    Calculates the value of the "cigar" function for a given input array.

    The "cigar" function is a common benchmark function in optimization studies.
    It is designed to simulate an elliptical-shaped function with one dominant
    dimension. It is used to evaluate optimization algorithms in high-dimensional
    spaces.

    Args:
        x (array-like): Input array for which the "cigar" function value is to
            be computed.

    Returns:
        float: Calculated "cigar" function value for the given input array.
    """
    factor = 1E6
    x = np.asarray(x)
    return x[0]*x[0] + factor * sum(xi*xi for xi in x);

def _sphere(x):
    """
    Calculates the sum of squares of elements in an input array.

    This function computes the sum of the squares of each element in the
    input array. It is typically used in mathematical or optimization contexts.

    Args:
        x: Input array or iterable whose squared elements will be summated.

    Returns:
        The sum of the squares of the elements in the input array or iterable.

    """
    x = np.asarray(x)
    return sum(xi*xi for xi in x);

def _elli(x):
    """
    Calculates the elliptic function value for the given input.

    The elliptic function is commonly used in optimization problems as a
    benchmark function. It is computed using the input array and a factor
    that exponentially grows based on the dimension of the input.

    Args:
        x (array-like): Input array representing the variables for which
            the elliptic function is calculated.

    Returns:
        float: The computed value of the elliptic function for the input.
    """
    dim = len(x)
    x = np.asarray(x)
    factor = 1E6
    f = 0
    for i in range(dim):
        f += factor ** (i / (dim - 1.)) * x[i] * x[i]
    return f
 
def _modify(x, delta):
    """
    Applies a random modification to an array.

    This function takes an array as input, and applies a random modification
    to each of its elements using a Gaussian (normal) distribution scaled
    by a specified delta factor.

    Args:
        x (list or np.ndarray): Input array to be modified.
        delta (float): Scaling factor for the random noise to be added.

    Returns:
        list: A new list containing the modified elements.
    """
    dim = len(x)
    modified = np.asarray(x) + delta * np.random.randn(dim)
    return modified.tolist()

def _rastrigin_mean(x, n):
    """
    Computes the mean output of the Rastrigin function over `n` modifications of
    input `x`. This function applies a slight modification to the input value `x`
    before passing it to `_rastrigin` and accumulates the results to compute
    the mean.

    Args:
        x: The initial input value for the Rastrigin function that will be slightly
            modified for each iteration.
        n: The number of iterations or modifications of input `x` for computing
            the average.

    Returns:
        The mean result of the Rastrigin function after applying it to `n`
        modified versions of the input `x`.
    """
    delta = 0.001
    sumy = 0
    for i in range(n):
        sumy += _rastrigin(_modify(x, delta))
    return sumy / n

def _eggholder(x):
    """
    Calculates the Eggholder function value for a given input. The Eggholder
    function is a complex mathematical function often used in optimization
    problems to test the performance of algorithms due to its multiple local
    minima.

    Args:
        x: A list or array-like object with two elements representing the
            input values for the Eggholder function.

    Returns:
        float: The result of evaluating the Eggholder function for the given
            input values.

    """
    return (-(x[1] + 47.0)
                        * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
                        - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
                        )
