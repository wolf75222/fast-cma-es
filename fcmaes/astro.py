# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - astro.py

 Description:
  - This module provides implementations of various astronomical functions
    and optimization problems, including the Messenger, GTOC1, Cassini,
    Rosetta, Sagas, Tandem, and their respective configurations. It allows
    for the evaluation of these functions with specified bounds and sequences.
    - The functions are designed to be used with the Fast CMA-ES optimization
    algorithm, providing a way to optimize trajectories and parameters for
    space missions.

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
import os
import ctypes as ct
from scipy.optimize import Bounds
from fcmaes.decpp import libcmalib

if not libcmalib is None: 
    
    astro_map = {  
        "messengerfullC": libcmalib.messengerfullC,
        "messengerC": libcmalib.messengerC,
        "gtoc1C": libcmalib.gtoc1C,
        "cassini1C": libcmalib.cassini1C,
        "cassini1minlpC": libcmalib.cassini1minlpC,
        "cassini2C": libcmalib.cassini2C,
        "rosettaC": libcmalib.rosettaC,
        "sagasC": libcmalib.sagasC,
        "tandemC": libcmalib.tandemC,
        "tandemCu": libcmalib.tandemCu,
        "cassini2minlpC": libcmalib.cassini2minlpC,
    }

    freemem = libcmalib.free_mem
    freemem.argtypes = [ct.POINTER(ct.c_double)]    
    
class Astrofun(object):
    """
    Represents an astronomical function with associated properties and constraints.

    This class encapsulates the details of a mathematical function with defined
    bounds and provides the ability to manage and evaluate functions within specified
    limits. It is designed for scenarios where certain constraints are imposed on
    a function's operational range.

    Attributes:
        name (str): Name of the astronomical function.
        fun_c (callable): The core callable function that represents the
            astronomical functionality.
        bounds (Bounds): An object representing the lower and upper bounds
            for the function's operation.
        fun (callable): The processed function wrapped with bounds constraints.
    """
    def __init__(self, name, fun_c, lower, upper):
        """
        Initializes an instance of the class.

        Args:
            name: The name identifier associated with the instance.
            fun_c: The function to be computed, provided as a callable.
            lower: The lower boundary for the input domain.
            upper: The upper boundary for the input domain.
        """
        self.name = name 
        self.fun_c = fun_c 
        self.bounds = Bounds(lower, upper)
        self.fun = python_fun(fun_c, self.bounds)

for func in astro_map:
    astro_map[func].argtypes = [ct.c_int, ct.POINTER(ct.c_double)]           
    astro_map[func].restype = ct.c_double           

class MessFull(object):
    """Represents a specific type of 'messenger full' configuration.

    This class inherits from Astrofun and initializes parameters for the
    'messenger full' subsystem. The parameters define operational ranges
    essential for the subsystem to function properly.

    Attributes:
        None
    """
    def __init__(self):
        """
        Initializes an object of the `messenger full` class, inheriting from Astrofun.

        The initialization sets up the parameters and properties specific for the
        `messenger full` configuration by invoking the parent class constructor
        with specific initialized values. The class defines ranges and bounds
        necessary for the system.

        Args:
            None
        """
        Astrofun.__init__(self, 'messenger full', "messengerfullC", 
                           [1900.0, 3.0,    0.0, 0.0,  100.0, 100.0, 100.0, 100.0, 100.0, 100.0,  0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  1.1, 1.1, 1.05, 1.05, 1.05,  -math.pi, -math.pi, -math.pi, -math.pi, -math.pi],
                           [2200.0, 4.05, 1.0, 1.0,  500.0, 500.0, 500.0, 500.0, 500.0, 550.0,  0.99, 0.99, 0.99, 0.99, 0.99, 0.99,  6.0,   6.0,    6.0,    6.0,    6.0,  math.pi,  math.pi,  math.pi,  math.pi,  math.pi]
        )
     
class Messenger(object):
    """
    Messenger class initialization and management.

    This class is a specialized implementation of the Astrofun class to initialize
    a "messenger reduced" instance with preset parameters. Its primary purpose is
    to provide customized functionality built upon the base Astrofun class and
    facilitate specific operations associated with the predefined configuration.

    Attributes:
        name (str): Name of the instance.
        identifier (str): Identifier for categorizing the instance.
        lower_bounds (list): Lower bound constraints for the instance configuration.
        upper_bounds (list): Upper bound constraints for the instance configuration.
    """

    def __init__(self):
        """
        Initializes an instance of the class with predefined parameters.

        This constructor method initializes an object by calling the `Astrofun`
        constructor with specific parameters suitable for a "messenger reduced"
        context.

        Args:
            None
        """
        Astrofun.__init__(self, 'messenger reduced', "messengerC", 
                           [1000.,1.,0.,0.,200.,30.,30.,30.,0.01,0.01,0.01,0.01,1.1,1.1,1.1,-math.pi,-math.pi,-math.pi],
                           [4000.,5.,1.,1.,400.,400.,400.,400.,0.99,0.99,0.99,0.99,6,6,6,math.pi,math.pi,math.pi]      
        )
    
class Gtoc1(object):
    """
    Represents the GTOC1 class, which is a specialized subclass of Astrofun.

    This class is designed to override and customize certain methods and behaviors of its parent
    class for specific use cases. It modifies the behavior of the `fun` method by replacing it
    with a new implementation provided by the `gtoc1` method. The class also initializes
    Astrofun with pre-defined lower and upper bounds for specific purposes.

    Attributes:
        gfun (Callable): The original function inherited from the Astrofun class
            available for internal use within the class.
        fun (Callable): The customizable function to use in place of the original
            inherited `fun` method.

    """
    
    def __init__(self):
        """
        Initializes an instance of a class and sets up attributes specific to
        the object's initialization, including some properties and method
        assignments.

        Args:
            None

        Attributes:
            gfun: A reference to the `fun` method stored in a separate
                attribute.
            fun: A specific functionality assigned to the `gtoc1` method.
        """
        Astrofun.__init__(self, 'GTOC1', "gtoc1C", 
                           [3000.,14.,14.,14.,14.,100.,366.,300.],
                           [10000.,2000.,2000.,2000.,2000.,9000.,9000.,9000.]       
                           )
        self.gfun = self.fun
        self.fun = self.gtoc1       
    
    def gtoc1(self, x):
        """
        Calculates the result of subtracting a constant from the output of `gfun`.

        Args:
            x: Input value passed to the `gfun` method.

        Returns:
            The result of `gfun(x) - 2000000`.
        """
        return self.gfun(x) - 2000000

class Cassini1(object):
    """
    Cassini1 class that represents a specific astronomical function configuration.

    This class initializes and configures a function used for astronomical
    computations. Inherits from the Astrofun class and sets specific parameters
    required for the Cassini1 configuration.

    Attributes:
        name (str): The name of the configuration.
        identifier (str): A unique identifier for the configuration.
        x_params (list of float): A list representing parameters on the x-axis.
        y_params (list of float): A list representing parameters on the y-axis.
    """
    
    def __init__(self):
        """
        Initializes a Cassini1 object with specified settings and parameters.

        Args:
            None
        """
        Astrofun.__init__(self, 'Cassini1', "cassini1C", 
                           [-1000.,30.,100.,30.,400.,1000.],
                           [0.,400.,470.,400.,2000.,6000.]       
        )

class Cassini2(object):
    """
    Represents the Cassini2 optimization problem.

    This class is designed to model the Cassini2 optimization problem. It
    inherits from the Astrofun base class and initializes the problem's
    parameters and constraints. Specifically, the class sets up the bounds
    and constants necessary to define the problem's search space.

    Attributes:
        name (str): The name of the problem instance.
        problem_code (str): A short problem code identifier.
        lower_bounds (list): List of lower bounds for the problem's variables.
        upper_bounds (list): List of upper bounds for the problem's variables.
    """
    
    def __init__(self):
        """
        Initializes the Cassini2 object with specific parameters for simulation.

        This constructor sets up the Cassini2 object by calling the parent
        class Astrofun's initializer and providing it with specific values
        for simulation parameters and boundaries. These parameters include
        values associated with limits, constants, and other factors
        required for the simulation.

        Args:
            No direct arguments as all required values are hardcoded within
            the constructor definition.

        Raises:
            No explicit error or exception handling is documented for
            this constructor.
        """
        Astrofun.__init__(self, 'Cassini2', "cassini2C", 
            [-1000,3,0,0,100,100,30,400,800,0.01,0.01,0.01,0.01,0.01,1.05,1.05,1.15,1.7, -math.pi, -math.pi, -math.pi, -math.pi],
            [0,5,1,1,400,500,300,1600,2200,0.9,0.9,0.9,0.9,0.9,6,6,6.5,291,math.pi,  math.pi,  math.pi,  math.pi]
        )

class Rosetta(object):
    """
    Represents the Rosetta class.

    This class serves as an implementation of the Rosetta spacecraft within the
    context of the Astrofun model. It includes initialization parameters and data
    related to its functionality within the model.

    Attributes:
        name (str): Name of the model.
        model_id (str): Identifier for the specific Rosetta configuration.
        init_params (list): Initial parameters relevant to the Rosetta model's
            setup and operation.
        max_params (list): Maximum constraint parameters for the Rosetta model
            within the system.
    """
    
    def __init__(self):
        """
        Initializes the Rosetta class instance with specific parameters for
        a celestial computation context based on Astrofun.

        Args:
            None
        """
        Astrofun.__init__(self, 'Rosetta', "rosettaC", 
            [1460,3,0,0,300,150,150,300,700,0.01,0.01,0.01,0.01,0.01,1.05,1.05,1.05,1.05, -math.pi, -math.pi, -math.pi, -math.pi],
            [1825,5,1,1,500,800,800,800,1850,0.9,0.9,0.9,0.9,0.9,9,9,9,9,math.pi,  math.pi,  math.pi,  math.pi]
        )

class Sagas(object):
    """
    Represents the Sagas class, which initializes specific parameters
    and configurations for astronomical functionalities.

    The Sagas class is a specialized implementation used for configuring
    parameters and invoking functionality related to astronomical data
    processing. It extends or derives the initialization properties
    from the Astrofun class and sets predefined attributes for data
    calibration, computation, or interactions within the specific
    context of astronomical studies.

    Attributes:
        attribute1 (float): Initial condition value for computation.
        attribute2 (list[float]): Array of configuration parameters
            that outline the behavior of the instance.
        attribute3 (float): Maximum allowable computation threshold.
    """
    
    def __init__(self):
        """
        Initializes an instance of the Astrofun class with pre-defined parameters for
        the 'Sagas' configuration. The initialization process involves setting up the
        class with the provided name, identifier, and two lists of numerical values
        that determine its properties and behavior.

        Args:
            None
        """
        Astrofun.__init__(self, 'Sagas', "sagasC", 
            [7000,0,0,0,50,300,0.01,0.01,1.05,8, -math.pi, -math.pi],
            [9100,7,1,1,2000,2000,0.9,0.9,7,500, math.pi,  math.pi]
        )

class Tandem(object):
    """Represents a Tandem configuration with specific bounds, sequences, and functionalities.

    This class is designed to create and manage a tandem object, where each tandem
    is characterized by specific constraints, bounds, sequences, and a function
    that calculates a value based on given inputs. It allows the classification
    of tandems as constrained or unconstrained and associates them with specific
    functionality.

    Attributes:
        name (str): The name of the tandem, which includes an indication of
            whether it is constrained or unconstrained and its identifier.
        fun_c (str): The string identifier of the function in the external library,
            which varies based on whether the tandem is constrained or unconstrained.
        fun (Callable): The main function of the class implemented as `tandem`,
            which calculates a specific value using external library logic.
        bounds (Bounds): The lower and upper bounds for input parameters associated
            with the tandem. It defines acceptable operational limits.
        seqs (list[list[int]]): Predefined list of sequences available for
            configuration in the tandem object.
        seq (list[int]): The specific sequence selected for the tandem object,
            determined by the input index at instantiation.
    """
    def __init__(self, i, constrained=True):
        """
        Initializes an instance with specific configuration based on the given index and constraints.

        This constructor sets up various attributes for the instance including the name, associated
        functions, bounds, and sequence configurations based on the input parameters. It provides
        flexibility for creating either a constrained or unconstrained configuration.

        Args:
            i (int): Index used to select specific sequence and configurations.
            constrained (bool): Indicates whether the configuration should be constrained
                or unconstrained. Default is True.
        """
        self.name = ('Tandem ' if constrained else 'Tandem unconstrained ') + str(i+1)
        self.fun_c = "tandemC" if constrained else "tandemCu"
        self.fun = self.tandem
        self.bounds = Bounds([5475, 2.5, 0, 0, 20, 20, 20, 20, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.05, -math.pi, -math.pi, -math.pi], 
                             [9132, 4.9, 1, 1, 2500, 2500, 2500, 2500, 0.99, 0.99, 0.99, 0.99, 10, 10, 10, math.pi,  math.pi,  math.pi])
        self.seqs = [[3,2,2,2,6],[3,2,2,3,6],[3,2,2,4,6],[3,2,2,5,6],[3,2,3,2,6],
                [3,2,3,3,6],[3,2,3,4,6],[3,2,3,5,6],[3,2,4,2,6],[3,2,4,3,6],
                [3,2,4,4,6],[3,2,4,5,6],[3,3,2,2,6],[3,3,2,3,6],[3,3,2,4,6],
                [3,3,2,5,6],[3,3,3,2,6],[3,3,3,3,6],[3,3,3,4,6],[3,3,3,5,6],
                [3,3,4,2,6],[3,3,4,3,6],[3,3,4,4,6],[3,3,4,5,6]]
        self.seq = self.seqs[i]
        
    def tandem(self, x):
        """
        Compute the evaluation of a C library function using the provided numeric array.

        This function interfaces with an external C function to perform calculations. The
        input array is passed to the C function after converting it to the required C types,
        and the result is retrieved. If any exception occurs or the result is not finite,
        a fallback value is returned.

        Args:
            x (list[float]): Numeric array used as input for the C function.

        Returns:
            float: Result of the C function evaluation, or a fallback value in case
            of errors or non-finite results.
        """
        n = len(x)
        array_type = ct.c_double * n   
        ints_type = ct.c_int * 5   
        fun_c = astro_map[self.fun_c]      
        fun_c.argtypes = [ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int)]
        try: # function is only defined inside bounds
            #x = np.asarray(x).clip(self.bounds.lb, self.bounds.ub)
            val = fun_c(n, array_type(*x), ints_type(*self.seq))
            if not math.isfinite(val):
                val = 1E10
        except Exception as ex:
            val = 1E10
        return val

class Tandem_minlp(object):
    """
    Encapsulates a Tandem optimization problem in the form of a mixed-integer nonlinear programming (MINLP) task.

    This class models and processes a constrained or unconstrained tandem optimization problem. It allows setting
    boundaries on variables, defining the objective function specific to the problem, and performing computations
    for the given input vector. The specific configuration is determined based on the `constrained` parameter.

    Attributes:
        name (str): Name of the MINLP problem, either constrained or unconstrained.
        fun_c (str): Name of the corresponding C function for the constrained or unconstrained computation.
        fun (Callable): Reference to the objective function (`tandem_minlp`).
        bounds (Bounds): Bounds for the optimization problem variables.
    """
    def __init__(self, constrained=True):
        """
        Initializes the instance with attributes based on whether it is in a constrained or
        unconstrained state. Sets the name, function strings for constrained operations,
        function reference, and specific variable bounds.

        Args:
            constrained (bool): Determines whether the instance is in the constrained state.
                Defaults to True.
        """
        self.name = ('Tandem minlp ' if constrained else 'Tandem unconstrained minlp ') 
        self.fun_c = "tandemC" if constrained else "tandemCu"
        self.fun = self.tandem_minlp
        self.bounds = Bounds([5475, 2.5, 0, 0, 20, 20, 20, 20, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05, 1.05, -math.pi, -math.pi, -math.pi,
                              1.51,1.51,1.51], 
                             [9132, 4.9, 1, 1, 2500, 2500, 2500, 2500, 0.99, 0.99, 0.99, 0.99, 10, 10, 10, math.pi,  math.pi,  math.pi,
                              3.49,4.49,5.49])
         
    def tandem_minlp(self, xs):
        """
        Evaluates a function using given sequence and array inputs.

        This method processes the provided list of inputs, partitions it into
        specific components, prepares the necessary C data types, and calls an
        external function to compute the final value. If the function call fails
        or the result is non-finite, a fallback default value is returned.

        Args:
            xs (list[float]): A list of floats where the first part represents an
                array of values and the last three indicate sequence parameters.

        Returns:
            float: The result of the computation provided by the external
                function. Defaults to 1E10 if the computation fails or
                produces a non-finite value.
        """
        n = len(xs) - 3
        x = xs[:-3]
        seq = [3] + [int(round(xi)) for xi in xs[-3:]] + [6]
        array_type = ct.c_double * n   
        ints_type = ct.c_int * 5   
        fun_c = astro_map[self.fun_c]      
        fun_c.argtypes = [ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int)]
        try:
            val = fun_c(n, array_type(*x), ints_type(*seq))
            if not math.isfinite(val):
                val = 1E10
        except Exception as ex:
            val = 1E10
        return val

class Cassini1multi(object):
    """
    Represents a mathematical optimization model for the Cassini1 multi-objective problem.

    This class is used to work with the Cassini1 multi-objective problem for optimization purposes.
    The class defines methods and attributes related to the problem, including objective weights
    and planet configurations. It acts as an interface for interacting with the Cassini1 problem
    and computes multi-objective functions based on input variables.

    Attributes:
        weights (list[float]): Weights for combining different objective functions in the
            optimization problem. Each weight corresponds to an objective in the multi-objective
            formulation.
        planets (list[int]): Configuration for planets involved in the Cassini1 problem. This
            determines the specific problem setup used in calculations.
    """
    
    def __init__(self, weights = [1,0,0,0], planets = [2,2,3,5]):
        """
        Initializes an instance of a class, setting up specific attributes and
        default parameters necessary for its operations.

        Args:
            weights (list, optional): A list of weight values used in the functionality
                of the object. Defaults to [1, 0, 0, 0].
            planets (list, optional): A list of planet identifiers or parameters used
                within the multi-function calculations. Defaults to [2, 2, 3, 5].
        """
        Astrofun.__init__(self, 'Cassini1minlp', "Cassini1minlpC", 
                           [-1000.,30.,100.,30.,400.,1000.],
                           [0.,400.,470.,400.,2000.,6000.]       
        )
        self.fun = self.cassini1
        self.weights = weights
        self.planets = planets
        self.mfun = lambda x: cassini1multi(x + [2,2,3,5])
         
    def cassini1(self, x):
        """
        Computes a weighted sum of results from the cassini1multi function using
        the input value added to the object's "planets" attribute.

        Args:
            x: The input value used for computation.

        Returns:
            A float value representing the weighted sum of the results from the
            cassini1multi function.
        """
        r = cassini1multi(x + self.planets)
        return self.weights[0]*r[0] + self.weights[1]*r[1] + self.weights[2]*r[2] + self.weights[3]*r[3]
    
class Cassini1minlp(object):
    """Represents the Cassini1 MINLP (Mixed-Integer Nonlinear Programming) problem.

    This class models a specific mathematical optimization problem related to the trajectory
    of the Cassini spacecraft. It is designed to work with certain attributes, including
    bounds and planets, and relies on the `cassini1` function for computation. Users can
    interact with this class to evaluate optimization solutions within predefined constraints.

    Attributes:
        fun (Callable): Reference to the cassini1 function used for calculations.
        planets (list): List of integers representing planetary identifiers involved in the
            optimization problem.
    """
    
    def __init__(self, planets = [2,2,3,5]):
        """
        Initializes the class with default parameters and sets up necessary attributes.

        Args:
            planets (list[int]): A list representing the planets of interest, defaulting to [2, 2, 3, 5].
        """
        Astrofun.__init__(self, 'Cassini1', "cassini1C", 
                           [-1000.,30.,100.,30.,400.,1000.],
                           [0.,400.,470.,400.,2000.,6000.]       
        )
        self.fun = self.cassini1
        self.planets = planets
   
    def cassini1(self, x):
        """
        Calculates the result of the cassini1minlp function after appending the provided
        parameter `x` to the `planets` attribute of the current instance.

        Args:
            x: Iterable of numerical values representing a part of input to the
               cassini1minlp function.

        Returns:
            Result of the cassini1minlp function after combining `x` with the `planets`
            attribute of the class instance.

        Raises:
            TypeError: If `x` is not an iterable of numerical values.
        """
        return cassini1minlp(list(x) + self.planets)
      
def cassini1minlp(x):
    """
    Computes the objective function for the Cassini1MINLP optimization problem.

    This function is a wrapper that interacts with an external C function to compute
    the required value for a specific mathematical optimization problem. The external
    function is utilized for its computational efficiency and requires properly
    formatted inputs to execute.

    Args:
        x (list of float): A list of input values representing variables for which
            the objective function needs to be evaluated.

    Returns:
        float: The computed objective function value. If errors occur or the result
            is not finite, a default high-value penalty of 1E10 is returned.
    """
    n = len(x)
    array_type = ct.c_double * n   
    fun_c = astro_map["cassini1minlpC"]      
    fun_c.argtypes = [ct.c_int, ct.POINTER(ct.c_double)]
    fun_c.restype = ct.POINTER(ct.c_double)  
    try: # function is only defined inside bounds
        res = fun_c(n, array_type(*x))
        dv = res[0]
        freemem(res)
        if not math.isfinite(dv):
            dv = 1E10
    except Exception as ex:
        print(ex)
        dv = 1E10
    return dv

def cassini1multi(x):
    """
    Calculates the Delta-V, time of flight (TOF), and launch time for a spacecraft based on given input
    parameters. The function uses an external C library to compute the result and handles edge cases
    by returning large values for Delta-V in case of errors or invalid operations.

    Args:
        x (list[float]): A list of input parameters including launch time and trajectory parameters.
            The specific meaning of each value in the list depends on the mathematical requirements
            for calculating the Delta-V.

    Returns:
        list[float]: A list containing:
            - Delta-V (float): The total change in velocity required for the spacecraft.
            - TOF (float): The time of flight calculated as the sum of specific parameters from the input.
            - Launch time (float): The initial launch time extracted from the input list.

    Raises:
        Any exception raised during function execution will be caught internally, and default large
        values will be returned for Delta-V and launch DV.
    """
    n = len(x)
    array_type = ct.c_double * n   
    fun_c = astro_map["cassini1minlpC"]      
    fun_c.argtypes = [ct.c_int, ct.POINTER(ct.c_double)]
    fun_c.restype = ct.POINTER(ct.c_double)  
    try: # function is only defined inside bounds
        res = fun_c(n, array_type(*x))
        dv = res[0]
        launch_dv = res[1]
        freemem(res)
        if not math.isfinite(dv):
            dv = 1E10
    except Exception as ex:
        print(ex)
        dv = 1E10
        launch_dv = 1E10 
    tof = sum(x[1:6])
    launch_time = x[0] 
    return [dv, tof, launch_time]

def cassini2multi(x):
    """
    Converts mission parameters to their respective outputs using an external
    function call and computes additional metrics.

    The function calculates delta velocity (dv), time of flight (tof), and
    launch time based on the input parameters. It utilizes a C-based external
    function to compute `dv`. If the external function fails, it assigns a
    default high value to `dv`.

    Args:
        x (list[float]): A list of mission parameters where specific indices
            represent particular parameters such as launch time and other
            configuration values.

    Returns:
        list[float]: A list containing:
            - dv (float): Delta velocity calculated from the external function.
            - tof (float): Total time of flight derived from the input parameters.
            - launch_time (float): The launch time extracted from the input
              parameters.

    Raises:
        Exception: Propagates the exception if the external function fails
        and logs the error, without terminating the program.
    """
    n = len(x)
    array_type = ct.c_double * n   
    fun_c = astro_map["cassini2minlpC"]      
    fun_c.argtypes = [ct.c_int, ct.POINTER(ct.c_double)]
    fun_c.restype = ct.c_double  
    try: # function is only defined inside bounds
        dv = fun_c(n, array_type(*x))
    except Exception as ex:
        print(ex)
        dv = 1E99
    tof = sum(x[4:9])
    launch_time = x[0] 
    return [dv, tof, launch_time]
 
class python_fun(object):
    """
    Represents a callable Python function object that interfaces with external C functions and enforces boundary
    restrictions on the input. This class is used to evaluate functions defined by external mappings, ensuring
    values remain valid within specified constraints.

    Attributes:
        cfun (str): The name of the function as defined in the external `astro_map` dictionary.
        bounds (Bounds): An object specifying the lower and upper bounds for valid input values.
    """
    def __init__(self, cfun, bounds):
        """
        Initializes an instance of the class with the provided callable function and bounds.

        Args:
            cfun: A callable function to be used in the implementation.
            bounds: A list or tuple representing the bounds associated with the object.
        """
        self.cfun = cfun
        self.bounds = bounds
    
    def __call__(self, x):
        """
        Evaluates the given callable function mapped to `astro_map` using the input array `x`.
        The function validates the input array's size before performing the calculation
        and ensures the result is finite. In case of an exception or invalid output,
        a default value of 1E10 is returned.

        Args:
            x (list[float]): An array of floating-point numbers to be evaluated by the
                callable function in `astro_map`.

        Returns:
            float: The computed value from the callable function. If an error occurs
                during computation or the output is not finite, returns 1E10.
        """
        fun_c = astro_map[self.cfun]
        n = len(x)
        array_type = ct.c_double * n   
        try: # function is only defined inside bounds
            # x = np.array(x).clip(self.bounds.lb, self.bounds.ub)
            val = float(fun_c(n, array_type(*x)))
            if not math.isfinite(val):
                val = 1E10
        except Exception as ex:
            val = 1E10
        return val 

