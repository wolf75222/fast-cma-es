# -*- coding: utf-8 -*-
"""
=============================================================================

 Fast CMA-ES - version 1.6.11

 (c) 2025 – Dietmar Wolz
 (c) 2025 – Latitude

 License: MIT

 File:
  - journal.py

 Description:
  - Simple Optuna Journal file generating wrapper for single and
    multiple objective fcmaes objective functions.
    Can be used to gain live insight into a long running optimization process.
  - Warning: Only use for slow Hyperparameter optimizations, otherwise the journal file will grow too big.
  - Usage example: [2]; See [3], [4].
    install optuna-dashboard
    - pip install optuna-dashboard
    optional:
    - pip install optuna-fast-fanova gunicorn
    Then call:
    - optuna-dashboard <path_to_journalfile>
    In your browser open:
    - http://127.0.0.1:8080/




 Authors:
  - Dietmar Wolz
  - romain.despoullains@latitude.eu
  - corentin.generet@latitude.eu

 References:
  - [1] https://github.com/dietmarwo/fast-cma-es
  - [2] https://github.com/dietmarwo/fast-cma-es/blob/master/examples/prophet_opt.py
  - [3] https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html
  - [4] https://optuna.readthedocs.io/en/latest/tutorial/20_recipes/011_journal_storage.html

 Documentation:
  -


=============================================================================
"""


from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from scipy.optimize import Bounds
import numpy as np
import multiprocessing as mp
from multiprocessing import Manager
import ctypes as ct
import json
import sys

from datetime import datetime

@dataclass
class Base_message:
    """
    Represents the base structure of a message.

    This class serves as a base for creating message objects. It defines common
    attributes that are essential for message handling, such as operation code and
    worker identifier. Intended to be used as a foundational data structure for
    various message-related operations.

    Attributes:
        op_code (int): The operation code indicating the type or purpose of
            the message.
        worker_id (str): The unique identifier corresponding to the worker
            associated with the message.
    """
    op_code: int
    worker_id: str

@dataclass
class Study_start(Base_message):
    """
    Represents the initial message to start a study.

    This class is used to define the starting parameters of a study.
    It holds critical information such as the name of the study and the
    directions for the study. The class ensures clear encapsulation of
    necessary data to initiate and guide the study process.

    Attributes:
        study_name (str): Name of the study.
        directions (List[int]): A list of integers representing the
            directions for the study.
    """
    study_name: str
    directions: List[int]

@dataclass
class Trial_start(Base_message):
    """
    Represents the start of a trial within a specific study.

    Provides details about the study identifier and the start date-time
    when the trial began.

    Attributes:
        study_id (int): Unique identifier of the study associated with
            the trial.
        datetime_start (str): Start date and time of the trial in ISO
            date-time format.
    """
    study_id: int
    datetime_start: str  # ISO format date-time

@dataclass
class Trial_param(Base_message):
    """
    Represents the parameters associated with a specific trial.

    This class is used to encapsulate details about a single trial parameter, with
    its unique identifier, name, internal value, and associated distribution.
    It extends from the Base_message class.

    Attributes:
        trial_id (int): Unique identifier for the trial to which this parameter belongs.
        param_name (str): Name of the trial parameter.
        distribution (Dict[str, Any]): A dictionary defining the distribution of the parameter values.
    """
    trial_id: int
    param_name: str
    param_value_internal: float
    distribution: Dict[str, Any]  # Represented as a dictionary

@dataclass
class Trial_value(Base_message):
    """Represents the value of a single trial in an experiment.

    This class is used to encapsulate and manage data regarding a specific
    trial, including its unique identifier, state, associated values, and
    completion date and time. It is intended to facilitate organization and
    manipulation of trial-related data, often in the context of optimization
    experiments or similar workflows.

    Attributes:
        trial_id (int): The unique identifier of the trial.
        state (int): The current state of the trial, typically represented
            as an integer mapping to specific status codes.
        values (List[float]): A list of values or outcomes associated with
            the trial. These could represent results of the trial or
            metrics of evaluation.
        datetime_complete (str): The completion date and time of the trial
            represented in ISO 8601 format.
    """
    trial_id: int
    state: int
    values: List[float]
    datetime_complete: str  # ISO format date-time

def message_to_json(message):
    """
    Converts a given message object to its JSON representation.

    This function serializes a message object based on its type. If the message type is
    `Trial_param`, its `distribution` field is serialized as a JSON string before converting
    the entire object to JSON format. For other message types, the object is directly
    converted to JSON format.

    Args:
        message: The message object to be serialized to JSON.

    Returns:
        str: A JSON string representation of the message.
    """
    if isinstance(message, Trial_param):
        data = asdict(message)
        # Serialize 'distribution' field as JSON string
        data['distribution'] = json.dumps(data['distribution'])
        return json.dumps(data, separators=(',', ':'))
    else:
        data = asdict(message)
        return json.dumps(data, separators=(',', ':'))
    
def distribution(low, high):
    """
    Generates a JSON representation of a float distribution within a specified range.

    The function creates a JSON object representing a uniform float distribution
    with specified lower and upper bounds. The resulting JSON is parsed and returned
    as a Python dictionary.

    Args:
        low (float): The lower bound of the distribution range.
        high (float): The upper bound of the distribution range.

    Returns:
        dict: A dictionary representation of the float distribution.
    """
    distribution_str =  f'{{"name": "FloatDistribution", "attributes": {{"step": null, "low": {low}, "high": {high}, "log": false}}}}'
    return json.loads(distribution_str)

def study_start(worker_id, study_name, dir):
    """
    Creates and returns a JSON representation of a study start message.

    This function builds a study start message using provided worker identification,
    study name, and study directions. If the `dir` parameter is scalar, it is converted
    into a list; otherwise, it is used as is. The message is then serialized into a
    JSON string for further communication or processing.

    Args:
        worker_id (int): Unique identifier for the worker initiating the study.
        study_name (str): Name of the study being started.
        dir (Union[str, List[str]]): Direction(s) for the study. Can be a single
            direction (string) or a list of directions.

    Returns:
        str: JSON-encoded representation of the constructed study start message.
    """
    msg = Study_start(
        op_code=0,
        worker_id=worker_id,
        study_name=study_name, 
        directions=[dir] if np.isscalar(dir) else dir,
    )
    return message_to_json(msg)

def trial_param(worker_id, trial_id, param_name, param_value_internal, low, high):
    """
    Generates a JSON message containing trial parameter details.

    This function creates a message using the given trial parameters and returns
    its JSON representation. The message contains information such as worker ID,
    trial ID, parameter name, internal parameter value, and the specified
    distribution range.

    Args:
        worker_id: The unique identifier for the worker.
        trial_id: The unique identifier for the trial.
        param_name: The name of the parameter.
        param_value_internal: The internal value of the parameter to be passed.
        low: The lower bound of the parameter's range.
        high: The upper bound of the parameter's range.

    Returns:
        str: A JSON-formatted string with the details of the trial parameter.
    """
    msg = Trial_param(
        op_code=5,
        worker_id=worker_id,
        trial_id=trial_id,
        param_name=str(param_name),
        param_value_internal=param_value_internal,
        distribution=distribution(low, high)
    )
    return message_to_json(msg)

def trial_start(worker_id, study_id):
    """
    Starts a trial by recording the worker ID, study ID, and the current timestamp, and
    generates a JSON message representation of the trial start event.

    Args:
        worker_id (int): Unique identifier for the worker initiating the trial.
        study_id (int): Unique identifier for the study related to the trial.

    Returns:
        str: JSON representation of the trial start event.
    """
    datetime_str = datetime.now().isoformat()
    msg = Trial_start(
        op_code=4,
        worker_id=worker_id,
        study_id=study_id, 
        datetime_start=datetime_str
    )
    return message_to_json(msg)

def trial_value(worker_id, trial_id, y):
    """
    Creates and returns a JSON representation of a trial value message.

    The function generates a message object for a specific trial and worker,
    incorporating information about the trial's state, its values, and a
    timestamp indicating when the operation was completed. It converts the
    message to a JSON format before returning.

    Args:
        worker_id (int): Identifier for the worker responsible for the trial.
        trial_id (int): Identifier for the specific trial whose value is being
            processed.
        y (Union[float, list[float]]): Numeric value(s) associated with the trial.
            This can be a scalar value or a list of values.

    Returns:
        str: JSON string representation of the trial value message.
    """
    datetime_str = datetime.now().isoformat()
    msg = Trial_value(
        op_code=6,
        worker_id=worker_id,
        trial_id=trial_id,
        state=1,
        values=[y] if np.isscalar(y) else y,
        datetime_complete=datetime_str
    )
    return message_to_json(msg)

class Journal:
    """
    Handles logging of study, trial, parameter, and value information to a file.

    This class facilitates logging structured data for studies and their associated
    trials into a file. Each method corresponds to specific types of log entries such
    as study start, trial start, trial parameters, and trial values. It ensures the
    data is written in real-time and provides methods for handling batch operations
    where multiple workers or trials need to be recorded.

    Attributes:
        filename (str): Path to the file where journal entries are logged.
        file (TextIO): File object used for writing journal entries.
    """
    def __init__(self, filename, study_name, dir):
        """
        Initializes an instance for managing file operations and conducting initial study setup.

        Args:
            filename (str): Name of the file to be created and written into.
            study_name (str): Name of the study to be initialized in the setup process.
            dir (str): Directory path for the study workspace or initialization.
        """
        self.filename = filename
        self.file = open(self.filename, 'w')
        self.study("main", study_name, dir)
    
    def study(self, worker_id, study_name, dir):
        """
        Writes the start of a study to a file.

        This method appends a formatted string, indicating the start of a study,
        to the specified file and flushes the file buffer to ensure the data is
        written to disk immediately.

        Args:
            worker_id: Worker identifier as a string or integer.
            study_name: Name of the study as a string.
            dir: Directory or path related to the study as a string.
        """
        self.file.write(study_start(worker_id, study_name, dir) + '\n')
        self.file.flush()  # Ensure that data is written to disk

    def trial(self, worker_id, study_id):
        """
        Initiates a trial session by logging the start of the trial and ensuring
        that the log data is immediately written to disk.

        Args:
            worker_id: Unique identifier for the worker participating in the trial.
            study_id: Unique identifier for the study to which the trial belongs.
        """
        self.file.write(trial_start(worker_id, study_id) + '\n')
        self.file.flush()  # Ensure that data is written to disk
    
    def param(self, worker_id, trial_id, param_name, param_value_internal, low, high):
        """
        Writes a parameter value to the associated file for a given trial and worker.

        This method serializes the parameter information using the `trial_param`
        function and writes the resulting string representation to the file object
        associated with this class. Additionally, it ensures that the data is promptly
        written to disk.

        Args:
            worker_id: Identifier of the worker requesting the execution.
            trial_id: Identifier of the trial being executed.
            param_name: Name of the parameter to be used in the computation.
            param_value_internal: Internal representation or reference of the parameter value.
            low: The lower bound of the parameter's acceptable range.
            high: The upper bound of the parameter's acceptable range.
        """
        self.file.write(trial_param(worker_id, trial_id, param_name, param_value_internal, low, high) + '\n')
        self.file.flush()  # Ensure that data is written to disk

    def value(self, worker_id, trial_id, y):
        """
        Writes the trial value to a file and ensures that data is written to disk.

        Args:
            worker_id: Identifier for the worker.
            trial_id: Identifier for the trial.
            y: Value associated with the trial.

        """
        self.file.write(trial_value(worker_id, trial_id, y) + '\n')
        self.file.flush()  # Ensure that data is written to disk
        
    def write_x(self, worker_id, trial_id, x, bounds):
        """
        Writes parameter values for a given trial and worker.

        This method iterates through the input vector `x` and assigns parameter values
        for corresponding indices, using lower and upper bounds provided in `bounds`.

        Args:
            worker_id: Identifier for the worker.
            trial_id: Identifier for the trial.
            x: List or array of parameter values.
            bounds: Object containing lower (`lb`) and upper (`ub`) bounds for the parameters.
        """
        for i, xi in enumerate(x):
            self.param(worker_id, trial_id, i, xi, bounds.lb[i], bounds.ub[i])
        
    def write_xs(self, trial_id, xs, bounds):
        """
        Writes a set of parameter values (xs) for different workers associated with a trial.

        This method distributes parameter values to workers by iterating through the
        list `xs`. Each parameter value in `xs` is assigned to a worker identified by
        an incremental ID. The `write_x` method is invoked for each worker, linking the
        trial, worker, and provided bounds.

        Args:
            trial_id (int): Identifier for the trial to which the parameters belong.
            xs (list): A list of parameter values to be assigned to workers.
            bounds (tuple): Bounds within which the parameter values are constrained.
        """
        for worker_id, x in enumerate(xs):
            self.write_x(str(worker_id+1), trial_id+worker_id, x, bounds)
    
    def write_ys(self, trial_id, ys):
        """
        Writes a list of values to corresponding trial IDs for all workers.

        Args:
            trial_id (int): The base trial ID to which worker IDs and their corresponding
                values will be associated.
            ys (list): A list of values to be written for each worker. The values in this
                list correspond to workers incrementally starting from worker ID 1.
        """
        for worker_id, y in enumerate(ys):
            self.value(str(worker_id+1), trial_id+worker_id, y)

    def write_starts(self, study_id, batch_size):
        """
        Executes a series of trials for a given study ID.

        The method iterates over a range defined by the batch size and performs trials
        by invoking the `trial` method with the respective worker ID and study ID.

        Args:
            study_id (str): The identifier of the study being processed.
            batch_size (int): The number of workers/trials to execute.
        """
        for worker_id in range(batch_size):
            self.trial(str(worker_id+1), study_id)

    def close(self):
        """
        Closes the associated file resource.

        This method safely closes the file resource associated with the instance,
        ensuring that any resources tied to it are properly released. It is crucial
        to call this method after completion of file operations to avoid resource
        leaks or locking issues.

        Raises:
            IOError: If an I/O operation error occurs while closing the file.
        """
        self.file.close()

class journal_wrapper(object):
    """
    A wrapper class for handling journaling operations related to batch processing in a
    parallelized optimization or search study.

    This class is responsible for managing concurrent journal entries that include trial
    start information, parameters, and trial evaluation results. It coordinates the storage,
    formatting, and writing of trial-related data into a journal file to support analytics
    or dashboards for the study. Synchronization mechanisms ensure thread-safe operations
    in multi-process environments.

    Attributes:
        fit (callable): A function that evaluates the input parameters `x` and returns the
            corresponding result.
        bounds (object): An object that contains lower (`lb`) and upper (`ub`) bounds for
            parameters.
        journal (Journal): An instance of the `Journal` class used for managing and writing
            to the journal file.
        study_id (str): A unique identifier for the entire study.
        batch_size (int): The number of trials to process in a batch.
    """

    def __init__(self, fit, bounds, jfname, study_name, study_id, batch_size):
        """
        Initializes the object with the given parameters and sets up the required attributes.

        Args:
            fit: Callable to perform fitting or evaluation tasks.
            bounds: Iterable specifying the bounds or constraints for the fitting
                or evaluation process.
            jfname: String specifying the file name for the journal.
            study_name: String representing the name of the study.
            study_id: Integer representing the identifier of the study.
            batch_size: Integer representing the batch size to be used.
        """
        self.fit = fit
        self.bounds = bounds
        self.journal = Journal(jfname, study_name, 1)
        self.study_id = study_id
        self.batch_size = batch_size
        self.trial_id = mp.RawValue(ct.c_int, 0)
        self.lock = mp.Lock()
        self.mgr = Manager()
        self.reset()

    def reset(self):
        """
        Resets the lists managed by the instance.

        This method reinitializes the lists `starts`, `xs`, and `ys` using the
        manager object to ensure they are shared lists in multiprocess scenarios.

        Args:
            None

        Returns:
            None
        """
        self.starts = self.mgr.list()
        self.xs = self.mgr.list()  
        self.ys = self.mgr.list()  

    def store_start(self, worker_id, study_id):
        """
        Stores the start of a trial by appending the formatted trial start string
        to the `starts` collection. This function helps track when each trial begins
        for a given worker and study.

        Args:
            worker_id: Identifier for the worker who starts the trial.
            study_id: Identifier for the study associated with the trial.
        """
        self.starts.append(trial_start(worker_id, study_id) + '\n')
         
    def store_x(self, worker_id, trial_id, x):
        """
        Stores the parameters of a specific trial for a worker.

        This method takes the worker ID, trial ID, and a list of parameters, formats
        them according to the trial's bounds, and appends the formatted string to the
        internal storage for later use.

        Args:
            worker_id: The ID of the worker to associate with the parameters.
            trial_id: The ID of the trial to associate with the parameters.
            x: A list of numerical parameters to be stored for the given worker and
                trial.
        """
        x_str = ''
        for i, xi in enumerate(x):
            x_str += trial_param(worker_id, trial_id, i, xi, self.bounds.lb[i], self.bounds.ub[i]) + '\n'
        self.xs.append(x_str)

    def store_y(self, worker_id, trial_id, y):
        """
        Stores the provided `y` value, associated with a worker and trial ID, in the list of `ys`.

        Args:
            worker_id (int): The unique identifier of the worker.
            trial_id (int): The unique identifier of the trial.
            y (Any): The value to store, associated with the worker and trial.

        """
        self.ys.append(trial_value(worker_id, trial_id, y) + '\n')
    
    #we need to reorder the journal output to get the dashboard working    
    def __call__(self, x):
        """
        Executes the callable functionality, managing and processing a batch of inputs
        and outputs concurrently, while ensuring proper thread safety and data
        handling.

        Args:
            x: Input value to process and fit.

        Returns:
            The processed output value as a result of `fit(x)`. If an exception occurs,
            returns the maximum possible float value as defined in `sys.float_info.max`.

        Raises:
            Not explicitly raised in documentation but exceptions are caught
            within the method.
        """
        try:
            with self.lock:
                n = self.batch_size
                if len(self.ys) >= n:      
                    for i in range(n):           
                        self.journal.file.write(self.starts.pop(0))
                    for i in range(n): 
                        self.journal.file.write(self.xs.pop(0))
                    for i in range(n): 
                        self.journal.file.write(self.ys.pop(0))
                    self.journal.file.flush()
                trial_id = self.trial_id.value
                self.trial_id.value += 1

            worker_id = str(trial_id % self.batch_size)
            self.store_start(worker_id, self.study_id)
            self.store_x(worker_id, trial_id, x)
            y = self.fit(x)
            self.store_y(worker_id, trial_id, y)                
            return y
        except Exception as ex:
            print(str(ex))  
            return sys.float_info.max  
        