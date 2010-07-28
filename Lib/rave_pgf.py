'''
Copyright (C) 2010- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
'''
## RAVE Product Generation Framework

## @file
## @author Daniel Michelson, SMHI
## @date 2010-07-09

import sys, os, traceback
from copy import deepcopy as copy
import logging
import Queue
import rave_pgf_logger
import rave_pgf_registry
import rave_pgf_qtools
import BaltradFrame
import _pyhl
from rave_defines import DEX_SPOE, DEX_CHANNEL, DEX_USER, LOG_ID, REGFILE


METHODS = {'generate' : '("algorithm",[files],[arguments])',
           'register': '("name", "module", "function", Help="", strings=",", ints=",", floats=",", seqs=",")',
           'deregister': '("name")',
           'Help': ''
           }


## The product generation framework class containing the product generation
# functionality.
class RavePGF():

  ## Constructor
  def __init__(self):
    self._jobid = 0
    self.logger = logging.getLogger(LOG_ID)
    self._init_logger()
    self._algorithm_registry = rave_pgf_registry.PGF_Registry(filename=REGFILE)
    self.queue = Queue.PriorityQueue()
    self._load_queue()
    self._run_all_jobs()


  ## This method must exist for system.listMethods to work.
  def _listMethods(self):
    self.logger.info("Someone called _listMethods.")
    return METHODS.keys()


  ## This method must exist for system.listMethods to work.
  # @param method string name of method for which to get help.
  def _methodHelp(self, method):
    self.logger.info("Someone called _methodHelp.")
    return "%s %s" % (method, METHODS[method])


  ## Built-in algorithm query.
  # @param name string, optionally the name of the algorithm being queried.
  # @return string a help text comprising the names of each registered
  # algorithm and its descriptive text.
  def Help(self, name=None):
    self.logger.info("Someone needs help.")
    return self._algorithm_registry.Help(name)


  ## Initializes the logger.
  def _init_logger(self):
    rave_pgf_logger.init_logger(self.logger)


  ## Registers a new item in the registry. Re-registering an item
  # is done in the same way.
  # @param name string algorithm's name.
  # @param module string name of the module to import.
  # @param function string name of the function to run in the module.
  # @param Help string containing explanatory text for this registry entry.
  # @param strings string of comma-separated argument names that are strings.
  # @param ints string of comma-separated argument names that are integers.
  # @param floats string of comma-separated argument names that are floats.
  # @param seqs string of comma-separated argument names that are sequences.
  # @return string
  def register(self, name, module, function, Help="",
               strings="", ints="", floats="", seqs=""):
    self.logger.info("Registering algorithm: %s" % name)
    self.deregister(name)
    self._algorithm_registry.register(name, module, function, Help=Help,
                                      strings=strings, ints=ints,
                                      floats=floats, seqs=seqs)
    return "Registered %s" % name


  ## Re-loads the generation registry, assuming it's been modified on file.
  # Is this ever needed or used?
  def update_registry(self):
    self._algorithm_registry = rave_pgf_registry.PGF_Registry(REGFILE)


  ## De-registers a new item in one of the registries.
  # @param name string name of the registry entry to de-register.
  # @return string
  def deregister(self, name):
    self.logger.info("De-registering algorithm: %s" % name)
    self._algorithm_registry.deregister(name)
    return "De-registered %s" % name


  ## Queue a job for processing. All jobs are calls to the \ref generate method.
  # @param algorithm, an Element object that contains all the information
  # required to run a job.
  # @param files, a list of input file strings.
  # @param arguments, a list of argument key-value pairs.
  # @param jobid int unique ID of that job.
  # @param priority, int the priority of the job, should be 0, 1, or 2, where 0
  # is the highest and 2 is the lowest.
  # @return nothing
  def _queue_job(self, algorithm, files, arguments, jobid, priority=0):
    # Once the merge files and arguments have been merged with the algorithm,
    # the result is referred to as a 'job'.
    rave_pgf_qtools.merge(algorithm, files, arguments, jobid, priority)
    self.queue.put((priority, algorithm))
    mod_name, func_name = algorithm.get('module'), algorithm.get('function')
    self.logger.info("ID=%i Queued %s.%s" % (jobid, mod_name, func_name))
    self._jobid += 1


  ## Dumps the job queue to XML file. Called automatically when the server
  # is stopped.
  def _dump_queue(self):
    self.logger.info("Dumping job queue containing %i jobs."%self.queue.qsize())
    rave_pgf_qtools.dump_queue(self.queue)
    self.logger.info("Shutting down.\n")


  ## Loads the job queue from XML file. Called automatically when the server
  # is started, even if the queue is empty.
  def _load_queue(self):
    self.logger.info("Loading job queue.")
    rave_pgf_qtools.load_queue(self.queue)
    if self.queue.qsize() > 0:
      self.logger.warning("Running jobs from dumped queue on file.")


  ## Internal executor of one product generation algorithm, according to how
  # they are queued. Check the job queue, get a job from it according to
  # how they are prioritized, then run it. Assume no exceptions will be raised
  # and declare that task done before running it.
  # TODO: Thread this.
  # @return outfile string, assuming it is the last item in the arguments list.
  def _run_one_job(self):
    if self.queue.qsize() > 0:
      priority, job = self.queue.get()
      self.queue.task_done()  # Count down queue even if job fails.
      algorithm, files, args = rave_pgf_qtools.split(job)
      self._run(algorithm, files, args)
      return args[-1]  


  ## Internal executor of the product generation algorithm queue, according to
  # how they are queued. Check the job queue, get jobs from it according to
  # how they are prioritized, then run them. Assuming no exceptions are raised,
  # declare each task as done after running it.
  # @return outfile string, assuming it is the last item in the arguments list.
  def _run_all_jobs(self):
    while self.queue.qsize() > 0:
      outfile = self._run_one_job()
      BaltradFrame.inject(outfile, channel=DEX_CHANNEL,
                          url=DEX_SPOE, sender=DEX_USER)
      self.logger.info("ID=%s Injected %s" % (algorithm_entry.get("jobid"),
                                              outfile))
      os.remove(outfile)


  ## Internal direct executor of the product generation algorithms. This will
  # automatically verify the module and reload it if it has been modified.
  # The module is loaded, and the function specified is executed with the
  # files and arguments lists. The function must not return anything.
  # @param algorithm Element copied from \ref self._algorithm_registry
  # @param files list of file strings
  # @param arguments list of verified arguments to pass to the generator
  # @return nothing
  def _run(self, algorithm, files, arguments):
    import imp
    mod_name, func_name = algorithm.get('module'), algorithm.get('function')
    jobid = algorithm.get('jobid')

    self.logger.info("ID=%s Running %s.%s" % (jobid, mod_name, func_name))

    fd, pathname, description = imp.find_module(mod_name)
    module = imp.load_module(mod_name, fd, pathname, description)
    fd.close()  # File descriptor is returned open, so close it.
    func = getattr(module, func_name)
    func(files, arguments)

    self.logger.info("ID=%s Finished %s.%s" % (jobid, mod_name, func_name))


  ## The mother method that coordinates the product generation calls.
  # This method verifies the integrity of the call and its arguments,
  # then adds an element to the job queue for execution.
  # @param algorithm string to the desired product generation call
  # @param files list of file strings
  # @param arguments list of strings, ordered as 'key-value' pairs, so that
  # even items are argument names and odd ones are their values. These must
  # be parsed into their corrects formats, ie. int, float, list, etc.
  # @return nothing
  def generate(self, algorithm, files, arguments):
    import rave_tempfile, rave_pgf_verify
    jobid = err_msg = None

    algorithm = algorithm.lower()
    self.logger.info("Request for generate algorithm: %s" % algorithm)
    fileno, outfile = rave_tempfile.mktemp(suffix='.h5', close="True")

    try:

      # Verify that the file list contains at least one file
      if len(files) == 0:
        raise IndexError("No input files given.")

      # Verify algorithm is registered
      algorithm_entry = copy(self._algorithm_registry.find(algorithm))
      if not algorithm_entry:
        raise LookupError('Algorithm "%s" not in registry' % algorithm)

      # Verify that the arguments check out. An IndexError will be raised if
      # the number of items in the argument list is odd.
      verified = rave_pgf_verify.verify_generate_args(arguments,
                                                      algorithm_entry)
      if not verified:
        raise TypeError('Erroneous arguments given to algorithm "%s"'%algorithm)
      # Add the outfile to the argument list. We need it later...
      arguments += ['outfile', outfile]

      # Queue this job and then run a job from the queue. If the job queue runs
      # assymetrically, then the outfile string could be different. Therefore,
      # \ref _run_one_job returns it.
      self._queue_job(algorithm_entry, files, arguments, self._jobid)
      mod_name  = algorithm_entry.get('module')
      func_name = algorithm_entry.get('function')
      jobid = algorithm_entry.get('jobid')

      outfile = self._run_one_job()

      # Inject the result.
      BaltradFrame.inject(outfile, channel='smhi_products',
                          url=DEX_SPOE, sender='smhi')

      # Log the result
      self.logger.info("ID=%s Injected %s" % (jobid, outfile))
    except Exception, err:
      # the 'err' itself is pretty useless
      err_msg = traceback.format_exc()
      self.logger.error("ID=%s failed. Check this out:\n%s" % (jobid, err_msg))

    if os.path.isfile(outfile): os.remove(outfile)
    return err_msg or "OK"


## Pretty useless method used to check argument types.
# @param arguments sequence of ints, floats, strings in arbitrary order.
  def echo_args(self, arguments):
    import types
    ret = ''
    for a in arguments:
      t = type(a)
      ret += 'Argument %s is a %s\n' % (a, t)
      if t == types.ListType:
        for e in a:
          ret += 'Argument %s in sequence is a %s\n' % (e, type(e))
    return ret


if __name__ == "__main__":
    print __doc__
