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

import sys, os, traceback, string, types
from copy import deepcopy as copy
import logging
import xmlrpclib
import multiprocessing
import rave_pgf_logger
import rave_pgf_registry
import rave_pgf_qtools
import BaltradFrame
import _pyhl
from rave_defines import DEX_SPOE, REGFILE, PGFs, LOGID, LOGLEVEL, PGF_HOST, PGF_PORT


METHODS = {'generate' :  '("algorithm",[files],[arguments])',
           'get_quality_controls' : '',
           'get_areas' : '',
           'execute' :   '("shell command")',
           'register':   '("name", "module", "function", Help="", strings=",", ints=",", floats=",", seqs=",")',
           'deregister': '("name")',
           'flush':      '("stupid_password")',
           'job_done':   '("jobid")',
           'Help': ''
           }

PGF_REGISTRY = rave_pgf_registry.PGF_Registry(filename=REGFILE)


## The product generation framework class containing the product generation
# functionality.
class RavePGF():

  ## Constructor
  def __init__(self):
    self.name = multiprocessing.current_process().name  # "%s %s" % (LOGID, multiprocessing.current_process().name)
    self._pid = os.getpid()
    self._job_counter = 0
    self._jobid = "%i-0" % self._pid
    self.logger = rave_pgf_logger.rave_pgf_syslog_client()
    self._algorithm_registry = None
    self.queue = None
    self.pool = None
    self._client = None


  ## This method must exist for system.listMethods to work.
  def _listMethods(self):
    self.logger.info("%s: Someone called _listMethods" % self.name)
    return METHODS.keys()


  ## This method must exist for system.listMethods to work.
  # @param method string name of method for which to get help.
  def _methodHelp(self, method):
    self.logger.info("%s: Someone called _methodHelp" % self.name)
    return "%s %s" % (method, METHODS[method])


  ## Built-in algorithm query.
  # @param name string, optionally the name of the algorithm being queried.
  # @return string a help text comprising the names of each registered
  # algorithm and its descriptive text.
  def Help(self, name=None):
    self.logger.info("%s: Someone needs help." % self.name)
    return self._algorithm_registry.Help(name)


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
    self.logger.info("%s: Registering algorithm: %s" % (self.name, name))
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
    self.logger.info("%s: De-registering algorithm: %s" % (self.name, name))
    self._algorithm_registry.deregister(name)
    return "De-registered %s" % name


  ## Public method for dumping queue to file.
  # @param stupid_password A rudimentary safeguard against sabotage.
  # @return string
  def flush(self, stupid_password):
    if stupid_password == "Killing me softly":
      self._dump_queue()
      if self.pool:
        self.pool.terminate()
      return "Killed softly"
    else:
      self.logger.warning("%s: Security breach? Did some scoundrel just try to shut us down?" % self.name)
      return ""


  ## Queue a job for processing. All jobs are calls to the \ref generate method.
  # @param algorithm, an Element object that contains all the information
  # required to run a job.
  # @param files, a list of input file strings.
  # @param arguments, a list of argument key-value pairs.
  # @param jobid string unique ID of that job.
  def _queue_job(self, algorithm, files, arguments, jobid):
    # Once the merge files and arguments have been merged with the algorithm,
    # the result is referred to as a 'job'.
    self.queue.queue_job(algorithm, files, arguments, jobid)
    mod_name, func_name = algorithm.get('module'), algorithm.get('function')
    self.logger.debug("%s: ID=%s Queued %s.%s" % (self.name, jobid, mod_name, func_name))
    self._job_counter += 1
    self._jobid = "%i-%i" % (self._pid, self._job_counter)


  ## Dumps the job queue to XML file. Called automatically when the server
  # is stopped.
  # @param filename string file name to white to dump the queue
  def _dump_queue(self, filename=rave_pgf_qtools.QFILE):
    self.logger.info("%s: Dumping job queue containing %i jobs" % (self.name, self.queue.qsize()))
    self.queue.dump()


  ## Loads the job queue from XML file. Called automatically when the server
  # is started, even if the queue is empty.
  def _load_queue(self):
    self.logger.info("%s: Loading job queue", self.name)
    self.queue.load()
    if self.queue.qsize() > 0:
      self.logger.warning("%s: Running %i jobs from dumped queue on file." % (self.name, self.queue.qsize()))


  ## Internal executor of the product generation algorithm queue, according to
  # how they are queued. Runs only once at startup, if there are dumped jobs in
  # the queue.
  def _run_all_jobs(self):
    if self.queue.qsize() > 0:
      for jobid, job in self.queue.items():
        algorithm, files, arguments = rave_pgf_qtools.split(job) 
        result = self.pool.apply_async(generate, (jobid, algorithm.tag, files, arguments))


  ## Internal direct executor of the product generation algorithms. This will
  # automatically verify the module and reload it if it has been modified.
  # The module is loaded, and the function specified is executed with the
  # files and arguments lists. The function must not return anything.
  # @param algorithm Element copied from \ref self._algorithm_registry
  # @param files list of file strings
  # @param arguments list of verified arguments to pass to the generator
  # @return the result from the algorithm, either filename or None
  def _run(self, algorithm, files, arguments):
    import imp
    mod_name, func_name = algorithm.get('module'), algorithm.get('function')

    self.logger.debug("%s: ID=%s Running %s.%s" % (self.name, self._jobid, mod_name, func_name))

    fd, pathname, description = imp.find_module(mod_name)
    module = imp.load_module(mod_name, fd, pathname, description)
    fd.close()  # File descriptor is returned open, so close it.
    func = getattr(module, func_name)
    outfile = func(files, arguments)

    self.logger.info("%s: ID=%s Finished %s.%s" % (self.name, self._jobid, mod_name, func_name))

    return outfile


  ## Dispatcher method. Calls the \ref generate function that in turn creates 
  # a provisional instance of a \ref PGF object to invoke the \ref _generate 
  # method that does the job. The algorithm's presence in the registry is checked
  # first. 
  # @param algorithm string to the desired product generation call
  # @param files list of file strings
  # @param arguments list of strings, ordered as 'key-value' pairs, so that
  # even items are argument names and odd ones are their values. These must
  # be parsed into their corrects formats, ie. int, float, list, etc.
  # @return string "OK" always, which is ignored because the real work is deferred.
  def generate(self, algorithm, files, arguments):
    err_msg = None
    jobid = self._jobid
    try:
      # Verify algorithm is registered
      algorithm = algorithm.lower()
      algorithm_entry = copy(self._algorithm_registry.find(algorithm))
      if not algorithm_entry:
        raise LookupError('Algorithm "%s" not in registry' % algorithm)
          
      # Format job. This queue will only ever have one entry; it is thus used
      # for conveniently formatting the job, not for actually queueing.
      self._queue_job(algorithm_entry, files, arguments, jobid)
      # Write job to resilient queue on disk.
      #self._dump_queue()  # Really necessary each time?

      self.logger.info("%s: ID=%s Dispatching request for %s" % (self.name, jobid, algorithm))
      result = self.pool.apply_async(generate, (jobid, algorithm, files, arguments))
    except Exception, err:
      #err_msg = traceback.format_exc()
      #self.logger.error("%s: ID=%s failed. Check this out:\n%s" % (self.name, jobid, err_msg))
      self.logger.exception("%s: ID=%s failed. Check this out:" % (self.name, jobid))

    if err_msg: return err_msg        
    return "OK"

  ##
  # Returns the registered controls
  #
  def get_quality_controls(self):
    result = []
    try:
      import rave_pgf_quality_registry
      names = rave_pgf_quality_registry.get_plugins()
      for n in names:
        result.append((n, "%s quality control"%n))
    except Exception, e:
      self.logger.exception("Failed to get quality controls")
    return result

  ##
  # Returns the areas available to the composite generator
  #
  def get_areas(self):
    result = {}
    try:
      import area_registry
      reg = area_registry.area_registry()
      keys = reg.get_area_names()
      for k in keys:
        a = reg.getarea(k)
        result[k] = {"id":a.id, "xsize":a.xsize, "ysize":a.ysize, "xscale":a.xscale, "yscale":a.yscale, "extent":a.extent, "pcs":a.projection.definition}
    except Exception, e:
      self.logger.exception("Failed to get areas")

    return result

  ## The mother method that coordinates the product generation calls.
  # This method verifies the integrity of the arguments, then runs the job.
  # If the result gets injected to the DEX, then this is also done.
  # @param algorithm string to the desired product generation call
  # @param files list of file strings
  # @param arguments list of strings, ordered as 'key-value' pairs, so that
  # even items are argument names and odd ones are their values. These must
  # be parsed into their corrects formats, ie. int, float, list, etc.
  # @return string either "OK" or an error with a corresponding Traceback
  def _generate(self, algorithm, files, arguments):
    import rave_pgf_verify, rave_pgf_protocol
    err_msg = None

    algorithm = algorithm.lower()
    self.logger.debug("%s: Request for generate algorithm: %s" % (self.name, algorithm))

    outfile = None
    
    try:
      # Verify that the file list contains at least one file
      if len(files) == 0:
        raise IndexError("No input files given.")

      algorithm_entry = copy(self._algorithm_registry.find(algorithm))

      # Convert arguments from one protocol type into rave protocol
      arguments = rave_pgf_protocol.convert_arguments(algorithm, algorithm_entry, arguments)
      
      # Verify arguments. An IndexError will be raised if
      # the number of items in the argument list is odd.
      verified = rave_pgf_verify.verify_generate_args(arguments,
                                                      algorithm_entry)
      if not verified:
        raise TypeError('Erroneous arguments given to algorithm "%s"' % algorithm)

      outfile = self._run(algorithm_entry, files, arguments)  # Don't queue, just do it! 

      if outfile != None:
        self.logger.debug("%s: ID=%s one job run, outfile=%s" % (self.name, self._jobid, outfile))
      else:
        self.logger.debug("%s: ID=%s one job run, no output file" % (self.name, self._jobid))
      
      # Inject the result if it is a file
      if outfile != None:
        BaltradFrame.inject_file(outfile, DEX_SPOE)
        # Log the result
        self.logger.debug("%s: ID=%s Injected %s" % (self.name, self._jobid, outfile))
        
    except Exception, err:
      # the 'err' itself is pretty useless
      #err_msg = traceback.format_exc()
      #self.logger.error("%s: ID=%s failed. Check this out:\n%s" % (self.name, self._jobid, err_msg))
      self.logger.exception("%s: ID=%s failed. Check this out:" % (self.name, self._jobid))

    if outfile != None:
      if os.path.isfile(outfile): os.remove(outfile)
    
    if err_msg != None:
      self.logger.debug("%s: ID=%s Returning: %s" % (self.name, self._jobid, err_msg))
      return err_msg
    self.logger.debug("%s: ID=%s Returning: OK" % (self.name, self._jobid))
    return "OK"


  # Removes a job from the queue
  # @param jobid string containing the job identifier
  def job_done(self, jobid):
    msg = self.queue.task_done(jobid)
    if msg != "OK":
      self.logger.error("%s: %s" % (self.name, msg))
    else:
      self.logger.debug("%s: ID=%s Dequeued job" % (self.name, jobid))


  ##
  # Executes a shell escape command
  # @param command: the shell command without &
  # @return: OK
  #
  def execute(self, command):
    import subprocess
    cmd = command.strip() 
    try:
      if not cmd.endswith("&"):
        cmd = "%s &" % cmd
      code = subprocess.call(cmd, shell=True)
      if code != 0:
        raise Exception, "Failure when executing %s" % command
    except Exception, err:
      #err_msg = traceback.format_exc()
      #self.logger.error("%s: Failed to execute command %s, msg: %s" % (self.name, command, err_msg))
      self.logger.exception("%s: Failed to execute command %s, msg:" % (self.name, command))
    
    self.logger.debug("%s: Returning OK" % self.name)
    return "OK"
    

  ## Pretty useless method used to check argument types.
  # @param arguments sequence of ints, floats, strings in arbitrary order.
  # @return string of arguments
  def echo_args(self, arguments):
    ret = ''
    for a in arguments:
      t = type(a)
      ret += 'Argument %s is a %s\n' % (a, t)
      if t == types.ListType:
        for e in a:
          ret += 'Argument %s in sequence is a %s\n' % (e, type(e))
    return ret


## Convenience function for running several jobs asynchronously with
# \multiprocessing.apply_async
# @param jobid string job ID, used to keep track of jobs
# @param algorithm string to the desired product generation call
# @param files list of file strings
# @param arguments list of strings, ordered as 'key-value' pairs, so that
# even items are argument names and odd ones are their values. These must
# be parsed into their corrects formats, ie. int, float, list, etc.
# @param host string URI of the RAVE PGF server to which to connect
# @param port int port of the RAVE PGF server to which to connect
def generate(jobid, algorithm, files, arguments, host=PGF_HOST, port=PGF_PORT):
    pgf = RavePGF()
    pgf._jobid = jobid
    pgf._algorithm_registry = copy(PGF_REGISTRY)  # Less flexible than reading it each time
    ret = pgf._generate(algorithm, files, arguments)
    pgf._client = xmlrpclib.ServerProxy("http://%s:%i/RAVE" % (host, port), verbose=False)
    pgf._client.job_done(jobid)


if __name__ == "__main__":
    print __doc__
