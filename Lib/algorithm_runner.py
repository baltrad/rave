import sys
import multiprocessing
import multiprocessing.pool
#import Queue as queue
import time
import re
import threading
from rave_mppool import *
import rave_pgf_logger

logger = rave_pgf_logger.create_logger()
if sys.version_info < (3,):
  import xmlrpclib as xmlrpc
  import Queue as queue
else:
  import queue
  
##
# Job that supports the wanted priority handling that we want from the PriorityQueue
# All jobs that has algorithm_id in the arguments will get priority = 0 instead of priority 1
# After that, the date and time are compared. Date & Time that are newer than now will always
# get higher priority that old files.
# 
class algorithm_job(object):
  def __init__(self, func, jobid, algorithm, files, arguments, jobdone=None):
    self._func = func
    self._jobid = jobid
    self._algorithm = algorithm
    self._files = files
    self._arguments = arguments
    self._priority = 1
    self._algorithmid = -1
    self._jobdone=jobdone
    algorithm_id = self.get_string_from_arguments(arguments, "--algorithm_id", None)
    if algorithm_id != None:
      self._priority = 0
      self._algorithmid = algorithm_id
    self._date = self.get_string_from_arguments(arguments, "--date", None)
    self._time = self.get_string_from_arguments(arguments, "--time", None)
    self._mergeable = "--merge=true" in arguments
    
  def get_arg_from_arguments(self, arguments, key):
    for arg in arguments:
      o = re.match(key+"=([^$]+)", arg)
      if o != None:
        return o.group(1)
    return None

  def get_string_from_arguments(self, arguments, key, defaultvalue):
    v = self.get_arg_from_arguments(arguments, key)
    if v != None:
      return v
    return defaultvalue
  
  def func(self):
    return self._func
  
  def jobid(self):
    return self._jobid
  
  def algorithm(self):
    return self._algorithm
  
  def files(self):
    return self._files
  
  def arguments(self):
    return self._arguments
  
  def date(self):
    return self._date
  
  def time(self):
    return self._time
  
  def priority(self):
    return self._priority
  
  def algorithmid(self):
    return self._algorithmid
  
  def setJobid(self, jobid):
    self._jobid = jobid
  
  def setArguments(self, arguments):
    self._arguments = arguments
  
  def setFiles(self, files):
    self._files = files
  
  def mergeable(self):
    return self._mergeable
  
  def jobdone(self):
    if self._jobdone != None:
      self._jobdone(self._jobid)

  def __eq__(self, other):
    if other == None:
      return False
    return (self._priority, self._algorithm, self._algorithmid, self._date, self._time) == \
      (other._priority, other._algorithm, other._algorithmid, other._date, other._time)
  
  def __ne__(self, other):
    return not self == other
  
  def __gt__(self, other):
    if self._priority > other._priority:
      return True
    self_datetime = "%s%s" % (str(self._date), str(self._time))
    other_datetime = "%s%s" % (str(other._date), str(other._time))
    return other_datetime > self_datetime
  
  def __lt__(self, other):
    if self._priority < other._priority:
      return True
    self_datetime = "%s%s" % (str(self._date), str(self._time))
    other_datetime = "%s%s" % (str(other._date), str(other._time))
    return other_datetime < self_datetime

  def __ge__(self, other):
    return (self > other) or (self == other)
  
  def __le__(self, other):
    return (self < other) or (self == other)

  def __str__(self):
    return "(p:%d - dt: %s%s - id: %d - a: %s)"%(self._priority, self._date, self._time, self._algorithmid, self._algorithm)

  def __repr__(self):
    return "%s"%self.__str__()

def run_algorithm(func, jobid, algorithm, files, arguments):
  try:
    return func(jobid, algorithm, files, arguments)
  except Exception:
    return None

##
# Wrapper around the multiprocessing pool used for generating products
# Since we want to avoid processing old jobs before newer jobs we have
# to prioritize the jobs and make sure that we are not draining computer
# resources on irrelevant jobs.
# Also, if a job with same algorithm_id, date and time is added when another job is in the
# queue, the queued job will be replaced with the new jobs jobid, arguments and files.
#
class algorithm_runner(object):
  Process = NonDaemonProcess    
  def __init__(self, nrprocesses):
    self.lock = threading.Lock()
    self.queue = queue.PriorityQueue()
    self.pool = RavePool(nrprocesses)
    self.nrprocesses = nrprocesses
    self.running_jobs = 0
    super(algorithm_runner, self).__init__()
  
  ##
  # Adds one job to be processed. If arguments contains an algorithm_id, the priority of
  # this job will be increased. Also, if the arguments contains date & time, this will
  # also increase priority.
  # If queue already contains an algorithm that is matching a previously added algorithm,
  # its contant will be replaced (files and arguments).
  #
  def add(self, func, jobid, algorithm, files, arguments, jobdone=None):
    self.lock.acquire()
    try:
      a = algorithm_job(func, jobid, algorithm, files, arguments, jobdone)
      ##
      # To avoid duplicate jobs
      if a.mergeable():
        for qi in self.queue.queue:
          if qi.date() != None and qi.time() != None and qi == a:
            logger.debug("[algorithm_runner] Merging %s with %s"%(a.jobid(), qi.jobid()))
            qi.jobdone() # We must let invoker know that this job is done
            qi.setJobid(jobid)
            qi.setArguments(arguments)
            qi.setFiles(files)
            a = None
            break

      if a != None:
        self.queue.put(a)
        logger.debug("[algorithm_runner] Added job (%s) - %s to queue, queue size = %d"%(a.jobid(), a.algorithm(), self.queue.qsize()))
      self._handle_queue()
    finally:
      self.lock.release()

  ##
  # Invoked by the async pool on answer
  #
  def async_callback(self, arg):
    self.lock.acquire()
    try:
      self.queue.task_done()
      self.running_jobs = self.running_jobs - 1
      logger.debug("[algorithm_runner] Finished with job %s"%(str(arg)))
      self._handle_queue()
    finally:
      self.lock.release()

  ##
  # Runs one job from the queue. Synchronization must be performed before entering this
  # method. Will increase running jobs with 1 if job added async to pool
  #
  def _handle_queue(self):
    if self.running_jobs < self.nrprocesses:
      job = None
      try:
        job = self.queue.get(False)
      except queue.Empty:
        pass
      if job:
        self.pool.apply_async(run_algorithm, (job.func(), job.jobid(), job.algorithm(), job.files(), job.arguments()), callback=self.async_callback)
        self.running_jobs = self.running_jobs + 1
        logger.info("[algorithm_runner] Applied job %s, currently %d jobs running and %d in queue"%(str(job.jobid()), self.running_jobs, self.queue.qsize()))
      else:
        logger.debug("[algorithm_runner] Queue empty")

  ##
  # Shutsdown and waits for pool to terminate
  def join(self):
    self.pool.close()
    self.pool.join()

  ##
  # Terminates the pool
  def terminate(self):
    self.pool.terminate()

def kalle(jobid, algorithm, files, arguments):
  time.sleep(0.5)
  fp = open("/tmp/slask.txt", "a")
  fp.write("ARGS: jobid=%d,algorithm=%s,files=%s,arguments=%s\n"%(jobid,algorithm, str(files), str(arguments)))
  fp.close()
  

if __name__ == "__main__":
  #self, func, jobid, algorithm, files, arguments
  job = algorithm_job(kalle, "1-123","se.sej",["a.h5","b.h5"],["--date=20150101","--time=101112","--algorithm_id=123"])
  print(job._date)
  print(job._time)
  print(job._algorithmid)
  
#   runner = algorithm_runner(2)  
#   
#   runner.add(kalle, 1, "a.1",[], ["algorithm_id",1,"date","20151001","time","100000"])
#   runner.add(kalle, 2, "a.2",[], ["date","20151001","time","100005"])
#   runner.add(kalle, 3, "a.1",[], ["algorithm_id",1,"date","20151001","time","090005"])
#   runner.add(kalle, 4, "a.3",[], ["algorithm_id",3,"date","20151001","time","100000"])
#   runner.add(kalle, 5, "a.5",[], [])
#   runner.add(kalle, 6, "a.2",[], ["date","20150921","time","100005"])
#   runner.add(kalle, 7, "a.3",["nils.h5","karl.h5"], ["algorithm_id",3,"date","20151001","time","100000"])
# 
#   time.sleep(2)
#   runner.join()
  
#  q = queue.PriorityQueue()
#  q.put(algorithm(1, "a.1",[], ["algorithm_id",1,"date","20151001","time","100000"]))
#  q.put(algorithm(2, "a.2",[], ["date","20151001","time","100005"]))
#  q.put(algorithm(3, "a.1",[], ["algorithm_id",1,"date","20151001","time","090005"]))
#  q.put(algorithm(4, "a.3",[], ["algorithm_id",3,"date","20151001","time","100000"]))
#  q.put(algorithm(5, "a.5",[], []))
#
#
#  print `q.get()`
#  print `q.get()`
#  print `q.get()`
#  print `q.get()`
#  print `q.get()`

  
