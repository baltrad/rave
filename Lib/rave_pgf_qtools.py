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
## Convenience functions for formatting processing queue messages.
# Note: Because the \ref rave_pgf_registry.PGF_Registry contains algorithms
# with an element called "arguments" and containing the argument names, the
# actual argument list is stored in the queue message with the tag "args".
# Otherwise the root Element will not cope with more than one "arguments".

## @file
## @author Daniel Michelson, SMHI
## @date 2010-07-23

import os
import traceback
import xmlrpclib
import Queue
from xml.etree import ElementTree as ET
from rave_defines import QFILE
import threading

## Job queue Exception
class PGF_JobQueue_isFull_Error(Exception):
    pass


## Queue object based on a dictionary
class PGF_JobQueue(dict):

    # Initializer
    # @parameter maxsize int maximum number of jobs allowed in the queue, defaults to 0
    # which means unlimited. 
    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self.lock = threading.Lock()

    ## Queue a job for processing. All jobs are calls to the \ref generate method.
    # @param algorithm_entry, an Element object that contains all the information
    # required to run a job.
    # @param files, a list of input file strings.
    # @param arguments, a list of argument key-value pairs.
    # @param jobid string unique ID of that job.
    def queue_job(self, algorithm_entry, files, arguments, jobid):
        self.lock.acquire()
        try:
            if self.maxsize == 0 or 0 < self.qsize() < self.maxsize:
                merge(algorithm_entry, files, arguments, jobid)
                self[jobid] = algorithm_entry
            else:
                raise PGF_JobQueue_isFull_Error
        finally:
          self.lock.release() 


    # @returns the number of jobs in the queue 
    def qsize(self):
        return self.__len__()


    # Removes a job from the queue
    # @param jobid string containing the job identifier
    # @returns string either OK or that the job can't be found
    def task_done(self, jobid):
        self.lock.acquire()
        try:
            if self.has_key(jobid):
                job = self.pop(jobid)
                return "OK"
            else:
                return "Job queue does not have job ID=%s" % jobid
        finally:
          self.lock.release()


    # Dumps the queue to XML file
    # @param filename string of the file to which to dump the queue
    def dump(self, filename=QFILE):
        self.lock.acquire()
        try:
            q = """<?xml version="1.0" encoding="UTF-8"?>\n"""
            root = ET.Element("generate-queue")
            for jobid, job in self.items():
                root.append(job)
            fd = open(filename, 'w')
            fd.write(q + ET.tostring(root))
            fd.close()
        finally:
          self.lock.release()


    # Loads the queue from XML file
    # @param filename string of the file from which to load the queue
    def load(self, filename=QFILE):
        from xml.parsers.expat import ExpatError
        if os.path.isfile(filename):
            try:
                elems =  ET.parse(filename).getroot()
            except Exception, err:
                err_msg = traceback.format_exc()
                print "Error trying to read PGF job queue: %sIgnoring, using empty job queue." % err_msg
                return  # queue is probably empty, just ignore
            for elem in elems.getchildren():
                self[elem.get('jobid')] = elem


## Adds Elements containing files and arguments to the message.
# @param algorithm Element in a \ref rave_pgf_registry.PGF_Registry instance.
# @param files list of input files
# @param arguments list of arguments
# @param jobid string unique ID of that job.
# several Elements in the queue with the same priority, they will be retrieved
# in alphabetical order, according to the tag (algorithm) names.
def merge(algorithm, files, arguments, jobid):
    algorithm.append(List2Element(files, "files"))
    algorithm.append(List2Element(arguments, "args"))
    algorithm.set("jobid", jobid)


## Convenience function for accessing files and arguments from a message.
# @param elem Element containing the algorithm, input file list, and argument
# list.
# @return tuple containing algorithm Element, list of files, list of arguments.
def split(elem):
    files = Element2List(elem, "files")
    args = Element2List(elem, "args")
    return elem, files, args


## Convenience function for retrieving an Element from a Python list.
# @param inlist input list.
# @param tagname string the name of the tag to create.
# @return Element
def List2Element(inlist, tagname):
    dumped = xmlrpclib.dumps(tuple(inlist))  # dumps takes a tuple, not a list
    e = ET.fromstring(dumped)
    e.tag = tagname  # xmlrpc.dumps creates tagname 'params' by default.
    return e

## Convenience function for retrieving a Python list from an Element.
# @param elem Element containing the list to retrieve.
# @param tagname string the name of the tag to read.
# @return list
def Element2List(elem, tagname):
    e = elem.find(tagname)
    tag = e.tag
    e.tag = 'params'  # xmlrpc.loads won't accept any other tagname. Hack...
    l = list(xmlrpclib.loads(ET.tostring(e))[0])
    e.tag = tag       # put back original tag
    return l 


if __name__ == "__main__":
    print __doc__
