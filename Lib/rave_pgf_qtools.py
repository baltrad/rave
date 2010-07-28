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
import xmlrpclib
from xml.etree import ElementTree as ET
from rave_defines import QFILE


## Adds Elements containing files and arguments to the message.
# @param algorithm Element in a \ref rave_pgf_registry.PGF_Registry instance.
# @param files list of input files
# @param arguments list of arguments
# @param jobid int unique ID of that job.
# @param priority int, where lower values denote higher priority. If there are
# several Elements in the queue with the same priority, they will be retrieved
# in alphabetical order, according to the tag (algorithm) names.
def merge(algorithm, files, arguments, jobid, priority):
    algorithm.append(List2Element(files, "files"))
    algorithm.append(List2Element(arguments, "args"))
    algorithm.set("priority", str(priority))
    algorithm.set("jobid", str(jobid))


## Convenience function for accessing files and arguments from a message.
# @param elem Element containing the algorithm, input file list, and argument
# list.
# @return tuple containing algorithm Element, list of files, list of arguments.
def split(elem):
    files = Element2List(elem, "files")
    args = Element2List(elem, "args")
    del elem.attrib["priority"]  # Don't need this any more...
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
    e.tag = 'params'  # xmlrpc.loads won't accept any other tagname.
    return list(xmlrpclib.loads(ET.tostring(e))[0])
    

## Dumps the queue to default XML file. This function should be called by atexit
# for saving the server's queue to file in case the server's shut down before
# the queue is emptied.
# @param queue \ref Queue.PriorityQueue used by \ref rave_pgf.RavePGF
# @return nothing
def dump_queue(queue):
    q = """<?xml version="1.0" encoding="UTF-8"?>\n"""
    root = ET.Element("generate-queue")
    while queue.qsize() > 0:
        prio, job = queue.get()
        root.append(job)
        queue.task_done()
    fd = open(QFILE, 'w')
    fd.write(q + ET.tostring(root))
    fd.close()


## Loads the queue that's been saved to file.
# @param queue \ref Queue.PriorityQueue used by \ref rave_pgf.RavePGF
# @param filename string of the dumped job queue.
# @param priority int, defaults to 0. This priority overrides that in the queue
# read from file.
# @return nothing
def load_queue(queue, filename=QFILE, priority=0):
    from xml.parsers.expat import ExpatError
    if os.path.isfile(filename):
        try:
            elems =  ET.parse(filename).getroot()
        except ExpatError:
            return  # queue is probably empty, just ignore
        for elem in elems.getchildren():
            queue.put((priority, elem))


if __name__ == "__main__":
    print __doc__
