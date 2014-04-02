'''
Copyright (C) 2014 Swedish Meteorological and Hydrological Institute, SMHI,

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/

Tests PGF Queue functionality

@file
@author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2014-03-21
'''
import unittest
import os, traceback
import _rave
import rave_pgf_qtools
import rave_pgf_registry
from xml.etree import ElementTree as ET
from rave_defines import REGFILE

class PyPgfQtoolsTest(unittest.TestCase):
    QFILE = "fixtures/rave_pgf_queue.xml"
    REG = rave_pgf_registry.PGF_Registry(filename=REGFILE)
    classUnderTest = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testQObject(self):
        Q = rave_pgf_qtools.PGF_JobQueue()
        self.assertTrue(str(type(Q)), "<class 'rave_pgf_qtools.PGF_JobQueue'>")
        self.assertEquals(Q.qsize(), 0)
        self.assertEquals(Q.maxsize, 0)

    def testQDumpEmpty(self):
        Q = rave_pgf_qtools.PGF_JobQueue()
        Q.dump(filename=self.QFILE)

    def testQLoadEmpty(self):
        Q = rave_pgf_qtools.PGF_JobQueue()
        Q.load(filename=self.QFILE)
        self.assertEquals(Q.qsize(), 0)
        os.remove(self.QFILE)

    def testQList2Element2List(self):
        l = ["bobbe.h5", "fubik.h5"]
        elem = ET.Element("myElem")
        e = rave_pgf_qtools.List2Element(l, "files")
        elem.append(e)
        self.assertEquals(e.tag, "files")
        self.assertEquals(len(e.getchildren()), 2)
        L = rave_pgf_qtools.Element2List(elem, "files")
        self.assertEquals(l, L)

    def testQMergeSplit(self):
        algorithm_entry = self.REG.find("se.smhi.rave.creategmapimage")
        rave_pgf_qtools.merge(algorithm_entry, ["bobbe.h5"], [], "1")
        self.assertEquals(algorithm_entry.get("jobid"), "1")
        elem, files, args = rave_pgf_qtools.split(algorithm_entry)
        self.assertEquals(files, ["bobbe.h5"])
        self.assertEquals(args, [])

    def testQJob(self):
        Q = rave_pgf_qtools.PGF_JobQueue()
        algorithm_entry = self.REG.find("se.smhi.rave.creategmapimage")
        Q.queue_job(algorithm_entry, ["bobbe.h5"], [], "1")
        self.assertEqual(Q.qsize(), 1)
        Q.dump(filename=self.QFILE+"1")

    def testQJobLoadOne(self):
        Q = rave_pgf_qtools.PGF_JobQueue()
        Q.load(filename=self.QFILE+"1")
        self.assertEquals(Q.qsize(), 1)

    def testQMaxsize(self):
        Q = rave_pgf_qtools.PGF_JobQueue()
        Q.maxsize = 1
        algorithm_entry = self.REG.find("se.smhi.rave.creategmapimage")
        try:
            Q.queue_job(algorithm_entry, ["fubik.h5"], [], "2")
        except rave_pgf_qtools.PGF_JobQueue_isFull_Error:
            err_msg = traceback.format_exc()
            err = err_msg.split("\n")[-2]
            self.assertEquals(err, "PGF_JobQueue_isFull_Error")

    def testQJobTaskDone(self):
        Q = rave_pgf_qtools.PGF_JobQueue()
        Q.load(filename=self.QFILE+"1")
        Q.task_done("1")
        self.assertEquals(Q.qsize(), 0)
        os.remove(self.QFILE+"1")
