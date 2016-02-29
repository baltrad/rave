# -*- coding: latin-1 -*-
'''
Copyright (C) 2016 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the volume plugin

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2016-02-29
'''
import unittest
import string
import math
import os
import algorithm_runner

class AlgorithmRunnerTest(unittest.TestCase):
  def setUp(self):
    pass #if os.path.isfile(self.TEMPORARY_FILE):
         #os.unlink(self.TEMPORARY_FILE)
      
  def tearDown(self):
    pass
    #if os.path.isfile(self.TEMPORARY_FILE):
    #os.unlink(self.TEMPORARY_FILE)
  
  #  def __init__(self, func, jobid, algorithm, files, arguments):

  def test_algorithm_job_1(self):
    job = algorithm_runner.algorithm_job(os.path, "123-432", "an.algorithm",["a.h5","b.h5"], ["--date=20150101","--time=101112","--algorithm_id=123"])
    self.assertEquals(os.path, job.func())
    self.assertEquals("an.algorithm", job.algorithm())
    self.assertEquals("123-432",job.jobid())
    self.assertTrue(set(["a.h5","b.h5"]) == set(job.files()))
    self.assertTrue(set(["--date=20150101","--time=101112","--algorithm_id=123"]) == set(job.arguments()))
    self.assertEquals(0, job.priority())
    self.assertEquals(123, job.algorithmid())
    self.assertEquals("20150101", job.date())
    self.assertEquals("101112", job.time())
    
  def test_algorithm_job_eq(self):
    job = algorithm_runner.algorithm_job(os.path, "123-432", "an.algorithm",["a.h5","b.h5"], ["--date=20150101","--time=101112","--algorithm_id=123"])
    job2 = algorithm_runner.algorithm_job(os.path, "123", "an.algorithm",["b.h5"], ["--date=20150101","--time=101112","--algorithm_id=123"])
    job3 = algorithm_runner.algorithm_job(os.path, "123-432", "an.algorithm",["a.h5","b.h5"], ["--date=20150101","--time=101112","--algorithm_id=123", "--pelle=321"])
    job4 = algorithm_runner.algorithm_job(os.path, "123-432", "an.algorithm",["a.h5","b.h5"], ["--date=20150102","--time=101112","--algorithm_id=123"])
    job5 = algorithm_runner.algorithm_job(os.path, "123-432", "an.algorithm",["a.h5","b.h5"], ["--date=20150101","--time=101113","--algorithm_id=123"])
    
    self.assertEquals(job, job2)
    self.assertEquals(job, job3)
    self.assertNotEquals(job, job4)
    self.assertNotEquals(job, job5)

  def test_algorithm_job_priority_algorithm(self):
    job = algorithm_runner.algorithm_job(os.path, "123-432", "an.algorithm",["a.h5","b.h5"], ["--date=20150101","--time=101112"])
    job2 = algorithm_runner.algorithm_job(os.path, "123", "an.algorithm",["b.h5"], ["--date=20150101","--time=101112","--algorithm_id=123"])
    
    self.assertTrue(job2 < job)
    self.assertTrue(job > job2)

  def test_algorithm_job_priority_datetime(self):
    job = algorithm_runner.algorithm_job(os.path, "123-432", "an.algorithm",["a.h5","b.h5"], ["--date=20150101","--time=101112","--algorithm_id=123"])
    job2 = algorithm_runner.algorithm_job(os.path, "123", "an.algorithm",["b.h5"], ["--date=20150102","--time=101112","--algorithm_id=123"])
    
    self.assertTrue(job2 < job)
    self.assertTrue(job > job2)

  def test_algorithm_job_priority_algorithm_datetime(self):
    job = algorithm_runner.algorithm_job(os.path, "123-432", "an.algorithm",["a.h5","b.h5"], ["--date=20150101","--time=101112","--algorithm_id=123"])
    job2 = algorithm_runner.algorithm_job(os.path, "123", "an.algorithm",["b.h5"], ["--date=20150102","--time=101112"])
    
    self.assertTrue(job < job2)
    self.assertTrue(job2 > job)
    
if __name__ == "__main__":
    unittest.main()


    