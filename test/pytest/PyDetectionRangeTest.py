'''
Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the polarscan module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-10-14
'''
import unittest
import os
import _detectionrange
import _polarscan
import _raveio
import _rave
import string
import numpy
import math

class PyDetectionRangeTest(unittest.TestCase):
  FIXTURE_VOLUME="fixture_ODIM_H5_pvol_ang_20090501T1200Z.h5"
  TEMPORARY_FILE="ravemodule_drtest.h5"
  
  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def test_new(self):
    obj = _detectionrange.new()
    
    isscan = string.find(`type(obj)`, "DetectionRangeCore")
    self.assertNotEqual(-1, isscan)

  def test_top(self):
    dr = _detectionrange.new()
    o = _raveio.open(self.FIXTURE_VOLUME)
    
    result = dr.top(o.object, 2000, -40.0)
    
    os = _raveio.new()
    os.filename = self.TEMPORARY_FILE
    os.object = result
    os.save()
    
  def test_top_filter(self):
    dr = _detectionrange.new()
    o = _raveio.open(self.FIXTURE_VOLUME)
    
    topfield = dr.top(o.object, 2000, -40.0)
    result = dr.filter(topfield)
    
    os = _raveio.new()
    os.filename = self.TEMPORARY_FILE
    os.object = result
    os.save()
