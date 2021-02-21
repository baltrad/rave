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

Tests the distance quality plugin

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2014-11-16
'''
import unittest
import os
import math
import string
import rave_distance_quality_plugin
import _raveio

class rave_distance_quality_plugin_test(unittest.TestCase):
  VOLUME_FIXTURE = "fixtures/pvol_seang_20090501T120000Z.h5"
  SCAN_FIXTURE = "fixtures/scan_sehuv_0.5_20110126T184500Z.h5"
  classUnderTest = None
    
  def setUp(self):
    self.classUnderTest = rave_distance_quality_plugin.rave_distance_quality_plugin() 

  def tearDown(self):
    self.classUnderTest = None

  def test_getQualityFields(self):
    result = self.classUnderTest.getQualityFields()
    self.assertEqual(1, len(result))
    self.assertEqual("se.smhi.composite.distance.radar", result[0])

  def test_process(self):
    vol = _raveio.open(self.VOLUME_FIXTURE).object
    result, qfield = self.classUnderTest.process(vol)
    self.assertTrue(vol == result)
    self.assertEqual(qfield, ["se.smhi.composite.distance.radar"], "Wrong qfield returned from process")
    
  def test_process_reprocess_true(self):
    vol = _raveio.open(self.VOLUME_FIXTURE).object
    result, _ = self.classUnderTest.process(vol, reprocess_quality_flag=True)
    self.assertTrue(vol == result)

  def test_process_reprocess_false(self):
    vol = _raveio.open(self.VOLUME_FIXTURE).object
    result, _ = self.classUnderTest.process(vol, reprocess_quality_flag=False)
    self.assertTrue(vol == result)

  def test_process_quality_control_mode(self):
    vol = _raveio.open(self.VOLUME_FIXTURE).object
    result, _ = self.classUnderTest.process(vol, reprocess_quality_flag=True, quality_control_mode="analyze")
    self.assertTrue(vol == result)

  def test_algorithm(self):
    result = self.classUnderTest.algorithm()
    self.assertTrue(result == None)
