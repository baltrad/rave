'''
Copyright (C) 2018 Swedish Meteorological and Hydrological Institute, SMHI

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

Tests the height quality plugin. Based on the distance quality plugin tests.

@file
@author Anders Henja (SMHI) and Daniel Michelson (ECCC)
@date 2018-02-09
'''
import unittest
import os
import math
import string
import rave_height_quality_plugin
import _raveio

class rave_height_quality_plugin_test(unittest.TestCase):
  VOLUME_FIXTURE = "fixtures/pvol_seang_20090501T120000Z.h5"
  classUnderTest = None
    
  def setUp(self):
    self.classUnderTest = rave_height_quality_plugin.rave_height_quality_plugin() 

  def tearDown(self):
    self.classUnderTest = None

  def test_getQualityFields(self):
    result = self.classUnderTest.getQualityFields()
    self.assertEquals(1, len(result))
    self.assertEquals("se.smhi.composite.height.radar", result[0])

  def test_process(self):
    vol = _raveio.open(self.VOLUME_FIXTURE).object
    result, qfield = self.classUnderTest.process(vol)
    self.assertTrue(vol == result)
    self.assertEquals(qfield, ["se.smhi.composite.height.radar"], "Wrong qfield returned from process")
    
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
