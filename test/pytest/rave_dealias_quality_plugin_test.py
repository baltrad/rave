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

Tests the dealias quality plugin

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2014-11-16
'''
import unittest
import os
import math
import string
import rave_dealias_quality_plugin
import _raveio

class rave_dealias_quality_plugin_test(unittest.TestCase):
  VOLUME_FIXTURE = "fixtures/pvol_seang_20090501T120000Z.h5"
  SCAN_FIXTURE = "fixtures/scan_sehuv_0.5_20110126T184500Z.h5"
  classUnderTest = None
    
  def setUp(self):
    self.classUnderTest = rave_dealias_quality_plugin.dealias_plugin() 

  def tearDown(self):
    self.classUnderTest = None

  def test_getQualityFields(self):
    result = self.classUnderTest.getQualityFields()
    self.assertEqual(1, len(result))
    self.assertEqual("se.smhi.detector.dealias", result[0])
    
  def test_process(self):
    scan = _raveio.open(self.SCAN_FIXTURE).object
    p1 = scan.removeParameter("DBZH")
    p1.quantity="VRAD"
    scan.addParameter(p1)

    result, qfield = self.classUnderTest.process(scan)
    
    self.assertEqual("True", result.getParameter("VRAD").getAttribute("how/dealiased"))
    self.assertEqual(qfield, ["se.smhi.detector.dealias"], "Wrong qfield returned from process")

  def test_process_reprocess(self):
    scan = _raveio.open(self.SCAN_FIXTURE).object
    p1 = scan.removeParameter("DBZH")
    p1.quantity="VRAD"
    scan.addParameter(p1)

    # Dealiasing always checks if this has been done before processing so reprocess is not relevant but we want to verify that
    # the API can handle the flag.
    result, qfield = self.classUnderTest.process(scan, reprocess_quality_flag=True)
    
    self.assertEqual("True", result.getParameter("VRAD").getAttribute("how/dealiased"))
    self.assertEqual(qfield, ["se.smhi.detector.dealias"], "Wrong qfield returned from process")
    
  def test_process_reprocess_with_quality_control_mode(self):
    scan = _raveio.open(self.SCAN_FIXTURE).object
    p1 = scan.removeParameter("DBZH")
    p1.quantity="VRAD"
    scan.addParameter(p1)

    # Dealiasing always checks if this has been done before processing so reprocess is not relevant but we want to verify that
    # the API can handle the flag.
    result, qfield = self.classUnderTest.process(scan, reprocess_quality_flag=True, quality_control_mode="analyze")
    
    self.assertEqual("True", result.getParameter("VRAD").getAttribute("how/dealiased"))
    self.assertEqual(qfield, ["se.smhi.detector.dealias"], "Wrong qfield returned from process")
    