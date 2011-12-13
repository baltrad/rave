'''
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the pgf volume plugin

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2011-12-13
'''
import unittest
import os
import math
import string
import rave_pgf_volume_plugin

class rave_pgf_volume_plugin_test(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_generateVolume(self):
    args={}
    args["date"] = "20110101"
    args["time"] = "100000"
    
    files=["fixtures/scan_sehud_0.5_20110126T184500Z.h5",
           "fixtures/scan_sehud_1.0_20110126T184600Z.h5",
           "fixtures/scan_sehud_1.5_20110126T184600Z.h5"]
    
    
    result = rave_pgf_volume_plugin.generateVolume(files, args)
    self.assertEquals(3, result.getNumberOfScans())
    self.assertAlmostEquals(61.5771, result.latitude * 180.0 / math.pi, 4)
    self.assertAlmostEquals(16.7144, result.longitude * 180.0 / math.pi, 4)
    self.assertAlmostEquals(389.0, result.height, 4)
    self.assertTrue(string.find(result.source, "RAD:SE44") >= 0)
    self.assertAlmostEquals(0.86, result.beamwidth * 180.0 / math.pi, 4)
    self.assertAlmostEquals(0.86, result.getScan(0).beamwidth * 180.0 / math.pi, 4)
    self.assertAlmostEquals(0.86, result.getScan(1).beamwidth * 180.0 / math.pi, 4)
    self.assertAlmostEquals(0.86, result.getScan(2).beamwidth * 180.0 / math.pi, 4)
