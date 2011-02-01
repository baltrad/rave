# -*- coding: latin-1 -*-
'''
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,

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
@date 2011-01-19
'''
import unittest
import string
import math
import _raveio
import os
import rave_pgf_volume_plugin

class PgfVolumePluginTest(unittest.TestCase):
  FIXTURES=["fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000000.h5",
            "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000001.h5",
            "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000002.h5",
            "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000003.h5",
            "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000004.h5",
            "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000005.h5"]
  
  TEMPORARY_FILE="ravemodule_pgfvolumetest.h5"
  
  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)
      
  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def testGenerateVolume(self):
    args = {"source":"selul","date":"20101023","time":"180000"}
    volume = rave_pgf_volume_plugin.generateVolume(self.FIXTURES, args)

    self.assertNotEqual(-1, string.find(`type(volume)`, "PolarVolumeCore"))
    self.assertAlmostEquals(66, volume.height)
    self.assertAlmostEquals(65.432, volume.latitude*180.0/math.pi,4)
    self.assertAlmostEquals(21.87, volume.longitude*180.0/math.pi,4)
    self.assertEquals(6, volume.getNumberOfScans())
    self.assertAlmostEquals(2.5, volume.getScan(0).elangle * 180.0/math.pi, 4)
    self.assertAlmostEquals(4.0, volume.getScan(1).elangle * 180.0/math.pi, 4)
    self.assertAlmostEquals(8.0, volume.getScan(2).elangle * 180.0/math.pi, 4)
    self.assertAlmostEquals(14.0, volume.getScan(3).elangle * 180.0/math.pi, 4)
    self.assertAlmostEquals(24.0, volume.getScan(4).elangle * 180.0/math.pi, 4)
    self.assertAlmostEquals(40.0, volume.getScan(5).elangle * 180.0/math.pi, 4)
    self.assertEquals("WMO:02092,RAD:SE41,PLC:Luleå,CMT:selul", volume.source)

  def testGenerateVolumeAndSave(self):
    args = {"source":"selul","date":"20101023","time":"180000"}
    volume = rave_pgf_volume_plugin.generateVolume(self.FIXTURES, args)
    
    ios = _raveio.new()
    ios.object = volume
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    ios = None
    
  def test_fix_source(self):
    s1 = "WMO:02451,RAD:SE46,PLC:Arlanda,CMT:searl"
    s2 = "WMO:02451,RAD:SE46,CMT:searl,PLC:Arlanda"
    s3 = "CMT:searl,WMO:02451,RAD:SE46,PLC:Arlanda"
    s4 = "WMO:02451,RAD:SE46,PLC:Arlanda"

    self.assertEquals("WMO:02451,RAD:SE46,PLC:Arlanda,CMT:selul", rave_pgf_volume_plugin.fix_source(s1, "selul"))
    self.assertEquals("WMO:02451,RAD:SE46,CMT:selul,PLC:Arlanda", rave_pgf_volume_plugin.fix_source(s2, "selul"))
    self.assertEquals("CMT:selul,WMO:02451,RAD:SE46,PLC:Arlanda", rave_pgf_volume_plugin.fix_source(s3, "selul"))
    self.assertEquals("WMO:02451,RAD:SE46,PLC:Arlanda,CMT:selul", rave_pgf_volume_plugin.fix_source(s4, "selul"))

if __name__ == "__main__":
    unittest.main()