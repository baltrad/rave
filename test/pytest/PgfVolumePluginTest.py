'''
Created on Jan 19, 2011

@author: anders
'''
import unittest
import string
import math
import rave_pgf_volume_plugin

class PgfVolumePluginTest(unittest.TestCase):
  FIXTURES=["fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000000.h5",
            "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000001.h5",
            "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000002.h5",
            "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000003.h5",
            "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000004.h5",
            "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000005.h5"]
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

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

if __name__ == "__main__":
    unittest.main()