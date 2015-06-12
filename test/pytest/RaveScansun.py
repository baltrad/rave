'''
Created on 20 January 2011

@author: Daniel Michelson
'''
import unittest
import _scansun
from numpy import nan

class RaveScansun(unittest.TestCase):
    # KNMI PVOL from Den Helder with a nice sun hit
    KNMI_TESTFILE="fixtures/KNMI-PVOL-Den_Helder.h5"

    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testSolarElevAzim(self):
        # Same metadata as in the lowest sweep of the KNMI volume
        lon, lat, yyyymmdd, hhmmss = 4.78997, 52.9533, 20110111, 75022
        valid = (-0.7758360955258381, 126.84009818438497, -0.18747813915648781)
        result = _scansun.solar_elev_azim(lon, lat, yyyymmdd, hhmmss)
        self.assertAlmostEquals(valid[0], result[0], 5)
        self.assertAlmostEquals(valid[1], result[1], 5)
        self.assertAlmostEquals(valid[2], result[2], 5)


    def testRefraction(self):
#        self.assertEquals(-0.19, round(_scansun.refraction(-0.78), 2))
        self.assertAlmostEquals(-0.19, _scansun.refraction(-0.78), 2)


    def testScansun(self):
        # The following validation values are:
        # Date    Time   Elevatn Azimuth ElevSun RelevSun  AzimSun dBSunFlux   SunMean SunStdd   ZdrMean ZdrStdd Refl  ZDR
        valid = ('RAD:NL51;PLC:nldhl', [(20110111, 75022, 0.30000001447042407, 126.5, 
                                         -0.77586040298515435, -0.18749940189096537, 
                                         126.84009882306225, 12.706416334302883, 
                                         -113.2001649139023, 0.78867323519645993, 
                                         nan, nan, 'DBZH', 'NA')])
        result = _scansun.scansun(self.KNMI_TESTFILE)
        self.assertEquals(valid[0][0], result[0][0])
        self.assertEquals(valid[1][0][0], result[1][0][0])
        self.assertEquals(valid[1][0][1], result[1][0][1])
        self.assertAlmostEquals(valid[1][0][2], result[1][0][2], 5)
        self.assertAlmostEquals(valid[1][0][3], result[1][0][3], 5)
        self.assertAlmostEquals(valid[1][0][4], result[1][0][4], 5)
        self.assertAlmostEquals(valid[1][0][5], result[1][0][5], 5)
        self.assertAlmostEquals(valid[1][0][6], result[1][0][6], 5)
        self.assertAlmostEquals(valid[1][0][7], result[1][0][7], 5)
        self.assertAlmostEquals(valid[1][0][8], result[1][0][8], 5)
        self.assertAlmostEquals(valid[1][0][9], result[1][0][9], 5)
        self.assertEquals(str(valid[1][0][10]), str(result[1][0][10]))
        self.assertEquals(str(valid[1][0][11]), str(result[1][0][11]))
        self.assertEquals(valid[1][0][12], result[1][0][12])
        self.assertEquals(valid[1][0][13], result[1][0][13])
	#self.assertEquals(valid, _scansun.scansun(self.KNMI_TESTFILE))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
