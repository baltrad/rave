'''
Created on 20 January 2011

@author: Daniel Michelson
'''
import unittest
import _scansun

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
        self.assertEquals(valid, _scansun.solar_elev_azim(lon, lat, yyyymmdd, hhmmss))


    def testRefraction(self):
        self.assertEquals(-0.19, round(_scansun.refraction(-0.78), 2))


    def testScansun(self):
        # The following validation values are:
        # Date    Time   Elevatn Azimuth ElevSun AzimSun dBmMHzSun dBmStdd RelevSun
        valid = [(20110111, 75022, 0.30000001192092901, 126.5, 
                  -0.77585925328048244, 126.84009776579752, -113.30817548053685, 
                  0.67217618344963492, -0.1874983807707411)]
        self.assertEquals(valid, _scansun.scansun(self.KNMI_TESTFILE))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()