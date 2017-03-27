'''
Created on 20 January 2011

@author: Daniel Michelson
'''
import os, unittest
import _scansun
import _raveio
from numpy import nan
import rave_pgf_scansun_plugin, odim_source
from rave_defines import UTF8

class RaveScansun(unittest.TestCase):
    # KNMI PVOL from Den Helder with a nice sun hit
    KNMI_TESTFILE="fixtures/KNMI-PVOL-Den_Helder.h5"
    # The following validation values are:
    # Date    Time   Elevatn Azimuth ElevSun AzimSun   N  dBSunFlux   SunMean SunStdd   ZdrMean ZdrStdd Refl  ZDR
    VALID = ('RAD:NL51;PLC:nldhl', [(20110111, 75022.0, 0.30000001447042407, 126.0, 
                                     -0.77586040298515435, 126.84009882306225, 
                                     98, 12.706416334302883, 
                                     -113.2001649139023, 0.78867323519645993, 
                                     nan, nan, 'DBZH', 'NA')])

    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testSolarElevAzim(self):
        # Same metadata as in the lowest sweep of the KNMI volume
        lon, lat, yyyymmdd, hhmmss = 4.78997, 52.9533, 20110111, 75022
        valid = (-0.7758360955258381, 126.84009818438497, -0.047247576587858942)
        result = _scansun.solar_elev_azim(lon, lat, yyyymmdd, hhmmss)
        self.assertAlmostEqual(valid[0], result[0], 5)
        self.assertAlmostEqual(valid[1], result[1], 5)
        self.assertAlmostEqual(valid[2], result[2], 5)


    def testRefraction(self):
        self.assertAlmostEqual(-0.05, _scansun.refraction(-0.78), 2)


    def testScansun(self):
        result = _scansun.scansun(self.KNMI_TESTFILE)
        self.assertEqual(self.VALID[0][0], result[0][0])
        self.assertEqual(self.VALID[1][0][0], result[1][0][0])
        self.assertEqual(self.VALID[1][0][1], result[1][0][1])
        self.assertAlmostEqual(self.VALID[1][0][2], result[1][0][2], 5)
        self.assertAlmostEqual(self.VALID[1][0][3], result[1][0][3], 5)
        self.assertAlmostEqual(self.VALID[1][0][4], result[1][0][4], 5)
        self.assertAlmostEqual(self.VALID[1][0][5], result[1][0][5], 5)
        self.assertEqual(self.VALID[1][0][6], result[1][0][6])
        self.assertAlmostEqual(self.VALID[1][0][7], result[1][0][7], 5)
        self.assertAlmostEqual(self.VALID[1][0][8], result[1][0][8], 5)
        self.assertAlmostEqual(self.VALID[1][0][9], result[1][0][9], 5)
        self.assertEqual(self.VALID[1][0][12], result[1][0][12])
        self.assertEqual(self.VALID[1][0][13], result[1][0][13])


    def testScansunFromObject(self):
        # Replicate the above test, but using in-memory processing.
        obj = _raveio.open(self.KNMI_TESTFILE).object
        result = _scansun.scansunFromObject(obj)
        self.assertEqual(self.VALID[0][0], result[0][0])
        self.assertEqual(self.VALID[1][0][0], result[1][0][0])
        self.assertEqual(self.VALID[1][0][1], result[1][0][1])
        self.assertAlmostEqual(self.VALID[1][0][2], result[1][0][2], 5)
        self.assertAlmostEqual(self.VALID[1][0][3], result[1][0][3], 5)
        self.assertAlmostEqual(self.VALID[1][0][4], result[1][0][4], 5)
        self.assertAlmostEqual(self.VALID[1][0][5], result[1][0][5], 5)
        self.assertEqual(self.VALID[1][0][6], result[1][0][6])
        self.assertAlmostEqual(self.VALID[1][0][7], result[1][0][7], 5)
        self.assertAlmostEqual(self.VALID[1][0][8], result[1][0][8], 5)
        self.assertAlmostEqual(self.VALID[1][0][9], result[1][0][9], 5)
        self.assertEqual(self.VALID[1][0][12], result[1][0][12])
        self.assertEqual(self.VALID[1][0][13], result[1][0][13])


    def testWriteHits(self):
        source = odim_source.SOURCE['nldhl'].encode(UTF8)  # Non-compliant ODIM source in fixture...
        hits = self.VALID[1]
        rave_pgf_scansun_plugin.RAVEETC = os.getcwd()+'/fixtures'
        fstr = '%s/scansun/nldhl.scansun' % rave_pgf_scansun_plugin.RAVEETC
        rave_pgf_scansun_plugin.writeHits(source, hits)
        fd = open(fstr)
        content = fd.read()
        fd.close()
        self.assertEqual(content, '#Date    Time        Elevatn Azimuth   ElevSun   AzimSun    N  dBSunFlux   SunMean SunStdd   ZdrMean ZdrStdd  Refl  ZDR\n20110111 075022.000    0.300  126.00   -0.7759  126.8401      98      12.71   -113.20   0.789       nan     nan  DBZH   NA\n')
        os.remove(fstr)
        os.rmdir(os.path.split(fstr)[0])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
