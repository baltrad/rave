# coding=iso-8859-1
'''
Created on Aug 12, 2009

@author: anders
'''
import unittest
import rave
import os
import _pyhl
import _helpers

class RaveTest(unittest.TestCase):
    # Old nordrad format for a pcappi, (not including IMAGE attributes)
    OLD_NRD_FORMAT_TESTFILE="fixture_old_pcappi-dbz-500.ang-gnom-2000.h5"
    TESTFILE = "testskrivning.h5"
    
    def setUp(self):
        _helpers.triggerMemoryStatus()
        if os.path.isfile(self.TESTFILE):
            os.unlink(self.TESTFILE)


    def tearDown(self):
        if os.path.isfile(self.TESTFILE):
            os.unlink(self.TESTFILE)


    def testReadOldNordradPcappi(self):
        obj = rave.open(self.OLD_NRD_FORMAT_TESTFILE)
        
        # Verify
        self.assertTrue(isinstance(obj, rave.RAVE))
        
        # /how
        self.assertEquals(2626, obj.get("/how/WMO"))
        self.assertEquals("Ängelholm", obj.get("/how/place"))
        self.assertEquals(1149076800, obj.get("/how/startepochs"))
        self.assertEquals("ang_gn", obj.get("/how/area"))
        self.assertEquals(0, obj.get("/how/doppler"))
        self.assertEquals(1149076800, obj.get("/how/endepochs"))
        self.assertEquals("ERIC", obj.get("/how/system"))
        self.assertEquals("IRIS", obj.get("/how/software"))
        self.assertAlmostEqual(5.35, obj.get("/how/wavelength"), 4)
        self.assertAlmostEqual(2, obj.get("/how/pulsewidth"), 4)
        self.assertEquals(250, obj.get("/how/lowprf"))
        self.assertEquals(250, obj.get("/how/highprf"))
        self.assertEquals("ang", obj.get("/how/nodes")[0])

        # /image1
        self.assertEquals("PCAPPI", obj.get("/image1/what/product"))
        self.assertAlmostEqual(500, obj.get("/image1/what/prodpar"), 4)
        self.assertEquals("DBZ", obj.get("/image1/what/quantity"))
        self.assertEquals("20060531", obj.get("/image1/what/startdate"))
        self.assertEquals("120000", obj.get("/image1/what/starttime"))
        self.assertEquals("20060531", obj.get("/image1/what/enddate"))
        self.assertEquals("120000", obj.get("/image1/what/endtime"))
        self.assertAlmostEqual(500, obj.get("/image1/what/prodpar"), 4)
        self.assertAlmostEqual(0.5, obj.get("/image1/what/gain"), 4)
        self.assertAlmostEqual(-32, obj.get("/image1/what/offset"), 4)
        self.assertAlmostEqual(255, obj.get("/image1/what/nodata"), 4)
        self.assertAlmostEqual(0, obj.get("/image1/what/undetect"), 4)
        
        # /what
        self.assertEquals("IMAGE", obj.get("/what/object"))
        self.assertEquals(1, obj.get("/what/sets"))
        self.assertEquals("H5rad 1.2", obj.get("/what/version"))
        self.assertEquals("20060531", obj.get("/what/date"))
        self.assertEquals("120000", obj.get("/what/time"))
        
        # /where
        self.assertEquals("+proj=gnom +a=6371000.0 +lat_0=56.367500 +lon_0=12.854400", obj.get("/where/projdef"))
        self.assertEquals(240, obj.get("/where/xsize"))
        self.assertEquals(240, obj.get("/where/ysize"))
        self.assertAlmostEqual(2000, obj.get("/where/xscale"), 4)
        self.assertAlmostEqual(2000, obj.get("/where/yscale"), 4)
        self.assertAlmostEqual(9.16975, obj.get("/where/LL_lon"), 4)
        self.assertAlmostEqual(54.1529, obj.get("/where/LL_lat"), 4)
        self.assertAlmostEqual(16.9802, obj.get("/where/UR_lon"), 4)
        self.assertAlmostEqual(58.4596, obj.get("/where/UR_lat"), 4)

    def testVerifyImageAttributesWritten(self):
        obj = rave.open(self.OLD_NRD_FORMAT_TESTFILE)
        self.assertEquals(None, obj.get("/image1/data/CLASS"))
        self.assertEquals(None, obj.get("/image1/data/IMAGE_VERSION"))

        obj.save(self.TESTFILE)
        
        # Verify
        nodelist = _pyhl.read_nodelist(self.TESTFILE)
        node = nodelist.fetchNode("/image1/data/CLASS")
        self.assertEquals("IMAGE", node.data())
        node = nodelist.fetchNode("/image1/data/IMAGE_VERSION")
        self.assertEquals("1.2", node.data())
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()