'''
Created on Aug 12, 2009

@author: anders
'''
import unittest
import rave_IO
import rave_info

class RaveIOTest(unittest.TestCase):
    # Old nordrad format for a pcappi, (not including IMAGE attributes)
    OLD_NRD_FORMAT_TESTFILE="fixture_old_pcappi-dbz-500.ang-gnom-2000.h5"

    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testOpenHdf5_oldNordradPcappi(self):
        info, data, items = rave_IO.open_hdf5(self.OLD_NRD_FORMAT_TESTFILE)
        self.assertTrue(isinstance(info, rave_info.INFO))
        self.assertTrue(isinstance(data, dict))
        self.assertTrue(isinstance(items, list))

        self.assertEquals(2626, info.get("/how/WMO"))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()