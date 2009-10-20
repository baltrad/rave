'''
Created on Oct 14, 2009
@author: Anders Henja
'''
import unittest
import os
import _rave
import string
import _helpers

class RaveModulePolarVolumeTest(unittest.TestCase):
  def setUp(self):
    _helpers.triggerMemoryStatus()

  def tearDown(self):
    pass
    
  def testNewVolume(self):
    obj = _rave.volume()
    
    result = string.find(`type(obj)`, "PolarVolumeCore")
    self.assertNotEqual(-1, result) 

  def testVolume_longitude(self):
    obj = _rave.volume()
    self.assertAlmostEquals(0.0, obj.longitude, 4)
    obj.longitude = 10.0
    self.assertAlmostEquals(10.0, obj.longitude, 4)

  def testVolume_longitude_typeError(self):
    obj = _rave.volume()
    self.assertAlmostEquals(0.0, obj.longitude, 4)
    try:
      obj.longitude = 10
      self.fail("Excepted TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.longitude, 4)

  def testVolume_latitude(self):
    obj = _rave.volume()
    self.assertAlmostEquals(0.0, obj.latitude, 4)
    obj.latitude = 10.0
    self.assertAlmostEquals(10.0, obj.latitude, 4)

  def testVolume_latitude_typeError(self):
    obj = _rave.volume()
    self.assertAlmostEquals(0.0, obj.latitude, 4)
    try:
      obj.latitude = 10
      self.fail("Excepted TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.latitude, 4)

  def testVolume_height(self):
    obj = _rave.volume()
    self.assertAlmostEquals(0.0, obj.height, 4)
    obj.height = 10.0
    self.assertAlmostEquals(10.0, obj.height, 4)

  def testVolume_height_typeError(self):
    obj = _rave.volume()
    self.assertAlmostEquals(0.0, obj.height, 4)
    try:
      obj.height = 10
      self.fail("Excepted TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.height, 4)
    
  def testVolume_addScan(self):
    obj = _rave.volume()
    obj.addScan(_rave.scan())
    self.assertEquals(1, obj.getNumberOfScans())

  def testVolume_getNumberOfScans(self):
    obj = _rave.volume()
    self.assertEquals(0, obj.getNumberOfScans())
    obj.addScan(_rave.scan())
    self.assertEquals(1, obj.getNumberOfScans())
    obj.addScan(_rave.scan())
    self.assertEquals(2, obj.getNumberOfScans())
    
  def testVolume_getScan(self):
    obj = _rave.volume()
    scan1 = _rave.scan()
    scan2 = _rave.scan()
    
    obj.addScan(scan1)
    obj.addScan(scan2)

    scanresult1 = obj.getScan(0)
    scanresult2 = obj.getScan(1)
    
    self.assertTrue (scan1 == scanresult1)
    self.assertTrue (scan2 == scanresult2)

  def testSortByElevations_ascending(self):
    obj = _rave.volume()
    scan1 = _rave.scan()
    scan1.elangle = 2.0
    scan2 = _rave.scan()
    scan2.elangle = 3.0
    scan3 = _rave.scan()
    scan3.elangle = 1.0
    
    obj.addScan(scan1)
    obj.addScan(scan2)
    obj.addScan(scan3)
    
    obj.sortByElevations(1)
    
    scanresult1 = obj.getScan(0)
    scanresult2 = obj.getScan(1)
    scanresult3 = obj.getScan(2)
    
    self.assertTrue (scan3 == scanresult1)
    self.assertTrue (scan1 == scanresult2)
    self.assertTrue (scan2 == scanresult3)

  def testSortByElevations_descending(self):
    obj = _rave.volume()
    scan1 = _rave.scan()
    scan1.elangle = 2.0
    scan2 = _rave.scan()
    scan2.elangle = 3.0
    scan3 = _rave.scan()
    scan3.elangle = 1.0
    
    obj.addScan(scan1)
    obj.addScan(scan2)
    obj.addScan(scan3)
    
    obj.sortByElevations(0)
    
    scanresult1 = obj.getScan(0)
    scanresult2 = obj.getScan(1)
    scanresult3 = obj.getScan(2)
    
    self.assertTrue (scan2 == scanresult1)
    self.assertTrue (scan1 == scanresult2)
    self.assertTrue (scan3 == scanresult3)
        
  def testVolume_ppi(self):
    pass
  

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()