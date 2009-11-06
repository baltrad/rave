'''
Created on Oct 14, 2009
@author: Anders Henja
'''
import unittest
import os
import _rave
import string
import _helpers
import math
import numpy

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

  def testVolume_erroneous_member(self):
    pass

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

  def testVolume_getScanNearestElevation(self):
    obj = _rave.volume()
    scan1 = _rave.scan()
    scan1.elangle = 0.1 * math.pi / 180.0
    obj.addScan(scan1)
    scan2 = _rave.scan()
    scan2.elangle = 0.5 * math.pi / 180.0
    obj.addScan(scan2)
    scan3 = _rave.scan()
    scan3.elangle = 2.0 * math.pi / 180.0
    obj.addScan(scan3)
    
    els = [(0.0, 0.1), (0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.31, 0.5), (1.0, 0.5), (2.0, 2.0)]

    for el in els:
      elevation = el[0]*math.pi / 180.0
      result = obj.getScanNearestElevation(elevation)
      self.assertAlmostEquals(el[1], result.elangle*180.0/math.pi, 5)

  def testVolume_getNearest(self):
    obj = _rave.volume()
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    scan1 = _rave.scan()
    scan1.elangle = 0.1 * math.pi / 180.0
    scan1.rstart = 0.0
    scan1.rscale = 5000.0
    scan1.nodata = 10.0
    scan1.undetect = 11.0
    data = numpy.zeros((100, 120), numpy.uint8)
    scan1.setData(data)
    
    scan2 = _rave.scan()
    scan2.elangle = 1.0 * math.pi / 180.0
    scan2.rstart = 0.0
    scan2.rscale = 5000.0
    scan2.nodata = 10.0
    scan2.undetect = 11.0    
    data = numpy.ones((100, 120), numpy.uint8)
    scan2.setData(data)
    
    obj.addScan(scan1)
    obj.addScan(scan2)
    
    t,v = obj.getNearest((12.0*math.pi/180.0, 60.45*math.pi/180.0), 1000.0)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(1.0, v, 4)

    t,v = obj.getNearest((12.0*math.pi/180.0, 62.00*math.pi/180.0), 1000.0)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(0.0, v, 4)

  def testVolume_getNearestForElevation(self):
    obj = _rave.volume()
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    scan1 = _rave.scan()
    scan1.elangle = 0.1 * math.pi / 180.0
    scan1.rstart = 0.0
    scan1.rscale = 5000.0
    scan1.nodata = 10.0
    scan1.undetect = 11.0
    data = numpy.zeros((10, 10), numpy.uint8)
    scan1.setData(data)
    
    scan2 = _rave.scan()
    scan2.elangle = 1.0 * math.pi / 180.0
    scan2.rstart = 0.0
    scan2.rscale = 5000.0
    scan2.nodata = 10.0
    scan2.undetect = 11.0    
    data = numpy.ones((10, 10), numpy.uint8)
    scan2.setData(data)
    
    obj.addScan(scan1)
    obj.addScan(scan2)
    
    t,v = obj.getNearestForElevation((12.0*math.pi/180.0, 60.40*math.pi/180.0), 0)
    self.assertEquals(_rave.RaveValueType_DATA, t)
    self.assertAlmostEquals(0.0, v, 4)

    t,v = obj.getNearestForElevation((12.0*math.pi/180.0, 61.00*math.pi/180.0), 0)
    self.assertEquals(_rave.RaveValueType_NODATA, t)
    self.assertAlmostEquals(10.0, v, 4)
    
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