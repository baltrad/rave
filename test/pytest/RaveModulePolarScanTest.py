'''
Created on Oct 14, 2009
@author: Anders Henja
'''
import unittest
import os
import _rave
import string
import numpy
import math

class RaveModulePolarScanTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testNewScan(self):
    obj = _rave.scan()
    
    isscan = string.find(`type(obj)`, "PolarScanCore")
    self.assertNotEqual(-1, isscan) 

  def testScan_elangle(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.elangle, 4)
    obj.elangle = 10.0
    self.assertAlmostEquals(10.0, obj.elangle, 4)

  def testScan_elangle_typeError(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.elangle, 4)
    try:
      obj.elangle = 10
      self.fail("Excepted TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.elangle, 4)

  def testScan_nbins(self):
    obj = _rave.scan()
    self.assertEquals(0, obj.nbins)
    obj.nbins = 10
    self.assertEquals(10, obj.nbins)

  def testScan_nbins_typeError(self):
    obj = _rave.scan()
    self.assertEquals(0, obj.nbins)
    try:
      obj.nbins = 10.0
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(0, obj.nbins)

  def testScan_rscale(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.rscale, 4)
    obj.rscale = 10.0
    self.assertAlmostEquals(10.0, obj.rscale, 4)

  def testScan_rscale_typeError(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.rscale, 4)
    try:
      obj.rscale = 10
      self.fail("Expected TypeError")
    except TypeError, e:
      pass
    self.assertAlmostEquals(0.0, obj.rscale, 4)

  def testScan_nrays(self):
    obj = _rave.scan()
    self.assertEquals(0, obj.nrays)
    obj.nrays = 10
    self.assertEquals(10, obj.nrays)

  def testScan_nrays_typeError(self):
    obj = _rave.scan()
    self.assertEquals(0, obj.nrays)
    try:
      obj.nrays = 10.0
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(0, obj.nrays)

  def testScan_rstart(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.rstart, 4)
    obj.rstart = 10.0
    self.assertAlmostEquals(10.0, obj.rstart, 4)

  def testScan_rstart_typeError(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.rstart, 4)
    try:
      obj.rstart = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.rstart, 4)

  def testScan_datatype(self):
    obj = _rave.scan()
    self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)
    obj.datatype = _rave.RaveDataType_INT
    self.assertEqual(_rave.RaveDataType_INT, obj.datatype)

  def testScan_setValidDataTypes(self):
    types = [_rave.RaveDataType_UNDEFINED, _rave.RaveDataType_CHAR, _rave.RaveDataType_UCHAR,
             _rave.RaveDataType_SHORT, _rave.RaveDataType_INT, _rave.RaveDataType_LONG,
             _rave.RaveDataType_FLOAT, _rave.RaveDataType_DOUBLE]

    obj = _rave.scan()
    for type in types:
      obj.datatype = type
      self.assertEqual(type, obj.datatype)

  def testScan_invalidDatatype(self):
    obj = _rave.scan()
    types = [99,100,-2,30]
    for type in types:
      try:
        obj.datatype = type
        self.fail("Expected ValueError")
      except ValueError, e:
        self.assertEqual(_rave.RaveDataType_UNDEFINED, obj.datatype)

  def testScan_a1gate(self):
    obj = _rave.scan()
    self.assertEquals(0, obj.a1gate)
    obj.a1gate = 10
    self.assertEquals(10, obj.a1gate)

  def testScan_a1gate_typeError(self):
    obj = _rave.scan()
    self.assertEquals(0, obj.a1gate)
    try:
      obj.a1gate = 10.0
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(0, obj.a1gate)

  def testScan_beamwidth(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.beamwidth, 4)
    obj.beamwidth = 10.0
    self.assertAlmostEquals(10.0, obj.beamwidth, 4)

  def testScan_beamwidth_typeError(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.beamwidth, 4)
    try:
      obj.beamwidth = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.beamwidth, 4)

  def testScan_quantity(self):
    obj = _rave.scan()
    self.assertEquals("", obj.quantity)
    obj.quantity = "DBZH"
    self.assertEquals("DBZH", obj.quantity)

  def testScan_quantity_typeError(self):
    obj = _rave.scan()
    self.assertEquals("", obj.quantity)
    try:
      obj.quantity = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals("", obj.quantity)

  def testScan_gain(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    obj.gain = 10.0
    self.assertAlmostEquals(10.0, obj.gain, 4)

  def testScan_gain_typeError(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    try:
      obj.gain = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.gain, 4)

  def testScan_offset(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    obj.offset = 10.0
    self.assertAlmostEquals(10.0, obj.offset, 4)

  def testScan_offset_typeError(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    try:
      obj.offset = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.offset, 4)

  def testScan_nodata(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    obj.nodata = 10.0
    self.assertAlmostEquals(10.0, obj.nodata, 4)

  def testScan_nodata_typeError(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    try:
      obj.nodata = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.nodata, 4)

  def testScan_undetect(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    obj.undetect = 10.0
    self.assertAlmostEquals(10.0, obj.undetect, 4)

  def testScan_undetect_typeError(self):
    obj = _rave.scan()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    try:
      obj.undetect = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.undetect, 4)

  def testScan_getRangeIndex(self):
    obj = _rave.scan()
    obj.nbins = 200
    obj.rscale = 1000.0

    # Ranges are tuples (range, expected index)
    ranges = [(499.0, 0),
              (501.0, 1),
              (199000.0, 199),
              (200000.0, -1)]
    
    for rr in ranges:
      result = obj.getRangeIndex(rr[0])
      self.assertEquals(rr[1], result)
  
  def testScan_getAzimuthIndex(self):
    obj = _rave.scan()
    obj.nrays = 400
    # Azimuths tuple is ordered by an azimuth in degrees and expected index
    azimuths = [(180.0, 200),
                (90.0, 100),
                (0.0, 0),
                (0.1, 0),
                (0.4, 0),
                (359.9, 0),
                (360.4, 0),
                (-0.1,0),
                (-1.0,399)]
    for azv in azimuths:
      result = obj.getAzimuthIndex(azv[0]*math.pi/180.0)
      self.assertEquals(azv[1], result)

  def testScan_getValueAtIndex(self):
    obj = _rave.scan()
    obj.nodata = 255.0
    obj.undetect = 0.0
    a=numpy.arange(30)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(5,6)).astype(numpy.float64)      
    a[0][0] = obj.undetect
    a[2][1] = obj.nodata
    a[4][5] = obj.undetect

    obj.setData(a)

    pts = [((0,0), (_rave.RaveValueType_UNDETECT, 0.0)),
           ((0,1), (_rave.RaveValueType_DATA, 1.0)),
           ((1,0), (_rave.RaveValueType_DATA, 6.0)),
           ((2,1), (_rave.RaveValueType_NODATA, obj.nodata)),
           ((4,4), (_rave.RaveValueType_DATA, 28.0)),
           ((4,5), (_rave.RaveValueType_UNDETECT, obj.undetect)),
           ((5,5), (_rave.RaveValueType_NODATA, obj.nodata))]
    
    for tval in pts:
      result = obj.getValueAtIndex(tval[0])
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

  def testScan_getValueAtAzimuthAndRange(self):
    obj = _rave.scan()
    obj.nodata = 255.0
    obj.undetect = 0.0
    obj.rscale = 1000.0
    a=numpy.arange(3600)
    a=numpy.array(a.astype(numpy.float64),numpy.float64)
    a=numpy.reshape(a,(360,10)).astype(numpy.float64)      
    a[0][0] = obj.undetect
    a[2][1] = obj.nodata
    a[4][5] = obj.undetect

    obj.setData(a)

    pts = [((0.0,0.0), (_rave.RaveValueType_UNDETECT, 0.0)),        #0, 0
           ((10.0,2000.0), (_rave.RaveValueType_DATA, 102.0)),      #10*10 + 2
           ((20.0,9000), (_rave.RaveValueType_DATA, 209.0)),        #20*10 + 9
           ((2.0,1000), (_rave.RaveValueType_NODATA, obj.nodata)),  #10*20 + 9
           ((4.0,5000), (_rave.RaveValueType_UNDETECT, obj.undetect))]
    
    for tval in pts:
      az = tval[0][0]*math.pi/180.0
      ra = tval[0][1]
      result = obj.getValueAtAzimuthAndRange((az,ra))
      self.assertEquals(tval[1][0], result[0])
      self.assertAlmostEquals(tval[1][1], result[1], 4)

  def testScan_setData_int8(self):
    obj = _rave.scan()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int8),numpy.int8)
    a=numpy.reshape(a,(12,10)).astype(numpy.int8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_CHAR, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)


  def testScan_setData_uint8(self):
    obj = _rave.scan()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint8),numpy.uint8)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint8)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)

  def testScan_setData_int16(self):
    obj = _rave.scan()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.int16),numpy.int16)
    a=numpy.reshape(a,(12,10)).astype(numpy.int16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_SHORT, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)

  def testScan_setData_uint16(self):
    obj = _rave.scan()
    a=numpy.arange(120)
    a=numpy.array(a.astype(numpy.uint16),numpy.uint16)
    a=numpy.reshape(a,(12,10)).astype(numpy.uint16)    
    
    obj.setData(a)
    
    self.assertEqual(_rave.RaveDataType_SHORT, obj.datatype)
    self.assertEqual(10, obj.nbins)
    self.assertEqual(12, obj.nrays)

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()