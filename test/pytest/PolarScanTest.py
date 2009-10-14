'''
Created on Oct 14, 2009
@author: Anders Henja
'''
import unittest
import os
import _polarvolume
import string

class PolarScanTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testNewScan(self):
    obj = _polarvolume.scan()
    
    isscan = string.find(`type(obj)`, "PolarScanCore")
    self.assertNotEqual(-1, isscan) 

  def testScan_elangle(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.elangle, 4)
    obj.elangle = 10.0
    self.assertAlmostEquals(10.0, obj.elangle, 4)

  def testScan_elangle_typeError(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.elangle, 4)
    try:
      obj.elangle = 10
      self.fail("Excepted TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.elangle, 4)

  def testScan_nbins(self):
    obj = _polarvolume.scan()
    self.assertEquals(0, obj.nbins)
    obj.nbins = 10
    self.assertEquals(10, obj.nbins)

  def testScan_nbins_typeError(self):
    obj = _polarvolume.scan()
    self.assertEquals(0, obj.nbins)
    try:
      obj.nbins = 10.0
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(0, obj.nbins)

  def testScan_rscale(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.rscale, 4)
    obj.rscale = 10.0
    self.assertAlmostEquals(10.0, obj.rscale, 4)

  def testScan_rscale_typeError(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.rscale, 4)
    try:
      obj.rscale = 10
      self.fail("Expected TypeError")
    except TypeError, e:
      pass
    self.assertAlmostEquals(0.0, obj.rscale, 4)

  def testScan_nrays(self):
    obj = _polarvolume.scan()
    self.assertEquals(0, obj.nrays)
    obj.nrays = 10
    self.assertEquals(10, obj.nrays)

  def testScan_nrays_typeError(self):
    obj = _polarvolume.scan()
    self.assertEquals(0, obj.nrays)
    try:
      obj.nrays = 10.0
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(0, obj.nrays)

  def testScan_rstart(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.rstart, 4)
    obj.rstart = 10.0
    self.assertAlmostEquals(10.0, obj.rstart, 4)

  def testScan_rstart_typeError(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.rstart, 4)
    try:
      obj.rstart = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.rstart, 4)

  def testScan_a1gate(self):
    obj = _polarvolume.scan()
    self.assertEquals(0, obj.a1gate)
    obj.a1gate = 10
    self.assertEquals(10, obj.a1gate)

  def testScan_a1gate_typeError(self):
    obj = _polarvolume.scan()
    self.assertEquals(0, obj.a1gate)
    try:
      obj.a1gate = 10.0
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals(0, obj.a1gate)

  def testScan_quantity(self):
    obj = _polarvolume.scan()
    self.assertEquals("", obj.quantity)
    obj.quantity = "DBZH"
    self.assertEquals("DBZH", obj.quantity)

  def testScan_quantity_typeError(self):
    obj = _polarvolume.scan()
    self.assertEquals("", obj.quantity)
    try:
      obj.quantity = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertEquals("", obj.quantity)

  def testScan_gain(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    obj.gain = 10.0
    self.assertAlmostEquals(10.0, obj.gain, 4)

  def testScan_gain_typeError(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.gain, 4)
    try:
      obj.gain = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.gain, 4)

  def testScan_offset(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    obj.offset = 10.0
    self.assertAlmostEquals(10.0, obj.offset, 4)

  def testScan_offset_typeError(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.offset, 4)
    try:
      obj.offset = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.offset, 4)

  def testScan_nodata(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    obj.nodata = 10.0
    self.assertAlmostEquals(10.0, obj.nodata, 4)

  def testScan_nodata_typeError(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.nodata, 4)
    try:
      obj.nodata = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.nodata, 4)

  def testScan_undetect(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    obj.undetect = 10.0
    self.assertAlmostEquals(10.0, obj.undetect, 4)

  def testScan_undetect_typeError(self):
    obj = _polarvolume.scan()
    self.assertAlmostEquals(0.0, obj.undetect, 4)
    try:
      obj.undetect = 10
      self.fail("Expected TypeError")
    except TypeError,e:
      pass
    self.assertAlmostEquals(0.0, obj.undetect, 4)

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()