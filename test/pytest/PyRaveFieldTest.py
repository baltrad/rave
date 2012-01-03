'''
Created on Jul 5, 2010

@author: anders
'''
import unittest
import _ravefield
import _rave
import string
import numpy

class PyRaveFieldTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _ravefield.new()
    self.assertNotEqual(-1, string.find(`type(obj)`, "RaveFieldCore"))
  
  def test_attribute_visibility(self):
    attrs = ['datatype', 'xsize', 'ysize']
    obj = _ravefield.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)

  def test_attributes(self):
    obj = _ravefield.new()
    obj.addAttribute("what/is", 10.0)
    obj.addAttribute("where/is", "that")
    obj.addAttribute("how/are", 5)

    names = obj.getAttributeNames()
    self.assertEquals(3, len(names))
    self.assertTrue("what/is" in names)
    self.assertTrue("where/is" in names)
    self.assertTrue("how/are" in names)
    
    self.assertAlmostEqual(10.0, obj.getAttribute("what/is"), 4)
    self.assertEquals("that", obj.getAttribute("where/is"))
    self.assertEquals(5, obj.getAttribute("how/are"))

  def test_bad_names(self):
    obj = _ravefield.new()
    BAD_NAMES = ["xyz/is", "what", "is"]
    for n in BAD_NAMES:
      try:
        obj.addAttribute(n, 5)
        self.fail("Expected AttributeError")
      except AttributeError, e:
        pass

  def test_data(self):
    obj = _ravefield.new()
    obj.setData(numpy.zeros((10,10), numpy.uint8))
    obj.setValue(0,1,10.0)
    obj.setValue(5,4,20.0)
    
    self.assertEquals(10, obj.xsize)
    self.assertEquals(10, obj.ysize)
    self.assertEquals(_rave.RaveDataType_UCHAR, obj.datatype)
    
    self.assertAlmostEquals(10.0, obj.getValue(0,1)[1], 4)
    self.assertAlmostEquals(20.0, obj.getValue(5,4)[1], 4)
    
    data = obj.getData()
    self.assertAlmostEquals(10.0, data[1][0], 4)
    self.assertAlmostEquals(20.0, data[4][5], 4)

  def test_concatx(self):
    obj = _ravefield.new()
    obj.setData(numpy.zeros((10,10), numpy.uint8))
    obj.setValue(0,1,10.0)
    obj.setValue(5,4,20.0)
    obj.setValue(9,4,35.0)

    obj2 = _ravefield.new()
    obj2.setData(numpy.zeros((10,6), numpy.uint8))
    obj2.setValue(0,1,15.0)
    obj2.setValue(5,4,25.0)

    obj2.setValue(0,4,36.0)
    obj2.setValue(1,4,37.0)

    result = obj.concatx(obj2)
    self.assertEquals(16, result.xsize)
    self.assertEquals(10, result.ysize)
    self.assertAlmostEquals(10.0, result.getValue(0,1)[1], 4)
    self.assertAlmostEquals(20.0, result.getValue(5,4)[1], 4)
    self.assertAlmostEquals(15.0, result.getValue(10,1)[1], 4)
    self.assertAlmostEquals(25.0, result.getValue(15,4)[1], 4)
    self.assertAlmostEquals(35.0, result.getValue(9,4)[1], 4)
    self.assertAlmostEquals(36.0, result.getValue(10,4)[1], 4)
    self.assertAlmostEquals(37.0, result.getValue(11,4)[1], 4)

  def test_concatx_differentY(self):
    obj = _ravefield.new()
    obj.setData(numpy.zeros((10,10), numpy.uint8))
    obj.setValue(0,1,10.0)
    obj.setValue(5,4,20.0)

    obj2 = _ravefield.new()
    obj2.setData(numpy.zeros((9,6), numpy.uint8))
    obj2.setValue(0,1,15.0)
    obj2.setValue(5,4,25.0)

    try:
      obj.concatx(obj2)
      self.fail("Expected ValueError")
    except ValueError:
      pass 

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()