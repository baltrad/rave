'''
Created on Jul 5, 2010

@author: anders
'''
import unittest
import _ravefield
import string
import numpy

class PyRaveFieldTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _ravefield.new()
    self.assertNotEqual(-1, str(type(obj)).find("RaveFieldCore"))
  
  def test_attribute_visibility(self):
    attrs = ['datatype', 'xsize', 'ysize']
    obj = _ravefield.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)

  def test_attributes(self):
    obj = _ravefield.new()
    obj.addAttribute("what/is", 10.0)
    obj.addAttribute("where/is", "that")
    obj.addAttribute("how/are", 5)

    names = obj.getAttributeNames()
    self.assertEqual(3, len(names))
    self.assertTrue("what/is" in names)
    self.assertTrue("where/is" in names)
    self.assertTrue("how/are" in names)
    
    self.assertAlmostEqual(10.0, obj.getAttribute("what/is"), 4)
    self.assertEqual("that", obj.getAttribute("where/is"))
    self.assertEqual(5, obj.getAttribute("how/are"))

  def test_removeAttributes(self):
    obj = _ravefield.new()
    obj.addAttribute("what/is", 10.0)
    obj.addAttribute("where/is", "that")
    obj.addAttribute("how/are", 5)

    obj.removeAttributes()
    
    names = obj.getAttributeNames()
    self.assertEqual(0, len(names))

  def test_howSubgroupAttribute(self):
    obj = _ravefield.new()

    obj.addAttribute("how/something", 1.0)
    obj.addAttribute("how/grp/something", 2.0)
    try:
      obj.addAttribute("how/grp/else/", 2.0)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

    self.assertAlmostEqual(1.0, obj.getAttribute("how/something"), 2)
    self.assertAlmostEqual(2.0, obj.getAttribute("how/grp/something"), 2)
    self.assertTrue(obj.hasAttribute("how/something"))
    self.assertTrue(obj.hasAttribute("how/grp/something"))

  def test_hasAttribute(self):
    obj = _ravefield.new()

    obj.addAttribute("how/something", 1.0)
    self.assertTrue(obj.hasAttribute("how/something"))
    self.assertFalse(obj.hasAttribute("how/somethingelse"))

  def test_bad_names(self):
    obj = _ravefield.new()
    BAD_NAMES = ["xyz/is", "what", "is"]
    for n in BAD_NAMES:
      try:
        obj.addAttribute(n, 5)
        self.fail("Expected AttributeError")
      except AttributeError:
        pass

  def test_getConvertedValue_with_default_values(self):
    obj = _ravefield.new()
    obj.setData(numpy.zeros((10,10), numpy.uint8))
    obj.setValue(0,1,1.0)
    obj.setValue(0,2,2.0)
    obj.setValue(0,3,3.0)
    obj.setValue(0,4,4.0)
    self.assertAlmostEqual(1.0, obj.getConvertedValue(0,1)[1], 4)
    self.assertAlmostEqual(2.0, obj.getConvertedValue(0,2)[1], 4)
    self.assertAlmostEqual(3.0, obj.getConvertedValue(0,3)[1], 4)
    self.assertAlmostEqual(4.0, obj.getConvertedValue(0,4)[1], 4)

  def test_getConvertedValue_with_offset_and_gain(self):
    obj = _ravefield.new()
    obj.setData(numpy.zeros((10,10), numpy.uint8))
    obj.setValue(0,1,1.0)
    obj.setValue(0,2,2.0)
    obj.setValue(0,3,3.0)
    obj.setValue(0,4,4.0)
    obj.addAttribute("what/offset", 5.0)
    obj.addAttribute("what/gain", 10.0)
    self.assertAlmostEqual(5 + 1.0 * 10.0, obj.getConvertedValue(0,1)[1], 4)
    self.assertAlmostEqual(5 + 2.0 * 10.0, obj.getConvertedValue(0,2)[1], 4)
    self.assertAlmostEqual(5 + 3.0 * 10.0, obj.getConvertedValue(0,3)[1], 4)
    self.assertAlmostEqual(5 + 4.0 * 10.0, obj.getConvertedValue(0,4)[1], 4)

  def Xtest_data(self):
    obj = _ravefield.new()
    obj.setData(numpy.zeros((10,10), numpy.uint8))
    obj.setValue(0,1,10.0)
    obj.setValue(5,4,20.0)
    
    self.assertEqual(10, obj.xsize)
    self.assertEqual(10, obj.ysize)
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)
    
    self.assertAlmostEqual(10.0, obj.getValue(0,1)[1], 4)
    self.assertAlmostEqual(20.0, obj.getValue(5,4)[1], 4)
    
    data = obj.getData()
    self.assertAlmostEqual(10.0, data[1][0], 4)
    self.assertAlmostEqual(20.0, data[4][5], 4)

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
    self.assertEqual(16, result.xsize)
    self.assertEqual(10, result.ysize)
    self.assertAlmostEqual(10.0, result.getValue(0,1)[1], 4)
    self.assertAlmostEqual(20.0, result.getValue(5,4)[1], 4)
    self.assertAlmostEqual(15.0, result.getValue(10,1)[1], 4)
    self.assertAlmostEqual(25.0, result.getValue(15,4)[1], 4)
    self.assertAlmostEqual(35.0, result.getValue(9,4)[1], 4)
    self.assertAlmostEqual(36.0, result.getValue(10,4)[1], 4)
    self.assertAlmostEqual(37.0, result.getValue(11,4)[1], 4)

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

  def test_circshiftData_00(self):
    obj = _ravefield.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.circshiftData(0,0)
    self.assertTrue((numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8)==obj.getData()).all())

  def test_circshiftData_y(self):
    obj = _ravefield.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.circshiftData(0,1)
    self.assertTrue((numpy.array([[12,13,14,15],[0,1,2,3],[4,5,6,7],[8,9,10,11]],numpy.uint8)==obj.getData()).all())

  def test_circshiftData_neg_y(self):
    obj = _ravefield.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.circshiftData(0,-1)
    self.assertTrue((numpy.array([[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,1,2,3]],numpy.uint8)==obj.getData()).all())

  def test_circshiftData_x(self):
    obj = _ravefield.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.circshiftData(1,0)
    self.assertTrue((numpy.array([[3,0,1,2],[7,4,5,6],[11,8,9,10],[15,12,13,14]],numpy.uint8)==obj.getData()).all())

  def test_circshiftData_neg_x(self):
    obj = _ravefield.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.circshiftData(-1,0)
    self.assertTrue((numpy.array([[1,2,3,0],[5,6,7,4],[9,10,11,8],[13,14,15,12]],numpy.uint8)==obj.getData()).all())


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()