'''
Created on Jul 5, 2010

@author: anders
'''
import unittest
import _ravedata2d
import string
import numpy

class PyRaveRaveData2DTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _ravedata2d.new()
    self.assertNotEqual(-1, str(type(obj)).find("RaveData2DCore"))
    self.assertEqual(0, obj.xsize)
    self.assertEqual(0, obj.ysize)

  def test_repr(self):
    obj = _ravedata2d.new(numpy.array([[ 0,  1,  2,  3,  5],
                             [ 4,  5,  6,  7,  8],
                             [ 9, 10, 11, 12, 13],
                             [14, 15, 16, 17, 18]], numpy.uint8))
    self.assertEqual("""(5 x 4) [
   [0.000, 1.000, 2.000, 3.000, 5.000],
   [4.000, 5.000, 6.000, 7.000, 8.000],
   [9.000, 10.000, 11.000, 12.000, 13.000],
   [14.000, 15.000, 16.000, 17.000, 18.000]];""", obj.str())
  
  def test_new_with_array(self):
    obj = _ravedata2d.new(numpy.array([[ 0,  1,  2,  3,  5],
                             [ 4,  5,  6,  7,  8],
                             [ 9, 10, 11, 12, 13],
                             [14, 15, 16, 17, 18]], numpy.uint8))
    self.assertNotEqual(-1, str(type(obj)).find("RaveData2DCore"))
    self.assertEqual(5, obj.xsize)
    self.assertEqual(4, obj.ysize)

  
  def test_attribute_visibility(self):
    attrs = ['datatype', 'xsize', 'ysize', 'nodata', 'useNodata']
    obj = _ravedata2d.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)

  def test_useNodata(self):
    obj = _ravedata2d.new()
    self.assertEqual(False, obj.useNodata);
    obj.useNodata = True
    self.assertEqual(True, obj.useNodata);

  def test_nodata(self):
    obj = _ravedata2d.new()
    self.assertEqual(255, obj.nodata);
    obj.nodata = 10
    self.assertEqual(10, obj.nodata);
    obj.nodata = 10.87
    self.assertAlmostEqual(10.87, obj.nodata, 3);

  def test_getConvertedValue_with_default_values(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.zeros((10,10), numpy.uint8))
    obj.setValue(0,1,1.0)
    obj.setValue(0,2,2.0)
    obj.setValue(0,3,3.0)
    obj.setValue(0,4,4.0)
    self.assertAlmostEqual(1.0, obj.getValue(0,1)[1], 4)
    self.assertAlmostEqual(2.0, obj.getValue(0,2)[1], 4)
    self.assertAlmostEqual(3.0, obj.getValue(0,3)[1], 4)
    self.assertAlmostEqual(4.0, obj.getValue(0,4)[1], 4)

  def test_fill(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.zeros((10,10), numpy.uint8))
    obj.fill(2.0)
    for x in range(obj.xsize):
      for y in range(obj.ysize):
        self.assertAlmostEqual(2.0, obj.getValue(x,y)[1], 4);

  def test_concatx(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.zeros((10,10), numpy.uint8))
    obj.setValue(0,1,10.0)
    obj.setValue(5,4,20.0)
    obj.setValue(9,4,35.0)

    obj2 = _ravedata2d.new()
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
    obj = _ravedata2d.new()
    obj.setData(numpy.zeros((10,10), numpy.uint8))
    obj.setValue(0,1,10.0)
    obj.setValue(5,4,20.0)

    obj2 = _ravedata2d.new()
    obj2.setData(numpy.zeros((9,6), numpy.uint8))
    obj2.setValue(0,1,15.0)
    obj2.setValue(5,4,25.0)

    try:
      obj.concatx(obj2)
      self.fail("Expected ValueError")
    except ValueError:
      pass 

  def test_min(self):
    obj = _ravedata2d.new(numpy.array([[ 3,  2,  1,  6,  7]], numpy.uint8))
    self.assertAlmostEqual(1.0, obj.min(), 2)

  def test_min_with_nodata(self):
    obj = _ravedata2d.new(numpy.array([[ 3,  2,  1,  6,  7]], numpy.uint8))
    obj.nodata = 1.0
    obj.useNodata = True
    self.assertAlmostEqual(2.0, obj.min(), 2)
        
  def test_max(self):
    obj = _ravedata2d.new(numpy.array([[ 3,  2,  1,  6,  7, 1]], numpy.uint8))
    self.assertAlmostEqual(7.0, obj.max(), 2)

  def test_max_with_nodata(self):
    obj = _ravedata2d.new(numpy.array([[ 3,  2,  1,  6,  7, 1]], numpy.uint8))
    obj.nodata = 7.0
    obj.useNodata = True
    self.assertAlmostEqual(6.0, obj.max(), 2)

  def test_circshift_1(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[ 0,  1,  2,  3,  5],
                             [ 4,  5,  6,  7,  8],
                             [ 9, 10, 11, 12, 13],
                             [14, 15, 16, 17, 18]], numpy.uint8))
    expected_array = numpy.array([[ 2,  3,  5,  0,  1],
                                  [ 6,  7,  8,  4,  5],
                                  [11, 12, 13,  9, 10],
                                  [16, 17, 18, 14, 15]], numpy.uint8)
    result = obj.circshift(3, 0);
    self.assertTrue((expected_array == result.getData()).all())

  def test_circshift_2(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[0, 1, 1, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], numpy.uint8))
    expected_array = numpy.array([[1, 1, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [1, 1, 0, 0]], numpy.uint8)
    result = obj.circshift(-1, -1);
    self.assertTrue((expected_array == result.getData()).all())

  def test_circshift_3(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[0, 1, 1, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], numpy.uint8))
    expected_array = numpy.array([[0, 1, 1, 0],
                                  [0, 1, 1, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]], numpy.uint8)
    result = obj.circshift(0, 0);
    self.assertTrue((expected_array == result.getData()).all())

  def test_circshift_4(self): # Unreasonable high shifting
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[0, 1, 1, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], numpy.uint8))
    expected_array = numpy.array([[0, 1, 1, 0],
                                  [0, 1, 1, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]], numpy.uint8)
    result = obj.circshift(-8, -8);
    self.assertTrue((expected_array == result.getData()).all())
    
  def test_circshift_5(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[0, 1, 1, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], numpy.uint8))
    expected_array = numpy.array([[0, 0, 1, 1],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 1, 1]], numpy.uint8)
    result = obj.circshift(1, -1);
    self.assertTrue((expected_array == result.getData()).all())

  def test_circshift_6(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[0, 1, 1, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], numpy.uint8))
    expected_array = numpy.array([[0, 0, 0, 0],
                                  [0, 0, 1, 1],
                                  [0, 0, 1, 1],
                                  [0, 0, 0, 0]], numpy.uint8)
    result = obj.circshift(1, 1);
    self.assertTrue((expected_array == result.getData()).all())

  def test_circshift_7(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 1, 1, 0],
                             [0, 0, 1, 1]], numpy.uint8))
    expected_array = numpy.array([[1, 0, 0, 1],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 1, 1]], numpy.uint8)
    result = obj.circshift(1, 1);
    self.assertTrue((expected_array == result.getData()).all())

  def test_circshiftData_00(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.circshiftData(0,0)
    self.assertTrue((numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8)==obj.getData()).all())

  def test_circshiftData_y(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.circshiftData(0,1)
    self.assertTrue((numpy.array([[12,13,14,15],[0,1,2,3],[4,5,6,7],[8,9,10,11]],numpy.uint8)==obj.getData()).all())

  def test_circshiftData_neg_y(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.circshiftData(0,-1)
    self.assertTrue((numpy.array([[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,1,2,3]],numpy.uint8)==obj.getData()).all())

  def test_circshiftData_x(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.circshiftData(1,0)
    self.assertTrue((numpy.array([[3,0,1,2],[7,4,5,6],[11,8,9,10],[15,12,13,14]],numpy.uint8)==obj.getData()).all())

  def test_circshiftData_neg_x(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],numpy.uint8))
    obj.circshiftData(-1,0)
    self.assertTrue((numpy.array([[1,2,3,0],[5,6,7,4],[9,10,11,8],[13,14,15,12]],numpy.uint8)==obj.getData()).all())

  def test_add_number(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    expected_array = numpy.array([[ 3.0, 4.0, 5.0, 6.0],
                                  [ 7.0, 8.0, 9.0, 10.0],
                                  [ 10.0, 9.0, 8.0, 7.0],
                                  [ 6.0, 5.0, 4.0, 3.0]], numpy.int64)
    result = obj.add(2.0);
    self.assertTrue((expected_array == result.getData()).all())

  def test_add_number_with_nodata(self):
    obj = _ravedata2d.new()
    obj.useNodata = True
    obj.nodata = 1.0
    
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    expected_array = numpy.array([[ 1.0, 4.0, 5.0, 6.0],
                                  [ 7.0, 8.0, 9.0, 10.0],
                                  [ 10.0, 9.0, 8.0, 7.0],
                                  [ 6.0, 5.0, 4.0, 1.0]], numpy.int64)
    result = obj.add(2.0);
    self.assertTrue((expected_array == result.getData()).all())


  def test_add_data2d(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    other.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [8.0, 7.0, 6.0, 5.0],
                               [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    expected_array = numpy.array([[ 2.0, 4.0, 6.0, 8.0],
                                  [ 10.0, 12.0, 14.0, 16.0],
                                  [ 16.0, 14.0, 12.0, 10.0],
                                  [ 8.0, 6.0, 4.0, 2.0]], numpy.int64)
    result = obj.add(other);
    self.assertTrue((expected_array == result.getData()).all())

  def test_add_data2d_with_nodata(self):
    obj = _ravedata2d.new()
    obj.nodata = 1.0
    obj.useNodata = True
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    other.setData(numpy.array([[3.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [8.0, 7.0, 6.0, 5.0],
                               [4.0, 3.0, 2.0, 4.0]], numpy.int64))
    
    expected_array = numpy.array([[ 1.0, 4.0, 6.0, 8.0],
                                  [ 10.0, 12.0, 14.0, 16.0],
                                  [ 16.0, 14.0, 12.0, 10.0],
                                  [ 8.0, 6.0, 4.0, 1.0]], numpy.int64)
    result = obj.add(other);
    #print(str(result.getData()))
    self.assertTrue((expected_array == result.getData()).all())

  def test_add_data2d_other_with_nodata(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    other.nodata = 3.0
    other.useNodata = True
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    other.setData(numpy.array([[3.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [8.0, 7.0, 6.0, 5.0],
                               [4.0, 3.0, 2.0, 4.0]], numpy.int64))
    
    expected_array = numpy.array([[ 3.0, 4.0, 3.0, 8.0],
                                  [ 10.0, 12.0, 14.0, 16.0],
                                  [ 16.0, 14.0, 12.0, 10.0],
                                  [ 8.0, 3.0, 4.0, 5.0]], numpy.int64)
    result = obj.add(other);
    self.assertTrue((expected_array == result.getData()).all())
    
  def test_add_data2d_xsize_1(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    other.setData(numpy.array([[1.0, 2.0, 3.0, 4.0]], numpy.int64))
    
    expected_array = numpy.array([[ 2.0, 4.0, 6.0, 8.0],
                                  [ 6.0, 8.0, 10.0, 12.0],
                                  [ 9.0, 9.0, 9.0, 9.0],
                                  [ 5.0, 5.0, 5.0, 5.0]], numpy.int64)
    result = obj.add(other);
    self.assertTrue((expected_array == result.getData()).all())

  def test_add_data2d_ysize_1(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    other.setData(numpy.array([[1.0], 
                               [2.0], 
                               [3.0], 
                               [4.0]], numpy.int64))
    
    expected_array = numpy.array([[ 2.0, 3.0, 4.0, 5.0],
                                  [ 7.0, 8.0, 9.0, 10.0],
                                  [ 11.0, 10.0, 9.0, 8.0],
                                  [ 8.0, 7.0, 6.0, 5.0]], numpy.int64)
    result = obj.add(other);
    self.assertTrue((expected_array == result.getData()).all())

  def test_sub_number(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    expected_array = numpy.array([[-1.0, 0.0, 1.0, 2.0],
                                  [ 3.0, 4.0, 5.0, 6.0],
                                  [ 6.0, 5.0, 4.0, 3.0],
                                  [ 2.0, 1.0, 0.0, -1.0]], numpy.int64)
    result = obj.sub(2.0);
    self.assertTrue((expected_array == result.getData()).all())

  def test_sub_data2d(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    other.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [8.0, 7.0, 6.0, 5.0],
                               [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    expected_array = numpy.array([[ 0.0, 0.0, 0.0, 0.0],
                                  [ 0.0, 0.0, 0.0, 0.0],
                                  [ 0.0, 0.0, 0.0, 0.0],
                                  [ 0.0, 0.0, 0.0, 0.0]], numpy.int64)
    result = obj.sub(other);
    self.assertTrue((expected_array == result.getData()).all())

  def test_sub_data2d_xsize_1(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    other.setData(numpy.array([[1.0, 2.0, 3.0, 4.0]], numpy.int64))
    
    expected_array = numpy.array([[ 0.0, 0.0, 0.0, 0.0],
                                  [ 4.0, 4.0, 4.0, 4.0],
                                  [ 7.0, 5.0, 3.0, 1.0],
                                  [ 3.0, 1.0, -1.0, -3.0]], numpy.int64)
    result = obj.sub(other);
    self.assertTrue((expected_array == result.getData()).all())

  def test_sub_data2d_ysize_1(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    other.setData(numpy.array([[1.0], 
                               [2.0], 
                               [3.0], 
                               [4.0]], numpy.int64))
    
    expected_array = numpy.array([[ 0.0, 1.0, 2.0, 3.0],
                                  [ 3.0, 4.0, 5.0, 6.0],
                                  [ 5.0, 4.0, 3.0, 2.0],
                                  [ 0.0, -1.0, -2.0, -3.0]], numpy.int64)
    result = obj.sub(other);
    self.assertTrue((expected_array == result.getData()).all())

  def test_emul_number(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    expected_array = numpy.array([[ 2.0, 4.0, 6.0, 8.0],
                                  [ 10.0, 12.0, 14.0, 16.0],
                                  [ 16.0, 14.0, 12.0, 10.0],
                                  [ 8.0, 6.0, 4.0, 2.0]], numpy.int64)
    result = obj.emul(2.0);
    self.assertTrue((expected_array == result.getData()).all())

  def test_emul_data2d(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    other.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [8.0, 7.0, 6.0, 5.0],
                               [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    expected_array = numpy.array([[ 1.0, 4.0, 9.0, 16.0],
                                  [ 25.0, 36.0, 49.0, 64.0],
                                  [ 64.0, 49.0, 36.0, 25.0],
                                  [ 16.0, 9.0, 4.0, 1.0]], numpy.int64)
    result = obj.emul(other);
    self.assertTrue((expected_array == result.getData()).all())

  def test_emul_data2d_xsize_1(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    other.setData(numpy.array([[1.0, 2.0, 3.0, 4.0]], numpy.int64))
    
    expected_array = numpy.array([[ 1.0, 4.0, 9.0, 16.0],
                                  [ 5.0, 12.0, 21.0, 32.0],
                                  [ 8.0, 14.0, 18.0, 20.0],
                                  [ 4.0, 6.0, 6.0, 4.0]], numpy.int64)
    result = obj.emul(other);
    self.assertTrue((expected_array == result.getData()).all())

  def test_emul_data2d_ysize_1(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    
    other.setData(numpy.array([[1.0], 
                               [2.0], 
                               [3.0], 
                               [4.0]], numpy.int64))
    
    expected_array = numpy.array([[ 1.0, 2.0, 3.0, 4.0],
                                  [ 10.0, 12.0, 14.0, 16.0],
                                  [ 24.0, 21.0, 18.0, 15.0],
                                  [ 16.0, 12.0, 8.0, 4.0]], numpy.int64)
    result = obj.emul(other);
    self.assertTrue((expected_array == result.getData()).all())

  def test_epow_number(self):
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
    expected_array = numpy.array([[ 1.0, 4.0, 9.0, 16.0],
                                  [ 25.0, 36.0, 49.0, 64.0],
                                  [ 64.0, 49.0, 36.0, 25.0],
                                  [ 16.0, 9.0, 4.0, 1.0]], numpy.int64)
    result = obj.epow(2.0);
    self.assertTrue((expected_array == result.getData()).all())

  def test_epow_data2d(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
     
    other.setData(numpy.array([[1.0, 2.0, 1.0, 2.0],
                               [2.0, 1.0, 2.0, 1.0],
                               [1.0, 2.0, 1.0, 2.0],
                               [2.0, 1.0, 2.0, 1.0]], numpy.int64))
     
    expected_array = numpy.array([[ 1.0, 4.0, 3.0, 16.0],
                                  [ 25.0, 6.0, 49.0, 8.0],
                                  [ 8.0, 49.0, 6.0, 25.0],
                                  [ 16.0, 3.0, 4.0, 1.0]], numpy.int64)
    result = obj.epow(other);
    self.assertTrue((expected_array == result.getData()).all())
 
  def test_epow_data2d_xsize_1(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
     
    other.setData(numpy.array([[1.0, 2.0, 2.0, 1.0]], numpy.int64))
     
    expected_array = numpy.array([[ 1.0, 4.0, 9.0, 4.0],
                                  [ 5.0, 36.0, 49.0, 8.0],
                                  [ 8.0, 49.0, 36.0, 5.0],
                                  [ 4.0, 9.0, 4.0, 1.0]], numpy.int64)
    result = obj.epow(other);
    self.assertTrue((expected_array == result.getData()).all())
 
  def test_epow_data2d_ysize_1(self):
    obj = _ravedata2d.new()
    other = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [5.0, 6.0, 7.0, 8.0],
                             [8.0, 7.0, 6.0, 5.0],
                             [4.0, 3.0, 2.0, 1.0]], numpy.int64))
     
    other.setData(numpy.array([[1.0], 
                               [2.0], 
                               [2.0], 
                               [1.0]], numpy.int64))
     
    expected_array = numpy.array([[ 1.0, 2.0, 3.0, 4.0],
                                  [ 25.0, 36.0, 49.0, 64.0],
                                  [ 64.0, 49.0, 36.0, 25.0],
                                  [ 4.0, 3.0, 2.0, 1.0]], numpy.int64)
    result = obj.epow(other);
    self.assertTrue((expected_array == result.getData()).all())

  def test_medfilt2(self):
    #import _rave
    #_rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [2.0, 3.0, 4.0, 5.0],
                             [6.0, 7.0, 8.0, 9.0],
                             [7.0, 8.0, 9.0, 10.0]], numpy.int64))
     
    result = obj.medfilt2(3, 3);
    expected_array = numpy.array([
      [0,   2,   3,   0],
      [2,   3,   4,   4],
      [3,   7,   8,   5],
      [0,   7,   8,   0]], numpy.int64)

    self.assertTrue((expected_array == result.getData()).all())

  def test_medfilt2_nodata(self):
    #import _rave
    #_rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
    obj = _ravedata2d.new(numpy.array([[-999.0, 2.0, 3.0, -999],
                             [2.0, 3.0, 4.0, 5.0],
                             [6.0, 7.0, 8.0, 9.0],
                             [7.0, 8.0, 9.0, 10.0]], numpy.int64))
    obj.nodata = -999;
    obj.useNodata = True
    result = obj.medfilt2(3,3)
    expected_array = numpy.array([
      [0,   2,   3,   0],
      [2,   4,   5,   4],
      [3,   7,   8,   5],
      [0,   7,   8,   0]], numpy.int64)
 
    self.assertTrue((expected_array == result.getData()).all())

  def test_medfilt2_nodata_2(self):
    #import _rave
    #_rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
    obj = _ravedata2d.new(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [2.0, -999.0, -999.0, -999.0],
                             [6.0, -999.0, 8.0, -999.0],
                             [7.0, -999.0, 9.0, 10.0]], numpy.int64))
    obj.nodata = -999;
    obj.useNodata = True
    result = obj.medfilt2(3,3)
    expected_array = numpy.array([
      [0,   1,   2,   0],
      [1,   0,   0,   0],
      [2,   0,   9,   0],
      [0,   0,   8,   0]], numpy.int64)
 
    self.assertTrue((expected_array == result.getData()).all())

  def test_cumsum_default(self):
    #import _rave
    #_rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [2.0, 3.0, 4.0, 5.0],
                             [6.0, 7.0, 8.0, 9.0],
                             [7.0, 8.0, 9.0, 10.0]], numpy.int64))
     
    result = obj.cumsum();
    expected_array = numpy.array([
      [1.0,   2.0,   3.0,   4.0],
      [3.0,   5.0,   7.0,   9.0],
      [9.0,   12.0,  15.0,  18.0],
      [16.0,  20.0,  24.0,  28.0]], numpy.float64)
    self.assertTrue((expected_array == result.getData()).all())

  def test_cumsum_colwise(self):
    #import _rave
    #_rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [2.0, 3.0, 4.0, 5.0],
                             [6.0, 7.0, 8.0, 9.0],
                             [7.0, 8.0, 9.0, 10.0]], numpy.int64))
     
    result = obj.cumsum();
    expected_array = numpy.array([
      [1.0,   2.0,   3.0,   4.0],
      [3.0,   5.0,   7.0,   9.0],
      [9.0,   12.0,  15.0,  18.0],
      [16.0,  20.0,  24.0,  28.0]], numpy.float64)
    self.assertTrue((expected_array == result.getData()).all())

  def test_cumsum_rowwise(self):
    #import _rave
    #_rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
    obj = _ravedata2d.new()
    obj.setData(numpy.array([[1.0, 2.0, 3.0, 4.0],
                             [2.0, 3.0, 4.0, 5.0],
                             [6.0, 7.0, 8.0, 9.0],
                             [7.0, 8.0, 9.0, 10.0]], numpy.int64))
     
    result = obj.cumsum(2);
    expected_array = numpy.array([
      [1.0,   3.0,   6.0,  10.0],
      [2.0,   5.0,   9.0,  14.0],
      [6.0,   13.0,  21.0, 30.0],
      [7.0,   15.0,  24.0, 34.0]], numpy.float64)
    self.assertTrue((expected_array == result.getData()).all())

  def test_movingstd_1(self):
    obj = _ravedata2d.new(numpy.array([
      [-7,-5,-3,-2],
      [-2,-1,0,1],
      [2,3,3,4],
      [5,5,6,7]], numpy.float64))
    obj.nodata = -999;
    obj.useNodata = True
    result = obj.movingstd(3, 3);
    #print(str(result.getData()))
    
    expected = numpy.array([
       [1.39847,   1.12384,   0.86653,   0.73775],
       [0.77055,   0.69347,   0.64010,   0.61237],
       [0.60381,   0.68592,   0.70465,   0.79495],
       [0.83593,   0.86753,   1.00821,   1.14754]], numpy.float64)
    
    for x in range(4):
      for y in range(4):
        self.assertAlmostEqual(result.getData()[y][x], expected[y][x], 3)


  def test_movingstd_noNodata(self):
    obj = _ravedata2d.new(numpy.array([
      [-7,-5,-3,-2],
      [-2,-1,0,1],
      [2,3,3,4],
      [5,5,6,7]], numpy.float64))
    obj.nodata = -999;
    obj.useNodata = False
    try:
      obj.movingstd(3,3)
      self.fail("Expected movingstd")
    except:
      pass
    #result = obj.movingstd(3, 3);
    #print(str(result.getData()))

  def test_movingstd_nodata(self):
    obj = _ravedata2d.new(numpy.array([
      [-999, -5, -3, -2],
      [-2,   -1,  0,  1],
      [ 2,   -999,3,  4],
      [ 5,    5,  6,  7]], numpy.float64))
    obj.nodata = -999;
    obj.useNodata = True
    result = obj.movingstd(1,1)
    #print(str(result.getData()))
    
  def test_hist(self):
    obj = _ravedata2d.new(numpy.array([
      [-7,-5,-3,-2],
      [-2,-1,0,1],
      [2,3,3,4],
      [5,5,6,7]], numpy.float64))
     
    result = obj.hist(3);
    self.assertEqual(3, len(result))
    self.assertEqual(3, result[0])
    self.assertEqual(6, result[1])
    self.assertEqual(7, result[2])

  def test_hist_2(self):
    obj = _ravedata2d.new(numpy.array([
      [-7,-5,-3,-2],
      [-2,-1,0,1],
      [2,3,3,4],
      [5,5,6,7]], numpy.float64))
     
    result = obj.hist(2);
    self.assertEqual(2, len(result))
    self.assertEqual(7, result[0])
    self.assertEqual(9, result[1])

  def test_hist_3(self):
    obj = _ravedata2d.new(numpy.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]], numpy.float64))
     
    result = obj.hist(2);
    self.assertEqual(2, len(result))
    self.assertEqual(8, result[0])
    self.assertEqual(7, result[1])

  def test_hist_4(self):
    obj = _ravedata2d.new(numpy.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]], numpy.float64))
    result = obj.hist(4);
    self.assertEqual(4, len(result))
    self.assertEqual(4, result[0])
    self.assertEqual(4, result[1])
    self.assertEqual(3, result[2])
    self.assertEqual(4, result[3])

  def test_hist_5(self):
    obj = _ravedata2d.new(numpy.array([[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]], numpy.float64))
    result = obj.hist(2);
    self.assertEqual(2, len(result))
    self.assertEqual(7, result[0])
    self.assertEqual(8, result[1])

  def test_hist_6(self):
    obj = _ravedata2d.new(numpy.array([[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]], numpy.float64))
    result = obj.hist(2);
    self.assertEqual(2, len(result))
    self.assertEqual(8, result[0])
    self.assertEqual(8, result[1])

  def test_hist_7(self):
    obj = _ravedata2d.new(numpy.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]], numpy.float64))
    result = obj.hist(2);
    self.assertEqual(2, len(result))
    self.assertEqual(16, result[0])
    self.assertEqual(0, result[1])

  def test_hist_8(self):
    obj = _ravedata2d.new(numpy.array([list(range(1,100))], numpy.float64))
    result = obj.hist(100);
    self.assertEqual(100, len(result))
    self.assertEqual(1, result[0])
    self.assertEqual(1, result[1])
    self.assertEqual(1, result[99])

  def test_hist_nodata(self):
    obj = _ravedata2d.new(numpy.array([[1,2,3,4,5,6,7,8,9,10,11,12,2,14,15]], numpy.float64))
    obj.nodata = 2
    obj.useNodata = True
    result = obj.hist(2);
    self.assertEqual(2, len(result))
    self.assertEqual(7, result[0])
    self.assertEqual(6, result[1])
  
  def test_entropy_1(self):
    obj = _ravedata2d.new(numpy.array([[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0]], numpy.float64))
    self.assertAlmostEqual(1.0, obj.entropy())

  def test_entropy_2(self):
    obj = _ravedata2d.new(numpy.array([[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]], numpy.float64))
    self.assertAlmostEqual(0.94945, obj.entropy(), 5)

  def test_entropy_3(self):
    obj = _ravedata2d.new(numpy.array([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]], numpy.float64))
    self.assertAlmostEqual(1.0, obj.entropy(2))

  def test_entropy_4(self):
    obj = _ravedata2d.new(numpy.array([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]], numpy.float64))
    self.assertAlmostEqual(2.0, obj.entropy(256))

  def test_entropy_nodata(self):
    obj = _ravedata2d.new(numpy.array([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]], numpy.float64))
    obj.useNodata = True
    obj.nodata = 2
    self.assertAlmostEqual(0.918, obj.entropy(), 3)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()