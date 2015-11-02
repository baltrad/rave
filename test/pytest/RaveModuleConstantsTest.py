'''
Created on Oct 16, 2009
@author: Anders Henja
'''
import unittest
import os
import _rave
import string

class RaveModuleConstantsTest(unittest.TestCase):
  def setUp(self):
    pass
  
  def tearDown(self):
    pass

  def testRaveDataTypes(self):
    self.assertEqual(-1, _rave.RaveDataType_UNDEFINED)
    self.assertEqual(0, _rave.RaveDataType_CHAR)
    self.assertEqual(1, _rave.RaveDataType_UCHAR)
    self.assertEqual(2, _rave.RaveDataType_SHORT)
    self.assertEqual(3, _rave.RaveDataType_USHORT)
    self.assertEqual(4, _rave.RaveDataType_INT)
    self.assertEqual(5, _rave.RaveDataType_UINT)
    self.assertEqual(6, _rave.RaveDataType_LONG)
    self.assertEqual(7, _rave.RaveDataType_ULONG)
    self.assertEqual(8, _rave.RaveDataType_FLOAT)
    self.assertEqual(9, _rave.RaveDataType_DOUBLE)

  def testRaveTransformMethods(self):
    self.assertEqual(1, _rave.NEAREST)
    self.assertEqual(2, _rave.BILINEAR)
    self.assertEqual(3, _rave.CUBIC)
    self.assertEqual(4, _rave.CRESSMAN)
    self.assertEqual(5, _rave.UNIFORM)
    self.assertEqual(6, _rave.INVERSE)
