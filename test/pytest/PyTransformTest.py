'''
Created on Oct 20, 2009
@author: Anders Henja
'''
import unittest
import os
import _rave
import _transform
import string
import numpy

class RaveModuleTransformTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testNewTransform(self):
    obj = _transform.new()
    
    istransform = string.find(`type(obj)`, "TransformCore")
    self.assertNotEqual(-1, istransform) 

  def testMethod(self):
    obj = _transform.new()
    self.assertEqual(_rave.NEAREST, obj.method)
    obj.method = _rave.CUBIC
    self.assertEqual(_rave.CUBIC, obj.method)
    
  def testValidMethods(self):
    obj = _transform.new()
    meths = [_rave.NEAREST, _rave.BILINEAR, _rave.CUBIC, _rave.CRESSMAN, _rave.UNIFORM, _rave.INVERSE]
    for method in meths:
      obj.method = method
      self.assertEqual(method, obj.method)
  
  def testInvalidMethods(self):
    obj = _transform.new()
    meths = [99, 33, 22, 11]
    for method in meths:
      try:
        obj.method = method
        self.fail("Expected ValueError")
      except ValueError, e:
        pass
      self.assertEqual(_rave.NEAREST, obj.method)
