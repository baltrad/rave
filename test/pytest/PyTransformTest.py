'''
Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/

Tests the transform module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-10-20
'''
import unittest
import os
import _rave
import _transform
import string
import numpy

class PyTransformTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _transform.new()
    
    istransform = string.find(`type(obj)`, "TransformCore")
    self.assertNotEqual(-1, istransform) 

  def test_attribute_visibility(self):
    attrs = ['method']
    obj = _transform.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)

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
