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

Tests the py rave value module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2025-02-14
'''
import unittest
import os
import _ravevalue
import string
import math

class PyRaveValueTest(unittest.TestCase):
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _ravevalue.new()
    
    isobj = str(type(obj)).find("RaveValueCore")
    self.assertNotEqual(-1, isobj)

  def test_isRaveValue(self):
    obj = _ravevalue.new()
    self.assertEqual(True, _ravevalue.isRaveValue(obj))
    

    self.assertEqual(False, _ravevalue.isRaveValue("abc"))
    
    self.assertEqual(False, _ravevalue.isRaveValue(None))
    

  def test_attribute_visibility(self):
    attrs = ['value']
    value = _ravevalue.new()
    alist = dir(value)
    for a in attrs:
      self.assertEqual(True, a in alist)

  def test_value_string(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = "HEJ"
    self.assertEqual("HEJ", classUnderTest.value)
    classUnderTest.value = None
    self.assertEqual(None, classUnderTest.value)

  def test_value_long(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = 1
    self.assertEqual(1, classUnderTest.value)
    classUnderTest.value = None
    self.assertEqual(None, classUnderTest.value)

  def test_value_double(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = 1.0
    self.assertEqual(1.0, classUnderTest.value, 4)
    classUnderTest.value = None
    self.assertEqual(None, classUnderTest.value)

  def test_value_string_array(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = ["1", "2", "3"]
    self.assertTrue(set(["1","2","3"]) == set(classUnderTest.value))
    classUnderTest.value = None
    self.assertEqual(None, classUnderTest.value)

  def test_value_long_array(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = [1,2,3]
    self.assertTrue(set([1,2,3]) == set(classUnderTest.value))
    classUnderTest.value = None
    self.assertEqual(None, classUnderTest.value)

  def test_value_double_array(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = [1.0,2.0,3.0]
    self.assertEqual(3, len(classUnderTest.value))
    self.assertEqual(1.0, classUnderTest.value[0], 4)
    self.assertEqual(2.0, classUnderTest.value[1], 4)
    self.assertEqual(3.0, classUnderTest.value[2], 4)
    classUnderTest.value = None
    self.assertEqual(None, classUnderTest.value)
