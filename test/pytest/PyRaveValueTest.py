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

  def test_value_long_array_from_tuple(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = (1,2,3)
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
  
  def test_value_hashtable_rave_values_doubles(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = {"a":1.1, "b":2.1, "c":3.1}

    self.assertTrue("a" in classUnderTest.value)
    self.assertEqual(1.1, classUnderTest.value["a"], 4)

    self.assertTrue("b" in classUnderTest.value)
    self.assertEqual(2.1, classUnderTest.value["b"], 4)

    self.assertTrue("c" in classUnderTest.value)
    self.assertEqual(3.1, classUnderTest.value["c"], 4)

  def test_value_hashtable_rave_values_strings(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = {"a":"a string", "b":"b string", "c":"c string"}

    self.assertTrue("a" in classUnderTest.value)
    self.assertEqual("a string", classUnderTest.value["a"])

    self.assertTrue("b" in classUnderTest.value)
    self.assertEqual("b string", classUnderTest.value["b"])

    self.assertTrue("c" in classUnderTest.value)
    self.assertEqual("c string", classUnderTest.value["c"])

  def test_value_hashtable_rave_values_tuple(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = {"a":(1.1,2.2), "b":(3.1,3.2), "c":(4.1,4.2,4.3)}

    self.assertTrue("a" in classUnderTest.value)
    value = classUnderTest.value["a"]
    self.assertEqual(1.1, value[0], 4)
    self.assertEqual(2.2, value[1], 4)

    self.assertTrue("b" in classUnderTest.value)
    value = classUnderTest.value["b"]
    self.assertEqual(3.1, value[0], 4)
    self.assertEqual(3.2, value[1], 4)

    self.assertTrue("c" in classUnderTest.value)
    value = classUnderTest.value["c"]
    self.assertEqual(4.1, value[0], 4)
    self.assertEqual(4.2, value[1], 4)
    self.assertEqual(4.3, value[2], 4)


  def test_value_hashtable_rave_values_hashtable(self):
    classUnderTest = _ravevalue.new()

    classUnderTest.value = {"a":{"zr_a":0.1, "zr_b":1.2}, "b":{"zr_a":1.1, "zr_b":2.2}}

    self.assertEqual(0.1, classUnderTest.value["a"]["zr_a"], 4)
    self.assertEqual(1.2, classUnderTest.value["a"]["zr_b"], 4)    

    self.assertEqual(1.1, classUnderTest.value["b"]["zr_a"], 4)
    self.assertEqual(2.2, classUnderTest.value["b"]["zr_b"], 4)    

  def test_new_from_long(self):
    classUnderTest = _ravevalue.new(33)
    self.assertEqual(33, classUnderTest.value)


  def test_new_from_double(self):
    classUnderTest = _ravevalue.new(2.1)
    self.assertEqual(2.1, classUnderTest.value, 4)

  def test_new_from_tuple(self):
    classUnderTest = _ravevalue.new((2.1, 3.2))
    self.assertEqual(2.1, classUnderTest.value[0], 4)
    self.assertEqual(3.2, classUnderTest.value[1], 4)
