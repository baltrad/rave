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
import json
import _rave

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

  def test_value_string_tokenize_1(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = "Testing,Tokenizer,With,Comma"
    result = classUnderTest.tokenize()
    self.assertEqual(4, len(result.value))
    self.assertEqual("Testing", result.value[0])
    self.assertEqual("Tokenizer", result.value[1])
    self.assertEqual("With", result.value[2])
    self.assertEqual("Comma", result.value[3])

  def test_value_string_tokenize_2(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = "Testing,Tokenizer,,With,Comma"
    result = classUnderTest.tokenize()
    self.assertEqual(5, len(result.value))
    self.assertEqual("Testing", result.value[0])
    self.assertEqual("Tokenizer", result.value[1])
    self.assertEqual("", result.value[2])
    self.assertEqual("With", result.value[3])
    self.assertEqual("Comma", result.value[4])

  def test_value_string_tokenize_3(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = ",,,With,"
    result = classUnderTest.tokenize()
    self.assertEqual(5, len(result.value))
    self.assertEqual("", result.value[0])
    self.assertEqual("", result.value[1])
    self.assertEqual("", result.value[2])
    self.assertEqual("With", result.value[3])
    self.assertEqual("", result.value[4])

  def test_value_string_tokenize_4(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = ""
    result = classUnderTest.tokenize()
    self.assertEqual(0, len(result.value))

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

  def test_value_boolean(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = True
    self.assertEqual(True, classUnderTest.value)
    classUnderTest.value = None
    self.assertEqual(None, classUnderTest.value)

  def test_value_null(self):
    classUnderTest = _ravevalue.new()
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

  def test_value_mixed_array(self):
    classUnderTest = _ravevalue.new()
    classUnderTest.value = [1.0,2,"abc"]
    self.assertEqual(3, len(classUnderTest.value))
    self.assertEqual(1.0, classUnderTest.value[0], 4)
    self.assertEqual(2, classUnderTest.value[1])
    self.assertEqual("abc", classUnderTest.value[2])
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

  def test_isStringArray_true(self):
    classUnderTest = _ravevalue.new(["a", "b", None, "4.0"])
    self.assertEqual(True, classUnderTest.isStringArray())
    self.assertEqual("a", classUnderTest.value[0])
    self.assertEqual("b", classUnderTest.value[1])
    self.assertEqual(None, classUnderTest.value[2])
    self.assertEqual("4.0", classUnderTest.value[3])

  def test_isStringArray_false(self):
    classUnderTest = _ravevalue.new(["a", "b", 1.0, "4.0"])
    self.assertEqual(False, classUnderTest.isStringArray())

  def test_isLongArray_true(self):
    classUnderTest = _ravevalue.new([1, 2, 3, 4])
    self.assertEqual(True, classUnderTest.isLongArray())
    self.assertEqual(1, classUnderTest.value[0])
    self.assertEqual(2, classUnderTest.value[1])
    self.assertEqual(3, classUnderTest.value[2])
    self.assertEqual(4, classUnderTest.value[3])

  def test_isLongArray_false(self):
    classUnderTest = _ravevalue.new([1, 2, 3, 4.0])
    self.assertEqual(False, classUnderTest.isLongArray())

    classUnderTest = _ravevalue.new([1, 2, 3, None])
    self.assertEqual(False, classUnderTest.isLongArray())

  def test_isDoubleArray_true(self):
    classUnderTest = _ravevalue.new([1.0, 2.0, 3, 4.0])
    self.assertEqual(True, classUnderTest.isDoubleArray())
    self.assertEqual(1.0, classUnderTest.value[0], 4)
    self.assertEqual(2.0, classUnderTest.value[1], 4)
    self.assertEqual(3.0, classUnderTest.value[2], 4)
    self.assertEqual(4.0, classUnderTest.value[3], 4)

  def test_isDoubleArray_false(self):
    classUnderTest = _ravevalue.new([1.0, 2.0, 3.0, None])
    self.assertEqual(False, classUnderTest.isDoubleArray())

    classUnderTest = _ravevalue.new([1, 2, "abc", 3])
    self.assertEqual(False, classUnderTest.isDoubleArray())

  def test_new_from_tuple(self):
    classUnderTest = _ravevalue.new((2.1, 3.2))
    self.assertEqual(2.1, classUnderTest.value[0], 4)
    self.assertEqual(3.2, classUnderTest.value[1], 4)

  def test_fromJSON_std_elements(self):
    if not _rave.isJsonSupported():
      return
    result = _ravevalue.fromJSON(json.dumps({"double":2.3, "long":123, "boolean": True, "null": None}))
    self.assertTrue("double" in result.value)
    self.assertTrue("long" in result.value)
    self.assertTrue("boolean" in result.value)
    self.assertTrue("null" in result.value)
    self.assertEqual(2.3, result.value["double"], 4)
    self.assertEqual(123, result.value["long"])
    self.assertEqual(True, result.value["boolean"])
    self.assertEqual(None, result.value["null"])


  def test_fromJSON_maps(self):
    if not _rave.isJsonSupported():
      return

    result = _ravevalue.fromJSON(json.dumps({"double":1.0, "inner":{"double":2.0}}))
    self.assertTrue("double" in result.value)
    self.assertTrue("inner" in result.value)
    self.assertEqual(1.0, result.value["double"], 4)
    self.assertTrue(type(result.value["inner"]) == type({}))

  def test_fromJSON_lists(self):
    if not _rave.isJsonSupported():
      return

    result = _ravevalue.fromJSON(json.dumps({"list":[1.0, 2.0, 3.0]}))
    self.assertTrue("list" in result.value)
    self.assertEqual(3, len(result.value["list"]))
    self.assertEqual(1.0, result.value["list"][0], 4)
    self.assertEqual(2.0, result.value["list"][1], 4)
    self.assertEqual(3.0, result.value["list"][2], 4)

  def test_fromJSON_maps_and_lists(self):
    if not _rave.isJsonSupported():
      return

    result = _ravevalue.fromJSON(json.dumps({"list":[{"another_list":["abc"]}]}))
    self.assertTrue("list" in result.value)
    self.assertEqual(1, len(result.value["list"]))
    self.assertTrue("another_list" in result.value["list"][0])
    self.assertEqual("abc", result.value["list"][0]["another_list"][0])

  def test_toJSON_string(self):
    if not _rave.isJsonSupported():
      return

    self.assertEqual('\"abc\"', _ravevalue.new("abc").toJSON())

  def test_toJSON_long(self):
    if not _rave.isJsonSupported():
      return

    self.assertEqual('2', _ravevalue.new(2).toJSON())

  def test_toJSON_float(self):
    if not _rave.isJsonSupported():
      return

    self.assertEqual('1.0', _ravevalue.new(1.0).toJSON())

  def test_toJSON_boolean(self):
    if not _rave.isJsonSupported():
      return

    self.assertEqual('true', _ravevalue.new(True).toJSON())

  def test_toJSON_null(self):
    if not _rave.isJsonSupported():
      return

    self.assertEqual('null', _ravevalue.new(None).toJSON())

  def test_toJSON_hash(self):
    if not _rave.isJsonSupported():
      return

    js = json.loads(_ravevalue.new({"abc":123, "def":2.2, "ghi":True}).toJSON())
    self.assertTrue("abc" in js)
    self.assertEqual(123, js["abc"])

    self.assertTrue("def" in js)
    self.assertEqual(2.2, js["def"], 4)

    self.assertTrue("ghi" in js)
    self.assertEqual(True, js["ghi"])

  def test_toJSON_list(self):
    if not _rave.isJsonSupported():
      return

    js = json.loads(_ravevalue.new([1, 2.2, True, "YES", None]).toJSON())
    self.assertEqual(5, len(js))
    self.assertEqual(1, js[0])
    self.assertEqual(2.2, js[1], 4)
    self.assertEqual(True, js[2])
    self.assertEqual("YES", js[3])
    self.assertEqual(None, js[4])

  def test_toJSON_mixed_hash(self):
    if not _rave.isJsonSupported():
      return

    js = json.loads(_ravevalue.new({"list1":[1, 2], "hash1":{"i":1, "list2":[3,4]}}).toJSON())
    self.assertTrue("list1" in js)
    self.assertTrue("hash1" in js)
    self.assertTrue("i" in js["hash1"])
    self.assertTrue("list2" in js["hash1"])
    self.assertEqual(1, js["list1"][0])
    self.assertEqual(2, js["list1"][1])
    self.assertEqual(1, js["hash1"]["i"])
    self.assertEqual(3, js["hash1"]["list2"][0])
    self.assertEqual(4, js["hash1"]["list2"][1])

