'''
Copyright (C) 2014 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the QI total module

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2014-02-27
'''

import unittest
import os
import _rave
import _raveio
import _qitotal
import _ravefield
import math
import string
import numpy

class PyQITotalTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _qitotal.new()
    isqitotal = string.find(`type(obj)`, "QITotalCore")
    self.assertNotEqual(-1, isqitotal)

  def test_attribute_visibility(self):
    attrs = ['gain', 'offset', 'datatype']
    obj = _qitotal.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)

  def test_undefined_attribute(self):
    obj = _qitotal.new()
    try:
      obj.no_such_attribute = 0
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass

  def test_gain(self):
    obj = _qitotal.new()
    self.assertAlmostEqual(1.0, obj.gain, 4)
    obj.gain = 2.0
    self.assertAlmostEqual(2.0, obj.gain, 4)
    try:
      obj.gain = 0.0
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    self.assertAlmostEqual(2.0, obj.gain, 4)

  def test_offset(self):
    obj = _qitotal.new()
    self.assertAlmostEqual(0.0, obj.offset, 4)
    obj.offset = 2.0
    self.assertAlmostEqual(2.0, obj.offset, 4)

  def test_datatype(self):
    obj = _qitotal.new()
    self.assertEqual(_rave.RaveDataType_DOUBLE, obj.datatype)
    obj.datatype = _rave.RaveDataType_UCHAR
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)

    
  def test_multiplicative(self):
    obj = _qitotal.new()
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((2,2), numpy.float64))
    f1.setValue(0,0,0.1)
    f1.setValue(0,1,0.2)
    f1.setValue(1,0,0.3)
    f1.setValue(1,1,0.4)

    f2 = _ravefield.new()
    f2.setData(numpy.zeros((2,2), numpy.float64))
    f2.setValue(0,0,0.1)
    f2.setValue(0,1,0.2)
    f2.setValue(1,0,0.5)
    f2.setValue(1,1,1.0)

    f3 = _ravefield.new()
    f3.setData(numpy.zeros((2,2), numpy.uint8))
    f3.addAttribute("what/gain", 0.1)
    f3.addAttribute("what/offset", 0.2)
    f3.setValue(0,0,1)  #0.2 + 1*0.1 = 0.3
    f3.setValue(0,1,2)  #0.2 + 2*0.1 = 0.4
    f3.setValue(1,0,3)  #0.2 + 3*0.1 = 0.5
    f3.setValue(1,1,4)  #0.2 + 4*0.1 = 0.6
    
    result = obj.multiplicative([f1,f2,f3])
    
    #0,0 = 0.1 * 0.1 * 0.3 = 0.003
    self.assertAlmostEqual(0.003, result.getValue(0,0)[1], 4)
    
    #0,1 = 0.2 * 0.2 * 0.4 = 0.016
    self.assertAlmostEqual(0.016, result.getValue(0,1)[1], 4)
    
    #1,0 = 0.3 * 0.5 * 0.5 = 0.075
    self.assertAlmostEqual(0.075, result.getValue(1,0)[1], 4)
    
    #1,1 = 0.4 * 1.0 * 0.6 = 0.24
    self.assertAlmostEqual(0.24, result.getValue(1,1)[1], 4)
    
  def test_multiplicative_inconsistent_dimensions(self):
    obj = _qitotal.new()
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((2,2), numpy.float64))

    f2 = _ravefield.new()
    f2.setData(numpy.zeros((2,2), numpy.float64))

    f3 = _ravefield.new()
    f3.setData(numpy.zeros((3,2), numpy.uint8))

    try:    
      obj.multiplicative([f1,f2,f3])
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass

  def test_multiplicative_wGainOffsetAndDatatype(self):
    obj = _qitotal.new()
    obj.gain = 0.001
    obj.offset = -0.01
    obj.datatype = _rave.RaveDataType_UCHAR
    
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((2,2), numpy.float64))
    f1.setValue(0,0,0.1)
    f1.setValue(0,1,0.2)
    f1.setValue(1,0,0.3)
    f1.setValue(1,1,0.4)

    f2 = _ravefield.new()
    f2.setData(numpy.zeros((2,2), numpy.float64))
    f2.setValue(0,0,0.1)
    f2.setValue(0,1,0.2)
    f2.setValue(1,0,0.5)
    f2.setValue(1,1,1.0)

    f3 = _ravefield.new()
    f3.setData(numpy.zeros((2,2), numpy.uint8))
    f3.addAttribute("what/gain", 0.1)
    f3.addAttribute("what/offset", 0.2)
    f3.setValue(0,0,1)  #0.2 + 1*0.1 = 0.3
    f3.setValue(0,1,2)  #0.2 + 2*0.1 = 0.4
    f3.setValue(1,0,3)  #0.2 + 3*0.1 = 0.5
    f3.setValue(1,1,4)  #0.2 + 4*0.1 = 0.6
    
    result = obj.multiplicative([f1,f2,f3])
    
    self.assertEquals(_rave.RaveDataType_UCHAR, result.datatype)
    self.assertAlmostEqual(0.001, result.getAttribute("what/gain"))
    self.assertAlmostEqual(-0.01, result.getAttribute("what/offset"))

    # Due to the conversion errors, we can not check on more than 2 decimals    
    #0,0 = 0.1 * 0.1 * 0.3 = 0.003
    self.assertAlmostEqual(0.003, result.getValue(0,0)[1] * 0.001 - 0.01, 2)
    
    #0,1 = 0.2 * 0.2 * 0.4 = 0.016
    self.assertAlmostEqual(0.016, result.getValue(0,1)[1] * 0.001 - 0.01, 2)
    
    #1,0 = 0.3 * 0.5 * 0.5 = 0.075
    self.assertAlmostEqual(0.075, result.getValue(1,0)[1] * 0.001 - 0.01, 2)
    
    #1,1 = 0.4 * 1.0 * 0.6 = 0.24
    self.assertAlmostEqual(0.24, result.getValue(1,1)[1] * 0.001 - 0.01, 2)

  def test_additive(self):
    obj = _qitotal.new()
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((2,2), numpy.float64))
    f1.setValue(0,0,0.1)
    f1.setValue(0,1,0.2)
    f1.setValue(1,0,0.3)
    f1.setValue(1,1,0.4)

    f2 = _ravefield.new()
    f2.setData(numpy.zeros((2,2), numpy.float64))
    f2.setValue(0,0,0.1)
    f2.setValue(0,1,0.2)
    f2.setValue(1,0,0.5)
    f2.setValue(1,1,1.0)

    f3 = _ravefield.new()
    f3.setData(numpy.zeros((2,2), numpy.uint8))
    f3.addAttribute("what/gain", 0.1)
    f3.addAttribute("what/offset", 0.2)
    f3.setValue(0,0,1)  #0.2 + 1*0.1 = 0.3
    f3.setValue(0,1,2)  #0.2 + 2*0.1 = 0.4
    f3.setValue(1,0,3)  #0.2 + 3*0.1 = 0.5
    f3.setValue(1,1,4)  #0.2 + 4*0.1 = 0.6
    
    result = obj.additive([f1,f2,f3])
    
    #0,0 = (0.1 + 0.1 + 0.3)/3 = 0.1667
    self.assertAlmostEqual(0.1667, result.getValue(0,0)[1], 4)
    
    #0,1 = (0.2 + 0.2 + 0.4)/3 = 0.2667
    self.assertAlmostEqual(0.2667, result.getValue(0,1)[1], 4)
    
    #1,0 = (0.3 + 0.5 + 0.5)/3 = 0.4333
    self.assertAlmostEqual(0.4333, result.getValue(1,0)[1], 4)
    
    #1,1 = (0.4 + 1.0 + 0.6)/3 = 0.6667
    self.assertAlmostEqual(0.6667, result.getValue(1,1)[1], 4)
    
  def test_additive_inconsistent_dimensions(self):
    obj = _qitotal.new()
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((2,2), numpy.float64))

    f2 = _ravefield.new()
    f2.setData(numpy.zeros((2,2), numpy.float64))

    f3 = _ravefield.new()
    f3.setData(numpy.zeros((3,2), numpy.uint8))

    try:    
      obj.additive([f1,f2,f3])
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass    

  def test_minimum(self):
    obj = _qitotal.new()
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((2,2), numpy.float64))
    f1.setValue(0,0,0.1)
    f1.setValue(0,1,0.2)
    f1.setValue(1,0,0.3)
    f1.setValue(1,1,0.4)

    f2 = _ravefield.new()
    f2.setData(numpy.zeros((2,2), numpy.float64))
    f2.setValue(0,0,0.1)
    f2.setValue(0,1,0.1)
    f2.setValue(1,0,0.5)
    f2.setValue(1,1,1.0)

    f3 = _ravefield.new()
    f3.setData(numpy.zeros((2,2), numpy.uint8))
    f3.addAttribute("what/gain", 0.1)
    f3.addAttribute("what/offset", 0.2)
    f3.setValue(0,0,1)  #0.2 + 1*0.1 = 0.3
    f3.setValue(0,1,2)  #0.2 + 2*0.1 = 0.4
    f3.setValue(1,0,3)  #0.2 + 3*0.1 = 0.5
    f3.setValue(1,1,4)  #0.2 + 4*0.1 = 0.6
    
    result = obj.minimum([f1,f2,f3])
    
    #0,0 = min(0.1, 0.1, 0.3) = 0.1
    self.assertAlmostEqual(0.1, result.getValue(0,0)[1], 4)
    
    #0,1 = min(0.2, 0.1, 0.4) = 0.1
    self.assertAlmostEqual(0.1, result.getValue(0,1)[1], 4)
    
    #1,0 = min(0.3, 0.5, 0.5) = 0.3
    self.assertAlmostEqual(0.3, result.getValue(1,0)[1], 4)
    
    #1,1 = min(0.4, 1.0, 0.6) = 0.4
    self.assertAlmostEqual(0.4, result.getValue(1,1)[1], 4)
    
  def test_minimum_inconsistent_dimensions(self):
    obj = _qitotal.new()
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((2,2), numpy.float64))

    f2 = _ravefield.new()
    f2.setData(numpy.zeros((2,2), numpy.float64))

    f3 = _ravefield.new()
    f3.setData(numpy.zeros((3,2), numpy.uint8))

    try:    
      obj.minimum([f1,f2,f3])
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass    
