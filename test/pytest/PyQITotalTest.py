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
    self.assertNotEqual(-1, str(type(obj)).find("QITotalCore"))

  def test_attribute_visibility(self):
    attrs = ['gain', 'offset', 'datatype']
    obj = _qitotal.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)

  def test_undefined_attribute(self):
    obj = _qitotal.new()
    try:
      obj.no_such_attribute = 0
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

  def test_gain(self):
    obj = _qitotal.new()
    self.assertAlmostEqual(1.0, obj.gain, 4)
    obj.gain = 2.0
    self.assertAlmostEqual(2.0, obj.gain, 4)
    try:
      obj.gain = 0.0
      self.fail("Expected AttributeError")
    except AttributeError:
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

  def test_weight(self):
    obj = _qitotal.new()
    try:
      obj.getWeight("se.some.task")
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

    obj.setWeight("se.some.task", 2.0)
    self.assertAlmostEqual(2.0, obj.getWeight("se.some.task"), 4)
    obj.removeWeight("se.some.task")

    try:
      obj.getWeight("se.some.task")
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

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

    self.assertEqual("pl.imgw.quality.qi_total", result.getAttribute("how/task"))
    self.assertEqual("method:multiplicative", result.getAttribute("how/task_args"))

    #0,0 = 0.1 * 0.1 * 0.3 = 0.003
    self.assertAlmostEqual(0.003, result.getValue(0,0)[1], 4)
    
    #0,1 = 0.2 * 0.2 * 0.4 = 0.016
    self.assertAlmostEqual(0.016, result.getValue(0,1)[1], 4)
    
    #1,0 = 0.3 * 0.5 * 0.5 = 0.075
    self.assertAlmostEqual(0.075, result.getValue(1,0)[1], 4)
    
    #1,1 = 0.4 * 1.0 * 0.6 = 0.24
    self.assertAlmostEqual(0.24, result.getValue(1,1)[1], 4)

  def Xtest_multiplicative_weighted_fields(self):
    obj = _qitotal.new()
    f1 = _ravefield.new()
    f1.addAttribute("how/task", "se.smhi.f1")
    f1.setData(numpy.zeros((2,2), numpy.float64))
    f1.setValue(0,0,0.1)
    f1.setValue(0,1,0.2)
    f1.setValue(1,0,0.3)
    f1.setValue(1,1,0.4)

    f2 = _ravefield.new()
    f1.addAttribute("how/task", "se.smhi.f2")
    f2.setData(numpy.zeros((2,2), numpy.float64))
    f2.setValue(0,0,0.1)
    f2.setValue(0,1,0.2)
    f2.setValue(1,0,0.5)
    f2.setValue(1,1,1.0)

    f3 = _ravefield.new()
    f3.setData(numpy.zeros((2,2), numpy.uint8))
    f3.addAttribute("how/task", "se.smhi.f3")
    f3.addAttribute("what/gain", 0.1)
    f3.addAttribute("what/offset", 0.2)
    f3.setValue(0,0,1)  #0.2 + 1*0.1 = 0.3
    f3.setValue(0,1,2)  #0.2 + 2*0.1 = 0.4
    f3.setValue(1,0,3)  #0.2 + 3*0.1 = 0.5
    f3.setValue(1,1,4)  #0.2 + 4*0.1 = 0.6
    
    obj.setWeight("se.smhi.f1", 10.0)
    obj.setWeight("se.smhi.f2", 5.0)
    
    result = obj.multiplicative([f1,f2,f3])

    self.assertEqual("pl.imgw.quality.qi_total", result.getAttribute("how/task"))
    self.assertEqual("method:multiplicative", result.getAttribute("how/task_args"))

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
    except AttributeError:
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
    
    self.assertEqual(_rave.RaveDataType_UCHAR, result.datatype)
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
    
    self.assertEqual("pl.imgw.quality.qi_total", result.getAttribute("how/task"))
    self.assertEqual("method:additive", result.getAttribute("how/task_args"))

    
    #0,0 = (0.1 + 0.1 + 0.3)/3 = 0.1667
    self.assertAlmostEqual(0.1667, result.getValue(0,0)[1], 4)
    
    #0,1 = (0.2 + 0.2 + 0.4)/3 = 0.2667
    self.assertAlmostEqual(0.2667, result.getValue(0,1)[1], 4)
    
    #1,0 = (0.3 + 0.5 + 0.5)/3 = 0.4333
    self.assertAlmostEqual(0.4333, result.getValue(1,0)[1], 4)
    
    #1,1 = (0.4 + 1.0 + 0.6)/3 = 0.6667
    self.assertAlmostEqual(0.6667, result.getValue(1,1)[1], 4)

  def test_additive_with_weights(self):
    obj = _qitotal.new()
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((2,2), numpy.float64))
    f1.addAttribute("how/task", "se.smhi.f1")
    f1.setValue(0,0,0.1)
    f1.setValue(0,1,0.2)
    f1.setValue(1,0,0.3)
    f1.setValue(1,1,0.4)

    f2 = _ravefield.new()
    f2.setData(numpy.zeros((2,2), numpy.float64))
    f2.addAttribute("how/task", "se.smhi.f2")
    f2.setValue(0,0,0.1)
    f2.setValue(0,1,0.2)
    f2.setValue(1,0,0.5)
    f2.setValue(1,1,1.0)

    f3 = _ravefield.new()
    f3.setData(numpy.zeros((2,2), numpy.uint8))
    f3.addAttribute("how/task", "se.smhi.f3")
    f3.addAttribute("what/gain", 0.1)
    f3.addAttribute("what/offset", 0.2)
    f3.setValue(0,0,1)  #0.2 + 1*0.1 = 0.3
    f3.setValue(0,1,2)  #0.2 + 2*0.1 = 0.4
    f3.setValue(1,0,3)  #0.2 + 3*0.1 = 0.5
    f3.setValue(1,1,4)  #0.2 + 4*0.1 = 0.6

    obj.setWeight("se.smhi.f1", 0.5)
    obj.setWeight("se.smhi.f2", 0.25)
    obj.setWeight("se.smhi.f3", 0.25)
    
    result = obj.additive([f1,f2,f3])
    
    self.assertEqual("pl.imgw.quality.qi_total", result.getAttribute("how/task"))
    self.assertEqual("method:additive", result.getAttribute("how/task_args"))

    
    #0,0 = (0.1*0.5 + 0.1*0.25 + 0.3*0.25) = 0.15
    self.assertAlmostEqual(0.15, result.getValue(0,0)[1], 4)
    
    #0,1 = (0.2*0.5 + 0.2*0.25 + 0.4*0.25) = 0.25
    self.assertAlmostEqual(0.25, result.getValue(0,1)[1], 4)
    
    #1,0 = (0.3*0.5 + 0.5*0.25 + 0.5*0.25) = 0.4
    self.assertAlmostEqual(0.4, result.getValue(1,0)[1], 4)
    
    #1,1 = (0.4*0.5 + 1.0*0.25 + 0.6*0.25) = 0.6
    self.assertAlmostEqual(0.6, result.getValue(1,1)[1], 4)

    
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
    except AttributeError:
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
    
    self.assertEqual("pl.imgw.quality.qi_total", result.getAttribute("how/task"))
    self.assertEqual("method:minimum", result.getAttribute("how/task_args"))

    #0,0 = min(0.1, 0.1, 0.3) = 0.1
    self.assertAlmostEqual(0.1, result.getValue(0,0)[1], 4)
    
    #0,1 = min(0.2, 0.1, 0.4) = 0.1
    self.assertAlmostEqual(0.1, result.getValue(0,1)[1], 4)
    
    #1,0 = min(0.3, 0.5, 0.5) = 0.3
    self.assertAlmostEqual(0.3, result.getValue(1,0)[1], 4)
    
    #1,1 = min(0.4, 1.0, 0.6) = 0.4
    self.assertAlmostEqual(0.4, result.getValue(1,1)[1], 4)
    
  def test_minimum_with_weights(self):
    obj = _qitotal.new()
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((2,2), numpy.float64))
    f1.addAttribute("how/task", "se.smhi.f1")
    f1.setValue(0,0,0.1)  # 0.1 * 0.5 = 0.05
    f1.setValue(0,1,0.2)  # 0.2 * 0.5 = 0.10
    f1.setValue(1,0,0.3)  # 0.3 * 0.5 = 0.15
    f1.setValue(1,1,0.4)  # 0.4 * 0.5 = 0.2

    f2 = _ravefield.new()
    f2.setData(numpy.zeros((2,2), numpy.float64))
    f2.addAttribute("how/task", "se.smhi.f2")
    f2.setValue(0,0,0.1) # 0.1 * 0.75 = 0.075
    f2.setValue(0,1,0.1) # 0.1 * 0.75 = 0.075
    f2.setValue(1,0,0.5) # 0.5 * 0.75 = 0.375
    f2.setValue(1,1,1.0) # 1.0 * 0.75 = 0.75

    f3 = _ravefield.new()
    f3.setData(numpy.zeros((2,2), numpy.uint8))
    f3.addAttribute("how/task", "se.smhi.f3")
    f3.addAttribute("what/gain", 0.1)
    f3.addAttribute("what/offset", 0.2)
    f3.setValue(0,0,1)  # (0.2 + 1*0.1) * 0.9 = 0.27
    f3.setValue(0,1,2)  # (0.2 + 2*0.1) * 0.9 = 0.36
    f3.setValue(1,0,3)  # (0.2 + 3*0.1) * 0.9 = 0.45
    f3.setValue(1,1,4)  # (0.2 + 4*0.1) * 0.9 = 0.54
    
    obj.setWeight("se.smhi.f1", 0.5)
    obj.setWeight("se.smhi.f2", 0.25)
    obj.setWeight("se.smhi.f3", 0.25)

    result = obj.minimum([f1,f2,f3])
    
    self.assertEqual("pl.imgw.quality.qi_total", result.getAttribute("how/task"))
    self.assertEqual("method:minimum", result.getAttribute("how/task_args"))

    #0,0 = min(0.05, 0.075, 0.27) = 0.05 => 0.1
    self.assertAlmostEqual(0.1, result.getValue(0,0)[1], 4)
    
    #0,1 = min(0.1, 0.075, 0.36) = 0.075 => 0.1
    self.assertAlmostEqual(0.1, result.getValue(0,1)[1], 4)
    
    #1,0 = min(0.15, 0.375, 0.45) = 0.15 => 0.3
    self.assertAlmostEqual(0.3, result.getValue(1,0)[1], 4)
    
    #1,1 = min(0.2, 0.75, 0.54) = 0.2 => 0.4
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
    except AttributeError:
      pass    
