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

Tests the compositing class

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2014-11-24
'''
import unittest
import _projection
import _area
import _rave,_polarscan
import string
import os
import compositing
import _pycomposite
import rave_quality_plugin, rave_pgf_quality_registry
import mock

class compositing_test(unittest.TestCase):
  def setUp(self):
    self.qc_check_1_mock = mock.Mock(spec=rave_quality_plugin.rave_quality_plugin)
    self.qc_check_2_mock = mock.Mock(spec=rave_quality_plugin.rave_quality_plugin)
    rave_pgf_quality_registry.add_plugin("qc.check.1", self.qc_check_1_mock)
    rave_pgf_quality_registry.add_plugin("qc.check.2", self.qc_check_2_mock)
    self.classUnderTest = compositing.compositing()

  def tearDown(self):
    self.classUnderTest = None
    rave_pgf_quality_registry.remove_plugin("qc.check.1")
    rave_pgf_quality_registry.remove_plugin("qc.check.2")

  def test_quality_control_objects(self):
    o1 = object()
    o2 = object()
    self.classUnderTest.detectors=["qc.check.1","qc.check.2"] 
    self.classUnderTest.reprocess_quality_field = True
    
    self.qc_check_1_mock.process.side_effect = lambda x,y: {o1:o1,o2:o2}[x]
    self.qc_check_1_mock.algorithm.return_value = None
    self.qc_check_2_mock.process.side_effect = lambda x,y: {o1:o1,o2:o2}[x]
    self.qc_check_2_mock.algorithm.return_value = None
    
    result, algorithm = self.classUnderTest.quality_control_objects({"s1.h5":o1,"s2.h5":o2})
    
    expected_qc_check_1_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True), mock.call.algorithm()]
    expected_qc_check_2_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True), mock.call.algorithm()]
    
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    self.assertTrue(expected_qc_check_2_calls == self.qc_check_2_mock.mock_calls)
    
    self.assertTrue(isinstance(result,dict))
    self.assertTrue(2 == len(result))
    self.assertTrue(result["s1.h5"] == o1)
    self.assertTrue(result["s2.h5"] == o2)
    self.assertTrue(algorithm == None)

  def test_quality_control_objects_algorithm_on_first(self):
    o1 = object()
    o2 = object()
    a1 = object()
    self.classUnderTest.detectors=["qc.check.1","qc.check.2"] 
    self.classUnderTest.reprocess_quality_field = True
    
    self.qc_check_1_mock.process.side_effect = lambda x,y: {o1:o1,o2:o2}[x]
    self.qc_check_1_mock.algorithm.return_value = a1
    self.qc_check_2_mock.process.side_effect = lambda x,y: {o1:o1,o2:o2}[x]
    self.qc_check_2_mock.algorithm.return_value = None
    
    result, algorithm = self.classUnderTest.quality_control_objects({"s1.h5":o1,"s2.h5":o2})
    
    expected_qc_check_1_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True), mock.call.algorithm()]
    expected_qc_check_2_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True), mock.call.algorithm()]
    
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    self.assertTrue(expected_qc_check_2_calls == self. qc_check_2_mock.mock_calls)
    
    self.assertTrue(isinstance(result,dict))
    self.assertTrue(2 == len(result))
    self.assertTrue(result["s1.h5"] == o1)
    self.assertTrue(result["s2.h5"] == o2)
    self.assertTrue(algorithm == a1)


  def test_quality_control_objects_processor_returns_both_object_and_algorithm(self):
    o1 = object()
    o2 = object()
    a2 = object()
    self.classUnderTest.detectors=["qc.check.1","qc.check.2"] 
    self.classUnderTest.reprocess_quality_field = True
    
    self.qc_check_1_mock.process.side_effect = lambda x,y: {o1:o1,o2:o2}[x]
    self.qc_check_1_mock.algorithm.return_value = None
    self.qc_check_2_mock.process.side_effect = lambda x,y: {o1:o1,o2:(o2,a2)}[x]
    self.qc_check_2_mock.algorithm.return_value = None
    
    result, algorithm = self.classUnderTest.quality_control_objects({"s1.h5":o1,"s2.h5":o2})
    
    expected_qc_check_1_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True), mock.call.algorithm()]
    expected_qc_check_2_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True)]
    
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    self.assertTrue(expected_qc_check_2_calls == self. qc_check_2_mock.mock_calls)
    
    self.assertTrue(isinstance(result,dict))
    self.assertTrue(2 == len(result))
    self.assertTrue(result["s1.h5"] == o1)
    self.assertTrue(result["s2.h5"] == o2)
    self.assertTrue(algorithm == a2)
    
    
  def test_set_product_from_string(self):
    prods = [("ppi", _rave.Rave_ProductType_PPI),
             ("cappi", _rave.Rave_ProductType_CAPPI),
             ("pcappi", _rave.Rave_ProductType_PCAPPI),
             ("pmax", _rave.Rave_ProductType_PMAX),
             ("max", _rave.Rave_ProductType_MAX)]

    for p in prods:
      self.classUnderTest.set_product_from_string(p[0])
      self.assertEquals(p[1], self.classUnderTest.product)

  def test_set_product_from_string_invalid(self):
    try:
      self.classUnderTest.set_product_from_string("nisse")
      self.fail("Expected ValueError")
    except ValueError, e:
      pass

  def test_set_method_from_string(self):
    methods = [("NEAREST_RADAR", _pycomposite.SelectionMethod_NEAREST),
               ("HEIGHT_ABOVE_SEALEVEL", _pycomposite.SelectionMethod_HEIGHT)]
    for m in methods:
      self.classUnderTest.set_method_from_string(m[0])
      self.assertEquals(m[1], self.classUnderTest.selection_method)
  
  def test_set_method_from_string_invalid(self):
    try:
      self.classUnderTest.set_method_from_string("nisse")
      self.fail("Expected ValueError")
    except ValueError, e:
      pass
