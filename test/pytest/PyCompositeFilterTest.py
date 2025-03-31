'''
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the py composite filter module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2025-01-27
'''
import unittest
import os
import _compositearguments, _compositefilter
import string
import math

class PyCompositeFilterTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _compositefilter.new()
    iscorrect = str(type(obj)).find("CompositeFilterCore")
    self.assertNotEqual(-1, iscorrect)

  def test_isCompositeFilter(self):
    obj = _compositefilter.new()
    self.assertEqual(True, _compositefilter.isCompositeFilter(obj))
    self.assertEqual(False, _compositefilter.isCompositeFilter("ABC"))

  def test_products(self):
    classUnderTest = _compositefilter.new()
    classUnderTest.products = ["CAPPI", "PCAPPI", "PPI"]
    self.assertTrue(set(classUnderTest.products) == set(["CAPPI", "PCAPPI", "PPI"]))
    classUnderTest.products = ["CAPPI"]
    self.assertTrue(set(classUnderTest.products) == set(["CAPPI"]))
    classUnderTest.products = []
    self.assertTrue(set(classUnderTest.products) == set([]))
    classUnderTest.products = None
    self.assertTrue(set(classUnderTest.products) == set([]))

  def test_quantities(self):
    classUnderTest = _compositefilter.new()
    classUnderTest.quantities = ["DBZH", "TH"]
    self.assertTrue(set(classUnderTest.quantities) == set(["DBZH", "TH"]))
    classUnderTest.quantities = ["TH"]
    self.assertTrue(set(classUnderTest.quantities) == set(["TH"]))
    classUnderTest.quantities = []
    self.assertTrue(set(classUnderTest.quantities) == set([]))
    classUnderTest.quantities = None
    self.assertTrue(set(classUnderTest.quantities) == set([]))

  def test_interpolation_methods(self):
    classUnderTest = _compositefilter.new()
    classUnderTest.interpolation_methods = ["NEAREST","LINEAR_HEIGHT","LINEAR_RANGE"]
    self.assertTrue(set(classUnderTest.interpolation_methods) == set(["NEAREST","LINEAR_HEIGHT","LINEAR_RANGE"]))
    classUnderTest.interpolation_methods = ["LINEAR_HEIGHT"]
    self.assertTrue(set(classUnderTest.interpolation_methods) == set(["LINEAR_HEIGHT"]))
    classUnderTest.interpolation_methods = []
    self.assertTrue(set(classUnderTest.interpolation_methods) == set([]))
    classUnderTest.interpolation_methods = None
    self.assertTrue(set(classUnderTest.interpolation_methods) == set([]))

  def test_match_product(self):
    classUnderTest = _compositefilter.new()
    classUnderTest.products = ["CAPPI", "PCAPPI", "PPI"]

    obj = _compositearguments.new()
    obj.product = "CAPPI"
    self.assertEqual(True, classUnderTest.match(obj))
    obj.product = "PCAPPI"
    self.assertEqual(True, classUnderTest.match(obj))
    obj.product = "PPI"
    self.assertEqual(True, classUnderTest.match(obj))
    obj.product = "ACQVA"
    self.assertEqual(False, classUnderTest.match(obj))
    obj.product = None
    self.assertEqual(False, classUnderTest.match(obj))

  def test_match_quantities(self):
    classUnderTest = _compositefilter.new()
    classUnderTest.quantities = ["DBZH", "TH"]

    obj = _compositearguments.new()
    obj.addParameter("DBZH", 0.0, 1.0)
    self.assertEqual(True, classUnderTest.match(obj))
    obj.addParameter("TH", 0.0, 1.0)
    self.assertEqual(True, classUnderTest.match(obj))
    obj.addParameter("VRADH", 0.0, 1.0)
    self.assertEqual(False, classUnderTest.match(obj))


  def test_match_interpolation_methods(self):
    classUnderTest = _compositefilter.new()
    classUnderTest.interpolation_methods = ["NEAREST","LINEAR_HEIGHT","LINEAR_RANGE"]

    obj = _compositearguments.new()
    obj.addArgument("interpolation_method", "NEAREST")
    self.assertEqual(True, classUnderTest.match(obj))
    obj.addArgument("interpolation_method", "LINEAR_HEIGHT")
    self.assertEqual(True, classUnderTest.match(obj))
    obj.addArgument("interpolation_method", "LINEAR_RANGE")
    self.assertEqual(True, classUnderTest.match(obj))
    obj.addArgument("interpolation_method", "3D")
    self.assertEqual(False, classUnderTest.match(obj))
    obj.removeArgument("interpolation_method")
    self.assertEqual(False, classUnderTest.match(obj))

  def test_match_combination_1(self):
    classUnderTest = _compositefilter.new()
    classUnderTest.products = ["CAPPI", "PCAPPI", "PPI"]
    classUnderTest.quantities = ["DBZH", "TH"]
    classUnderTest.interpolation_methods = ["NEAREST","LINEAR_HEIGHT","LINEAR_RANGE"]

    self.assertEqual(True, classUnderTest.match(self.create_argument("CAPPI", ["DBZH","TH"], "NEAREST")))
    self.assertEqual(True, classUnderTest.match(self.create_argument("PCAPPI", ["DBZH"], "LINEAR_HEIGHT")))
    self.assertEqual(True, classUnderTest.match(self.create_argument("PPI", ["TH","DBZH"], "LINEAR_RANGE")))
    self.assertEqual(True, classUnderTest.match(self.create_argument("CAPPI", ["DBZH","TH"], "LINEAR_RANGE")))
    self.assertEqual(True, classUnderTest.match(self.create_argument("PCAPPI", ["DBZH"], "NEAREST")))
    self.assertEqual(True, classUnderTest.match(self.create_argument("PPI", ["TH","DBZH"], "LINEAR_HEIGHT")))

  def test_match_combination_2(self):
    classUnderTest = _compositefilter.new()
    classUnderTest.products = ["CAPPI", "PCAPPI", "PPI"]
    classUnderTest.interpolation_methods = ["NEAREST"]

    self.assertEqual(True, classUnderTest.match(self.create_argument("CAPPI", ["DBZH","TH"], "NEAREST")))
    self.assertEqual(True, classUnderTest.match(self.create_argument("PCAPPI", ["KALLE"], "NEAREST")))
    self.assertEqual(True, classUnderTest.match(self.create_argument("PPI", ["TH","VRADH"], "NEAREST")))
    self.assertEqual(True, classUnderTest.match(self.create_argument("CAPPI", ["KALLE","TH"], "NEAREST")))
    self.assertEqual(True, classUnderTest.match(self.create_argument("PCAPPI", ["DBZH"], "NEAREST")))
    self.assertEqual(True, classUnderTest.match(self.create_argument("PPI", ["TH","DBZH"], "NEAREST")))

  def test_match_combination_failure_1(self):
    classUnderTest = _compositefilter.new()
    classUnderTest.products = ["CAPPI", "PCAPPI", "PPI"]
    classUnderTest.quantities = ["DBZH", "TH"]
    classUnderTest.interpolation_methods = ["NEAREST","LINEAR_HEIGHT","LINEAR_RANGE"]

    self.assertEqual(False, classUnderTest.match(self.create_argument("CAPPI", ["DBZH","TH"], "3D")))
    self.assertEqual(False, classUnderTest.match(self.create_argument("PCAPPI", ["DBZH"], "QUADRATIC")))
    self.assertEqual(False, classUnderTest.match(self.create_argument("ACQVA", ["TH","DBZH"], "NEAREST")))
    self.assertEqual(False, classUnderTest.match(self.create_argument("KALLE", ["DBZH","TH"], "NEAREST")))
    self.assertEqual(False, classUnderTest.match(self.create_argument("PCAPPI", ["VRADH"], "NEAREST")))
    self.assertEqual(False, classUnderTest.match(self.create_argument("PPI", ["TH","VRADH"], "NEAREST")))

  def create_argument(self, product, quantities, interpolation_method):
    obj = _compositearguments.new()
    if product:
      obj.product = product
    if quantities:
      for q in quantities:
        obj.addParameter(q, 0, 1)
    if interpolation_method:
      obj.addArgument("interpolation_method", interpolation_method)
    return obj