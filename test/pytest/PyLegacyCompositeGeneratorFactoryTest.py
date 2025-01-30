'''
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the py legacy composite generatory factory.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2024-12-15
'''
import unittest
import os
import _compositearguments
import _compositegenerator
import _area
import _projection
import _polarscan
import _polarvolume
import _legacycompositegeneratorfactory
import _raveio
import string
import math

class PyLegacyCompositeGeneratorFactoryTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _legacycompositegeneratorfactory.new()
    iscorrect = str(type(obj)).find("CompositeGeneratorFactoryCore")
    self.assertNotEqual(-1, iscorrect)

  def test_getName(self):
    obj = _legacycompositegeneratorfactory.new()
    self.assertEqual("LegacyCompositeGenerator", obj.getName())

  def test_getDefaultId(self):
    obj = _legacycompositegeneratorfactory.new()
    self.assertEqual("legacy", obj.getDefaultId())

  def test_canHandle_products(self):
    # Rave_ProductType_MAX & NEAREST
    # Rave_ProductType_PMAX & NEAREST
    #
    # Rave_ProductType_MAX
    # Rave_ProductType_PPI
    # Rave_ProductType_PCAPPI
    # Rave_ProductType_CAPPI

    classUnderTest = _legacycompositegeneratorfactory.new()
    for product in ["PPI", "PCAPPI", "CAPPI"]:
      args = _compositearguments.new()
      args.product = product
      self.assertEqual(True, classUnderTest.canHandle(args))

    for product in ["MAX", "PMAX"]:
      args = _compositearguments.new()
      args.product = product
      args.addArgument("interpolation_method", "NEAREST")
      self.assertEqual(True, classUnderTest.canHandle(args))

    for product in ["SCAN", "ETOP", "RHI"]:
      args = _compositearguments.new()
      args.product = product
      self.assertEqual(False, classUnderTest.canHandle(args))

    for product in ["MAX", "PMAX"]:
      args = _compositearguments.new()
      args.product = product
      args.addArgument("interpolation_method", "3D")
      self.assertEqual(False, classUnderTest.canHandle(args))

  def test_create(self):
    classUnderTest = _legacycompositegeneratorfactory.new()
    obj = classUnderTest.create()
    self.assertEqual("LegacyCompositeGenerator", obj.getName())
    self.assertTrue(classUnderTest != obj)
