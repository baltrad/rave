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

Tests the py composite arguments module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2024-12-13
'''
import unittest
import os
import _compositearguments
import _compositegenerator
import _compositefilter
import _area
import _projection
import _polarscan
import _polarvolume
import _legacycompositegeneratorfactory
import _acqvacompositegeneratorfactory
import _compositefactorymanager
import string
import math

class PyCompositeGeneratorTest(unittest.TestCase):
  GENERATOR_FIXTURE="fixtures/composite_generator_factories.xml"
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _compositegenerator.new()
    iscorrect = str(type(obj)).find("CompositeGeneratorCore")
    self.assertNotEqual(-1, iscorrect)

  def test_generate_bad_argument(self):
    obj = _compositegenerator.new()
    args = _compositearguments.new()
    try:
      result = obj.generate(None)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

  def test_create(self):
    obj = _compositegenerator.create()
    ids = obj.getFactoryIDs() # The factories ID in default case should be what the compositegeneratorfactories has set as defaultID

    self.assertEqual(3, len(ids))
    self.assertTrue("legacy" in ids)
    self.assertTrue("acqva" in ids)
    self.assertTrue("nearest" in ids)

  def test_create_with_manager(self):
    manager = _compositefactorymanager.new()
    manager.remove("LegacyCompositeGenerator")
    manager.remove("NearestCompositeGenerator")
    obj = _compositegenerator.create(manager)
    ids = obj.getFactoryIDs()
    self.assertEqual(1, len(ids))
    self.assertTrue("acqva" in ids)

  def test_create_with_empty_manager(self):
    manager = _compositefactorymanager.new()
    manager.remove("LegacyCompositeGenerator")
    manager.remove("AcqvaCompositeGenerator")
    manager.remove("NearestCompositeGenerator")

    try:
      _compositegenerator.create(manager)
      self.fail("Expected ValueError")
    except ValueError:
      pass

  def test_create_with_manager_and_xml(self):
    manager = _compositefactorymanager.new()
    obj = _compositegenerator.create(manager, self.GENERATOR_FIXTURE)
    ids = obj.getFactoryIDs()
    self.assertEqual(3, len(ids))
    self.assertTrue("acqva" in ids)
    self.assertTrue("legacy" in ids)
    self.assertTrue("legacy2" in ids)

  def test_register_without_filter(self):
    obj = _compositegenerator.new()
    self.assertEqual(0, len(obj.getFactoryIDs()))
    obj.register("1", _legacycompositegeneratorfactory.new())
    self.assertEqual(1, len(obj.getFactoryIDs()))
    obj.register("2", _acqvacompositegeneratorfactory.new())
    self.assertEqual(2, len(obj.getFactoryIDs()))
    self.assertTrue("1" in obj.getFactoryIDs())
    self.assertTrue("2" in obj.getFactoryIDs())

    args = _compositearguments.new()
    args.product = "PCAPPI"
    factory = obj.identify(args)
    self.assertEqual("LegacyCompositeGenerator", factory.getName())

    args.product = "ACQVA"
    factory = obj.identify(args)
    self.assertEqual("AcqvaCompositeGenerator", factory.getName())

  def test_register_with_filter(self):
    obj = _compositegenerator.new()
    filter1 = _compositefilter.new()
    filter1.products = ["ACQVA"]

    filter2 = _compositefilter.new()
    filter2.products = ["PPI"]

    obj.register("1", _legacycompositegeneratorfactory.new(), [filter1])
    obj.register("2", _acqvacompositegeneratorfactory.new(), [filter2])

    args = _compositearguments.new()
    args.product = "ACQVA"
    factory = obj.identify(args)
    self.assertEqual("LegacyCompositeGenerator", factory.getName())

  # def test_uregister(self):
  #   obj = _compositegenerator.new()
  #   obj.register("1", _legacycompositegeneratorfactory.new())
  #   obj.register("2", _acqvacompositegeneratorfactory.new())
  #   obj.unregister("1")
  #   self.assertEqual(1, len(obj.getFactoryIDs()))
  #   self.assertTrue("2" in obj.getFactoryIDs())
  #   obj.unregister("2")
  #   self.assertEqual(0, len(obj.getFactoryIDs()))

  # def Xtest_generate(self):
  #   obj = _compositegenerator.create()
  #   obj.register("legacy", _legacycompositegeneratorfactory.new())

  #   args = _compositearguments.new()
  #   args.area = self.create_area("eua_gmaps")
  #   args.product = "PPI"
  #   args.elangle = 0.0
  #   args.time = "120000"
  #   args.date = "20090501"
  #   #args.strategy = "acqva"
  #   args.addParameter("DBZH", 0.1, -30.0)
  #   #args.quality_flags = ["se.smhi.composite.distance.radar"]

  #   # arguments are usualy method specific in some way
  #   args.addArgument("selection_method", "HEIGHT_ABOVE_SEALEVEL")
  #   args.addArgument("interpolation_method", "NEAREST")

  #   # args.addObject(_raveio.open(....).object)
  #   # args.addObject(_raveio.open(....).object)
  #   # args.addObject(_raveio.open(....).object)

  #   result = obj.generate(args)

