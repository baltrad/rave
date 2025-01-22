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
import string
import math

class PyLegacyCompositeGeneratorFactoryTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def create_area(self, areaid):
    area = _area.new()
    if areaid == "eua_gmaps":
      area.id = "eua_gmaps"
      area.xsize = 800
      area.ysize = 1090
      area.xscale = 6223.0
      area.yscale = 6223.0
      #               llX           llY            urX        urY
      area.extent = (-3117.83526,-6780019.83039,4975312.43200,3215.41216)
      area.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +x_0=1335833 +y_0=-11000715 +a=6378137.0 +b=6378137.0 +no_defs +datum=WGS84")
    else:
      raise Exception("No such area")
    
    return area

  def test_new(self):
    obj = _legacycompositegeneratorfactory.new()
    iscorrect = str(type(obj)).find("CompositeGeneratorFactoryCore")
    self.assertNotEqual(-1, iscorrect)

  def test_getName(self):
    obj = _legacycompositegeneratorfactory.new()
    self.assertEqual("LegacyCompositeGenerator", obj.getName())

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


  """
  def test_generate_bad_argument(self):
    obj = _compositegenerator.new()
    args = _compositearguments.new()
    try:
      result = obj.generate(None)
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

  def test_isCompositeArguments(self):
    obj = _compositearguments.new()
    self.assertEqual(True, _compositearguments.isCompositeArguments(obj))
    
    self.assertEqual(False, _compositearguments.isCompositeArguments("ABC"))

  def test_area(self):
    obj = _compositearguments.new()
    self.assertTrue(obj.area is None)
    obj.area = _area.new()
    self.assertTrue(_area.isArea(obj.area))
    try:
      obj.area = "a2"
      fail("Expected TypeError")
    except TypeError:
      pass
    obj.area = None
    self.assertTrue(obj.area is None)

  def test_product(self):
    obj = _compositearguments.new()
    self.assertTrue(obj.product is None)
    obj.product = "CAPPI"
    self.assertEqual("CAPPI", obj.product)
    obj.product = "NISSE"
    self.assertEqual("NISSE", obj.product)
    obj.product = None
    self.assertTrue(obj.product is None)
    try:
      obj.product = 123
      fail("Expected ValueError")
    except ValueError:
      pass

  def test_time(self):
    obj = _compositearguments.new()
    self.assertEqual(None, obj.time)
    obj.time = "200500"
    self.assertEqual("200500", obj.time)
    obj.time = None
    self.assertEqual(None, obj.time)

  def test_time_badValues(self):
    obj = _compositearguments.new()
    values = ["10101", "1010101", "1010ab", "1010x0", "abcdef", 123456]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError:
        pass

  def test_date(self):
    obj = _compositearguments.new()
    self.assertEqual(None, obj.date)
    obj.date = "20050101"
    self.assertEqual("20050101", obj.date)
    obj.date = None
    self.assertEqual(None, obj.date)

  def test_date_badValues(self):
    obj = _compositearguments.new()
    values = ["200910101", "2001010", "200a1010", 20091010]
    for val in values:
      try:
        obj.time = val
        self.fail("Expected ValueError")
      except ValueError:
        pass

  def test_height(self):
    obj = _compositearguments.new()
    self.assertAlmostEqual(1000.0, obj.height, 4)
    obj.height = 1.0
    self.assertAlmostEqual(1.0, obj.height, 4)
     
  def test_range(self):
    obj = _compositearguments.new()
    self.assertAlmostEqual(500000.0, obj.range, 4)
    obj.range = 1.0
    self.assertAlmostEqual(1.0, obj.range, 4)

  def test_elangle(self):
    obj = _compositearguments.new()
    self.assertAlmostEqual(0.0, obj.elangle, 4)
    obj.elangle = 1.0
    self.assertAlmostEqual(1.0, obj.elangle, 4)

  def test_methodId(self):
    obj = _compositearguments.new()
    self.assertTrue(obj.method_id is None)
    obj.method_id = "MyMethod"
    self.assertEqual("MyMethod", obj.method_id)
    obj.method_id = None
    self.assertTrue(obj.method_id is None)

  def test_arguments(self):
    obj = _compositearguments.new()
    obj.addArgument("interpolation_method", "3d")
    obj.addArgument("interpolation_range", 300.0)
    obj.addArgument("interpolation_height", 300)
    self.assertEqual("3d", obj.getArgument("interpolation_method"))
    self.assertAlmostEqual(300.0, obj.getArgument("interpolation_range"), 4)
    self.assertEqual(300, obj.getArgument("interpolation_height"))

  def test_arguments_notfound(self):
    obj = _compositearguments.new()
    try:
      self.assertEqual("3d", obj.getArgument("interpolation"))
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

    obj.addArgument("interpolation_method", "3d")
    try:
      self.assertEqual("3d", obj.getArgument("interpolation"))
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    
  def test_parameter(self):
    obj = _compositearguments.new()
    try:
      obj.getParameter("DBZH")
    except KeyError:
      pass
    self.assertEqual(0, obj.getParameterCount())
    obj.addParameter("DBZH", 1.0, 0.0)
    self.assertEqual(1, obj.getParameterCount())
    obj.addParameter("TH", 2.0, 1.0)
    self.assertEqual(2, obj.getParameterCount())
    (gain, offset) = obj.getParameter("DBZH")
    self.assertAlmostEqual(1.0, gain, 4)
    self.assertAlmostEqual(0.0, offset, 4)

    (quantity, gain, offset) = obj.getParameterAtIndex(0)
    self.assertEqual("DBZH", quantity)
    self.assertAlmostEqual(1.0, gain, 4)
    self.assertAlmostEqual(0.0, offset, 4)

    (gain, offset) = obj.getParameter("TH")
    self.assertAlmostEqual(2.0, gain, 4)
    self.assertAlmostEqual(1.0, offset, 4)

    (quantity, gain, offset) = obj.getParameterAtIndex(1)
    self.assertEqual("TH", quantity)
    self.assertAlmostEqual(2.0, gain, 4)
    self.assertAlmostEqual(1.0, offset, 4)

  def test_addObject_scan(self):
    obj = _compositearguments.new()
    s1 = _polarscan.new()
    s1.date="20240101"
    obj.addObject(s1)
    self.assertEqual(1, obj.getNumberOfObjects())

  def test_addObject_volume(self):
    obj = _compositearguments.new()
    s1 = _polarvolume.new()
    s1.date="20240101"
    obj.addObject(s1)
    self.assertEqual(1, obj.getNumberOfObjects())

  def test_objects(self):
    obj = _compositearguments.new()
    self.assertEqual(0, obj.getNumberOfObjects())
    s1 = _polarscan.new()
    s1.date="20240101"
    obj.addObject(s1)
    self.assertEqual(1, obj.getNumberOfObjects())
    self.assertEqual("20240101", obj.getObject(0).date)
    s2 = _polarscan.new()
    s2.date="20240102"
    obj.addObject(s2)
    self.assertEqual(2, obj.getNumberOfObjects())
    self.assertEqual("20240102", obj.getObject(1).date)

  def test_create(self):
    obj = _compositegenerator.create()
    ids = obj.getFactoryIDs()
    self.assertEqual(2, len(ids))
    self.assertTrue("legacy" in ids)
    self.assertTrue("acqva" in ids)

  def test_new_no_factories(self):
    obj = _compositegenerator.new()
    ids = obj.getFactoryIDs()
    self.assertEqual(0, len(ids))

  def test_register(self):
    obj = _compositegenerator.new()
    obj.register("1", _legacycompositegeneratorfactory.new())
    self.assertEqual(1, len(obj.getFactoryIDs()))
    obj.register("2", _acqvacompositegeneratorfactory.new())
    self.assertEqual(2, len(obj.getFactoryIDs()))
    self.assertTrue("1" in obj.getFactoryIDs())
    self.assertTrue("2" in obj.getFactoryIDs())

  def test_uregister(self):
    obj = _compositegenerator.new()
    obj.register("1", _legacycompositegeneratorfactory.new())
    obj.register("2", _acqvacompositegeneratorfactory.new())
    obj.unregister("1")
    self.assertEqual(1, len(obj.getFactoryIDs()))
    self.assertTrue("2" in obj.getFactoryIDs())
    obj.unregister("2")
    self.assertEqual(0, len(obj.getFactoryIDs()))

  def test_generate(self):
    obj = _compositegenerator.create()

    args = _compositearguments.new()
    args.area = self.create_area("eua_gmaps")
    args.product = "PPI"
    args.elangle = 0.0
    args.time = "120000"
    args.date = "20090501"
    #args.strategy = "acqva"
    args.addParameter("DBZH", 0.1, -30.0)
    #args.quality_flags = ["se.smhi.composite.distance.radar"]

    # arguments are usualy method specific in some way
    args.addArgument("selection_method", "HEIGHT_ABOVE_SEALEVEL")
    args.addArgument("interpolation_method", "NEAREST")

    # args.addObject(_raveio.open(....).object)
    # args.addObject(_raveio.open(....).object)
    # args.addObject(_raveio.open(....).object)

    result = obj.generate(args)
  """

