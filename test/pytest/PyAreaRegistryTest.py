'''
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the py area registry module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2010-12-20
'''
import unittest
import _projection
import _area
import _rave
import string
import os

from xml.etree import ElementTree

class PyAreaRegistryTest(unittest.TestCase):
  PROJ_FIXTURE="fixtures/fixture_projections.xml"
  AREA_FIXTURE="fixtures/fixture_areas.xml"
  TEMPORARY_FILE="pyprojtest.xml"
  
  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def test_new(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry
    obj = _arearegistry.new()
    self.assertNotEqual(-1, str(type(obj)).find("AreaRegistryCore"))

  def test_load(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry
    registry = _arearegistry.load(self.AREA_FIXTURE)
    self.assertTrue(registry != None)
    
    self.assertEqual(2, registry.size())
    self.assertEqual("nrd2km", registry.get(0).id)
    self.assertEqual("nrd2km_laea20e60n", registry.get(1).id)

  def test_load_laea(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry
    registry = _arearegistry.load(self.AREA_FIXTURE)
    self.assertTrue(registry != None)
    
    area = registry.get(1)
    self.assertEqual("laea20e60n", area.pcsid)
    self.assertEqual(None, area.projection)
    self.assertEqual(987, area.xsize)
    self.assertEqual(543, area.ysize)
    self.assertAlmostEqual(2000.0, area.xscale, 4)
    self.assertAlmostEqual(1000.0, area.yscale, 4)
    self.assertAlmostEqual(-738816.513333, area.extent[0], 4)
    self.assertAlmostEqual(-3995515.596160, area.extent[1], 4)
    self.assertAlmostEqual(955183.48666699999, area.extent[2], 4)
    self.assertAlmostEqual(-1787515.59616, area.extent[3], 4)
    
  def test_load_laea_withprojregistry(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry
    import _projectionregistry
    projregistry = _projectionregistry.load(self.PROJ_FIXTURE)
    registry = _arearegistry.load(self.AREA_FIXTURE, projregistry)
    self.assertTrue(registry != None)
    
    area = registry.get(1)
    self.assertEqual("laea20e60n", area.pcsid)
    self.assertTrue(area.projection != None)
    self.assertEqual("Nordic, all radars, 2 km, laea", area.description)
    self.assertEqual(987, area.xsize)
    self.assertEqual(543, area.ysize)
    self.assertAlmostEqual(2000.0, area.xscale, 4)
    self.assertAlmostEqual(1000.0, area.yscale, 4)
    self.assertAlmostEqual(-738816.513333, area.extent[0], 4)
    self.assertAlmostEqual(-3995515.596160, area.extent[1], 4)
    self.assertAlmostEqual(955183.48666699999, area.extent[2], 4)
    self.assertAlmostEqual(-1787515.59616, area.extent[3], 4)
    
    
  def test_getByName(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry    
    registry = _arearegistry.load(self.AREA_FIXTURE)
    self.assertTrue(registry != None)

    self.assertEqual("nrd2km", registry.getByName("nrd2km").id)
    self.assertEqual("nrd2km_laea20e60n", registry.getByName("nrd2km_laea20e60n").id)

  def test_add(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry    
    registry = _arearegistry.load(self.AREA_FIXTURE)
    a = _area.new()
    a.id = "nisse"
    a.xsize = 111
    a.ysize = 222
    a.xscale = 1000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.pcsid = "laea20e60n"
    registry.add(a)
    self.assertEqual(3, registry.size())
    self.assertEqual("nisse", registry.getByName("nisse").id)

  def test_remove(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry    
    registry = _arearegistry.load(self.AREA_FIXTURE)
    registry.remove(0)
    self.assertEqual(1, registry.size())
    self.assertEqual("nrd2km_laea20e60n", registry.get(0).id)

  def test_removeByName(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry    
    registry = _arearegistry.load(self.AREA_FIXTURE)
    registry.removeByName("nrd2km_laea20e60n")
    self.assertEqual(1, registry.size())
    self.assertEqual("nrd2km", registry.get(0).id)

  def test_write(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry    
    registry = _arearegistry.load(self.AREA_FIXTURE)
    a = _area.new()
    a.id = "nisse"
    a.description = "nisses test"
    a.xsize = 111
    a.ysize = 222
    a.xscale = 1000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.pcsid = "laea20e60n"
    registry.add(a)

    registry.write(self.TEMPORARY_FILE)
    
    tree = ElementTree.parse(self.TEMPORARY_FILE)
    areas = tree.findall("area")
    self.assertEqual(3, len(areas))
    self.assertEqual("nisse", areas[2].get('id'))
    args = areas[2].findall("areadef/arg")
    self.assertEqual("laea20e60n", self.findArgElements(args, "id", "pcs").text.strip())
    self.assertEqual("111", self.findArgElements(args, "id", "xsize").text.strip())
    self.assertEqual("222", self.findArgElements(args, "id", "ysize").text.strip())
    self.assertEqual("1000.0", self.findArgElements(args, "id", "xscale").text.strip()[:6])
    self.assertEqual("2000.0", self.findArgElements(args, "id", "yscale").text.strip()[:6])
    extent = self.findArgElements(args, "id", "extent").text.strip().split(",")
    self.assertEqual("-738816.5", extent[0].strip()[:9])
    self.assertEqual("-3995515.5", extent[1].strip()[:10])
    self.assertEqual("955183.4", extent[2].strip()[:8])
    self.assertEqual("-1787515.5", extent[3].strip()[:10])

  def test_write_2(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry    
    registry = _arearegistry.load(self.AREA_FIXTURE)
    a = _area.new()
    a.id = "nisse"
    a.description = "nisses test"
    a.xsize = 111
    a.ysize = 222
    a.xscale = 1000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.pcsid = "laea20e60n"
    registry.add(a)

    registry.write(self.TEMPORARY_FILE)

    newreg = _arearegistry.load(self.TEMPORARY_FILE)
    self.assertEqual(3, newreg.size())
    self.assertEqual("nrd2km",newreg.get(0).id)
    self.assertEqual("nrd2km_laea20e60n",newreg.get(1).id)
    self.assertEqual("nisse",newreg.get(2).id)
    

  def findArgElements(self, args, aname, avalue):
    for arg in args:
      attr = arg.get(aname)
      if attr != None and attr == avalue:
        return arg
    return None
        