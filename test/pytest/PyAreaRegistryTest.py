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
    
    isok = string.find(`type(obj)`, "AreaRegistryCore")
    self.assertNotEqual(-1, isok)

  def test_load(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry
    registry = _arearegistry.load(self.AREA_FIXTURE)
    self.assertTrue(registry != None)
    
    self.assertEquals(2, registry.size())
    self.assertEquals("nrd2km", registry.get(0).id)
    self.assertEquals("nrd2km_laea20e60n", registry.get(1).id)

  def test_load_laea(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry
    registry = _arearegistry.load(self.AREA_FIXTURE)
    self.assertTrue(registry != None)
    
    area = registry.get(1)
    self.assertEquals("laea20e60n", area.pcsid)
    self.assertEquals(None, area.projection)
    self.assertEquals(987, area.xsize)
    self.assertEquals(543, area.ysize)
    self.assertAlmostEquals(2000.0, area.xscale, 4)
    self.assertAlmostEquals(1000.0, area.yscale, 4)
    self.assertAlmostEquals(-738816.513333, area.extent[0], 4)
    self.assertAlmostEquals(-3995515.596160, area.extent[1], 4)
    self.assertAlmostEquals(955183.48666699999, area.extent[2], 4)
    self.assertAlmostEquals(-1787515.59616, area.extent[3], 4)
    
  def test_load_laea_withprojregistry(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry
    import _projectionregistry
    projregistry = _projectionregistry.load(self.PROJ_FIXTURE)
    registry = _arearegistry.load(self.AREA_FIXTURE, projregistry)
    self.assertTrue(registry != None)
    
    area = registry.get(1)
    self.assertEquals("laea20e60n", area.pcsid)
    self.assertTrue(area.projection != None)
    self.assertEquals("Nordic, all radars, 2 km, laea", area.description)
    self.assertEquals(987, area.xsize)
    self.assertEquals(543, area.ysize)
    self.assertAlmostEquals(2000.0, area.xscale, 4)
    self.assertAlmostEquals(1000.0, area.yscale, 4)
    self.assertAlmostEquals(-738816.513333, area.extent[0], 4)
    self.assertAlmostEquals(-3995515.596160, area.extent[1], 4)
    self.assertAlmostEquals(955183.48666699999, area.extent[2], 4)
    self.assertAlmostEquals(-1787515.59616, area.extent[3], 4)
    
    
  def test_getByName(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry    
    registry = _arearegistry.load(self.AREA_FIXTURE)
    self.assertTrue(registry != None)

    self.assertEquals("nrd2km", registry.getByName("nrd2km").id)
    self.assertEquals("nrd2km_laea20e60n", registry.getByName("nrd2km_laea20e60n").id)

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
    self.assertEquals(3, registry.size())
    self.assertEquals("nisse", registry.getByName("nisse").id)

  def test_remove(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry    
    registry = _arearegistry.load(self.AREA_FIXTURE)
    registry.remove(0)
    self.assertEquals(1, registry.size())
    self.assertEquals("nrd2km_laea20e60n", registry.get(0).id)

  def test_removeByName(self):
    if not _rave.isXmlSupported():
      return
    import _arearegistry    
    registry = _arearegistry.load(self.AREA_FIXTURE)
    registry.removeByName("nrd2km_laea20e60n")
    self.assertEquals(1, registry.size())
    self.assertEquals("nrd2km", registry.get(0).id)

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
    self.assertEquals(3, len(areas))
    self.assertEquals("nisse", areas[2].get('id'))
    args = areas[2].findall("areadef/arg")
    self.assertEquals("laea20e60n", string.strip(self.findArgElements(args, "id", "pcs").text))
    self.assertEquals("111", string.strip(self.findArgElements(args, "id", "xsize").text))
    self.assertEquals("222", string.strip(self.findArgElements(args, "id", "ysize").text))
    self.assertEquals("1000.0", string.strip(self.findArgElements(args, "id", "xscale").text)[:6])
    self.assertEquals("2000.0", string.strip(self.findArgElements(args, "id", "yscale").text)[:6])
    extent = string.strip(self.findArgElements(args, "id", "extent").text).split(",")
    self.assertEquals("-738816.5", string.strip(extent[0])[:9])
    self.assertEquals("-3995515.5", string.strip(extent[1])[:10])
    self.assertEquals("955183.4", string.strip(extent[2])[:8])
    self.assertEquals("-1787515.5", string.strip(extent[3])[:10])

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
    self.assertEquals(3, newreg.size())
    self.assertEquals("nrd2km",newreg.get(0).id)
    self.assertEquals("nrd2km_laea20e60n",newreg.get(1).id)
    self.assertEquals("nisse",newreg.get(2).id)
    

  def findArgElements(self, args, aname, avalue):
    for arg in args:
      attr = arg.get(aname)
      if attr != None and attr == avalue:
        return arg
    return None
        