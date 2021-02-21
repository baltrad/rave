'''
Copyright (C) 2012 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the python version of the area registry.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2012-03-16
'''
import unittest
import _projection
import _area
import _rave
import string
import os
import area_registry

from xml.etree import ElementTree

class area_registry_test(unittest.TestCase):
  PROJ_FIXTURE="fixtures/fixture_projections.xml"
  AREA_FIXTURE="fixtures/fixture_areas.xml"
  TEMPORARY_FILE="pyprojtest.xml"
  
  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def test_test_c_loading(self):
    if not _rave.isXmlSupported():
      return
    registry = area_registry.area_registry(self.AREA_FIXTURE, self.PROJ_FIXTURE)

    nrd2kmarea = registry.getarea("nrd2km")
    self.assertNotEqual(-1, str(type(nrd2kmarea)).find("AreaCore"))
    self.assertEqual("nrd2km", nrd2kmarea.id)
    self.assertEqual(848, nrd2kmarea.xsize)

  