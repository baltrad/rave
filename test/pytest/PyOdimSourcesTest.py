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

Tests the py area module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2025-01-16
'''
import unittest
import os
import _odimsources
import string
import math
import _rave

class PyOdimSourcesTest(unittest.TestCase):
  FIXTURE="fixtures/odim_sources_fixture.xml"

  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _odimsources.new()
    
    isarea = str(type(obj)).find("OdimSourcesCore")
    self.assertNotEqual(-1, isarea)

  def test_load(self):
    expected_sources = ['seosd', 'dkvir', 'dkhor', 'sella', 'sekaa', 'dkaal', 'sebaa', 'dkege', 'seatv', 'dkste', 'dksin', 'seang', 'dkbor', 'dkhvi', 'sevax', 'dkrom', 'dkode', 'sehem', 'dkvej', 'dkvix', 'sekrn', 'dkaar', 'seoer', 'selek', 'sehuv']

    obj = _odimsources.load(self.FIXTURE)

    loaded_sources = obj.nods()

    # We expect that both expected and loaded will be the same and hence the number of unique sources should be 0
    unique_sources = set(expected_sources) - set(loaded_sources)
    self.assertEqual(0, len(unique_sources))

  def test_load_sekrn(self):
    obj = _odimsources.load(self.FIXTURE)
    sekrn = obj.get("sekrn")
    self.assertEqual("sekrn", sekrn.nod)
    self.assertEqual("02032", sekrn.wmo)
    self.assertEqual("0-20000-0-2032", sekrn.wigos)
    self.assertEqual("Kiruna", sekrn.plc)
    self.assertEqual("SE40", sekrn.rad)
    self.assertEqual("ESWI", sekrn.cccc)
    self.assertEqual("82", sekrn.org)

  def test_load_dkaar(self):
    obj = _odimsources.load(self.FIXTURE)
    site = obj.get("dkaar")
    self.assertEqual("dkaar", site.nod)
    self.assertEqual("00000", site.wmo)
    self.assertEqual(None, site.wigos)
    self.assertEqual("Aarhus", site.plc)
    self.assertEqual("DN98", site.rad)
    self.assertEqual("EKMI", site.cccc)
    self.assertEqual("94", site.org)

  def test_get_nosuch_nod(self):
    obj = _odimsources.load(self.FIXTURE)
    try:
      site = obj.get("fivan")
      self.fail("Expected KeyError")
    except KeyError:
      pass

  def test_get_wmo(self):
    obj = _odimsources.load(self.FIXTURE)
    site = obj.get_wmo("02032")
    self.assertEqual("sekrn", site.nod)

  def test_get_wigos(self):
    obj = _odimsources.load(self.FIXTURE)
    site = obj.get_wigos("0-20000-0-2032")
    self.assertEqual("sekrn", site.nod)

  def test_identify_source_byNOD(self):
    obj = _odimsources.load(self.FIXTURE)
    site = obj.identify("NOD:sekrn")
    self.assertEqual("sekrn", site.nod)

  def test_identify_source_byWMO(self):
    obj = _odimsources.load(self.FIXTURE)
    site = obj.identify("WMO:02032")
    self.assertEqual("sekrn", site.nod)

  def test_identify_source_byWIGOS(self):
    obj = _odimsources.load(self.FIXTURE)
    site = obj.identify("WIGOS:0-20000-0-2032")
    self.assertEqual("sekrn", site.nod)

  def test_identify_source_byRAD(self):
    obj = _odimsources.load(self.FIXTURE)
    site = obj.identify("RAD:SE40")
    self.assertEqual("sekrn", site.nod)

  def test_identify_source_byWIGOS_WMO(self):
    obj = _odimsources.load(self.FIXTURE)
    site = obj.identify("WIGOS:0-20000-0-2032,WMO:02032")
    self.assertEqual("sekrn", site.nod)

  def test_identify_source_byWMO_RAD(self):
    obj = _odimsources.load(self.FIXTURE)
    site = obj.identify("WMO:00000,RAD:DN99")
    self.assertEqual("dkaal", site.nod)
