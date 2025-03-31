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
import _odimsource
import string
import math

class PyOdimSourceTest(unittest.TestCase):
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _odimsource.new("sekkr")
    isodimsource = str(type(obj)).find("OdimSourceCore")
    self.assertNotEqual(-1, isodimsource)

  def test_new_wo_nod(self):
    try:
      obj = _odimsource.new()
      self.fail("Expected TypeError")
    except TypeError:
      pass

  def test_isOdimSource(self):
    obj = _odimsource.new("sekkr")
    self.assertEqual(True, _odimsource.isOdimSource(obj))
    
    self.assertEqual(False, _odimsource.isOdimSource("abc"))

  def test_attribute_visibility(self):
    attrs = ['nod', 'wmo', 'wigos', 'plc', 'rad', 'cccc', 'org']
    source = _odimsource.new("sekkr")
    alist = dir(source)
    for a in attrs:
      self.assertEqual(True, a in alist)

  def test_nod(self):
    obj = _odimsource.new("sekkr")

    self.assertEqual("sekkr", obj.nod)
    try:
      obj.nod = "sehuv"
      self.fail("Expected TypeError")
    except TypeError:
      pass
    self.assertEqual("sekkr", obj.nod)

  def test_wmo(self):
    obj = _odimsource.new("sekkr")

    self.assertEqual(None, obj.wmo)
    obj.wmo = "02032"
    self.assertEqual("02032", obj.wmo)
    obj.wmo = None
    self.assertEqual(None, obj.wmo)

    try:
      obj.wmo = 1
      self.fail("Expected TypeError")
    except TypeError:
      pass

  def test_wigos(self):
    obj = _odimsource.new("sekkr")

    self.assertEqual(None, obj.wigos)
    obj.wigos = "0-20000-0-2032"
    self.assertEqual("0-20000-0-2032", obj.wigos)
    obj.wigos = None
    self.assertEqual(None, obj.wigos)

    try:
      obj.wigos = 1
      self.fail("Expected TypeError")
    except TypeError:
      pass

  def test_plc(self):
    obj = _odimsource.new("sekkr")

    self.assertEqual(None, obj.plc)
    obj.plc = "Kiruna"
    self.assertEqual("Kiruna", obj.plc)
    obj.plc = None
    self.assertEqual(None, obj.plc)

    try:
      obj.plc = 1
      self.fail("Expected TypeError")
    except TypeError:
      pass

  def test_rad(self):
    obj = _odimsource.new("sekkr")

    self.assertEqual(None, obj.rad)
    obj.rad = "SE40"
    self.assertEqual("SE40", obj.rad)
    obj.rad = None
    self.assertEqual(None, obj.rad)

    try:
      obj.rad = 1
      self.fail("Expected TypeError")
    except TypeError:
      pass

  def test_cccc(self):
    obj = _odimsource.new("sekkr")

    self.assertEqual(None, obj.cccc)
    obj.cccc = "ESWI"
    self.assertEqual("ESWI", obj.cccc)
    obj.cccc = None
    self.assertEqual(None, obj.cccc)

    try:
      obj.cccc = 1
      self.fail("Expected TypeError")
    except TypeError:
      pass

  def test_org(self):
    obj = _odimsource.new("sekkr")

    self.assertEqual(None, obj.org)
    obj.org = "82"
    self.assertEqual("82", obj.org)
    obj.org = None
    self.assertEqual(None, obj.org)

    try:
      obj.org = 1
      self.fail("Expected TypeError")
    except TypeError:
      pass

  def test_new(self):
    obj = _odimsource.new("sekkr", "02032", "0-20000-0-2032", "Kiruna", "SE40", "ESWI", "82")

    self.assertEqual("sekkr", obj.nod)
    self.assertEqual("02032", obj.wmo)
    self.assertEqual("0-20000-0-2032", obj.wigos)
    self.assertEqual("Kiruna", obj.plc)
    self.assertEqual("SE40", obj.rad)
    self.assertEqual("ESWI", obj.cccc)
    self.assertEqual("82", obj.org)

  def test_new_nonod(self):
    try:
      _odimsource.new(None, "02032", "0-20000-0-2032", "Kiruna", "SE40", "ESWI", "82")
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

  def test_source_seknr(self):
    obj = _odimsource.new("seknr", "02032", "0-20000-0-2032", "Kiruna", "SE40", "ESWI", "82")

    self.assertEqual("NOD:seknr,WMO:02032,RAD:SE40,PLC:Kiruna,WIGOS:0-20000-0-2032", obj.source)

  def test_source_seknr_wmo00000(self):
    obj = _odimsource.new("seknr", "00000", "0-20000-0-2032", "Kiruna", "SE40", "ESWI", "82")

    self.assertEqual("NOD:seknr,RAD:SE40,PLC:Kiruna,WIGOS:0-20000-0-2032", obj.source)

  def test_source_seknr_nowmo(self):
    obj = _odimsource.new("seknr", None, "0-20000-0-2032", "Kiruna", "SE40", "ESWI", "82")

    self.assertEqual("NOD:seknr,RAD:SE40,PLC:Kiruna,WIGOS:0-20000-0-2032", obj.source)

  def test_source_seknr_onlynod(self):
    obj = _odimsource.new("seknr", None, None, None, None, None, None)

    self.assertEqual("NOD:seknr", obj.source)

  def test_source_sella(self):
    obj = _odimsource.new("sella", "02092", "0-20000-0-2092", "Luleå", "SE41", "ESWI", "82")

    self.assertEqual("NOD:sella,WMO:02092,RAD:SE41,PLC:Luleå,WIGOS:0-20000-0-2092", obj.source)
