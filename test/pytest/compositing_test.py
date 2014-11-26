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
import _rave
import string
import os
import compositing
import _pycomposite
import area_registry

from xml.etree import ElementTree

class compositing_test(unittest.TestCase):
  def setUp(self):
    self.classUnderTest = compositing.compositing() 

  def tearDown(self):
    self.classUnderTest = None
  
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
