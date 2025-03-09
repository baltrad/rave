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

Tests the py composite factory manager module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2025-01-30
'''
import unittest
import os
import _compositefactorymanager
import _legacycompositegeneratorfactory
import _acqvacompositegeneratorfactory

class PyCompositeFactoryManagerTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _compositefactorymanager.new()
    iscorrect = str(type(obj)).find("CompositeFactoryManagerCore")
    self.assertNotEqual(-1, iscorrect)

  def test_getRegisteredFactoryNames_initialized(self):
    obj = _compositefactorymanager.new()
    names = obj.getRegisteredFactoryNames()
    self.assertTrue(3, len(names))
    self.assertTrue("LegacyCompositeGenerator" in names)
    self.assertTrue("AcqvaCompositeGenerator" in names)
    self.assertTrue("NearestCompositeGenerator" in names)

  def test_remove(self):
    obj = _compositefactorymanager.new()
    obj.remove("LegacyCompositeGenerator")
    names = obj.getRegisteredFactoryNames()
    self.assertEqual(2, len(names))
    self.assertTrue("AcqvaCompositeGenerator" in names)
    self.assertTrue("NearestCompositeGenerator" in names)

    obj.remove("AcqvaCompositeGenerator")
    names = obj.getRegisteredFactoryNames()
    self.assertEqual(1, len(names))

    self.assertTrue("NearestCompositeGenerator" in names)
    obj.remove("NearestCompositeGenerator")
    names = obj.getRegisteredFactoryNames()
    self.assertEqual(0, len(names))

  def test_add(self):
    obj = _compositefactorymanager.new()
    obj.remove("LegacyCompositeGenerator")
    obj.remove("AcqvaCompositeGenerator")
    obj.remove("NearestCompositeGenerator")

    obj.add(_legacycompositegeneratorfactory.new())
    names = obj.getRegisteredFactoryNames()
    self.assertEqual(1, len(names))
    self.assertTrue("LegacyCompositeGenerator" in names)

    obj.add(_legacycompositegeneratorfactory.new())
    names = obj.getRegisteredFactoryNames()
    self.assertEqual(1, len(names))
    self.assertTrue("LegacyCompositeGenerator" in names)

    obj.add(_acqvacompositegeneratorfactory.new())
    names = obj.getRegisteredFactoryNames()
    self.assertEqual(2, len(names))
    self.assertTrue("LegacyCompositeGenerator" in names)
    self.assertTrue("AcqvaCompositeGenerator" in names)

  def test_get(self):
    obj = _compositefactorymanager.new()
    legacyfactory = obj.get("LegacyCompositeGenerator")
    acqvafactory = obj.get("AcqvaCompositeGenerator")

    self.assertEqual("LegacyCompositeGenerator", legacyfactory.getName())
    self.assertEqual("AcqvaCompositeGenerator", acqvafactory.getName())    

  def test_isRegistered(self):
    obj = _compositefactorymanager.new()

    self.assertEqual(True, obj.isRegistered("LegacyCompositeGenerator"))
    self.assertEqual(True, obj.isRegistered("AcqvaCompositeGenerator"))
    self.assertEqual(True, obj.isRegistered("NearestCompositeGenerator"))
    self.assertEqual(False, obj.isRegistered("NoSuchGenerator"))
    obj.remove("AcqvaCompositeGenerator")
    self.assertEqual(False, obj.isRegistered("AcqvaCompositeGenerator"))

  def test_size(self):
    obj = _compositefactorymanager.new()
    self.assertEqual(3, obj.size())
    obj.remove("AcqvaCompositeGenerator")
    self.assertEqual(2, obj.size())
    obj.remove("LegacyCompositeGenerator")
    self.assertEqual(1, obj.size())
    obj.remove("NearestCompositeGenerator")
    self.assertEqual(0, obj.size())
