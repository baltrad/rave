'''
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the py poo composite algorithm module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2011-10-31
'''
import _poocompositealgorithm
import unittest
import string

class PyPooCompositeAlgorithmTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass
  
  def test_new(self):
    obj = _poocompositealgorithm.new()
    self.assertNotEqual(-1, str(type(obj)).find("CompositeAlgorithmCore"))

  def test_getName(self):
    obj = _poocompositealgorithm.new()
    self.assertEqual("POO", obj.getName())
  
  def test_process(self):
    obj = _poocompositealgorithm.new()
    obj.process()
        