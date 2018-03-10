'''
Copyright (C) 2018 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the python version of the pgf qtools handling

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2018-03-10
'''
import unittest
import string
import os
import rave_pgf_registry
import rave_pgf_protocol
from copy import deepcopy as copy

class rave_pgf_protocol_test(unittest.TestCase):
  FIXTURE="fixtures/sample_rave_pgf_registry.xml"
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_convert_beast_arguments(self):
    registry = rave_pgf_registry.PGF_Registry(filename=self.FIXTURE)
    algorithm_entry = copy(registry.find("eu.baltrad.beast.generatevolume"))
    arguments = rave_pgf_protocol.convert_arguments("eu.baltrad.beast.generatevolume", algorithm_entry, ["--source=selul","--date=20180101","--time=100000","--algorithm_id=123"])
    self.assertEqual(8, len(arguments))
    self.assertEqual("source", arguments[0])
    self.assertEqual("selul", arguments[1])
    self.assertEqual("date", arguments[2])
    self.assertEqual("20180101", arguments[3])
    self.assertEqual("time", arguments[4])
    self.assertEqual("100000", arguments[5])
    self.assertEqual("algorithm_id", arguments[6])
    self.assertEqual("123", arguments[7])
    
  
  def test_convert_beast_arguments_floats(self):
    registry = rave_pgf_registry.PGF_Registry(filename=self.FIXTURE)
    algorithm_entry = copy(registry.find("eu.baltrad.beast.generatecomposite"))
    arguments = rave_pgf_protocol.convert_arguments("eu.baltrad.beast.generatecomposite", algorithm_entry, ["--height=10.0","--range=123.0","--zrA=1.0","--zrb=2.0"])
    self.assertEqual(8, len(arguments))
    self.assertEqual("height", arguments[0])
    self.assertAlmostEqual(10.0, arguments[1],4)
    self.assertEqual("range", arguments[2])
    self.assertAlmostEqual(123.0, arguments[3],4)
    self.assertEqual("zrA", arguments[4])
    self.assertAlmostEqual(1.0, arguments[5],4)
    self.assertEqual("zrb", arguments[6])
    self.assertAlmostEqual(2.0, arguments[7],4)
