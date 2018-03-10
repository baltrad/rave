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

Tests the pgf argument verification

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2018-03-10
'''
import unittest
import string
import os
import rave_pgf_registry
import rave_pgf_verify
from copy import deepcopy as copy

class rave_pgf_verify_test(unittest.TestCase):
  FIXTURE="fixtures/sample_rave_pgf_registry.xml"
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_verify_generate_arguments_volume(self):
    registry = rave_pgf_registry.PGF_Registry(filename=self.FIXTURE)
    algorithm_entry = copy(registry.find("eu.baltrad.beast.generatevolume"))
    result = rave_pgf_verify.verify_generate_args(["source","selul","date","20180101","time","100000","algorithm_id","123"], algorithm_entry)
    self.assertEqual(True, result)

  def test_verify_generate_arguments_composite(self):
    registry = rave_pgf_registry.PGF_Registry(filename=self.FIXTURE)
    algorithm_entry = copy(registry.find("eu.baltrad.beast.generatecomposite"))
    result = rave_pgf_verify.verify_generate_args(["height",1.0,"range",2.0,"zrA",3.0,"zrb",4.0], algorithm_entry)
    self.assertEqual(True, result)

  def test_verify_generate_arguments_composite_invalid_arg(self):
    registry = rave_pgf_registry.PGF_Registry(filename=self.FIXTURE)
    algorithm_entry = copy(registry.find("eu.baltrad.beast.generatecomposite"))
    result = rave_pgf_verify.verify_generate_args(["height","abc","range",2.0,"zrA",3.0,"zrb",4.0], algorithm_entry)
    self.assertEqual(False, result)
