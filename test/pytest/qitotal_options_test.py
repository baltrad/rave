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

Tests the qitotal options reader

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2014-03-22
'''
import unittest
import os
import math
import string
import numpy
import qitotal_options
import _raveio, _ravefield

class qitotal_options_test(unittest.TestCase):
  OPTIONS_FIXTURE = "fixtures/test_qitotal_options.xml"
  
  def setUp(self):
    pass
  
  def tearDown(self):
    pass
  
  def test_parse_qitotal_site_information(self):
    result = qitotal_options.parse_qitotal_site_information(self.OPTIONS_FIXTURE)
    self.assertTrue("default" in result.keys())
    self.assertTrue("sehuv" in result.keys())
    v = result["sehuv"]
    self.assertEqual(2, len(v.qifields()))
    self.assertEqual("se.smhi.test.1", v.qifields()[0].name())
    self.assertAlmostEqual(0.3, v.qifields()[0].weight(), 4)
    self.assertEqual("se.smhi.test.2", v.qifields()[1].name())
    self.assertAlmostEqual(0.7, v.qifields()[1].weight(), 4)

    v = result["default"]
    self.assertEqual(2, len(v.qifields()))
    self.assertEqual("se.smhi.test.1", v.qifields()[0].name())
    self.assertAlmostEqual(0.5, v.qifields()[0].weight(), 4)
    self.assertEqual("se.smhi.test.2", v.qifields()[1].name())
    self.assertAlmostEqual(0.5, v.qifields()[1].weight(), 4)
    