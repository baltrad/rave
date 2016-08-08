'''
Copyright (C) 2016- Swedish Meteorological and Hydrological Institute, SMHI,

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

Test case for the rave_hexquant name conversion

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2016-08-08
'''

import unittest, os, datetime
from rave_hexquant import *

class rave_hexquant_test(unittest.TestCase):

  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_q2hex_DBZH_VRAD(self):
    result = q2hex(["DBZH","VRAD"])
    self.assertEqual("0x5", result)

  def test_q2hex_Nothing(self):
    result = q2hex([])
    self.assertEqual("0x0", result)

  def test_q2hex_UnknownQuantity(self):
    result = q2hex(["NONAME", "DBZH"])
    self.assertEqual("0x1", result)