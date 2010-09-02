'''
Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,

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
------------------------------------------------------------------------

Tests the rave module list

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-12-10
'''

import unittest
import os
import _rave

class RaveModuleListTest(unittest.TestCase):
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def XtestNewRaveList(self):
    obj = _rave.list()
    isravelist = string.find(`type(obj)`, "RaveListCore")
    self.assertNotEqual(-1, isravelist)

  def XtestAdd(self):
    obj = _rave.list()
    scan = _rave.scan()
    scan.elangle = 1.0
    obj.add(scan)
    self.assertEquals(1, obj.size())
    
