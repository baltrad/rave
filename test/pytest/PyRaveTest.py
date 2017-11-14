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

Tests the rave module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2011-07-06
'''
import unittest

import _rave

class PyRaveTest(unittest.TestCase):
  def testIsXMlSupported(self):
    modulebuilt=False
    try:
      import _projectionregistry
      import _arearegistry
      modulebuilt=True
    except:
      pass
    self.assertEquals(modulebuilt, _rave.isXmlSupported())

  def testCompareDateTime_1(self):
    self.assertEqual(0, _rave.compare_datetime("20171030","013059", "20171030","013059")) 
    self.assertEqual(0, _rave.compare_datetime("20171231","235959", "20171231","235959")) 
    self.assertEqual(0, _rave.compare_datetime("20170101","000000", "20170101","000000")) 

  def testCompareDateTime_2(self):
    self.assertEqual(-1, _rave.compare_datetime("20171030","013059", "20181130","013059")) 
    self.assertEqual(-1, _rave.compare_datetime("20171231","235959", "20180101","000000")) 
    self.assertEqual(-1, _rave.compare_datetime("20170101","000000", "20180101","000000")) 
    
  def testCompareDateTime_3(self):
    self.assertEqual(1, _rave.compare_datetime("20171030","013059", "20170930","013059")) 
    self.assertEqual(1, _rave.compare_datetime("20180101","000000", "20171231","235959")) 
    self.assertEqual(1, _rave.compare_datetime("20180101","000000", "20170101","000000")) 

if __name__ == "__main__":
  unittest.main()