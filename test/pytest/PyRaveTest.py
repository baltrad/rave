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
    self.assertEqual(modulebuilt, _rave.isXmlSupported())

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

  def check_value_in_list(self, result, name, value):
    for r in result:
      if r[0] == name:
        if isinstance(value, str):
          self.assertEqual(value, r[1])
        elif isinstance(value, float):
          self.assertAlmostEqual(value, r[1], 4)
        elif isinstance(value, list):
          self.assertEqual(len(value), len(r[1]))
          idx = 0
          for i in value:
            self.assertAlmostEqual(i, r[1][idx])
            idx=idx+1
        return
    self.fail("Expected a match for name %s"%name)


  def test_translate_from_projection_to_wkt_aea(self):
    proj = _rave.projection("myid", "aea", "+proj=aea +lat_0=1 +lon_0=2 +x_0=14 +y_0=60 +lat_1=12 +lat_2=13 +a=6378160 +b=6356775")
    result = _rave.translate_from_projection_to_wkt(proj)
    self.check_value_in_list(result, "grid_mapping_name", "albers_conical_equal_area")
    self.check_value_in_list(result, "latitude_of_projection_origin", 1.0)
    self.check_value_in_list(result, "longitude_of_central_meridian", 2.0)
    self.check_value_in_list(result, "false_easting", 14.0)
    self.check_value_in_list(result, "false_northing", 60.0)
    self.check_value_in_list(result, "standard_parallel", [12.0,13.0])

  def test_translate_from_projection_to_wkt_aeqd(self):
    proj = _rave.projection("myid", "aeqd", "+proj=aeqd +lat_0=1 +lon_0=2 +x_0=14 +y_0=60 +R=6378137.0")
    result = _rave.translate_from_projection_to_wkt(proj)
    self.check_value_in_list(result, "grid_mapping_name", "azimuthal_equidistant")
    self.check_value_in_list(result, "latitude_of_projection_origin", 1.0)
    self.check_value_in_list(result, "longitude_of_projection_origin", 2.0)
    self.check_value_in_list(result, "false_easting", 14.0)
    self.check_value_in_list(result, "false_northing", 60.0)
    self.check_value_in_list(result, "earth_radius", 6378137.0)
 
  def test_translate_from_projection_to_wkt_laea(self):
    proj = _rave.projection("myid", "laea", "+proj=laea +lat_0=1 +lon_0=2 +x_0=14 +y_0=60 +R=6378137.0")
    result = _rave.translate_from_projection_to_wkt(proj)
    self.check_value_in_list(result, "grid_mapping_name", "lambert_azimuthal_equal_area")
    self.check_value_in_list(result, "latitude_of_projection_origin", 1.0)
    self.check_value_in_list(result, "longitude_of_projection_origin", 2.0)
    self.check_value_in_list(result, "false_easting", 14.0)
    self.check_value_in_list(result, "false_northing", 60.0)
    self.check_value_in_list(result, "earth_radius", 6378137.0)
    
  def test_translate_from_projection_to_wkt_lcc(self):
    proj = _rave.projection("myid", "lcc", "+proj=lcc +lat_0=1 +lon_0=2 +x_0=14 +y_0=60 +lat_1=12 +lat_2=13 +a=6378160 +b=6356775")
    result = _rave.translate_from_projection_to_wkt(proj)
    self.check_value_in_list(result, "grid_mapping_name", "lambert_conformal_conic")
    self.check_value_in_list(result, "latitude_of_projection_origin", 1.0)
    self.check_value_in_list(result, "longitude_of_central_meridian", 2.0)
    self.check_value_in_list(result, "false_easting", 14.0)
    self.check_value_in_list(result, "false_northing", 60.0)
    self.check_value_in_list(result, "standard_parallel", [12.0,13.0])

  def Xtest_translate_from_projection_to_wkt_leac(self):
    proj = _rave.projection("myid", "leac", "+proj=leac +lat_ts=0 +lon_0=0 +k_0=1.0 +R=6378137.0 +no_defs")
    result = _rave.translate_from_projection_to_wkt(proj)
    self.check_value_in_list(result, "grid_mapping_name", "lambert_cylindrical_equal_area")
    self.check_value_in_list(result, "longitude_of_projection_origin", 0.0)
    self.check_value_in_list(result, "standard_parallel", 0.0)
    self.check_value_in_list(result, "scale_factor_at_projection_origin", 1.0)
    self.check_value_in_list(result, "earth_radius", 6378137.0)

    
  def test_translate_from_projection_to_wkt_mercator(self):
    proj = _rave.projection("myid", "mercator", "+proj=merc +lat_ts=0 +lon_0=0 +k_0=1.0 +R=6378137.0 +no_defs")
    result = _rave.translate_from_projection_to_wkt(proj)
    self.check_value_in_list(result, "grid_mapping_name", "mercator")
    self.check_value_in_list(result, "longitude_of_projection_origin", 0.0)
    self.check_value_in_list(result, "standard_parallel", 0.0)
    self.check_value_in_list(result, "scale_factor_at_projection_origin", 1.0)
    self.check_value_in_list(result, "earth_radius", 6378137.0)

  def test_ebase_product_type(self):
    self.assertEqual(19, _rave.Rave_ProductType_EBASE)

if __name__ == "__main__":
  unittest.main()
