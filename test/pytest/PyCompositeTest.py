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
------------------------------------------------------------------------*/

Tests the composite module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2010-01-29
'''
import unittest
import _pycomposite
import _rave
import _area
import _projection, _projectionpipeline
import _raveio
import _ravefield
import _polarscan, _polarvolume, _polarscanparam, _poocompositealgorithm
import numpy
import math
import string
import sys
from numpy import cos, sin

EVEN_RANGE_DUMMY_VALUE = 10
UNEVEN_RANGE_DUMMY_VALUE = 190
EVEN_AZIMUTH_DUMMY_VALUE = 20
UNEVEN_AZIMUTH_DUMMY_VALUE = 230

def fill_pvol_with_dummy_data(pvol_object, get_dummy_data_value_func, quantities, gain):
  i = 0
  while i <  pvol_object.getNumberOfScans():
    scan = pvol_object.getScan(i)
    elangle_deg = math.degrees(scan.elangle)
    
    for parameterName in scan.getParameterNames():
      if parameterName in quantities:
        parameter = scan.getParameter(parameterName)
        nbins = parameter.nbins
        nrays = parameter.nrays
        parameter.gain = 1.0
        parameter.offset = 0.0
        data = numpy.zeros((nrays, nbins), numpy.int16)
        parameter.setData(data)
        
        height_field = scan.getHeightField()
        distance_field = scan.getDistanceField()
        for bin_no in range(nbins):
          for ray_no in range(nrays):
            dummy_value = get_dummy_data_value_func(elangle_deg, height_field, distance_field, bin_no, ray_no)
            adjusted_value = min((dummy_value / gain), 254)
            scan.setParameterValue(parameterName, (bin_no, ray_no), adjusted_value)
      else:
        scan.removeParameter(parameterName)          
    
    i += 1
    
def get_height_dummy_data(elangle_deg, height_field, distance_field, bin_no, ray_no):
  return height_field.getValue(bin_no, 0)[1]

def get_range_alternating_dummy_data(elangle_deg, height_field, distance_field, bin_no, ray_no):
  if bin_no % 2 == 0:
    return EVEN_RANGE_DUMMY_VALUE
  else:
    return UNEVEN_RANGE_DUMMY_VALUE
  
def get_azimuth_alternating_dummy_data(elangle_deg, height_field, distance_field, bin_no, ray_no):
  if ray_no % 2 == 0:
    return EVEN_AZIMUTH_DUMMY_VALUE
  else:
    return UNEVEN_AZIMUTH_DUMMY_VALUE
  
def get_range_and_azimuth_alternating_dummy_data(elangle_deg, height_field, distance_field, bin_no, ray_no):
  range_val = 0
  azimuth_val = 0
  if bin_no % 2 == 0:
    range_val = EVEN_RANGE_DUMMY_VALUE
  else:
    range_val = UNEVEN_RANGE_DUMMY_VALUE
  if ray_no % 2 == 0:
    azimuth_val = EVEN_AZIMUTH_DUMMY_VALUE
  else:
    azimuth_val = UNEVEN_AZIMUTH_DUMMY_VALUE
  return (range_val + azimuth_val) / 2

def get_3d_alternating_dummy_data(elangle_deg, height_field, distance_field, bin_no, ray_no):
  height_val = height_field.getValue(bin_no, 0)[1]
  value = height_val
  if bin_no % 2 == 0:
    value += 500
  else:
    value -= 500
  if ray_no % 2 == 0:
    value += 700
  else:
    value -= 700
  return value
  
  
class PyCompositeTest(unittest.TestCase):
  SWEDISH_VOLUMES = ["fixtures/pvol_seang_20090501T120000Z.h5",
                     "fixtures/pvol_searl_20090501T120000Z.h5",
                     "fixtures/pvol_sease_20090501T120000Z.h5",
                     "fixtures/pvol_sehud_20090501T120000Z.h5",
                     "fixtures/pvol_sekir_20090501T120000Z.h5",
                     "fixtures/pvol_sekkr_20090501T120000Z.h5",
                     "fixtures/pvol_selek_20090501T120000Z.h5",
                     "fixtures/pvol_selul_20090501T120000Z.h5",
                     "fixtures/pvol_seosu_20090501T120000Z.h5",
                     "fixtures/pvol_seovi_20090501T120000Z.h5",
                     "fixtures/pvol_sevar_20090501T120000Z.h5",
                     "fixtures/pvol_sevil_20090501T120000Z.h5"]

  QC_VOLUMES = ["fixtures/searl_qcvol_20120131T0000Z.h5",
                "fixtures/sease_qcvol_20120131T0000Z.h5",
                "fixtures/sevil_qcvol_20120131T0000Z.h5",
                "fixtures/seang_qcvol_20120131T0000Z.h5"
                ]
  
  QC_VOLUMES_2016 = ["fixtures/pvol_qc_searl_20161103T0900.h5",
                     "fixtures/pvol_qc_sease_20161103T0900.h5",
                     "fixtures/pvol_qc_seosd_20161103T0900.h5",
                     "fixtures/pvol_qc_sevil_20161103T0900.h5"
                    ]
  
  MAX_VOLUMES = ["fixtures/prepared_max_fixture_hud.h5",
                 "fixtures/prepared_max_fixture_osu.h5",
                 "fixtures/prepared_max_fixture_ovi.h5"]
  
  DUMMY_DATA_FIXTURES = ["fixtures/sehem_qcvol_pn129_20180129T100000Z_0x73fc7b_dummydata.h5"]
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _pycomposite.new()
    self.assertNotEqual(-1, str(type(obj)).find("CompositeCore"))

  def test_attribute_visibility(self):
    attrs = ['height', 'range', 'elangle', 'product', 'date', 'time', 'selection_method', 'quality_indicator_field_name']
    obj = _pycomposite.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)
 
  def test_height(self):
    obj = _pycomposite.new()
    self.assertAlmostEqual(1000.0, obj.height, 4)
    obj.height = 1.0
    self.assertAlmostEqual(1.0, obj.height, 4)
     
  def test_range(self):
    obj = _pycomposite.new()
    self.assertAlmostEqual(500000.0, obj.range, 4)
    obj.range = 1.0
    self.assertAlmostEqual(1.0, obj.range, 4)
  
  def test_quality_indicator_field_name(self):
    obj = _pycomposite.new()
    self.assertEqual(None, obj.quality_indicator_field_name)
    obj.quality_indicator_field_name = "se.some.field"
    self.assertEqual("se.some.field", obj.quality_indicator_field_name)
    obj.quality_indicator_field_name = None
    self.assertEqual(None, obj.quality_indicator_field_name)
  
  def test_elangle(self):
    obj = _pycomposite.new()
    self.assertAlmostEqual(0.0, obj.elangle, 4)
    obj.elangle = 1.0
    self.assertAlmostEqual(1.0, obj.elangle, 4)
     
  def test_product(self):
    obj = _pycomposite.new()
    self.assertEqual(_rave.Rave_ProductType_PCAPPI, obj.product)
 
  def test_product_valid(self):
    valid_products = [_rave.Rave_ProductType_PCAPPI, 
                      _rave.Rave_ProductType_CAPPI, 
                      _rave.Rave_ProductType_PPI, 
                      _rave.Rave_ProductType_PMAX, 
                      _rave.Rave_ProductType_MAX]
    obj = _pycomposite.new()
    for p in valid_products:
      obj.product = p
      self.assertEqual(p, obj.product)
 
  def test_product_invalid(self):
    invalid_products = [_rave.Rave_ProductType_SCAN, 
                        _rave.Rave_ProductType_ETOP, 
                        _rave.Rave_ProductType_RR, 
                        _rave.Rave_ProductType_VIL, 
                        _rave.Rave_ProductType_COMP,
                        _rave.Rave_ProductType_VP,
                        _rave.Rave_ProductType_RHI,
                        _rave.Rave_ProductType_XSEC,
                        _rave.Rave_ProductType_VSP,
                        _rave.Rave_ProductType_HSP,
                        _rave.Rave_ProductType_RAY,
                        _rave.Rave_ProductType_AZIM,
                        _rave.Rave_ProductType_QUAL,
                        _rave.Rave_ProductType_HSP]
    obj = _pycomposite.new()
    obj.product = _rave.Rave_ProductType_PCAPPI
    for p in invalid_products:
      obj.product = p
      self.assertEqual(_rave.Rave_ProductType_PCAPPI, obj.product)
 
  def test_selection_method(self):
    obj = _pycomposite.new()
    self.assertEqual(_pycomposite.SelectionMethod_NEAREST, obj.selection_method)
    obj.selection_method = _pycomposite.SelectionMethod_HEIGHT
    self.assertEqual(_pycomposite.SelectionMethod_HEIGHT, obj.selection_method)
 
  def test_date(self):
    obj = _pycomposite.new()
    self.assertEqual(None, obj.date)
    obj.date = "20130101"
    self.assertEqual("20130101", obj.date)
    obj.date = None
    self.assertEqual(None, obj.date)
 
  def test_time(self):
    obj = _pycomposite.new()
    self.assertEqual(None, obj.time)
    obj.time = "101010"
    self.assertEqual("101010", obj.time)
    obj.time = None
    self.assertEqual(None, obj.time)
 
  def test_selection_method_invalid(self):
    obj = _pycomposite.new()
    try:
      obj.selection_method = 99
      self.fail("Expected ValueError")
    except ValueError:
      pass
    self.assertEqual(_pycomposite.SelectionMethod_NEAREST, obj.selection_method)
   
  def test_addParameter(self):
    obj = _pycomposite.new()
    obj.addParameter("DBZH", 2.0, 3.0, -30.0)
    result = obj.getParameter(0)
    self.assertEqual("DBZH", result[0])
    self.assertAlmostEqual(2.0, result[1], 4)
    self.assertAlmostEqual(3.0, result[2], 4)
 
  def test_addParameter_duplicate(self):
    obj = _pycomposite.new()
    obj.addParameter("DBZH", 2.0, 3.0, -30.0)
    obj.addParameter("DBZH", 3.0, 4.0, -30.0)
    self.assertEqual(1, obj.getParameterCount())
    result = obj.getParameter(0)
    self.assertEqual("DBZH", result[0])
    self.assertAlmostEqual(3.0, result[1], 4)
    self.assertAlmostEqual(4.0, result[2], 4)
 
  def test_addParameter_multiple(self):
    obj = _pycomposite.new()
    obj.addParameter("DBZH", 2.0, 3.0, -30.0)
    obj.addParameter("MMH", 3.0, 4.0, -30.0)
    self.assertEqual(2, obj.getParameterCount())
    result = obj.getParameter(0)
    self.assertEqual("DBZH", result[0])
    self.assertAlmostEqual(2.0, result[1], 4)
    self.assertAlmostEqual(3.0, result[2], 4)
    result = obj.getParameter(1)
    self.assertEqual("MMH", result[0])
    self.assertAlmostEqual(3.0, result[1], 4)
    self.assertAlmostEqual(4.0, result[2], 4)
 
  def test_hasParameter(self):
    obj = _pycomposite.new()
    self.assertEqual(False, obj.hasParameter("DBZH"))
    obj.addParameter("DBZH", 2.0, 3.0, -30.0)
    self.assertEqual(True, obj.hasParameter("DBZH"))
 
  def test_getParameterCount(self):
    obj = _pycomposite.new()
    self.assertEqual(0, obj.getParameterCount())
    obj.addParameter("DBZH", 2.0, 3.0, -30.0)
    self.assertEqual(1, obj.getParameterCount())
    obj.addParameter("MMH", 1.0, 2.0, -30.0)
    self.assertEqual(2, obj.getParameterCount())

  def test_rix_nearest(self):
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
      
    for fname in ["fixtures/eesyr_volume.h5", "fixtures/rix_volume.h5"]:
      rio = _raveio.open(fname)
      generator.add(rio.object)
      
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
      
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.generate(a)
      
    self.assertEqual("DBZH", result.getParameter("DBZH").quantity)
    self.assertAlmostEqual(1.0, result.getParameter("DBZH").gain, 4)
    self.assertAlmostEqual(0.0, result.getParameter("DBZH").offset, 4)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEqual(0.0, prodpar, 4)
    self.assertEqual(_rave.Rave_ProductType_PPI, result.product)
    self.assertEqual(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEqual("nrd2km", result.source);
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "rixeecomposite.h5"
    ios.save()
  
  def verify_common_swe_pcappi_values(self, result):
    dbzh_param = result.getParameter("DBZH")
    self.assertNotEqual(255, dbzh_param.getData()[829][603], "PCAPPI should fill with data at position in composite.")
    self.assertNotEqual(255, dbzh_param.getData()[979][243], "PCAPPI should fill with data at position in composite.")
    self.assertNotEqual(255, dbzh_param.getData()[164][506], "PCAPPI should fill with data at position in composite.")
    self.assertNotEqual(255, dbzh_param.getData()[548][281], "PCAPPI should fill with data at position in composite.")
     
    self.assertEqual(255, dbzh_param.getData()[561][261], "No data should be found at position in composite.")
    self.assertEqual(255, dbzh_param.getData()[615][567], "No data should be found at position in composite.")
    self.assertEqual(255, dbzh_param.getData()[371][427], "No data should be found at position in composite.")
     
    self.assertNotEqual(0, dbzh_param.getData()[829][603], "Precipitation should be detected at position in composite.")
    self.assertNotEqual(0, dbzh_param.getData()[243][556], "Precipitation should be detected at position in composite.")
    self.assertNotEqual(0, dbzh_param.getData()[870][458], "Precipitation should be detected at position in composite.")
    self.assertNotEqual(0, dbzh_param.getData()[467][205], "Precipitation should be detected at position in composite.")
   
  def do_swe_pcappi_test(self, interpolation_method):  
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +towgs84=0,0,0")
      
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
      
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PCAPPI
    generator.height = 1000.0
    generator.time = "120000"
    generator.date = "20090501"
    generator.interpolation_method = interpolation_method
    result = generator.generate(a)
      
    self.assertEqual("DBZH", result.getParameter("DBZH").quantity)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEqual(1000.0, prodpar, 4)
    self.assertEqual(_rave.Rave_ProductType_PCAPPI, result.product)
    self.assertEqual(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEqual("nrd2km", result.source);
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite.h5"
    ios.save()
     
    self.verify_common_swe_pcappi_values(result)
     
    return result
   
  def test_nearest(self):
    result = self.do_swe_pcappi_test(_pycomposite.InterpolationMethod_NEAREST)
     
    dbzh_param = result.getParameter("DBZH")
    self.assertEqual(0, dbzh_param.getData()[402][524], "No precipitation should be detected at position in composite.")
    self.assertEqual(2, dbzh_param.getData()[421][538], "Wrong value at position in composite.")
    self.assertEqual(1, dbzh_param.getData()[246][564], "Wrong value at position in composite.")
    self.assertEqual(29, dbzh_param.getData()[244][552], "Wrong value at position in composite.")
    self.assertEqual(14, dbzh_param.getData()[862][608], "Wrong value at position in composite.")
    self.assertEqual(31, dbzh_param.getData()[850][597], "Wrong value at position in composite.")
     
  def test_pcappi_linear_height_interpolation(self):
    result = self.do_swe_pcappi_test(_pycomposite.InterpolationMethod_LINEAR_HEIGHT)
     
    dbzh_param = result.getParameter("DBZH")
    self.assertEqual(2, dbzh_param.getData()[402][524], "Wrong value at position in composite.")
    self.assertEqual(0, dbzh_param.getData()[421][538], "Wrong value at position in composite.")
    self.assertEqual(1, dbzh_param.getData()[246][564], "Wrong value at position in composite.")
    self.assertEqual(29, dbzh_param.getData()[244][552], "Wrong value at position in composite.")
    self.assertEqual(14, dbzh_param.getData()[862][608], "Wrong value at position in composite.")
    self.assertEqual(31, dbzh_param.getData()[850][597], "Wrong value at position in composite.")
     
  def test_pcappi_linear_range_and_azimuth_interpolation(self):
    result = self.do_swe_pcappi_test(_pycomposite.InterpolationMethod_LINEAR_RANGE_AND_AZIMUTH)
     
    dbzh_param = result.getParameter("DBZH")
    self.assertEqual(0, dbzh_param.getData()[402][524], "Wrong value at position in composite.")
    self.assertEqual(0, dbzh_param.getData()[421][538], "Wrong value at position in composite.")
    self.assertEqual(0, dbzh_param.getData()[246][564], "Wrong value at position in composite.")
    self.assertEqual(28, dbzh_param.getData()[244][552], "Wrong value at position in composite.")
    self.assertEqual(0, dbzh_param.getData()[862][608], "Wrong value at position in composite.")
    self.assertEqual(25, dbzh_param.getData()[850][597], "Wrong value at position in composite.")
     
  def test_pcappi_quadratic_3d_interpolation(self):
    result = self.do_swe_pcappi_test(_pycomposite.InterpolationMethod_QUADRATIC_3D)
     
    dbzh_param = result.getParameter("DBZH")
    self.assertEqual(0, dbzh_param.getData()[402][524], "Wrong value at position in composite.")
    self.assertEqual(0, dbzh_param.getData()[421][538], "Wrong value at position in composite.")
    self.assertEqual(1, dbzh_param.getData()[246][564], "Wrong value at position in composite.")
    self.assertEqual(29, dbzh_param.getData()[244][552], "Wrong value at position in composite.")
    self.assertEqual(0, dbzh_param.getData()[862][608], "Wrong value at position in composite.")
    self.assertEqual(25, dbzh_param.getData()[850][597], "Wrong value at position in composite.")

  def deg2rad(self, xy):
    return (xy[0]*math.pi/180, xy[1]*math.pi/180.0)
  
  def rad2deg(self, xy):
    return (xy[0]*180.0/math.pi, xy[1]*180.0/math.pi)
  
  def generate_sehem_test_comp(self, dummy_data_func, value_gain, height, interpolation_method):
    generator = _pycomposite.new()
    a = _area.new()
    a.id = "hem_ps14e60n_500m"
    a.xsize = 981
    a.ysize = 978
    a.xscale = 500.0
    a.yscale = 500.0
    a.extent = (23785.852938, -3734664.464654, 514285.852938, -3245664.464654) # sehem projection extent
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +towgs84=0,0,0")
    
    fname = self.DUMMY_DATA_FIXTURES[0]
    pvol_object = _raveio.open(fname).object
    fill_pvol_with_dummy_data(pvol_object, dummy_data_func, ["DBZH"], value_gain)
    for i in range(pvol_object.getNumberOfScans()):
      pvol_object.getScan(i).use_azimuthal_nav_information = False
    generator.add(pvol_object)
 
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_CAPPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    generator.height = height
    generator.interpolation_method = interpolation_method
    result = generator.generate(a, ["fi.fmi.ropo.detector.classification", "se.smhi.detector.beamblockage", "se.smhi.composite.distance.radar"])
     
    return result

  def validate_sehem_distance_field(self, param):
    distance_field = param.getQualityFieldByHowTask("se.smhi.composite.distance.radar")
    self.assertEqual(3, distance_field.getData()[494][500], "Wrong quality value at position in composite.")
    self.assertEqual(59, distance_field.getData()[494][250], "Wrong quality value at position in composite.")
    self.assertEqual(34, distance_field.getData()[494][350], "Wrong quality value at position in composite.")
    self.assertEqual(52, distance_field.getData()[494][700], "Wrong quality value at position in composite.")
  

  def test_linear_height_interpolation_cappi(self):
    height = 2600
    height_gain = 100.0
    allowed_height_diff = 1
     
    result = self.generate_sehem_test_comp(get_height_dummy_data, height_gain, height, _pycomposite.InterpolationMethod_LINEAR_HEIGHT)
     
    dbzh_param = result.getParameter("DBZH")
    data = dbzh_param.getData()
    for x in range(260, 470):
      for y in range(340, 650):
        height_diff = abs(data[y, x] - (height/height_gain))
        self.assertTrue(height_diff <= allowed_height_diff)
    
    ropo_field = dbzh_param.getQualityFieldByHowTask("fi.fmi.ropo.detector.classification")
    self.assertEqual(243, ropo_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[423][549], "Wrong quality value at position in composite.")
    self.assertEqual(242, ropo_field.getData()[486][373], "Wrong quality value at position in composite.")
    self.assertEqual(246, ropo_field.getData()[487][461], "Wrong quality value at position in composite.")
    self.assertEqual(69, ropo_field.getData()[252][430], "Wrong quality value at position in composite.")
    self.assertEqual(252, ropo_field.getData()[249][460], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[416][408], "Wrong quality value at position in composite.")
     
    beamb_field = dbzh_param.getQualityFieldByHowTask("se.smhi.detector.beamblockage")
    self.assertEqual(255, beamb_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(254, beamb_field.getData()[425][336], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[387][475], "Wrong quality value at position in composite.")
    self.assertEqual(252, beamb_field.getData()[390][621], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[446][531], "Wrong quality value at position in composite.")
    self.assertEqual(251, beamb_field.getData()[312][394], "Wrong quality value at position in composite.")
    self.assertEqual(252, beamb_field.getData()[194][490], "Wrong quality value at position in composite.")
     
    self.validate_sehem_distance_field(dbzh_param)

  def test_linear_range_interpolation_cappi(self):
    height = 2600
    range_gain = 1.0
     
    result = self.generate_sehem_test_comp(get_range_alternating_dummy_data, range_gain, height, _pycomposite.InterpolationMethod_LINEAR_RANGE)
     
    dbzh_param = result.getParameter("DBZH")
    data = dbzh_param.getData()
    value_sum = 0.0
    no_of_values = 0
    max_value = max(EVEN_RANGE_DUMMY_VALUE, UNEVEN_RANGE_DUMMY_VALUE)
    min_value = min(EVEN_RANGE_DUMMY_VALUE, UNEVEN_RANGE_DUMMY_VALUE)
    for x in range(260, 470):
      for y in range(340, 650):
        value = data[y, x]
        self.assertTrue(value <= max_value)
        self.assertTrue(value >= min_value)
        value_sum += value
        no_of_values += 1
         
    mean_value = value_sum / no_of_values
    expected_mean = (max_value + min_value) / 2
    self.assertAlmostEqual(expected_mean, mean_value, msg="Unexpected mean value of values in composite", delta=0.01) 
     
    # check some fixed known positions
    self.assertEqual(106, dbzh_param.getData()[398][490], "Wrong value at position in composite.")
    self.assertEqual(135, dbzh_param.getData()[423][549], "Wrong value at position in composite.")
    self.assertEqual(11, dbzh_param.getData()[486][373], "Wrong value at position in composite.")
     
    ropo_field = dbzh_param.getQualityFieldByHowTask("fi.fmi.ropo.detector.classification")
    self.assertEqual(243, ropo_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[423][549], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[486][373], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[487][461], "Wrong quality value at position in composite.")
    self.assertEqual(52, ropo_field.getData()[252][430], "Wrong quality value at position in composite.")
    self.assertEqual(159, ropo_field.getData()[249][460], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[416][408], "Wrong quality value at position in composite.")
     
    beamb_field = dbzh_param.getQualityFieldByHowTask("se.smhi.detector.beamblockage")
    self.assertEqual(255, beamb_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(253, beamb_field.getData()[425][336], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[387][475], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[390][621], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[446][531], "Wrong quality value at position in composite.")
    self.assertEqual(249, beamb_field.getData()[312][394], "Wrong quality value at position in composite.")
    self.assertEqual(252, beamb_field.getData()[194][490], "Wrong quality value at position in composite.")
     
    self.validate_sehem_distance_field(dbzh_param)

  def test_linear_azimuth_interpolation_cappi(self):
    height = 2600
    azimuth_gain = 1.0
    histogram_bins = 10
     
    result = self.generate_sehem_test_comp(get_azimuth_alternating_dummy_data, azimuth_gain, height, _pycomposite.InterpolationMethod_LINEAR_AZIMUTH)
     
    dbzh_param = result.getParameter("DBZH")
    data = dbzh_param.getData()
    value_sum = 0.0
    no_of_values = 0
    max_value = max(EVEN_AZIMUTH_DUMMY_VALUE, UNEVEN_AZIMUTH_DUMMY_VALUE)
    min_value = min(EVEN_AZIMUTH_DUMMY_VALUE, UNEVEN_AZIMUTH_DUMMY_VALUE)
    values = []
    for x in range(260, 710):
      for y in range(340, 460):
        value = data[y, x]
        self.assertTrue(value <= max_value)
        self.assertTrue(value >= min_value)
        value_sum += value
        no_of_values += 1
        values.append(value)
         
    hist = numpy.histogram(values, histogram_bins, (min_value, max_value))
    # check even distribution
    expected_count = no_of_values/histogram_bins
    allowed_diff = expected_count * 0.02 # allow 2% diff
    for bin_count in hist[0]:
      self.assertAlmostEqual(expected_count, bin_count, msg="Unexpected distribution of values in composite", delta=allowed_diff) 
          
    mean_value = value_sum / no_of_values
    expected_mean = (max_value + min_value) / 2
    self.assertAlmostEqual(expected_mean, mean_value, msg="Unexpected mean value of values in composite", delta=1)
 
    # check some fixed known positions   
    self.assertEqual(202, dbzh_param.getData()[398][490], "Wrong value at position in composite.")
    self.assertEqual(34, dbzh_param.getData()[423][549], "Wrong value at position in composite.")
    self.assertEqual(23, dbzh_param.getData()[486][373], "Wrong value at position in composite.")
    self.assertEqual(52, dbzh_param.getData()[205][467], "Wrong value at position in composite.")
    self.assertEqual(32, dbzh_param.getData()[205][468], "Wrong value at position in composite.")
     
    ropo_field = dbzh_param.getQualityFieldByHowTask("fi.fmi.ropo.detector.classification")
    self.assertEqual(243, ropo_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[423][549], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[486][373], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[487][461], "Wrong quality value at position in composite.")
    self.assertEqual(62, ropo_field.getData()[252][430], "Wrong quality value at position in composite.")
    self.assertEqual(252, ropo_field.getData()[249][460], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[416][408], "Wrong quality value at position in composite.")
     
    beamb_field = dbzh_param.getQualityFieldByHowTask("se.smhi.detector.beamblockage")
    self.assertEqual(255, beamb_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(253, beamb_field.getData()[425][336], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[387][475], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[390][621], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[446][531], "Wrong quality value at position in composite.")
    self.assertEqual(249, beamb_field.getData()[312][394], "Wrong quality value at position in composite.")
    self.assertEqual(252, beamb_field.getData()[194][490], "Wrong quality value at position in composite.")
     
    self.validate_sehem_distance_field(dbzh_param)

  def test_linear_rangeandazimuth_interpolation_cappi(self):
    height = 2600
    gain = 1.0
     
    result = self.generate_sehem_test_comp(get_range_and_azimuth_alternating_dummy_data, gain, height, _pycomposite.InterpolationMethod_LINEAR_RANGE_AND_AZIMUTH)
     
    dbzh_param = result.getParameter("DBZH")
    data = dbzh_param.getData()
    value_sum = 0.0
    no_of_values = 0
    max_value = max(EVEN_AZIMUTH_DUMMY_VALUE, UNEVEN_AZIMUTH_DUMMY_VALUE, EVEN_RANGE_DUMMY_VALUE, UNEVEN_RANGE_DUMMY_VALUE)
    min_value = min(EVEN_AZIMUTH_DUMMY_VALUE, UNEVEN_AZIMUTH_DUMMY_VALUE, EVEN_RANGE_DUMMY_VALUE, UNEVEN_RANGE_DUMMY_VALUE)
    for x in range(260, 710):
      for y in range(340, 460):
        value = data[y, x]
        self.assertTrue(value <= max_value)
        self.assertTrue(value >= min_value)
        value_sum += value
        no_of_values += 1
          
    mean_value = value_sum / no_of_values
    expected_mean = sum([EVEN_AZIMUTH_DUMMY_VALUE, UNEVEN_AZIMUTH_DUMMY_VALUE, EVEN_RANGE_DUMMY_VALUE, UNEVEN_RANGE_DUMMY_VALUE]) / 4
    self.assertAlmostEqual(expected_mean, mean_value, msg="Unexpected mean value of values in composite", delta=1)
     
    # check some fixed known positions
    self.assertEqual(154, dbzh_param.getData()[398][490], "Wrong value at position in composite.")
    self.assertEqual(84, dbzh_param.getData()[423][549], "Wrong value at position in composite.")
    self.assertEqual(17, dbzh_param.getData()[486][373], "Wrong value at position in composite.")
    self.assertEqual(36, dbzh_param.getData()[205][467], "Wrong value at position in composite.")
    self.assertEqual(23, dbzh_param.getData()[205][468], "Wrong value at position in composite.")
     
    ropo_field = dbzh_param.getQualityFieldByHowTask("fi.fmi.ropo.detector.classification")
    self.assertEqual(243, ropo_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[423][549], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[486][373], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[487][461], "Wrong quality value at position in composite.")
    self.assertEqual(76, ropo_field.getData()[252][430], "Wrong quality value at position in composite.")
    self.assertEqual(183, ropo_field.getData()[249][460], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[416][408], "Wrong quality value at position in composite.")
     
    beamb_field = dbzh_param.getQualityFieldByHowTask("se.smhi.detector.beamblockage")
    self.assertEqual(255, beamb_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(253, beamb_field.getData()[425][336], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[387][475], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[390][621], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[446][531], "Wrong quality value at position in composite.")
    self.assertEqual(249, beamb_field.getData()[312][394], "Wrong quality value at position in composite.")
    self.assertEqual(252, beamb_field.getData()[194][490], "Wrong quality value at position in composite.")
     
    self.validate_sehem_distance_field(dbzh_param)

  def test_linear_3d_interpolation_cappi(self):
    gain = 20.0
    height = 2000.0
    histogram_bins = 20
     
    result = self.generate_sehem_test_comp(get_3d_alternating_dummy_data, gain, height, _pycomposite.InterpolationMethod_LINEAR_3D)
     
    dbzh_param = result.getParameter("DBZH")
    data = dbzh_param.getData()
    value_sum = 0.0
    no_of_values = 0
    values = []
    for x in range(310, 670):
      for y in range(340, 460):
        value = data[y, x]
        value_sum += value
        no_of_values += 1
        values.append(value)
          
    height_val = height / gain
    mean_value = value_sum / no_of_values
    expected_mean = height_val
    self.assertAlmostEqual(expected_mean, mean_value, msg="Unexpected mean value of values in composite", delta=4)
     
    # check distribution
    hist = numpy.histogram(values, histogram_bins, (height_val - (height_val / 2), height_val + (height_val / 2)))
    previous_bin_count = 0
    for bin_no in range(len(hist[0])):
      bin_count = hist[0][bin_no]     
      if bin_no < len(hist[0]) / 2:
        self.assertTrue(bin_count > previous_bin_count)
      else:
        self.assertTrue(bin_count < previous_bin_count)
      if previous_bin_count > 0:
        self.assertTrue(abs(bin_count - previous_bin_count) < 400)
      previous_bin_count = bin_count
     
    # check some fixed known positions
    self.assertEqual(75, dbzh_param.getData()[398][490], "Wrong value at position in composite.")
    self.assertEqual(117, dbzh_param.getData()[423][549], "Wrong value at position in composite.")
    self.assertEqual(156, dbzh_param.getData()[486][373], "Wrong value at position in composite.")
    self.assertEqual(67, dbzh_param.getData()[487][461], "Wrong value at position in composite.")
     
    ropo_field = dbzh_param.getQualityFieldByHowTask("fi.fmi.ropo.detector.classification")
    self.assertEqual(241, ropo_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(241, ropo_field.getData()[423][549], "Wrong quality value at position in composite.")
    self.assertEqual(239, ropo_field.getData()[486][373], "Wrong quality value at position in composite.")
    self.assertEqual(245, ropo_field.getData()[487][461], "Wrong quality value at position in composite.")
    self.assertEqual(72, ropo_field.getData()[252][430], "Wrong quality value at position in composite.")
    self.assertEqual(189, ropo_field.getData()[249][460], "Wrong quality value at position in composite.")
    self.assertEqual(216, ropo_field.getData()[416][408], "Wrong quality value at position in composite.")
     
    beamb_field = dbzh_param.getQualityFieldByHowTask("se.smhi.detector.beamblockage")
    self.assertEqual(255, beamb_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(254, beamb_field.getData()[425][336], "Wrong quality value at position in composite.")
    self.assertEqual(253, beamb_field.getData()[387][475], "Wrong quality value at position in composite.")
    self.assertEqual(251, beamb_field.getData()[390][621], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[446][531], "Wrong quality value at position in composite.")
    self.assertEqual(0, beamb_field.getData()[194][490], "Wrong quality value at position in composite.")
     
    self.validate_sehem_distance_field(dbzh_param)

  def test_quad_height_interpolation_cappi(self):
    height = 2600
    height_gain = 100.0
    allowed_height_diff = 15
    allowed_margin_at_radius = 1
     
    result = self.generate_sehem_test_comp(get_height_dummy_data, height_gain, height, _pycomposite.InterpolationMethod_QUADRATIC_HEIGHT)
     
    # check known positions, to ensure that correct values are set
    dbzh_param = result.getParameter("DBZH")
    data = dbzh_param.getData()
    for x in range(260, 470):
      for y in range(340, 650):
        height_diff = abs(data[y, x] - (height/height_gain))
        self.assertTrue(height_diff <= allowed_height_diff)
         
    centre_x = 489
    centre_y = 486
    radii = [20, 50, 100, 200, 280]
    for radius in radii:
      first_value = None
      for angle_deg in range(0, 360):
        angle_rad = math.radians(angle_deg)
        x = int(round(centre_x + (radius * cos(angle_rad))))
        y = int(round(centre_y + (radius * sin(angle_rad))))
        value = data[y, x]
        if not first_value:
          first_value = value
        value_diff = abs(float(first_value) - float(value))
        self.assertTrue(value_diff <= allowed_margin_at_radius, "Values circling the centre at the same radius should be the same")
         
    ropo_field = dbzh_param.getQualityFieldByHowTask("fi.fmi.ropo.detector.classification")
    self.assertEqual(243, ropo_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[423][549], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[486][373], "Wrong quality value at position in composite.")
    self.assertEqual(244, ropo_field.getData()[487][461], "Wrong quality value at position in composite.")
    self.assertEqual(43, ropo_field.getData()[252][430], "Wrong quality value at position in composite.")
    self.assertEqual(252, ropo_field.getData()[249][460], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[416][408], "Wrong quality value at position in composite.")
     
    beamb_field = dbzh_param.getQualityFieldByHowTask("se.smhi.detector.beamblockage")
    self.assertEqual(255, beamb_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(254, beamb_field.getData()[425][336], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[387][475], "Wrong quality value at position in composite.")
    self.assertEqual(252, beamb_field.getData()[390][621], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[446][531], "Wrong quality value at position in composite.")
     
    self.validate_sehem_distance_field(dbzh_param)
     
  def test_quad_3d_interpolation_cappi(self):
    gain = 20.0
    height = 2000.0
    expected_bin_counts = [1483, 1566, 1719, 2065, 2014, 1944, 2106, 2351, 2337, 2162, 2123, 1911, 1799, 1760, 1716, 1446, 1492, 1251,  858,  863]
    histogram_bins = len(expected_bin_counts)
     
    result = self.generate_sehem_test_comp(get_3d_alternating_dummy_data, gain, height, _pycomposite.InterpolationMethod_QUADRATIC_3D)
     
    dbzh_param = result.getParameter("DBZH")
    data = dbzh_param.getData()
    values = []
    for x in range(310, 670):
      for y in range(340, 460):
        value = data[y, x]
        values.append(value)
          
    height_val = height / gain
    # check distribution
    hist = numpy.histogram(values, histogram_bins, (height_val - (height_val / 2), height_val + (height_val / 2)))
     
    for bin_no in range(len(hist[0])):
      bin_count = hist[0][bin_no]
      self.assertEqual(expected_bin_counts[bin_no], bin_count, "Unexpected distribution of values in composite.")
      
    # check some fixed known positions
    self.assertEqual(73, dbzh_param.getData()[398][490], "Wrong value at position in composite.")
    self.assertEqual(116, dbzh_param.getData()[423][549], "Wrong value at position in composite.")
    self.assertEqual(169, dbzh_param.getData()[486][373], "Wrong value at position in composite.")
    self.assertEqual(38, dbzh_param.getData()[487][461], "Wrong value at position in composite.")
     
    ropo_field = dbzh_param.getQualityFieldByHowTask("fi.fmi.ropo.detector.classification")    
    self.assertEqual(243, ropo_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[423][549], "Wrong quality value at position in composite.")
    self.assertEqual(240, ropo_field.getData()[486][373], "Wrong quality value at position in composite.")
    self.assertEqual(243, ropo_field.getData()[487][461], "Wrong quality value at position in composite.")
    self.assertEqual(40, ropo_field.getData()[252][430], "Wrong quality value at position in composite.")
    self.assertEqual(193, ropo_field.getData()[249][460], "Wrong quality value at position in composite.")
    self.assertEqual(229, ropo_field.getData()[416][408], "Wrong quality value at position in composite.")
     
    beamb_field = dbzh_param.getQualityFieldByHowTask("se.smhi.detector.beamblockage")
    self.assertEqual(255, beamb_field.getData()[398][490], "Wrong quality value at position in composite.")
    self.assertEqual(253, beamb_field.getData()[425][336], "Wrong quality value at position in composite.")
    self.assertEqual(254, beamb_field.getData()[387][475], "Wrong quality value at position in composite.")
    self.assertEqual(250, beamb_field.getData()[390][621], "Wrong quality value at position in composite.")
    self.assertEqual(255, beamb_field.getData()[446][531], "Wrong quality value at position in composite.")
     
    self.validate_sehem_distance_field(dbzh_param)
 
  def test_nearest_with_radarindex(self):
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
     
    ids = "" 
    radarIndex = 1
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      wmoIndex = rio.object.source.find("WMO:")
      wmostr = rio.object.source[wmoIndex+4:wmoIndex+9]
      generator.add(rio.object)
      ids = "%s,%s:%d"%(ids,wmostr, radarIndex)
      radarIndex = radarIndex + 1
    ids = ids[1:]
 
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PCAPPI
    generator.height = 1000.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.generate(a, ["se.smhi.composite.index.radar"])
      
    self.assertEqual("DBZH", result.getParameter("DBZH").quantity)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEqual(1000.0, prodpar, 4)
    self.assertEqual(_rave.Rave_ProductType_PCAPPI, result.product)
    self.assertEqual(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEqual("nrd2km", result.source);
     
    nodes = result.getParameter("DBZH").getQualityFieldByHowTask("se.smhi.composite.index.radar").getAttribute("how/task_args")
    self.assertEqual(ids, nodes)
     
    import _transform
    t=_transform.new()
     
    result = t.combine_tiles(a, [result])
     
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_with_index.h5"
    ios.save()
  
  def test_nearest_with_reversed_radarindex(self):
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
 
    ids = "" 
    radarIndex = len(self.SWEDISH_VOLUMES)
    radarIndexMapping={}
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      wmoIndex = rio.object.source.find("WMO:")
      wmostr = rio.object.source[wmoIndex+4:wmoIndex+9]
      generator.add(rio.object)
      ids = "%s,%s:%d"%(ids,wmostr, radarIndex)
      radarIndexMapping["WMO:%s"%wmostr] = radarIndex
      radarIndex = radarIndex - 1
    ids = ids[1:]
 
    generator.applyRadarIndexMapping(radarIndexMapping)
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PCAPPI
    generator.height = 1000.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.generate(a, ["se.smhi.composite.index.radar"])
      
    self.assertEqual("DBZH", result.getParameter("DBZH").quantity)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEqual(1000.0, prodpar, 4)
    self.assertEqual(_rave.Rave_ProductType_PCAPPI, result.product)
    self.assertEqual(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEqual("nrd2km", result.source);
     
    nodes = result.getParameter("DBZH").getQualityFieldByHowTask("se.smhi.composite.index.radar").getAttribute("how/task_args")
    self.assertEqual(ids, nodes)
     
    import _transform
    t=_transform.new()
     
    result = t.combine_tiles(a, [result])
     
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_with_reversedindex.h5"
    ios.save()
  
  def test_nearest_multicomposite(self):
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
      
    for fname in self.QC_VOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
      
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.addParameter("TH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "000000"
    generator.date = "20120131"
    result = generator.generate(a, ["fi.fmi.ropo.detector.classification", "se.smhi.detector.poo", "se.smhi.composite.distance.radar"])
      
    self.assertEqual("DBZH", result.getParameter("DBZH").quantity)
    self.assertEqual("000000", result.time)
    self.assertEqual("20120131", result.date)
      
    self.assertEqual("TH", result.getParameter("TH").quantity)
    self.assertEqual("000000", result.time)
    self.assertEqual("20120131", result.date)
      
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEqual(0.0, prodpar, 4)
    self.assertEqual(_rave.Rave_ProductType_PPI, result.product)
    self.assertEqual(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEqual("nrd2km", result.source);
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swemulticomposite.h5"
    ios.save()
  
  def test_nearest_max(self):
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
      
    for fname in self.MAX_VOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
      
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_MAX
    generator.height = 0.0
    generator.range = 0.0
    generator.time = "120000"
    generator.date = "20090501"    
    result = generator.generate(a, ["se.smhi.composite.distance.radar"])
      
    self.assertEqual("DBZH", result.getParameter("DBZH").quantity)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
      
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEqual(0.0, prodpar, 4)
    self.assertEqual(_rave.Rave_ProductType_MAX, result.product)
    self.assertEqual(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEqual("nrd2km", result.source);
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swemaxcomposite.h5"
    ios.save()
  
  def create_simple_scan(self, sizes, quantity, v, qis, elangle, lon, lat, height, src):
    s = _polarscan.new()
    s.longitude = lon
    s.latitude = lat
    s.height = height
    s.elangle = elangle
    s.rstart = 0.0
    s.rscale = 10000.0
    s.source = src
    p = _polarscanparam.new()
    p.quantity = "DBZH"
    p.nodata = 255.0
    p.undetect = 0.0
    p.offset = 0.0
    p.gain = 1.0
    data = numpy.zeros(sizes, numpy.uint8)
    data = data + v
    p.setData(data)
      
    for k in qis.keys():
      data = numpy.zeros(sizes, numpy.float64)
      data = data + qis[k]
      qf = _ravefield.new()
      qf.addAttribute("how/task", k)
      qf.setData(data)
      p.addQualityField(qf)
      
    s.addParameter(p)
    return s
  
  def test_nearest_max_slim(self):
    a = _area.new()
    a.id = "test10km"
    a.xsize = 23
    a.ysize = 19
    a.xscale = 10000.0
    a.yscale = 10000.0
    a.extent = (1229430.993379, 8300379.564361, 1459430.993379, 8490379.564361)
    a.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +R=6378137.0 +nadgrids=@null +no_defs")
  
    s1 = self.create_simple_scan((4,4), "DBZH", 5, {"se.smhi.detector.poo": 0.1, "qf":0.6}, 0.1 * math.pi / 180.0, 12.0*math.pi/180.0, 60.0*math.pi/180.0, 0.0, "NOD:se1")
    s2 = self.create_simple_scan((4,4), "DBZH", 10, {"qf":0.5}, 0.2 * math.pi / 180.0, 12.0*math.pi/180.0, 60.0*math.pi/180.0, 0.0, "NOD:se1")
    v1 = _polarvolume.new()
    v1.longitude = 12.0*math.pi/180.0
    v1.latitude = 60.0*math.pi/180.0
    v1.height = 0.0
    v1.source = "NOD:se1"
    v1.addScan(s1)
    v1.addScan(s2)
  
    s1 = self.create_simple_scan((4,4), "DBZH", 5, {"se.smhi.detector.poo": 0.2, "qf":0.4}, 0.1 * math.pi / 180.0, 12.1*math.pi/180.0, 60.0*math.pi/180.0, 0.0, "NOD:sek")
    s2 = self.create_simple_scan((4,4), "DBZH", 10, {"qf":0.3}, 0.2 * math.pi / 180.0, 12.1*math.pi/180.0, 60.0*math.pi/180.0, 0.0, "NOD:sek")
    v2 = _polarvolume.new()
    v2.longitude = 12.1*math.pi/180.0
    v2.latitude = 60.0*math.pi/180.0
    v2.height = 0.0
    v2.source = "NOD:sek"
    v2.addScan(s1)
    v2.addScan(s2)
  
    generator = _pycomposite.new()
    generator.add(v1)
    generator.add(v2)
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_MAX
    generator.height = 0.0
    generator.range = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    generator.quality_indicator_field_name="qf"
    generator.algorithm = _poocompositealgorithm.new()
    result = generator.generate(a, ["se.smhi.detector.poo", "qf"])
  
  def test_nearest_max_polgmaps(self):
    a = _area.new()
    a.id = "polgmaps_2000"
    a.xsize = 754
    a.ysize = 810
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (1335807.096179,6065547.928434,2843807.096179,7685547.928434)
    a.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +R=6378137.0 +nadgrids=@null +no_defs")
  
    PLVOLUMES=["fixtures/plbrz_pvol_20120205T0430Z.h5",
               "fixtures/plgda_pvol_20120205T0430Z.h5",
               "fixtures/plleg_pvol_20120205T0430Z.h5",
               "fixtures/plpas_pvol_20120205T0430Z.h5",
               "fixtures/plpoz_pvol_20120205T0430Z.h5",
               "fixtures/plram_pvol_20120205T0430Z.h5",
               "fixtures/plrze_pvol_20120205T0430Z.h5"]
  
    generator = _pycomposite.new()
    #_rave.setDebugLevel(_rave.Debug_RAVE_DEBUG)
    for fname in PLVOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
    #GAIN = 0.4
    #OFFSET = -30.0
  
    generator.addParameter("DBZH", 0.4, -30.0, -30.0)
    generator.product = _rave.Rave_ProductType_MAX
    generator.height = 0.0
    generator.range = 0.0
    generator.time = "043000"
    generator.date = "20120205"
    generator.quality_indicator_field_name="pl.imgw.quality.qi_total"
    generator.algorithm = _poocompositealgorithm.new()
    result = generator.generate(a, ["se.smhi.detector.poo", "pl.imgw.quality.qi_total"])
  
    ios = _raveio.new()
    ios.object = result
    ios.filename = "polgmaps_max_qitotal.h5"
    ios.save()
  
  def test_nearest_pseudomax(self):
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
      
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
      
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PMAX
    generator.height = 1000.0
    generator.range = 70000.0
    generator.time = "120000"
    generator.date = "20090501"    
    result = generator.generate(a, ["se.smhi.composite.distance.radar"])
      
    self.assertEqual("DBZH", result.getParameter("DBZH").quantity)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
      
    prodpar = result.getAttribute("what/prodpar")
    v = prodpar.split(",")
    vh = float(v[0])
    vr = float(v[1])
    self.assertAlmostEqual(1000.0, vh, 4)
    self.assertAlmostEqual(70000.0, vr, 4)
    #self.assertAlmostEqual(70000.0, prodpar, 4)
    self.assertEqual(_rave.Rave_ProductType_PMAX, result.product)
    self.assertEqual(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEqual("nrd2km", result.source);
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swepseudomaxcomposite.h5"
    ios.save()
  
  # To verify ticket 96
  def test_nearest_gmapproj(self):
    generator = _pycomposite.new()
  
    a = _area.new()
    a.id = "eua_gmaps"
    a.xsize = 800
    a.ysize = 1090
    a.xscale = 6223.0
    a.yscale = 6223.0
    #               llX           llY            urX        urY
    a.extent = (-3117.83526,-6780019.83039,4975312.43200,3215.41216)
    # You can also add  +nadgrids=@null, for usage see PROJ.4 documentation
    a.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +x_0=1335833 +y_0=-11000715 +a=6378137.0 +b=6378137.0 +no_defs +datum=WGS84")
  
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
      
    generator.addParameter("DBZH", 0.4, -30.0, -30.0)
    generator.product = _rave.Rave_ProductType_PCAPPI
    generator.height = 1000.0
    result = generator.generate(a)
      
    result.time = "120000"
    result.date = "20090501"
    result.source = "eua_gmaps"
      
    ios = _raveio.new()
    ios.object = result
    ios.save("swecomposite_gmap.h5")

  def test_nearest_ppi(self):
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
      
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
      
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.generate(a)
      
    self.assertEqual("DBZH", result.getParameter("DBZH").quantity)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEqual(0.0, prodpar, 4)
    self.assertEqual(_rave.Rave_ProductType_PPI, result.product)
    self.assertEqual(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEqual("nrd2km", result.source);
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_ppi.h5"
    ios.save()
      
  def test_nearest_ppi_fromscans(self):
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
      
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      scan = rio.object.getScanClosestToElevation(0.0, 0)
      generator.add(scan)
      
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.generate(a)
      
    self.assertEqual("DBZH", result.getParameter("DBZH").quantity)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEqual(0.0, prodpar, 4)
    self.assertEqual(_rave.Rave_ProductType_PPI, result.product)
    self.assertEqual(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEqual("nrd2km", result.source);
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_ppi_fromscan.h5"
    ios.save()    

  def test_nearest_ppi_fromscans_with_startazA(self):
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
    
    startazA = numpy.arange(-0.5,359.5,360/420.0)
    startazA[0] = 359.5
    stopazA = numpy.arange(0.5,360.5,360/420.0)

    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      scan = rio.object.getScanClosestToElevation(0.0, 0)
      scan.addAttribute("how/startazA", startazA)
      scan.addAttribute("how/stopazA", stopazA)
      generator.add(scan)
    
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.generate(a)
      
    self.assertEqual("DBZH", result.getParameter("DBZH").quantity)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEqual(0.0, prodpar, 4)
    self.assertEqual(_rave.Rave_ProductType_PPI, result.product)
    self.assertEqual(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEqual("nrd2km", result.source);
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_ppi_fromscan_with_startaza.h5"
    ios.save()  

  def test_nearest_ppi_fromscans_byHeight(self):
    generator = _pycomposite.new()
    generator.selection_method = _pycomposite.SelectionMethod_HEIGHT
      
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
      
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      scan = rio.object.getScanClosestToElevation(0.0, 0)
      generator.add(scan)
      
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.generate(a)
      
    self.assertEqual("DBZH", result.getParameter("DBZH").quantity)
    self.assertEqual("120000", result.time)
    self.assertEqual("20090501", result.date)
    prodpar = result.getAttribute("what/prodpar")
    self.assertAlmostEqual(0.0, prodpar, 4)
    self.assertEqual(_rave.Rave_ProductType_PPI, result.product)
    self.assertEqual(_rave.Rave_ObjectType_COMP, result.objectType)
    self.assertEqual("nrd2km", result.source);
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_ppi_fromscan_byheight.h5"
    ios.save()    
  
  def test_overlapping_objects(self):
    # tests ticket 355, composite selection criterion: nearest
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "eua_gmaps"
    a.xsize = 800
    a.ysize = 1090
    a.xscale = 6223.0
    a.yscale = 6223.0
    #               llX           llY            urX        urY
    a.extent = (-3117.83526,-6780019.83039,4975312.43200,3215.41216)
    a.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +x_0=1335833 +y_0=-11000715 +a=6378137.0 +b=6378137.0 +no_defs +datum=WGS84")
      
    rio = _raveio.open("fixtures/pvol_seang_20090501T120000Z.h5")
    scan = rio.object.getScanClosestToElevation(40.0, 0) # Take highest elevation from angelholm
    generator.add(scan)
      
    rio = _raveio.open("fixtures/pvol_sekkr_20090501T120000Z.h5")
    scan = rio.object.getScanClosestToElevation(0.0, 0) # Take lowest elevation from karlskrona
    generator.add(scan)
  
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.generate(a)
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_gmap_overlapping.h5"
    ios.save()
  
  def test_nearest_by_quality_indicator(self):
    # tests ticket 355, composite selection criterion: nearest
    generator = _pycomposite.new()
      
    a = _area.new()
    a.id = "eua_gmaps"
    a.xsize = 800
    a.ysize = 1090
    a.xscale = 6223.0
    a.yscale = 6223.0
    #               llX           llY            urX        urY
    a.extent = (-3117.83526,-6780019.83039,4975312.43200,3215.41216)
    a.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +x_0=1335833 +y_0=-11000715 +a=6378137.0 +b=6378137.0 +no_defs +datum=WGS84")
      
    rio = _raveio.open("fixtures/pvol_seang_20090501T120000Z.h5")
    scan = rio.object.getScanClosestToElevation(0.0, 0)
    for x in range(scan.nbins):
       for y in range(scan.nrays):
         scan.setValue((x,y), 50)
      
    f1 = _ravefield.new()
    d = numpy.zeros((scan.nrays, scan.nbins), numpy.float64)
    for x in range(scan.nbins):
      for y in range(scan.nrays):
        d[y][x] = 0.99
    f1.setData(d)
    f1.addAttribute("how/task", "a.test.field")
    scan.addQualityField(f1)
    generator.add(scan)
      
    rio = _raveio.open("fixtures/pvol_sekkr_20090501T120000Z.h5")
    scan = rio.object.getScanClosestToElevation(0.0, 0)
    for x in range(scan.nbins):
       for y in range(scan.nrays):
         scan.setValue((x,y), 200)
    f2 = _ravefield.new()
    d = numpy.zeros((scan.nrays, scan.nbins), numpy.float64)
    for x in range(scan.nbins):
      for y in range(scan.nrays):
        d[y][x] = 0.10
    f2.addAttribute("how/task", "a.test.field")
    f2.setData(d)
    scan.addQualityField(f2)
           
    generator.add(scan)
  
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    generator.quality_indicator_field_name = "a.test.field"
      
    result = generator.generate(a)
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_quality_field.h5"
    ios.save()
 
  def test_nearest_ppi_fromscans_adddistancequality(self):
    generator = _pycomposite.new()
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
      
    for fname in self.SWEDISH_VOLUMES:
      rio = _raveio.open(fname)
      scan = rio.object.getScanClosestToElevation(0.0, 0)
      generator.add(scan)
      
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.generate(a, ["se.smhi.composite.distance.radar"])
      
    field = result.getParameter("DBZH").getQualityField(0)
    self.assertEqual("se.smhi.composite.distance.radar", field.getAttribute("how/task"))
    self.assertEqual(2000, field.getAttribute("what/gain"), "Wrong setting of gain in distance quality field")
    self.assertEqual(0, field.getAttribute("what/offset"), "Wrong setting of offset in distance quality field")
      
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_ppi_distancequality.h5"
    ios.save()
 
  def verify_qc_volumes_2016(self, result):
    dbzh_param = result.getParameter("DBZH")
    self.assertEqual(dbzh_param.getNumberOfQualityFields(), 5, "Wrong number of quality fields")
     
    qfield_expected_gain = 1.0/255.0
     
    field = dbzh_param.getQualityField(0)
    self.assertEqual("fi.fmi.ropo.detector.classification", field.getAttribute("how/task"))
    self.assertAlmostEqual(qfield_expected_gain, field.getAttribute("what/gain"), 6, "Wrong setting of gain in ropo quality field")
    self.assertEqual(0, field.getAttribute("what/offset"), "Wrong setting of offset in ropo quality field")
    data = field.getData()
    # check one known point in quality field where the algorithm has detected an anomaly
    self.assertEqual(data[911][541], 63, "Invalid quality value for ropo quality field")
    # check one point in quality field located outside radar range. here quality should be 0
    self.assertEqual(data[709][359], 0, "Invalid quality value for ropo quality field")
    # check one point within radar range where no anomaly is detected. here quality should be the maximum of 255
    self.assertEqual(data[780][487], 255, "Invalid quality value for ropo quality field")
  
    field = dbzh_param.getQualityField(1)
    self.assertEqual("se.smhi.detector.poo", field.getAttribute("how/task"))
    self.assertAlmostEqual(qfield_expected_gain, field.getAttribute("what/gain"), 6, "Wrong setting of gain in poo quality field")
    self.assertEqual(0, field.getAttribute("what/offset"), "Wrong setting of offset in poo quality field")
    data = field.getData()
    # check one known point in quality field where the algorithm has detected an anomaly
    self.assertEqual(data[633][382], 191, "Invalid quality value for poo quality field")
    # check one point in quality field located outside radar range. here quality should be 0
    self.assertEqual(data[709][359], 0, "Invalid quality value for poo quality field")
    # check one point within radar range where no anomaly is detected. here quality should be the maximum of 255
    self.assertEqual(data[780][487], 255, "Invalid quality value for poo quality field")
     
    field = dbzh_param.getQualityField(2)
    self.assertEqual("se.smhi.detector.beamblockage", field.getAttribute("how/task"))
    self.assertAlmostEqual(qfield_expected_gain, field.getAttribute("what/gain"), 6, "Wrong setting of gain in beamb quality field")
    self.assertEqual(0, field.getAttribute("what/offset"), "Wrong setting of offset in beamb quality field")
    data = field.getData()
    # check one known point in quality field where the algorithm has detected an anomaly
    self.assertEqual(data[576][311], 122, "Invalid quality value for beamb quality field")
    # check one point in quality field located outside radar range. here quality should be 0
    self.assertEqual(data[709][359], 0, "Invalid quality value for beamb quality field")
    # check one point within radar range where no anomaly is detected. here quality should be the maximum of 255
    self.assertEqual(data[780][487], 255, "Invalid quality value for beamb quality field")
     
    field = dbzh_param.getQualityField(3)
    self.assertEqual("se.smhi.composite.distance.radar", field.getAttribute("how/task"))
    self.assertEqual(2000, field.getAttribute("what/gain"), "Wrong setting of gain in distance quality field")
    self.assertEqual(0, field.getAttribute("what/offset"), "Wrong setting of offset in distance quality field")
    data = field.getData()
    # check one known point in quality field where the algorithm has detected an anomaly
    self.assertEqual(data[425][366], 101, "Invalid quality value for distance quality field")
    # check one point in quality field located outside radar range. here quality should be 0
    self.assertEqual(data[709][359], 0, "Invalid quality value for distance quality field")
 
    field = dbzh_param.getQualityField(4)
    self.assertEqual("se.smhi.composite.height.radar", field.getAttribute("how/task"))
    self.assertEqual(100, field.getAttribute("what/gain"), "Wrong setting of gain in height quality field")
    self.assertEqual(0, field.getAttribute("what/offset"), "Wrong setting of offset in height quality field")
    data = field.getData()
    # check one known point in quality field where the algorithm has detected an anomaly
    self.assertEqual(data[425][366], 47, "Invalid quality value for height quality field")
    # check one point in quality field located outside radar range. here quality should be 0
    self.assertEqual(data[709][359], 0, "Invalid quality value for height quality field")
     
  def test_quality_fields_for_scans(self):
    generator = _pycomposite.new()
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
     
    for fname in self.QC_VOLUMES_2016:
      rio = _raveio.open(fname)
      scan = rio.object.getScanClosestToElevation(0.0, 0)
      generator.add(scan)
     
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.generate(a, ["fi.fmi.ropo.detector.classification", "se.smhi.detector.poo", "se.smhi.detector.beamblockage", "se.smhi.composite.distance.radar", "se.smhi.composite.height.radar"])
     
    self.verify_qc_volumes_2016(result)
     
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_scans_qfields.h5"
    ios.save()  
 
  def test_quality_fields_for_pvols(self):
    generator = _pycomposite.new()
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
     
    for fname in self.QC_VOLUMES_2016:
      rio = _raveio.open(fname)
      generator.add(rio.object)
     
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_PPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    result = generator.generate(a, ["fi.fmi.ropo.detector.classification", "se.smhi.detector.poo", "se.smhi.detector.beamblockage", "se.smhi.composite.distance.radar", "se.smhi.composite.height.radar"])
     
    self.verify_qc_volumes_2016(result)
     
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_pvols_qfields.h5"
    ios.save()
 
  def test_cappi_for_unsorted_volume(self):
    generator = _pycomposite.new()
    a = _area.new()
    a.id = "nrd2km"
    a.xsize = 848
    a.ysize = 1104
    a.xscale = 2000.0
    a.yscale = 2000.0
    a.extent = (-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616)
    a.projection = _projection.new("x", "y", "+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
     
    for fname in self.DUMMY_DATA_FIXTURES:
      rio = _raveio.open(fname)
      generator.add(rio.object)
 
    generator.addParameter("DBZH", 1.0, 0.0, -30.0)
    generator.product = _rave.Rave_ProductType_CAPPI
    generator.elangle = 0.0
    generator.time = "120000"
    generator.date = "20090501"
    generator.height = 5000
    result = generator.generate(a, ["fi.fmi.ropo.detector.classification", "se.smhi.detector.beamblockage", "se.smhi.composite.distance.radar"])
 
    # check known positions, to ensure that correct values are set
    dbzh_param = result.getParameter("DBZH")
    data = dbzh_param.getData()
    self.assertEqual(data[855][507], 80, "Invalid data value in CAPPI.")
    self.assertEqual(data[869][468], 5, "Invalid data value in CAPPI.")
    self.assertEqual(data[931][474], 1, "Invalid data value in CAPPI.")
    self.assertEqual(data[849][505], 255, "Invalid data value in CAPPI.")
 
    ios = _raveio.new()
    ios.object = result
    ios.filename = "swecomposite_cappi_unsorted_pvols.h5"
    ios.save()
