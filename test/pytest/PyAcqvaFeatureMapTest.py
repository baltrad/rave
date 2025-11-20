'''
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the acqva feature map

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2025-11-12
'''
import unittest
import _rave
import _acqvafeaturemap
import _pyhl
import string
import os
import numpy as np
import math

class PyAcqvaFeatureMapTest(unittest.TestCase):
  TEMPORARY_FILE="acqvafeaturemap_iotest.h5"

  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def test_map(self):
    obj = _acqvafeaturemap.map()
    self.assertNotEqual(-1, str(type(obj)).find("AcqvaFeatureMapCore"))

  def test_elevation(self):
    obj = _acqvafeaturemap.elevation()
    self.assertNotEqual(-1, str(type(obj)).find("AcqvaFeatureMapElevationCore"))

  def test_field(self):
    obj = _acqvafeaturemap.field()
    self.assertNotEqual(-1, str(type(obj)).find("AcqvaFeatureMapFieldCore"))

  def test_field_elangle(self):
    obj = _acqvafeaturemap.field()
    self.assertEqual(obj.elangle, 0.0, 4)
    obj.elangle = 1.0
    self.assertEqual(obj.elangle, 1.0, 4)

  def test_field_rscale(self):
    obj = _acqvafeaturemap.field()
    self.assertEqual(obj.rscale, 0.0, 4)
    obj.rscale = 1.0
    self.assertEqual(obj.rscale, 1.0, 4)

  def test_field_rstart(self):
    obj = _acqvafeaturemap.field()
    self.assertEqual(obj.rstart, 0.0, 4)
    obj.rstart = 1.0
    self.assertEqual(obj.rstart, 1.0, 4)

  def test_field_beamwidth(self):
    obj = _acqvafeaturemap.field()
    self.assertEqual(obj.beamwidth, 0.0, 4)
    obj.beamwidth = 1.0
    self.assertEqual(obj.beamwidth, 1.0, 4)

  def test_field_addGetAttribute(self):
    obj = _acqvafeaturemap.field()
    obj.addAttribute("how/string", "some")
    obj.addAttribute("how/double", 1.1)
    obj.addAttribute("how/long", 1)

    self.assertEqual("some", obj.getAttribute("how/string"))
    self.assertAlmostEqual(1.1, obj.getAttribute("how/double"))
    self.assertEqual(1, obj.getAttribute("how/long"))

  def test_field_addGetSubAttributes(self):
    obj = _acqvafeaturemap.field()
    obj.addAttribute("how/acqva/string", "some")
    obj.addAttribute("how/acqva/double", 1.1)
    obj.addAttribute("how/acqva/long", 1)

    self.assertEqual("some", obj.getAttribute("how/acqva/string"))
    self.assertAlmostEqual(1.1, obj.getAttribute("how/acqva/double"))
    self.assertEqual(1, obj.getAttribute("how/acqva/long"))

  def test_field_hasAttribute(self):
    obj = _acqvafeaturemap.field()
    obj.addAttribute("how/string", "some")
    obj.addAttribute("how/double", 1.1)
    obj.addAttribute("how/long", 1)

    self.assertTrue(obj.hasAttribute("how/string"))
    self.assertTrue(obj.hasAttribute("how/double"))
    self.assertTrue(obj.hasAttribute("how/long"))
    self.assertFalse(obj.hasAttribute("how/not"))

  def test_fieldremoveAttribute(self):
    obj = _acqvafeaturemap.field()
    obj.addAttribute("how/string", "some")
    obj.addAttribute("how/double", 1.1)
    obj.addAttribute("how/long", 1)

    obj.removeAttribute("how/double")
    self.assertFalse(obj.hasAttribute("how/double"))


  def test_field_createData(self):
    obj = _acqvafeaturemap.field()
    obj.createData((1, 360), _rave.RaveDataType_UCHAR)
    self.assertEqual(1, obj.nbins)
    self.assertEqual(360, obj.nrays)
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)

    data = obj.getData()
    self.assertEqual(360, data.shape[0])  # Note that numpyarray and differs in axis
    self.assertEqual(1, data.shape[1])  # Note that numpyarray and differs in axis
    self.assertEqual(np.uint8, data.dtype)
    self.assertTrue(np.all(data == 0))

  def test_field_createData_UINT(self):
    obj = _acqvafeaturemap.field()
    obj.createData((1, 360), _rave.RaveDataType_UINT)
    self.assertEqual(1, obj.nbins)
    self.assertEqual(360, obj.nrays)
    self.assertEqual(_rave.RaveDataType_UINT, obj.datatype)

    data = obj.getData()
    self.assertEqual(360, data.shape[0])  # Note that numpyarray and differs in axis
    self.assertEqual(1, data.shape[1])  # Note that numpyarray and differs in axis
    self.assertEqual(np.uint32, data.dtype)
    self.assertTrue(np.all(data == 0))

  def test_field_setData_UCHAR(self):
    obj = _acqvafeaturemap.field()

    obj.setData(np.zeros((360,1), np.uint8))
    self.assertEqual(1, obj.nbins)
    self.assertEqual(360, obj.nrays)
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)

  def test_field_setData_UINT(self):
    obj = _acqvafeaturemap.field()

    obj.setData(np.zeros((360,1), np.uint32))
    self.assertEqual(1, obj.nbins)
    self.assertEqual(360, obj.nrays)
    self.assertEqual(_rave.RaveDataType_UINT, obj.datatype)

  def test_field_fill(self):
    obj = _acqvafeaturemap.field()
    obj.createData((1, 360), _rave.RaveDataType_UCHAR).fill(1)
    self.assertEqual(1, obj.nbins)
    self.assertEqual(360, obj.nrays)
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)

    data = obj.getData()
    self.assertEqual(360, data.shape[0])  # Note that numpyarray and differs in axis
    self.assertEqual(1, data.shape[1])  # Note that numpyarray and differs in axis
    self.assertEqual(np.uint8, data.dtype)
    self.assertTrue(np.all(data == 1))

  def test_field_fill_2(self):
    obj = _acqvafeaturemap.field()
    obj.createData((1, 360), _rave.RaveDataType_UCHAR).fill(2)
    self.assertEqual(1, obj.nbins)
    self.assertEqual(360, obj.nrays)
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)

    data = obj.getData()
    self.assertEqual(360, data.shape[0])  # Note that numpyarray and differs in axis
    self.assertEqual(1, data.shape[1])  # Note that numpyarray and differs in axis
    self.assertEqual(np.uint8, data.dtype)
    self.assertTrue(np.all(data == 2))

  def test_field_setGetValue(self):
    obj = _acqvafeaturemap.field()

    obj.createData((1, 360), _rave.RaveDataType_UINT)
    obj.setValue((0,5), 1)
    obj.setValue((0,10), 2)

    self.assertEqual(0, obj.getValue((0,0)))
    self.assertEqual(1, obj.getValue((0,5)))
    self.assertEqual(2, obj.getValue((0,10)))

    data = obj.getData()
    self.assertEqual(0, data[0,0])
    self.assertEqual(1, data[5,0])
    self.assertEqual(2, data[10,0])

  def test_field_with_arguments(self):
    obj = _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 0.5)
    self.assertEqual(1, obj.nbins)
    self.assertEqual(360, obj.nrays)
    self.assertEqual(_rave.RaveDataType_UCHAR, obj.datatype)
    self.assertEqual(0.5, obj.elangle, 4)

  def test_field_toRaveField(self):
    obj = _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 0.5).fill(1)
    result = obj.toRaveField()
    self.assertNotEqual(-1, str(type(result)).find("RaveFieldCore"))
    self.assertEqual(1, result.xsize)
    self.assertEqual(360, result.ysize)
    self.assertEqual(_rave.RaveDataType_UCHAR, result.datatype)
    self.assertTrue(np.all(result.getData() == np.ones((360,1), np.uint8)))

  def test_field_toScan(self):
    obj = _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 0.5).fill(1)
    obj.elangle = 0.5
    obj.rscale = 1.0
    obj.rstart = 2.0
    obj.beamwidth = 3.0

    result = obj.toScan("DBZH", 2.1, 3.2, 55.1)
    self.assertAlmostEqual(0.5, result.elangle, 4)
    self.assertAlmostEqual(1.0, result.rscale, 4)
    self.assertAlmostEqual(2.0, result.rstart, 4)
    self.assertAlmostEqual(3.0, result.beamwidth, 4)
    self.assertAlmostEqual(2.1, result.longitude, 4)
    self.assertAlmostEqual(3.2, result.latitude, 4)
    self.assertAlmostEqual(55.1, result.height, 4)

    self.assertNotEqual(-1, str(type(result)).find("PolarScanCore"))
    self.assertEqual(1, result.nbins)
    self.assertEqual(360, result.nrays)

    param = result.getParameter("DBZH")
    self.assertEqual(_rave.RaveDataType_UCHAR, param.datatype)
    self.assertTrue(np.all(param.getData() == np.ones((360,1), np.uint8)))

  def test_elevation_elangle(self):
    obj = _acqvafeaturemap.elevation()
    self.assertEqual(obj.elangle, 0.0, 4)
    obj.elangle = 1.0
    self.assertEqual(obj.elangle, 1.0, 4)

  def test_elevation_add(self):
    obj = _acqvafeaturemap.elevation()
    obj.elangle = 1.0
    self.assertEqual(0, obj.size())
    obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0))
    self.assertEqual(1, obj.size())
    obj.add( _acqvafeaturemap.field((2,360), _rave.RaveDataType_UCHAR, 1.0))
    self.assertEqual(2, obj.size())

  def test_elevation_add_conflicting_dimensions(self):
    obj = _acqvafeaturemap.elevation()
    obj.elangle = 1.0
    self.assertEqual(0, obj.size())
    obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0))
    self.assertEqual(1, obj.size())
    try:
        obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0))
        self.fail("Expected AttributeError")
    except AttributeError:
        pass
            
    self.assertEqual(1, obj.size())

  def test_elevation_add_conflicting_elangles(self):
    obj = _acqvafeaturemap.elevation()
    obj.elangle = 1.0
    self.assertEqual(0, obj.size())
    obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0))
    self.assertEqual(1, obj.size())
    try:
        obj.add( _acqvafeaturemap.field((1,361), _rave.RaveDataType_UCHAR, 1.5))
        self.fail("Expected AttributeError")
    except AttributeError as e:
        pass
            
    self.assertEqual(1, obj.size())

  def test_elevation_get(self):
    obj = _acqvafeaturemap.elevation()
    obj.elangle = 1.0
    obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0))
    obj.add( _acqvafeaturemap.field((2,360), _rave.RaveDataType_UCHAR, 1.0))
    obj.add( _acqvafeaturemap.field((3,360), _rave.RaveDataType_UCHAR, 1.0))

    self.assertEqual(1, obj.get(0).nbins)
    self.assertEqual(360, obj.get(0).nrays)
    self.assertEqual(2, obj.get(1).nbins)
    self.assertEqual(360, obj.get(1).nrays)
    self.assertEqual(3, obj.get(2).nbins)
    self.assertEqual(360, obj.get(2).nrays) 

  def test_elevation_remove(self):
    obj = _acqvafeaturemap.elevation()
    obj.elangle = 1.0
    self.assertEqual(0, obj.size())
    obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0))
    obj.add( _acqvafeaturemap.field((2,360), _rave.RaveDataType_UCHAR, 1.0))
    obj.add( _acqvafeaturemap.field((3,360), _rave.RaveDataType_UCHAR, 1.0))

    obj.remove(1)
    self.assertEqual(2, obj.size())
    self.assertEqual(1, obj.get(0).nbins)
    self.assertEqual(360, obj.get(0).nrays)
    self.assertEqual(3, obj.get(1).nbins)
    self.assertEqual(360, obj.get(1).nrays) 

  def test_elevation_find(self):
    obj = _acqvafeaturemap.elevation()
    obj.elangle = 1.0
    obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0, 1.0, 2.0, 3.0))
    obj.add( _acqvafeaturemap.field((2,360), _rave.RaveDataType_UCHAR, 1.0, 1.0, 2.0, 3.0))
    obj.add( _acqvafeaturemap.field((3,360), _rave.RaveDataType_UCHAR, 1.0, 1.0))


    f1 = obj.find((2,360))
    self.assertEqual(2, f1.nbins)
    self.assertEqual(360, f1.nrays)
    self.assertAlmostEqual(1.0, f1.rscale, 4)
    self.assertAlmostEqual(2.0, f1.rstart, 4)
    self.assertAlmostEqual(3.0, f1.beamwidth, 4)

    f2 = obj.find((3,360))
    self.assertEqual(3, f2.nbins)
    self.assertEqual(360, f2.nrays)
    self.assertAlmostEqual(1.0, f2.rscale, 4)
    self.assertAlmostEqual(0.0, f2.rstart, 4)
    self.assertAlmostEqual(0.0, f2.beamwidth, 4)

    self.assertTrue(obj.find((4,360)) is None)
    self.assertTrue(obj.find((1,361)) is None)

  def test_elevation_find_with_optionals(self):
    obj = _acqvafeaturemap.elevation()
    obj.elangle = 1.0
    obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0, 1.0, 2.0, 3.0))
    obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0, 1.0, 3.0, 3.0))
    obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0, 1.0, 2.0, 4.0))
    obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0, 2.0, 3.0, 4.0))

    # RSCALE
    f1 = obj.find((1,360), 1.0)  # Will take first found with correct rscale
    self.assertEqual(1, f1.nbins)
    self.assertEqual(360, f1.nrays)
    self.assertAlmostEqual(1.0, f1.rscale, 4)
    self.assertAlmostEqual(2.0, f1.rstart, 4)
    self.assertAlmostEqual(3.0, f1.beamwidth, 4)

    # RSCALE & RSTART
    f1 = obj.find((1,360), 1.0, 3.0)  # Will take first found with correct rscale & rstart
    self.assertEqual(1, f1.nbins)
    self.assertEqual(360, f1.nrays)
    self.assertAlmostEqual(1.0, f1.rscale, 4)
    self.assertAlmostEqual(3.0, f1.rstart, 4)
    self.assertAlmostEqual(3.0, f1.beamwidth, 4)

    # RSCALE & RSTART
    f1 = obj.find((1,360), 2.0, 3.0)  # Will take first found with correct rscale & rstart
    self.assertEqual(1, f1.nbins)
    self.assertEqual(360, f1.nrays)
    self.assertAlmostEqual(2.0, f1.rscale, 4)
    self.assertAlmostEqual(3.0, f1.rstart, 4)
    self.assertAlmostEqual(4.0, f1.beamwidth, 4)

    # RSCALE & RSTART & BEAMWIDTH
    f1 = obj.find((1,360), 1.0, 2.0, 4)  # Will take first found with correct rscale & rstart
    self.assertEqual(1, f1.nbins)
    self.assertEqual(360, f1.nrays)
    self.assertAlmostEqual(1.0, f1.rscale, 4)
    self.assertAlmostEqual(2.0, f1.rstart, 4)
    self.assertAlmostEqual(4.0, f1.beamwidth, 4)

  def test_map_nod(self):
    obj = _acqvafeaturemap.map()
    self.assertEqual("", obj.nod)
    obj.nod="seang"
    self.assertEqual("seang", obj.nod)

  def test_map_longitude(self):
    obj = _acqvafeaturemap.map()
    self.assertAlmostEqual(0.0, obj.longitude, 4)
    obj.longitude=1
    self.assertAlmostEqual(1.0, obj.longitude, 4)

  def test_map_latitude(self):
    obj = _acqvafeaturemap.map()
    self.assertAlmostEqual(0.0, obj.latitude, 4)
    obj.latitude=1.0
    self.assertAlmostEqual(1.0, obj.latitude, 4)

  def test_map_height(self):
    obj = _acqvafeaturemap.map()
    self.assertAlmostEqual(0.0, obj.height)
    obj.height=1.0
    self.assertAlmostEqual(1.0, obj.height, 4)

  def test_map_startdate(self):
    obj = _acqvafeaturemap.map()
    self.assertEqual(None, obj.startdate)
    obj.startdate="20250101"
    self.assertEqual("20250101", obj.startdate)

  def test_map_enddate(self):
    obj = _acqvafeaturemap.map()
    self.assertEqual(None, obj.enddate)
    obj.enddate="20250101"
    self.assertEqual("20250101", obj.enddate)

  def test_map_getNumberOfElevations(self):
    obj = _acqvafeaturemap.map()
    self.assertEqual(0, obj.getNumberOfElevations())

  def test_map_createElevation(self):
    obj = _acqvafeaturemap.map()
    e1 = obj.createElevation(0.1)
    e2 = obj.createElevation(0.2)
    e3 = obj.createElevation(0.3)

    self.assertEqual(3, obj.getNumberOfElevations())
    self.assertAlmostEqual(0.1, obj.getElevation(0).elangle, 4)
    self.assertAlmostEqual(0.2, obj.getElevation(1).elangle, 4)
    self.assertAlmostEqual(0.3, obj.getElevation(2).elangle, 4)

  def test_map_createElevation_duplicate(self):
    obj = _acqvafeaturemap.map()
    e1 = obj.createElevation(0.1)
    e2 = obj.createElevation(0.1)
    self.assertEqual(1, obj.getNumberOfElevations())
    self.assertAlmostEqual(0.1, obj.getElevation(0).elangle, 4)

  def test_map_removeElevation(self):
    obj = _acqvafeaturemap.map()
    e1 = obj.createElevation(0.1)
    e2 = obj.createElevation(0.2)
    e3 = obj.createElevation(0.3)
    obj.removeElevation(1)
    self.assertEqual(2, obj.getNumberOfElevations())
    self.assertAlmostEqual(0.1, obj.getElevation(0).elangle, 4)
    self.assertAlmostEqual(0.3, obj.getElevation(1).elangle, 4)

  def test_map_findElevation(self):
    obj = _acqvafeaturemap.map()
    e1 = obj.createElevation(0.1)
    e2 = obj.createElevation(0.2)
    e3 = obj.createElevation(0.3)

    result = obj.findElevation(0.2)
    self.assertAlmostEqual(0.2, result.elangle, 4)

    result = obj.findElevation(0.4)
    self.assertEqual(None, result)

  def test_map_createField(self):
    obj = _acqvafeaturemap.map()
    f1 = obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0, 500.0, 0.0, 1.0)
    f2 = obj.createField((1,361), _rave.RaveDataType_UCHAR, 1.0, 500.0, 1.0, 2.0)
    self.assertEqual(1, obj.getNumberOfElevations())
    self.assertAlmostEqual(1.0, obj.getElevation(0).elangle, 4)
    self.assertAlmostEqual(1.0, obj.getElevation(0).get(0).elangle, 4)
    self.assertAlmostEqual(500.0, obj.getElevation(0).get(0).rscale, 4)
    self.assertAlmostEqual(0.0, obj.getElevation(0).get(0).rstart, 4)
    self.assertAlmostEqual(1.0, obj.getElevation(0).get(0).beamwidth, 4)
    self.assertAlmostEqual(500.0, obj.getElevation(0).get(1).rscale, 4)
    self.assertAlmostEqual(1.0, obj.getElevation(0).get(1).rstart, 4)
    self.assertAlmostEqual(2.0, obj.getElevation(0).get(1).beamwidth, 4)

  def test_map_createField_with_same_resolution_different_scale(self):
    obj = _acqvafeaturemap.map()
    f1 = obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0, 500.0, 0.0, 1.0)
    f2 = obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0, 400.0, 0.0, 1.0)
    self.assertEqual(1, obj.getNumberOfElevations())
    self.assertAlmostEqual(1.0, obj.getElevation(0).elangle, 4)
    self.assertAlmostEqual(1.0, obj.getElevation(0).get(0).elangle, 4)
    self.assertAlmostEqual(500.0, obj.getElevation(0).get(0).rscale, 4)
    self.assertAlmostEqual(1.0, obj.getElevation(0).get(1).elangle, 4)
    self.assertAlmostEqual(400.0, obj.getElevation(0).get(1).rscale, 4)

  def test_map_findField(self):
    obj = _acqvafeaturemap.map()
    f1 = obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0, 200.0, 0.0, 1.0)
    f2 = obj.createField((1,361), _rave.RaveDataType_UCHAR, 1.0, 300.0, 0.0, 1.0)
    f3 = obj.createField((1,360), _rave.RaveDataType_UCHAR, 2.0, 400.0, 0.0, 1.0)
    f4 = obj.createField((1,361), _rave.RaveDataType_UCHAR, 2.0, 500.0, 0.0, 1.0)

    result = obj.findField((1,360), 2.0)
    self.assertEqual(1, result.nbins)
    self.assertEqual(360, result.nrays)
    self.assertAlmostEqual(2.0, result.elangle, 4)

    result = obj.findField((5,360), 2.0)
    self.assertEqual(None, result)

    result = obj.findField((1,360), 3.0)
    self.assertEqual(None, result)

  def test_map_addGetAttribute(self):
    obj = _acqvafeaturemap.map()
    obj.addAttribute("how/string", "some")
    obj.addAttribute("how/double", 1.1)
    obj.addAttribute("how/long", 1)

    self.assertEqual("some", obj.getAttribute("how/string"))
    self.assertAlmostEqual(1.1, obj.getAttribute("how/double"))
    self.assertEqual(1, obj.getAttribute("how/long"))

  def test_map_addGetSubAttributes(self):
    obj = _acqvafeaturemap.map()
    obj.addAttribute("how/acqva/string", "some")
    obj.addAttribute("how/acqva/double", 1.1)
    obj.addAttribute("how/acqva/long", 1)

    self.assertEqual("some", obj.getAttribute("how/acqva/string"))
    self.assertAlmostEqual(1.1, obj.getAttribute("how/acqva/double"))
    self.assertEqual(1, obj.getAttribute("how/acqva/long"))

  def test_map_hasAttribute(self):
    obj = _acqvafeaturemap.map()
    obj.addAttribute("how/string", "some")
    obj.addAttribute("how/double", 1.1)
    obj.addAttribute("how/long", 1)

    self.assertTrue(obj.hasAttribute("how/string"))
    self.assertTrue(obj.hasAttribute("how/double"))
    self.assertTrue(obj.hasAttribute("how/long"))
    self.assertFalse(obj.hasAttribute("how/not"))

  def test_map_removeAttribute(self):
    obj = _acqvafeaturemap.map()
    obj.addAttribute("how/string", "some")
    obj.addAttribute("how/double", 1.1)
    obj.addAttribute("how/long", 1)

    obj.removeAttribute("how/double")
    self.assertFalse(obj.hasAttribute("how/double"))

  def test_map_save_1_group(self):
    obj = _acqvafeaturemap.map()
    obj.nod = "seang"
    obj.startdate="20220101"
    obj.enddate="20220201"
    obj.latitude = 60.0 * math.pi / 180.0
    obj.longitude = 14.0 * math.pi / 180.0
    obj.height = 300.0
    obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0, 200, 0.0, 1.0 * math.pi/180.0)
    obj.createField((1,362), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0, 300, 100.0, 2.0 * math.pi/180.0)

    obj.save(self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("ACQVA Feature Map 1.0", nodelist.getNode("/Conventions").data())

    # What
    self.assertEqual("seang", nodelist.getNode("/what/nod").data())
    self.assertEqual("20220101", nodelist.getNode("/what/startdate").data())
    self.assertEqual("20220201", nodelist.getNode("/what/enddate").data())

    #Where
    self.assertAlmostEqual(60.0, nodelist.getNode("/where/lat").data(), 4)
    self.assertAlmostEqual(14.0, nodelist.getNode("/where/lon").data(), 4)
    self.assertAlmostEqual(300, nodelist.getNode("/where/height").data(), 4)

    # Elevation group 1
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/where/elangle").data(), 4)

    # Field 1
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/where/elangle").data(), 4)
    f1data = nodelist.getNode("/dataset1/data1/data").data()
    self.assertEqual((360,1), f1data.shape)
    self.assertAlmostEqual(200.0, nodelist.getNode("/dataset1/data1/where/rscale").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/where/rstart").data(), 4)
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/how/beamwidth").data(), 4)

    # Field 2
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data2/where/elangle").data(), 4)
    f2data = nodelist.getNode("/dataset1/data2/data").data()
    self.assertEqual((362,1), f2data.shape)
    self.assertAlmostEqual(300.0, nodelist.getNode("/dataset1/data2/where/rscale").data(), 4)
    self.assertAlmostEqual(100.0, nodelist.getNode("/dataset1/data2/where/rstart").data(), 4)
    self.assertAlmostEqual(2.0, nodelist.getNode("/dataset1/data2/how/beamwidth").data(), 4)

  def test_map_save_2_groups(self):
    obj = _acqvafeaturemap.map()
    obj.nod = "seang"
    obj.startdate="20220101"
    obj.enddate="20220201"
    obj.latitude = 60.0 * math.pi / 180.0
    obj.longitude = 14.0 * math.pi / 180.0
    obj.height = 300.0
    obj.addAttribute("how/acqva/string", "some")
    obj.addAttribute("how/acqva/double", 1.1)
    obj.addAttribute("how/acqva/long", 1)    
    obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0, 200, 0.0, 1.0* math.pi / 180.0).addAttribute("how/acqvafield/string", "some")
    obj.createField((1,362), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0, 300, 0.0, 2.0* math.pi / 180.0).addAttribute("how/acqvafield/double", 1.1)
    obj.createField((1,361), _rave.RaveDataType_UCHAR, 2.0 * math.pi / 180.0, 400, 100.0, 3.0* math.pi / 180.0).addAttribute("how/acqvafield/long", 1)
    obj.createField((1,363), _rave.RaveDataType_UCHAR, 2.0 * math.pi / 180.0, 500, 100.0, 4.0* math.pi / 180.0)

    obj.save(self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("ACQVA Feature Map 1.0", nodelist.getNode("/Conventions").data())

    # What
    self.assertEqual("seang", nodelist.getNode("/what/nod").data())
    self.assertEqual("20220101", nodelist.getNode("/what/startdate").data())
    self.assertEqual("20220201", nodelist.getNode("/what/enddate").data())

    #Where
    self.assertAlmostEqual(60.0, nodelist.getNode("/where/lat").data(), 4)
    self.assertAlmostEqual(14.0, nodelist.getNode("/where/lon").data(), 4)
    self.assertAlmostEqual(300, nodelist.getNode("/where/height").data(), 4)

    # How
    self.assertEqual("some", nodelist.getNode("/how/acqva/string").data(), 4)
    self.assertAlmostEqual(1.1, nodelist.getNode("/how/acqva/double").data(), 4)
    self.assertAlmostEqual(1, nodelist.getNode("/how/acqva/long").data(), 4)

    # Elevation group 1
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/where/elangle").data(), 4)

    # Field 1
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/where/elangle").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/where/rstart").data(), 4)
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/how/beamwidth").data(), 4)
    f1data = nodelist.getNode("/dataset1/data1/data").data()
    self.assertEqual((360,1), f1data.shape)
    self.assertEqual("some", nodelist.getNode("/dataset1/data1/how/acqvafield/string").data())

    # Field 2
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data2/where/elangle").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data2/where/rstart").data(), 4)
    self.assertAlmostEqual(2.0, nodelist.getNode("/dataset1/data2/how/beamwidth").data(), 4)
    f2data = nodelist.getNode("/dataset1/data2/data").data()
    self.assertEqual((362,1), f2data.shape)
    self.assertAlmostEqual(1.1, nodelist.getNode("/dataset1/data2/how/acqvafield/double").data(), 4)

    # Elevation group 2
    self.assertAlmostEqual(2.0, nodelist.getNode("/dataset2/where/elangle").data(), 4)

    # Field 1
    self.assertAlmostEqual(2.0, nodelist.getNode("/dataset2/data1/where/elangle").data(), 4)
    self.assertAlmostEqual(100.0, nodelist.getNode("/dataset2/data1/where/rstart").data(), 4)
    self.assertAlmostEqual(100.0, nodelist.getNode("/dataset2/data1/where/rstart").data(), 4)
    f1data = nodelist.getNode("/dataset2/data1/data").data()
    self.assertEqual((361,1), f1data.shape)
    self.assertEqual(1, nodelist.getNode("/dataset2/data1/how/acqvafield/long").data())

    # Field 2
    self.assertAlmostEqual(2.0, nodelist.getNode("/dataset2/data2/where/elangle").data(), 4)
    f2data = nodelist.getNode("/dataset2/data2/data").data()
    self.assertEqual((363,1), f2data.shape)

  def test_map_save_bad_metadata(self):
    obj = _acqvafeaturemap.map()
    try:
        obj.save(self.TEMPORARY_FILE)
        self.fail("Expected IOError")
    except IOError:
        pass

    obj.nod = "seang"
    try:
        obj.save(self.TEMPORARY_FILE)
        self.fail("Expected IOError")
    except IOError:
        pass

    obj.startdate="20220101"
    try:
        obj.save(self.TEMPORARY_FILE)
        self.fail("Expected IOError")
    except IOError:
        pass

    obj.enddate="20220201"
    obj.save(self.TEMPORARY_FILE)

  def test_load(self):
    obj = _acqvafeaturemap.map()
    obj.nod = "seang"
    obj.startdate="20220101"
    obj.enddate="20220201"
    obj.latitude = 60.0 * math.pi / 180.0
    obj.longitude = 14.0 * math.pi / 180.0
    obj.height = 300.0
    obj.addAttribute("how/acqva/string", "some")
    obj.addAttribute("how/acqva/double", 1.1)
    obj.addAttribute("how/acqva/long", 1)    

    obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0, 200.0, 0.0, 1.0).addAttribute("how/acqvafield/string", "some")
    obj.createField((1,362), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0, 300.0, 1.0, 2.0).addAttribute("how/acqvafield/double", 1.1)
    obj.createField((1,361), _rave.RaveDataType_UCHAR, 2.0 * math.pi / 180.0, 400.0, 2.0, 3.0).addAttribute("how/acqvafield/long", 1)
    obj.createField((1,363), _rave.RaveDataType_UCHAR, 2.0 * math.pi / 180.0, 500.0, 3.0, 4.0)

    obj.save(self.TEMPORARY_FILE)

    result = _acqvafeaturemap.load(self.TEMPORARY_FILE)
    self.assertEqual("seang", result.nod)
    self.assertEqual("20220101", result.startdate)
    self.assertEqual("20220201", result.enddate)
    self.assertAlmostEqual(60.0, result.latitude*180.0/math.pi, 4)
    self.assertAlmostEqual(14.0, result.longitude*180.0/math.pi, 4)
    self.assertAlmostEqual(300, result.height, 4)
    self.assertEqual("some", result.getAttribute("how/acqva/string"))
    self.assertAlmostEqual(1.1, result.getAttribute("how/acqva/double"))
    self.assertEqual(1, result.getAttribute("how/acqva/long"))

    self.assertEqual(2, result.getNumberOfElevations())
    self.assertAlmostEqual(1.0 * math.pi / 180.0, result.getElevation(0).elangle, 4)
    self.assertAlmostEqual(2.0 * math.pi / 180.0, result.getElevation(1).elangle, 4)

    self.assertTrue(np.all(result.getElevation(0).get(0).getData() == np.zeros((360, 1), np.uint8)))
    self.assertAlmostEqual(1.0 * math.pi / 180.0, result.getElevation(0).get(0).elangle, 4)
    self.assertAlmostEqual(200.0, result.getElevation(0).get(0).rscale, 4)
    self.assertEqual("some", result.getElevation(0).get(0).getAttribute("how/acqvafield/string"))

    self.assertTrue(np.all(result.getElevation(0).get(1).getData() == np.zeros((362, 1), np.uint8)))
    self.assertAlmostEqual(1.0 * math.pi / 180.0, result.getElevation(0).get(1).elangle, 4)
    self.assertAlmostEqual(300.0, result.getElevation(0).get(1).rscale, 4)
    self.assertAlmostEqual(1.1, result.getElevation(0).get(1).getAttribute("how/acqvafield/double"))

    self.assertTrue(np.all(result.getElevation(1).get(0).getData() == np.zeros((361, 1), np.uint8)))    
    self.assertAlmostEqual(2.0 * math.pi / 180.0, result.getElevation(1).get(0).elangle, 4)
    self.assertAlmostEqual(400.0, result.getElevation(1).get(0).rscale, 4)
    self.assertEqual(1, result.getElevation(1).get(0).getAttribute("how/acqvafield/long"))

    self.assertTrue(np.all(result.getElevation(1).get(1).getData() == np.zeros((363, 1), np.uint8)))
    self.assertAlmostEqual(2.0 * math.pi / 180.0, result.getElevation(1).get(1).elangle, 4)
    self.assertAlmostEqual(500.0, result.getElevation(1).get(1).rscale, 4)
