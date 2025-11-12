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

class PyAreaRegistryTest(unittest.TestCase):
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
    obj.add( _acqvafeaturemap.field((1,360), _rave.RaveDataType_UCHAR, 1.0))
    obj.add( _acqvafeaturemap.field((2,360), _rave.RaveDataType_UCHAR, 1.0))
    obj.add( _acqvafeaturemap.field((3,360), _rave.RaveDataType_UCHAR, 1.0))


    f1 = obj.find((2,360))
    self.assertEqual(2, f1.nbins)
    self.assertEqual(360, f1.nrays)

    f2 = obj.find((3,360))
    self.assertEqual(3, f2.nbins)
    self.assertEqual(360, f2.nrays)

    self.assertTrue(obj.find((4,360)) is None)
    self.assertTrue(obj.find((1,361)) is None)

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
    f1 = obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0)
    f2 = obj.createField((1,361), _rave.RaveDataType_UCHAR, 1.0)
    self.assertEqual(1, obj.getNumberOfElevations())
    self.assertAlmostEqual(1.0, obj.getElevation(0).elangle, 4)

  def test_map_findField(self):
    obj = _acqvafeaturemap.map()
    f1 = obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0)
    f2 = obj.createField((1,361), _rave.RaveDataType_UCHAR, 1.0)
    f3 = obj.createField((1,360), _rave.RaveDataType_UCHAR, 2.0)
    f4 = obj.createField((1,361), _rave.RaveDataType_UCHAR, 2.0)

    result = obj.findField((1,360), 2.0)
    self.assertEqual(1, result.nbins)
    self.assertEqual(360, result.nrays)
    self.assertAlmostEqual(2.0, result.elangle, 4)

    result = obj.findField((5,360), 2.0)
    self.assertEqual(None, result)

    result = obj.findField((1,360), 3.0)
    self.assertEqual(None, result)

  def test_map_save_1_group(self):
    obj = _acqvafeaturemap.map()
    obj.nod = "seang"
    obj.startdate="20220101"
    obj.enddate="20220201"
    obj.latitude = 60.0 * math.pi / 180.0
    obj.longitude = 14.0 * math.pi / 180.0
    obj.height = 300.0
    obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0)
    obj.createField((1,362), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0)

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

    # Field 2
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data2/where/elangle").data(), 4)
    f2data = nodelist.getNode("/dataset1/data2/data").data()
    self.assertEqual((362,1), f2data.shape)

  def test_map_save_2_groups(self):
    obj = _acqvafeaturemap.map()
    obj.nod = "seang"
    obj.startdate="20220101"
    obj.enddate="20220201"
    obj.latitude = 60.0 * math.pi / 180.0
    obj.longitude = 14.0 * math.pi / 180.0
    obj.height = 300.0
    obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0)
    obj.createField((1,362), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0)
    obj.createField((1,361), _rave.RaveDataType_UCHAR, 2.0 * math.pi / 180.0)
    obj.createField((1,363), _rave.RaveDataType_UCHAR, 2.0 * math.pi / 180.0)

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

    # Field 2
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data2/where/elangle").data(), 4)
    f2data = nodelist.getNode("/dataset1/data2/data").data()
    self.assertEqual((362,1), f2data.shape)

    # Elevation group 2
    self.assertAlmostEqual(2.0, nodelist.getNode("/dataset2/where/elangle").data(), 4)

    # Field 1
    self.assertAlmostEqual(2.0, nodelist.getNode("/dataset2/data1/where/elangle").data(), 4)
    f1data = nodelist.getNode("/dataset2/data1/data").data()
    self.assertEqual((361,1), f1data.shape)

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
    obj.createField((1,360), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0)
    obj.createField((1,362), _rave.RaveDataType_UCHAR, 1.0 * math.pi / 180.0)
    obj.createField((1,361), _rave.RaveDataType_UCHAR, 2.0 * math.pi / 180.0)
    obj.createField((1,363), _rave.RaveDataType_UCHAR, 2.0 * math.pi / 180.0)

    obj.save(self.TEMPORARY_FILE)

    result = _acqvafeaturemap.load(self.TEMPORARY_FILE)
    self.assertEqual("seang", result.nod)
    self.assertEqual("20220101", result.startdate)
    self.assertEqual("20220201", result.enddate)
    self.assertAlmostEqual(60.0, result.latitude*180.0/math.pi, 4)
    self.assertAlmostEqual(14.0, result.longitude*180.0/math.pi, 4)
    self.assertAlmostEqual(300, result.height, 4)

    self.assertEqual(2, result.getNumberOfElevations())
    self.assertAlmostEqual(1.0 * math.pi / 180.0, result.getElevation(0).elangle, 4)
    self.assertAlmostEqual(2.0 * math.pi / 180.0, result.getElevation(1).elangle, 4)

    self.assertTrue(np.all(result.getElevation(0).get(0).getData() == np.zeros((360, 1), np.uint8)))
    self.assertTrue(np.all(result.getElevation(0).get(1).getData() == np.zeros((362, 1), np.uint8)))
    self.assertTrue(np.all(result.getElevation(1).get(0).getData() == np.zeros((361, 1), np.uint8)))    
    self.assertTrue(np.all(result.getElevation(1).get(1).getData() == np.zeros((363, 1), np.uint8)))