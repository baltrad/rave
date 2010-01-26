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

Tests the PyRaveIO module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-10-16
'''
import unittest
import os
import _raveio
import _cartesian
import _projection
import _polarvolume
import _polarscan
import _polarscanparam
import _rave
import string
import numpy
import _pyhl
import math

class PyRaveIOTest(unittest.TestCase):
  FIXTURE_VOLUME="fixture_ODIM_H5_pvol_ang_20090501T1200Z.h5"
  FIXTURE_IMAGE="fixture_old_pcappi-dbz-500.ang-gnom-2000.h5"
  FIXTURE_CVOL_CAPPI="fixture_ODIM_cvol_cappi.h5"
  
  TEMPORARY_FILE="ravemodule_iotest.h5"
  
  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def test_new(self):
    obj = _raveio.new()
    israveio = string.find(`type(obj)`, "RaveIOCore")
    self.assertNotEqual(-1, israveio)

  def test_load(self):
    obj = _raveio.new()
    self.assertTrue(obj.object == None)
    obj.filename = self.FIXTURE_VOLUME
    obj.load()
    self.assertTrue(obj.object != None)

  def test_open_noSuchFile(self):
    try:
      _raveio.open("No_Such_File_Fixture.h5")
      self.fail("Expected IOError")
    except IOError, e:
      pass
  
  def test_objectType(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    self.assertEquals(_raveio.Rave_ObjectType_PVOL, obj.objectType)

  def test_objectType_notSettable(self):
    obj = _raveio.new()
    self.assertEquals(_raveio.Rave_ObjectType_UNDEFINED, obj.objectType)
    try:
      obj.objectType = _raveio.Rave_ObjectType_PVOL
      self.fail("Expected AttributeError")
    except AttributeError, e:
      pass
    self.assertEquals(_raveio.Rave_ObjectType_UNDEFINED, obj.objectType)
  
  def test_load_volume(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    result = string.find(`type(vol)`, "PolarVolumeCore")
    self.assertNotEqual(-1, result)     

  def test_load_volume_checkData(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    self.assertEquals(10, vol.getNumberOfScans())
    self.assertAlmostEquals(56.3675, vol.latitude*180.0/math.pi, 4)
    self.assertAlmostEquals(12.8544, vol.longitude*180.0/math.pi, 4)
    self.assertAlmostEquals(209, vol.height, 4)
    
    # Verify the scan
    scan = vol.getScan(0)
    self.assertEquals(0, scan.a1gate)
    self.assertAlmostEquals(0.5, scan.elangle*180.0/math.pi, 4)
    self.assertEquals(120, scan.nbins)
    self.assertEquals(420, scan.nrays)
    self.assertAlmostEquals(2000.0, scan.rscale, 4)
    self.assertAlmostEquals(0.0, scan.rstart, 4)

    #Inherited volume position
    self.assertAlmostEquals(56.3675, scan.latitude*180.0/math.pi, 4)
    self.assertAlmostEquals(12.8544, scan.longitude*180.0/math.pi, 4)
    self.assertAlmostEquals(209, scan.height, 4)

    # Verify the DBZH
    dbzhParam = scan.getParameter("DBZH")
    self.assertAlmostEquals(0.4, dbzhParam.gain, 4)
    self.assertAlmostEquals(-30.0, dbzhParam.offset, 4)
    self.assertAlmostEquals(255.0, dbzhParam.nodata, 4)
    self.assertAlmostEquals(0.0, dbzhParam.undetect, 4)
    self.assertEquals(120, dbzhParam.nbins, 4)
    self.assertEquals(420, dbzhParam.nrays, 4)
    self.assertEquals("DBZH", dbzhParam.quantity)

    # And verify the VRAD
    vradParam = scan.getParameter("VRAD")
    self.assertAlmostEquals(0.1875, vradParam.gain, 4)
    self.assertAlmostEquals(-24.0, vradParam.offset, 4)
    self.assertAlmostEquals(255.0, vradParam.nodata, 4)
    self.assertAlmostEquals(0.0, vradParam.undetect, 4)
    self.assertEquals("VRAD", vradParam.quantity)

    # Verify last scan
    scan = vol.getScan(9)
    self.assertEquals(0, scan.a1gate)
    self.assertAlmostEquals(40.0, scan.elangle*180.0/math.pi, 4)
    self.assertEquals(120, scan.nbins)
    self.assertEquals(420, scan.nrays)
    self.assertAlmostEquals(1000.0, scan.rscale, 4)
    self.assertAlmostEquals(0.0, scan.rstart, 4)
    #Inherited volume position
    self.assertAlmostEquals(56.3675, scan.latitude*180.0/math.pi, 4)
    self.assertAlmostEquals(12.8544, scan.longitude*180.0/math.pi, 4)
    self.assertAlmostEquals(209, scan.height, 4)  

    # Verify last scans DBZH parameter
    dbzhParam = scan.getParameter("DBZH")
    self.assertAlmostEquals(0.4, dbzhParam.gain, 4)
    self.assertAlmostEquals(-30.0, dbzhParam.offset, 4)
    self.assertAlmostEquals(255.0, dbzhParam.nodata, 4)
    self.assertAlmostEquals(0.0, dbzhParam.undetect, 4)
    self.assertEquals("DBZH", dbzhParam.quantity)
    self.assertEquals(120, dbzhParam.nbins)
    self.assertEquals(420, dbzhParam.nrays)

    # Verify last scans VRAD parameter
    vradParam = scan.getParameter("VRAD")
    self.assertAlmostEquals(0.375, vradParam.gain, 4)
    self.assertAlmostEquals(-48.0, vradParam.offset, 4)
    self.assertAlmostEquals(255.0, vradParam.nodata, 4)
    self.assertAlmostEquals(0.0, vradParam.undetect, 4)
    self.assertEquals("VRAD", vradParam.quantity)

  def test_save_cartesian(self):
    obj = _cartesian.new()
    obj.time = "100000"
    obj.date = "20091010"
    obj.objectType = _rave.Rave_ObjectType_CVOL
    obj.product = _rave.Rave_ProductType_CAPPI
    obj.source = "PLC:123"
    obj.quantity = "DBZH"
    obj.gain = 1.0
    obj.offset = 0.0
    obj.nodata = 255.0
    obj.undetect = 0.0
    obj.xscale = 2000.0
    obj.yscale = 2000.0
    obj.areaextent = (-240000.0, -240000.0, 238000.0, 238000.0)
    projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    obj.projection = projection
    data = numpy.zeros((240,240),numpy.uint8)
    obj.setData(data)

    ios = _raveio.new()
    ios.object = obj
    ios.filename = self.TEMPORARY_FILE
    ios.save()
    
    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEquals("ODIM_H5/V2_0", nodelist.getNode("/Conventions").data())
    # What
    self.assertEquals("100000", nodelist.getNode("/what/time").data())
    self.assertEquals("20091010", nodelist.getNode("/what/date").data())
    self.assertEquals("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEquals("CVOL", nodelist.getNode("/what/object").data())
    self.assertEquals("H5rad 2.0", nodelist.getNode("/what/version").data())
    
    #Where
    self.assertEquals("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", nodelist.getNode("/where/projdef").data())
    self.assertEquals(240, nodelist.getNode("/where/xsize").data())
    self.assertEquals(240, nodelist.getNode("/where/ysize").data())
    self.assertAlmostEquals(2000.0, nodelist.getNode("/where/xscale").data(), 4)
    self.assertAlmostEquals(2000.0, nodelist.getNode("/where/yscale").data(), 4)

    #LL = (9.171399, 54.153937)
    #UR = (16.941843,58.441852)
    #UL = (8.732727,58.440757)
    #LR = (16.506793,54.154869)
    #print "%f,%f"%(self.rad2deg(projection.inv((238000.0, -240000.0))))
    
    self.assertAlmostEquals(9.1714, nodelist.getNode("/where/LL_lon").data(), 4)
    self.assertAlmostEquals(54.1539, nodelist.getNode("/where/LL_lat").data(), 4)
    self.assertAlmostEquals(8.7327, nodelist.getNode("/where/UL_lon").data(), 4)
    self.assertAlmostEquals(58.4408, nodelist.getNode("/where/UL_lat").data(), 4)
    self.assertAlmostEquals(16.9418, nodelist.getNode("/where/UR_lon").data(), 4)
    self.assertAlmostEquals(58.4419, nodelist.getNode("/where/UR_lat").data(), 4)
    self.assertAlmostEquals(16.5068, nodelist.getNode("/where/LR_lon").data(), 4)
    self.assertAlmostEquals(54.1549, nodelist.getNode("/where/LR_lat").data(), 4)

    #dataset1
    self.assertEquals("DBZH", nodelist.getNode("/dataset1/what/quantity").data())
    self.assertEquals("100000", nodelist.getNode("/dataset1/what/starttime").data())
    self.assertEquals("20091010", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEquals("100000", nodelist.getNode("/dataset1/what/endtime").data())
    self.assertEquals("20091010", nodelist.getNode("/dataset1/what/enddate").data())
    self.assertAlmostEquals(1.0, nodelist.getNode("/dataset1/what/gain").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset1/what/offset").data(), 4)
    self.assertAlmostEquals(255.0, nodelist.getNode("/dataset1/what/nodata").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset1/what/undetect").data(), 4)
    
    self.assertEquals(numpy.uint8, nodelist.getNode("/dataset1/data1/data").data().dtype)
    self.assertEquals("IMAGE", nodelist.getNode("/dataset1/data1/data/CLASS").data())
    self.assertEquals("1.2", nodelist.getNode("/dataset1/data1/data/IMAGE_VERSION").data())
  
  def test_load_cartesian(self):
    obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
    self.assertEquals(_raveio.RaveIO_ODIM_Version_2_0, obj.version)
    self.assertEquals(_raveio.RaveIO_ODIM_H5rad_Version_2_0, obj.h5radversion)
    self.assertEquals(_rave.Rave_ObjectType_CVOL, obj.objectType)
    
    cvol = obj.object
    self.assertEquals("100000", cvol.time)
    self.assertEquals("20091010", cvol.date)
    self.assertEquals(_rave.Rave_ObjectType_CVOL, cvol.objectType)
    self.assertEquals(_rave.Rave_ProductType_CAPPI, cvol.product)
    self.assertEquals("PLC:123", cvol.source)
    self.assertEquals("DBZH", cvol.quantity)
    self.assertAlmostEquals(1.0, cvol.gain, 4)
    self.assertAlmostEquals(0.0, cvol.offset, 4)
    self.assertAlmostEquals(255.0, cvol.nodata, 4)
    self.assertAlmostEquals(0.0, cvol.undetect, 4)
    self.assertAlmostEquals(2000.0, cvol.xscale, 4)
    self.assertAlmostEquals(2000.0, cvol.yscale, 4)
    self.assertAlmostEquals(-240000.0, cvol.areaextent[0], 4)
    self.assertAlmostEquals(-240000.0, cvol.areaextent[1], 4)
    self.assertAlmostEquals(238000.0, cvol.areaextent[2], 4)
    self.assertAlmostEquals(238000.0, cvol.areaextent[3], 4)
    self.assertEquals("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", cvol.projection.definition)
    self.assertEquals(240, cvol.xsize)
    self.assertEquals(240, cvol.ysize)
    self.assertEquals(numpy.uint8, cvol.getData().dtype)

  def test_save_polar_volume(self):
    obj = _polarvolume.new()
    obj.time = "100000"
    obj.date = "20091010"
    obj.source = "PLC:123"
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    
    scan1 = _polarscan.new()
    scan1.elangle = 0.1 * math.pi / 180.0
    scan1.a1gate = 2
    scan1.rstart = 0.0
    scan1.rscale = 5000.0
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 10.0
    dbzhParam.undetect = 11.0
    dbzhParam.quantity = "DBZH"
    dbzhParam.gain = 1.0
    dbzhParam.offset = 0.0
    scan1.time = "100001"
    scan1.date = "20091010"
    data = numpy.zeros((100, 120), numpy.uint8)
    dbzhParam.setData(data)
    scan1.addParameter(dbzhParam)
    obj.addScan(scan1)

    scan2 = _polarscan.new()
    scan2.elangle = 0.5 * math.pi / 180.0
    scan2.a1gate = 1
    scan2.rstart = 1000.0
    scan2.rscale = 2000.0
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 255.0
    dbzhParam.undetect = 0.0
    dbzhParam.quantity = "MMM"
    dbzhParam.gain = 1.0
    dbzhParam.offset = 0.0
    scan2.time = "100002"
    scan2.date = "20091010"
    data = numpy.zeros((100, 120), numpy.uint8)
    dbzhParam.setData(data)
    scan2.addParameter(dbzhParam)
    obj.addScan(scan2)
    
    ios = _raveio.new()
    ios.object = obj
    ios.filename = self.TEMPORARY_FILE
    ios.save()
    
    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEquals("ODIM_H5/V2_0", nodelist.getNode("/Conventions").data())
    # What
    self.assertEquals("100000", nodelist.getNode("/what/time").data())
    self.assertEquals("20091010", nodelist.getNode("/what/date").data())
    self.assertEquals("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEquals("PVOL", nodelist.getNode("/what/object").data())
    self.assertEquals("H5rad 2.0", nodelist.getNode("/what/version").data())
    
    #Where
    self.assertAlmostEquals(12.0, nodelist.getNode("/where/lon").data(), 4)
    self.assertAlmostEquals(60.0, nodelist.getNode("/where/lat").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/where/height").data(), 4)

    #
    # dataset1 (scan1)
    #
    self.assertEquals("100001", nodelist.getNode("/dataset1/what/starttime").data())
    self.assertEquals("20091010", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEquals("100001", nodelist.getNode("/dataset1/what/endtime").data())
    self.assertEquals("20091010", nodelist.getNode("/dataset1/what/enddate").data())
    self.assertEquals("SCAN", nodelist.getNode("/dataset1/what/product").data())
    
    # dataset1/where
    self.assertAlmostEquals(0.1, nodelist.getNode("/dataset1/where/elangle").data(), 4)
    self.assertEquals(2, nodelist.getNode("/dataset1/where/a1gate").data())
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset1/where/rstart").data(), 4)
    self.assertAlmostEquals(5000.0, nodelist.getNode("/dataset1/where/rscale").data(), 4)
    self.assertEquals(120, nodelist.getNode("/dataset1/where/nbins").data())
    self.assertEquals(100, nodelist.getNode("/dataset1/where/nrays").data())
    
    # dataset1/data1/what
    self.assertEquals("DBZH", nodelist.getNode("/dataset1/data1/what/quantity").data())
    self.assertAlmostEquals(1.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEquals(10.0, nodelist.getNode("/dataset1/data1/what/nodata").data(), 4)
    self.assertAlmostEquals(11.0, nodelist.getNode("/dataset1/data1/what/undetect").data(), 4)
    
    # dataset1/data1/data
    self.assertEquals(numpy.uint8, nodelist.getNode("/dataset1/data1/data").data().dtype)
    self.assertEquals("IMAGE", nodelist.getNode("/dataset1/data1/data/CLASS").data())
    self.assertEquals("1.2", nodelist.getNode("/dataset1/data1/data/IMAGE_VERSION").data())

    #
    # dataset2 (scan2)
    #
    self.assertEquals("100002", nodelist.getNode("/dataset2/what/starttime").data())
    self.assertEquals("20091010", nodelist.getNode("/dataset2/what/startdate").data())
    self.assertEquals("100002", nodelist.getNode("/dataset2/what/endtime").data())
    self.assertEquals("20091010", nodelist.getNode("/dataset2/what/enddate").data())
    self.assertEquals("SCAN", nodelist.getNode("/dataset2/what/product").data())
    
    # dataset2/where
    self.assertAlmostEquals(0.5, nodelist.getNode("/dataset2/where/elangle").data(), 4)
    self.assertEquals(1, nodelist.getNode("/dataset2/where/a1gate").data())
    self.assertAlmostEquals(1000.0, nodelist.getNode("/dataset2/where/rstart").data(), 4)
    self.assertAlmostEquals(2000.0, nodelist.getNode("/dataset2/where/rscale").data(), 4)
    self.assertEquals(120, nodelist.getNode("/dataset2/where/nbins").data())
    self.assertEquals(100, nodelist.getNode("/dataset2/where/nrays").data())
    
    # dataset2/data1/what
    self.assertEquals("MMM", nodelist.getNode("/dataset2/data1/what/quantity").data())
    self.assertAlmostEquals(1.0, nodelist.getNode("/dataset2/data1/what/gain").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset2/data1/what/offset").data(), 4)
    self.assertAlmostEquals(255.0, nodelist.getNode("/dataset2/data1/what/nodata").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset2/data1/what/undetect").data(), 4)
    
    # dataset2/data1/data
    self.assertEquals(numpy.uint8, nodelist.getNode("/dataset2/data1/data").data().dtype)
    self.assertEquals("IMAGE", nodelist.getNode("/dataset2/data1/data/CLASS").data())
    self.assertEquals("1.2", nodelist.getNode("/dataset2/data1/data/IMAGE_VERSION").data())

    
  def addGroupNode(self, nodelist, name):
    node = _pyhl.node(_pyhl.GROUP_ID, name)
    nodelist.addNode(node)
    
  def addAttributeNode(self, nodelist, name, type, value):
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, name)
    node.setScalarValue(-1,value,type,-1)
    nodelist.addNode(node)

  def rad2deg(self, coord):
    return (coord[0]*180.0/math.pi, coord[1]*180.0/math.pi)
  