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
import _cartesianvolume
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
  FIXTURE_SCAN="fixtures/scan_sevil_20100702T113200Z.h5"
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

  def test_attribute_visibility(self):
    attrs = ['version', 'h5radversion', 'objectType', 'filename', 'object']
    obj = _raveio.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)

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
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 238000.0, 238000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.quantity = "DBZH"
    image.gain = 1.0
    image.offset = 0.0
    image.nodata = 255.0
    image.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    image.setData(data)
    
    ios = _raveio.new()
    ios.object = image
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEquals("ODIM_H5/V2_0", nodelist.getNode("/Conventions").data())
    # What
    self.assertEquals("100000", nodelist.getNode("/what/time").data())
    self.assertEquals("20100101", nodelist.getNode("/what/date").data())
    self.assertEquals("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEquals("IMAGE", nodelist.getNode("/what/object").data())
    self.assertEquals("H5rad 2.0", nodelist.getNode("/what/version").data())
    
    #Where
    self.assertEquals("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", nodelist.getNode("/where/projdef").data())
    self.assertEquals(240, nodelist.getNode("/where/xsize").data())
    self.assertEquals(240, nodelist.getNode("/where/ysize").data())
    self.assertAlmostEquals(2000.0, nodelist.getNode("/where/xscale").data(), 4)
    self.assertAlmostEquals(2000.0, nodelist.getNode("/where/yscale").data(), 4)

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
    self.assertEquals("20100101", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEquals("100000", nodelist.getNode("/dataset1/what/endtime").data())
    self.assertEquals("20100101", nodelist.getNode("/dataset1/what/enddate").data())
    self.assertAlmostEquals(1.0, nodelist.getNode("/dataset1/what/gain").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset1/what/offset").data(), 4)
    self.assertAlmostEquals(255.0, nodelist.getNode("/dataset1/what/nodata").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset1/what/undetect").data(), 4)
    
    self.assertEquals(numpy.uint8, nodelist.getNode("/dataset1/data1/data").data().dtype)
    self.assertEquals("IMAGE", nodelist.getNode("/dataset1/data1/data/CLASS").data())
    self.assertEquals("1.2", nodelist.getNode("/dataset1/data1/data/IMAGE_VERSION").data())

  def test_save_cartesian_startandstoptime(self):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 238000.0, 238000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.quantity = "DBZH"
    image.gain = 1.0
    image.offset = 0.0
    image.nodata = 255.0
    image.undetect = 0.0
    image.addAttribute("what/starttime", "110000")
    image.addAttribute("what/startdate", "20110101")
    image.addAttribute("what/endtime", "110005")
    image.addAttribute("what/enddate", "20110101")
    
    data = numpy.zeros((240,240),numpy.uint8)
    image.setData(data)
    
    ios = _raveio.new()
    ios.object = image
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEquals("ODIM_H5/V2_0", nodelist.getNode("/Conventions").data())
    # What
    self.assertEquals("100000", nodelist.getNode("/what/time").data())
    self.assertEquals("20100101", nodelist.getNode("/what/date").data())
    self.assertEquals("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEquals("IMAGE", nodelist.getNode("/what/object").data())
    self.assertEquals("H5rad 2.0", nodelist.getNode("/what/version").data())
    
    #Where
    self.assertEquals("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", nodelist.getNode("/where/projdef").data())
    self.assertEquals(240, nodelist.getNode("/where/xsize").data())
    self.assertEquals(240, nodelist.getNode("/where/ysize").data())
    self.assertAlmostEquals(2000.0, nodelist.getNode("/where/xscale").data(), 4)
    self.assertAlmostEquals(2000.0, nodelist.getNode("/where/yscale").data(), 4)

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
    self.assertEquals("110000", nodelist.getNode("/dataset1/what/starttime").data())
    self.assertEquals("20110101", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEquals("110005", nodelist.getNode("/dataset1/what/endtime").data())
    self.assertEquals("20110101", nodelist.getNode("/dataset1/what/enddate").data())
    self.assertAlmostEquals(1.0, nodelist.getNode("/dataset1/what/gain").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset1/what/offset").data(), 4)
    self.assertAlmostEquals(255.0, nodelist.getNode("/dataset1/what/nodata").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset1/what/undetect").data(), 4)
    
    self.assertEquals(numpy.uint8, nodelist.getNode("/dataset1/data1/data").data().dtype)
    self.assertEquals("IMAGE", nodelist.getNode("/dataset1/data1/data/CLASS").data())
    self.assertEquals("1.2", nodelist.getNode("/dataset1/data1/data/IMAGE_VERSION").data())

  def test_save_cartesian_volume(self):
    cvol = _cartesianvolume.new()
    cvol.time = "100000"
    cvol.date = "20091010"
    cvol.objectType = _rave.Rave_ObjectType_CVOL
    cvol.source = "PLC:123"
    cvol.xscale = 2000.0
    cvol.yscale = 2000.0
    cvol.areaextent = (-240000.0, -240000.0, 238000.0, 238000.0)
    projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    cvol.projection = projection

    image = _cartesian.new()
    image.product = _rave.Rave_ProductType_CAPPI
    image.quantity = "DBZH"
    image.gain = 1.0
    image.offset = 0.0
    image.nodata = 255.0
    image.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    image.setData(data)

    cvol.addImage(image)

    image = _cartesian.new()
    image.product = _rave.Rave_ProductType_CAPPI
    image.quantity = "MMH"
    image.gain = 1.0
    image.offset = 0.0
    image.nodata = 255.0
    image.undetect = 0.0
    image.addAttribute("what/starttime", "110000")
    image.addAttribute("what/startdate", "20110101")
    image.addAttribute("what/endtime", "110005")
    image.addAttribute("what/enddate", "20110101")
    
    data = numpy.zeros((240,240),numpy.uint8)
    image.setData(data)

    cvol.addImage(image)

    ios = _raveio.new()
    ios.object = cvol
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

    #dataset2
    self.assertEquals("MMH", nodelist.getNode("/dataset2/what/quantity").data())
    self.assertEquals("110000", nodelist.getNode("/dataset2/what/starttime").data())
    self.assertEquals("20110101", nodelist.getNode("/dataset2/what/startdate").data())
    self.assertEquals("110005", nodelist.getNode("/dataset2/what/endtime").data())
    self.assertEquals("20110101", nodelist.getNode("/dataset2/what/enddate").data())
    self.assertAlmostEquals(1.0, nodelist.getNode("/dataset2/what/gain").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset2/what/offset").data(), 4)
    self.assertAlmostEquals(255.0, nodelist.getNode("/dataset2/what/nodata").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode("/dataset2/what/undetect").data(), 4)
    
    self.assertEquals(numpy.uint8, nodelist.getNode("/dataset2/data1/data").data().dtype)
    self.assertEquals("IMAGE", nodelist.getNode("/dataset2/data1/data/CLASS").data())
    self.assertEquals("1.2", nodelist.getNode("/dataset2/data1/data/IMAGE_VERSION").data())

  
  def test_load_cartesian_volume(self):
    obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
    self.assertEquals(_raveio.RaveIO_ODIM_Version_2_0, obj.version)
    self.assertEquals(_raveio.RaveIO_ODIM_H5rad_Version_2_0, obj.h5radversion)
    self.assertEquals(_rave.Rave_ObjectType_CVOL, obj.objectType)
    
    cvol = obj.object
    self.assertEquals(_rave.Rave_ObjectType_CVOL, cvol.objectType)
    self.assertEquals("100000", cvol.time)
    self.assertEquals("20091010", cvol.date)
    self.assertEquals("PLC:123", cvol.source)
    self.assertEquals("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", cvol.projection.definition)
    self.assertAlmostEquals(-240000.0, cvol.areaextent[0], 4)
    self.assertAlmostEquals(-240000.0, cvol.areaextent[1], 4)
    self.assertAlmostEquals(238000.0, cvol.areaextent[2], 4)
    self.assertAlmostEquals(238000.0, cvol.areaextent[3], 4)
    self.assertAlmostEquals(2000.0, cvol.xscale, 4)
    self.assertAlmostEquals(2000.0, cvol.yscale, 4)

    self.assertEquals(1, cvol.getNumberOfImages())

    image = cvol.getImage(0)
    self.assertEquals(_rave.Rave_ProductType_CAPPI, image.product)
    self.assertEquals("DBZH", image.quantity)
    self.assertAlmostEquals(1.0, image.gain, 4)
    self.assertAlmostEquals(0.0, image.offset, 4)
    self.assertAlmostEquals(255.0, image.nodata, 4)
    self.assertAlmostEquals(0.0, image.undetect, 4)
    self.assertEquals(240, image.xsize)
    self.assertEquals(240, image.ysize)
    self.assertEquals(numpy.uint8, image.getData().dtype)

  def test_load_cartesian_volume_save_cartesian_image(self):
    obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
    image = obj.object.getImage(0)
    ios = _raveio.new()
    ios.object = image
    ios.filename = self.TEMPORARY_FILE
    ios.save()    
    
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEquals("ODIM_H5/V2_0", nodelist.getNode("/Conventions").data())

    # What
    self.assertEquals("100000", nodelist.getNode("/what/time").data())
    self.assertEquals("20091010", nodelist.getNode("/what/date").data())
    self.assertEquals("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEquals("IMAGE", nodelist.getNode("/what/object").data())
    self.assertEquals("H5rad 2.0", nodelist.getNode("/what/version").data())
    
    #Where
    self.assertEquals("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", nodelist.getNode("/where/projdef").data())
    self.assertEquals(240, nodelist.getNode("/where/xsize").data())
    self.assertEquals(240, nodelist.getNode("/where/ysize").data())
    self.assertAlmostEquals(2000.0, nodelist.getNode("/where/xscale").data(), 4)
    self.assertAlmostEquals(2000.0, nodelist.getNode("/where/yscale").data(), 4)

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
    scan1.time = "100001"
    scan1.date = "20091010"
    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 10.0
    dbzhParam.undetect = 11.0
    dbzhParam.quantity = "DBZH"
    dbzhParam.gain = 1.0
    dbzhParam.offset = 0.0
    data = numpy.zeros((100, 120), numpy.uint8)
    dbzhParam.setData(data)
    scan1.addParameter(dbzhParam)

    mmhParam = _polarscanparam.new()
    mmhParam.nodata = 12.0
    mmhParam.undetect = 13.0
    mmhParam.quantity = "MMH"
    mmhParam.gain = 10.0
    mmhParam.offset = 20.0
    data = numpy.zeros((100, 120), numpy.int16)
    mmhParam.setData(data)
    scan1.addParameter(mmhParam)

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
    
    # Verify that both DBZH and MMH has been stored properly.
    d1field = nodelist.getNode("/dataset1/data1/what/quantity").data()
    d2field = nodelist.getNode("/dataset1/data2/what/quantity").data()
    dbzhname = "/dataset1/data1"
    mmhname = "/dataset1/data2"
    if d1field == "MMH":
      dbzhname = "/dataset1/data2"
      mmhname = "/dataset1/data1"
    
    # dbzh field
    self.assertEquals("DBZH", nodelist.getNode(dbzhname + "/what/quantity").data())
    self.assertAlmostEquals(1.0, nodelist.getNode(dbzhname + "/what/gain").data(), 4)
    self.assertAlmostEquals(0.0, nodelist.getNode(dbzhname + "/what/offset").data(), 4)
    self.assertAlmostEquals(10.0, nodelist.getNode(dbzhname + "/what/nodata").data(), 4)
    self.assertAlmostEquals(11.0, nodelist.getNode(dbzhname + "/what/undetect").data(), 4)
    
    # 
    self.assertEquals(numpy.uint8, nodelist.getNode(dbzhname + "/data").data().dtype)
    self.assertEquals("IMAGE", nodelist.getNode(dbzhname + "/data/CLASS").data())
    self.assertEquals("1.2", nodelist.getNode(dbzhname + "/data/IMAGE_VERSION").data())

    # mmh field
    self.assertEquals("MMH", nodelist.getNode(mmhname + "/what/quantity").data())
    self.assertAlmostEquals(10.0, nodelist.getNode(mmhname + "/what/gain").data(), 4)
    self.assertAlmostEquals(20.0, nodelist.getNode(mmhname + "/what/offset").data(), 4)
    self.assertAlmostEquals(12.0, nodelist.getNode(mmhname + "/what/nodata").data(), 4)
    self.assertAlmostEquals(13.0, nodelist.getNode(mmhname + "/what/undetect").data(), 4)
    
    # dataset1/data2/data
    self.assertEquals(numpy.int16, nodelist.getNode(mmhname + "/data").data().dtype)

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

  # (RT: Ticket 8)
  def test_loadCartesian_differentXYSize(self):
    src = _cartesian.new()
    src.time = "100000"
    src.date = "20091010"
    src.objectType = _rave.Rave_ObjectType_IMAGE
    src.product = _rave.Rave_ProductType_COMP
    src.source = "PLC:123"
    src.quantity = "DBZH"
    src.gain = 1.0
    src.offset = 0.0
    src.nodata = 255.0
    src.undetect = 0.0
    src.xscale = 2000.0
    src.yscale = 2000.0
    src.areaextent = (-240000.0, -240000.0, 238000.0, 238000.0)
    projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    src.projection = projection
    data = numpy.zeros((100,90),numpy.int16)
    src.setData(data)

    ios = _raveio.new()
    ios.object = src
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertEquals(_rave.Rave_ObjectType_IMAGE, obj.object.objectType);
  
  def test_load_scan(self):
    obj = _raveio.open(self.FIXTURE_SCAN)
    self.assertNotEqual(-1, string.find(`type(obj.object)`, "PolarScanCore"))
    scan = obj.object

    self.assertAlmostEquals(40.0, scan.elangle*180.0/math.pi, 4)

    p1 = scan.getParameter("DBZH")
    self.assertAlmostEquals(0.4, p1.gain, 4)
    self.assertAlmostEquals(-30.0, p1.offset, 4)
    
    p2 = scan.getParameter("VRAD")
    self.assertAlmostEquals(0.375, p2.gain, 4)
    self.assertAlmostEquals(-48.0, p2.offset, 4)
  
  def addGroupNode(self, nodelist, name):
    node = _pyhl.node(_pyhl.GROUP_ID, name)
    nodelist.addNode(node)
    
  def addAttributeNode(self, nodelist, name, type, value):
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, name)
    node.setScalarValue(-1,value,type,-1)
    nodelist.addNode(node)

  def rad2deg(self, coord):
    return (coord[0]*180.0/math.pi, coord[1]*180.0/math.pi)
  