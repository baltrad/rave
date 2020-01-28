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

@co-author Ulf Nordh (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2017-10-27. Updated code with more fields for vertical profiles
'''
import unittest
import os
import _raveio
import _cartesian
import _cartesianparam
import _cartesianvolume
import _projection
import _polarvolume
import _polarscan
import _polarscanparam
import _rave
import _ravefield
import _verticalprofile
import string
import numpy
import _pyhl
import math

class PyRaveIOTest(unittest.TestCase):
  FIXTURE_VOLUME="fixture_ODIM_H5_pvol_ang_20090501T1200Z.h5"
  FIXTURE_IMAGE="fixture_old_pcappi-dbz-500.ang-gnom-2000.h5"
  FIXTURE_CVOL_CAPPI="fixture_ODIM_cvol_cappi.h5"
  FIXTURE_SCAN="fixtures/scan_sevil_20100702T113200Z.h5"
  FIXTURE_SCAN_WITH_ARRAYS="fixtures/scan_with_arrays.h5"
  FIXTURE_CARTESIAN_IMAGE="fixtures/cartesian_image.h5"
  FIXTURE_CARTESIAN_VOLUME="fixtures/cartesian_volume.h5"
  FIXTURE_VP="fixtures/vp_fixture.h5"
  FIXTURE_VP_NEW_VERSION="fixtures/selek_vp_20170901T000000Z.h5"
  FIXTURE_VP_NEW_VERSION_EXTRA_HOW="fixtures/selek_vp_only_UWND_and_extra_how.h5"
  FIXTURE_BUFR_PVOL="fixtures/odim_polar_ref.bfr"
  FIXTURE_BUFR_COMPO="fixtures/odim_compo_ref.bfr"
  FIXTURE_BUFR_2_2="fixtures/odim_2_2_ref.bfr"
  
  TEMPORARY_FILE="ravemodule_iotest.h5"
  TEMPORARY_FILE2="ravemodule_iotest2.h5"
  
  
  def setUp(self):
    #_rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)
    if os.path.isfile(self.TEMPORARY_FILE2):
      os.unlink(self.TEMPORARY_FILE2)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)
    if os.path.isfile(self.TEMPORARY_FILE2):
      os.unlink(self.TEMPORARY_FILE2)

  def test_new(self):
    obj = _raveio.new()
    israveio = str(type(obj)).find("RaveIOCore")
    self.assertNotEqual(-1, israveio)

  def test_attribute_visibility(self):
    attrs = ['version', 'h5radversion', 'objectType', 'filename', 'object', 'compression_level', 'fcp_userblock',
             'fcp_sizes', 'fcp_symk', 'fcp_istorek', 'fcp_metablocksize']
    obj = _raveio.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)

  def test_no_filename(self):
    obj = _raveio.new()
    # According to issue obj.filename should crash
    obj.filename
    

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
    except IOError:
      pass
  
  def test_objectType(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    self.assertEqual(_raveio.Rave_ObjectType_PVOL, obj.objectType)

  def test_objectType_notSettable(self):
    obj = _raveio.new()
    self.assertEqual(_raveio.Rave_ObjectType_UNDEFINED, obj.objectType)
    try:
      obj.objectType = _raveio.Rave_ObjectType_PVOL
      self.fail("Expected AttributeError")
    except AttributeError:
      pass
    self.assertEqual(_raveio.Rave_ObjectType_UNDEFINED, obj.objectType)
  
  def test_compression_level(self):
    obj = _raveio.new()
    self.assertEqual(6, obj.compression_level)
    obj.compression_level = 9
    self.assertEqual(9, obj.compression_level)
    obj.compression_level = 0
    self.assertEqual(0, obj.compression_level)
    obj.compression_level = 10
    self.assertEqual(0, obj.compression_level)
    obj.compression_level = -1
    self.assertEqual(0, obj.compression_level)
    
  def test_fcp_userblock(self):
    obj = _raveio.new()
    self.assertEqual(0, obj.fcp_userblock)
    obj.fcp_userblock = 2
    self.assertEqual(2, obj.fcp_userblock)
    
  def test_fcp_sizes(self):
    obj = _raveio.new()
    self.assertEqual(4, obj.fcp_sizes[0])
    self.assertEqual(4, obj.fcp_sizes[1])
    obj.fcp_sizes = (8, 2)
    self.assertEqual(8, obj.fcp_sizes[0])
    self.assertEqual(2, obj.fcp_sizes[1])
    
  def test_fcp_symk(self):
    obj = _raveio.new()
    self.assertEqual(1, obj.fcp_symk[0])
    self.assertEqual(1, obj.fcp_symk[1])
    obj.fcp_symk = (2, 4)
    self.assertEqual(2, obj.fcp_symk[0])
    self.assertEqual(4, obj.fcp_symk[1])
    
  def test_fcp_istorek(self):
    obj = _raveio.new()
    self.assertEqual(1, obj.fcp_istorek)
    obj.fcp_istorek = 2
    self.assertEqual(2, obj.fcp_istorek)
    
  def test_fcp_metablocksize(self):
    obj = _raveio.new()
    self.assertEqual(0, obj.fcp_metablocksize)
    obj.fcp_metablocksize = 1
    self.assertEqual(1, obj.fcp_metablocksize)
    
  def test_load_volume(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    self.assertNotEqual(-1, str(type(vol)).find("PolarVolumeCore"))     

  def test_load_volume_checkData(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    self.assertEqual(10, vol.getNumberOfScans())
    self.assertAlmostEqual(56.3675, vol.latitude*180.0/math.pi, 4)
    self.assertAlmostEqual(12.8544, vol.longitude*180.0/math.pi, 4)
    self.assertAlmostEqual(209, vol.height, 4)

    # Verify the scan
    scan = vol.getScan(0)
    self.assertEqual(0, scan.a1gate)
    self.assertAlmostEqual(0.5, scan.elangle*180.0/math.pi, 4)
    self.assertEqual(120, scan.nbins)
    self.assertEqual(420, scan.nrays)
    self.assertAlmostEqual(2000.0, scan.rscale, 4)
    self.assertAlmostEqual(0.0, scan.rstart, 4)

    #Inherited volume position
    self.assertAlmostEqual(56.3675, scan.latitude*180.0/math.pi, 4)
    self.assertAlmostEqual(12.8544, scan.longitude*180.0/math.pi, 4)
    self.assertAlmostEqual(209, scan.height, 4)

    # Verify the DBZH
    dbzhParam = scan.getParameter("DBZH")
    self.assertAlmostEqual(0.4, dbzhParam.gain, 4)
    self.assertAlmostEqual(-30.0, dbzhParam.offset, 4)
    self.assertAlmostEqual(255.0, dbzhParam.nodata, 4)
    self.assertAlmostEqual(0.0, dbzhParam.undetect, 4)
    self.assertEqual(120, dbzhParam.nbins, 4)
    self.assertEqual(420, dbzhParam.nrays, 4)
    self.assertEqual("DBZH", dbzhParam.quantity)

    # And verify the VRAD
    vradParam = scan.getParameter("VRADH")
    self.assertAlmostEqual(0.1875, vradParam.gain, 4)
    self.assertAlmostEqual(-24.0, vradParam.offset, 4)
    self.assertAlmostEqual(255.0, vradParam.nodata, 4)
    self.assertAlmostEqual(0.0, vradParam.undetect, 4)
    self.assertEqual("VRADH", vradParam.quantity)

    # Verify last scan
    scan = vol.getScan(9)
    self.assertEqual(0, scan.a1gate)
    self.assertAlmostEqual(40.0, scan.elangle*180.0/math.pi, 4)
    self.assertEqual(120, scan.nbins)
    self.assertEqual(420, scan.nrays)
    self.assertAlmostEqual(1000.0, scan.rscale, 4)
    self.assertAlmostEqual(0.0, scan.rstart, 4)
    #Inherited volume position
    self.assertAlmostEqual(56.3675, scan.latitude*180.0/math.pi, 4)
    self.assertAlmostEqual(12.8544, scan.longitude*180.0/math.pi, 4)
    self.assertAlmostEqual(209, scan.height, 4)  

    # Verify last scans DBZH parameter
    dbzhParam = scan.getParameter("DBZH")
    self.assertAlmostEqual(0.4, dbzhParam.gain, 4)
    self.assertAlmostEqual(-30.0, dbzhParam.offset, 4)
    self.assertAlmostEqual(255.0, dbzhParam.nodata, 4)
    self.assertAlmostEqual(0.0, dbzhParam.undetect, 4)
    self.assertEqual("DBZH", dbzhParam.quantity)
    self.assertEqual(120, dbzhParam.nbins)
    self.assertEqual(420, dbzhParam.nrays)

    # Verify last scans VRAD parameter
    vradParam = scan.getParameter("VRADH")
    self.assertAlmostEqual(0.375, vradParam.gain, 4)
    self.assertAlmostEqual(-48.0, vradParam.offset, 4)
    self.assertAlmostEqual(255.0, vradParam.nodata, 4)
    self.assertAlmostEqual(0.0, vradParam.undetect, 4)
    self.assertEqual("VRADH", vradParam.quantity)

  def test_save_cartesian(self):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI

    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    dbzhqfield = _ravefield.new()
    dbzhqfield.addAttribute("what/dbzhqfield", "a quality field")
    dbzhqfield.setData(numpy.zeros((240,240), numpy.uint8))
    param.addQualityField(dbzhqfield)    
    image.addParameter(param)

    param = _cartesianparam.new()
    param.quantity = "MMH"
    param.gain = 2.0
    param.offset = 1.0
    param.nodata = 254.0
    param.undetect = 3.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    image.addParameter(param)
    
    qfield1 = _ravefield.new()
    qfield1.addAttribute("what/sthis", "a quality field")
    qfield1.setData(numpy.zeros((240,240), numpy.uint8))
    qfield2 = _ravefield.new()
    qfield2.addAttribute("what/sthat", "another quality field")
    qfield2.setData(numpy.zeros((240,240), numpy.uint8))
    
    image.addQualityField(qfield1)
    image.addQualityField(qfield2)
    
    ios = _raveio.new()
    ios.object = image
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("ODIM_H5/V2_2", nodelist.getNode("/Conventions").data())
    # What
    self.assertEqual("100000", nodelist.getNode("/what/time").data())
    self.assertEqual("20100101", nodelist.getNode("/what/date").data())
    self.assertEqual("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEqual("IMAGE", nodelist.getNode("/what/object").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    
    #Where
    self.assertEqual("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", nodelist.getNode("/where/projdef").data())
    self.assertEqual(240, nodelist.getNode("/where/xsize").data())
    self.assertEqual(240, nodelist.getNode("/where/ysize").data())
    self.assertAlmostEqual(2000.0, nodelist.getNode("/where/xscale").data(), 4)
    self.assertAlmostEqual(2000.0, nodelist.getNode("/where/yscale").data(), 4)

    self.assertAlmostEqual(9.1714, nodelist.getNode("/where/LL_lon").data(), 4)
    self.assertAlmostEqual(54.1539, nodelist.getNode("/where/LL_lat").data(), 4)
    self.assertAlmostEqual(8.73067, nodelist.getNode("/where/UL_lon").data(), 4)
    self.assertAlmostEqual(58.45867, nodelist.getNode("/where/UL_lat").data(), 4)
    self.assertAlmostEqual(16.9781, nodelist.getNode("/where/UR_lon").data(), 4)
    self.assertAlmostEqual(58.45867, nodelist.getNode("/where/UR_lat").data(), 4)
    self.assertAlmostEqual(16.5374, nodelist.getNode("/where/LR_lon").data(), 4)
    self.assertAlmostEqual(54.1539, nodelist.getNode("/where/LR_lat").data(), 4)

    #dataset1
    self.assertEqual("100000", nodelist.getNode("/dataset1/what/starttime").data())
    self.assertEqual("20100101", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEqual("100000", nodelist.getNode("/dataset1/what/endtime").data())
    self.assertEqual("20100101", nodelist.getNode("/dataset1/what/enddate").data())

    dbzhdata = 1
    mmhdata = 2
    if nodelist.getNode("/dataset1/data1/what/quantity").data() == "MMH":
      dbzhdata = 2
      mmhdata = 1
    
    self.assertEqual("DBZH", nodelist.getNode("/dataset1/data%d/what/quantity"%dbzhdata).data())
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data%d/what/gain"%dbzhdata).data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data%d/what/offset"%dbzhdata).data(), 4)
    self.assertAlmostEqual(255.0, nodelist.getNode("/dataset1/data%d/what/nodata"%dbzhdata).data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data%d/what/undetect"%dbzhdata).data(), 4)
    self.assertEqual("a quality field", nodelist.getNode("/dataset1/data%d/quality1/what/dbzhqfield"%dbzhdata).data())
    d = nodelist.getNode("/dataset1/data%d/quality1/data"%dbzhdata).data()
    self.assertTrue(240, numpy.shape(d)[0])
    self.assertTrue(240, numpy.shape(d)[0])

    self.assertEqual("MMH", nodelist.getNode("/dataset1/data%d/what/quantity"%mmhdata).data())
    self.assertAlmostEqual(2.0, nodelist.getNode("/dataset1/data%d/what/gain"%mmhdata).data(), 4)
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data%d/what/offset"%mmhdata).data(), 4)
    self.assertAlmostEqual(254.0, nodelist.getNode("/dataset1/data%d/what/nodata"%mmhdata).data(), 4)
    self.assertAlmostEqual(3.0, nodelist.getNode("/dataset1/data%d/what/undetect"%mmhdata).data(), 4)
    
    self.assertEqual(numpy.uint8, nodelist.getNode("/dataset1/data1/data").data().dtype)
    self.assertEqual("IMAGE", nodelist.getNode("/dataset1/data1/data/CLASS").data())
    self.assertEqual("1.2", nodelist.getNode("/dataset1/data1/data/IMAGE_VERSION").data())

    #quality information
    self.assertEqual("a quality field", nodelist.getNode("/dataset1/quality1/what/sthis").data())
    d = nodelist.getNode("/dataset1/quality1/data").data()
    self.assertTrue(d is not None)
    self.assertEqual(240, numpy.shape(d)[0])
    self.assertEqual(240, numpy.shape(d)[1])

    self.assertEqual("another quality field", nodelist.getNode("/dataset1/quality2/what/sthat").data())
    d = nodelist.getNode("/dataset1/quality2/data").data()
    self.assertTrue(d is not None)
    self.assertEqual(240, numpy.shape(d)[0])
    self.assertEqual(240, numpy.shape(d)[1])


  def test_save_cartesian_SURF(self):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.product = _rave.Rave_ProductType_SURF
    image.source = "PLC:123"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 238000.0, 238000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")

    param = _cartesianparam.new()
    param.quantity = "PROB"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    image.addParameter(param)
    
    ios = _raveio.new()
    ios.object = image
    ios.filename = self.TEMPORARY_FILE
    _rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)

    ios.save()

    _rave.setDebugLevel(_rave.Debug_RAVE_SILENT)
    
    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    # Assume that rest is working as expected according to full cartesian test, now we only want to know that
    # SURF can be written and read
    self.assertEqual("SURF", nodelist.getNode("/dataset1/what/product").data())

    robj = _raveio.open(self.TEMPORARY_FILE).object
    self.assertEqual(True, _cartesian.isCartesian(robj))
    self.assertEqual(_rave.Rave_ProductType_SURF, robj.product)
    

  def test_save_cartesian_startandstoptime(self):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.starttime = "110000"
    image.startdate = "20110101"
    image.endtime = "110005"
    image.enddate = "20110101"

    param = _cartesianparam.new()    
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    image.addParameter(param)
    
    ios = _raveio.new()
    ios.object = image
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("ODIM_H5/V2_2", nodelist.getNode("/Conventions").data())
    # What
    self.assertEqual("100000", nodelist.getNode("/what/time").data())
    self.assertEqual("20100101", nodelist.getNode("/what/date").data())
    self.assertEqual("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEqual("IMAGE", nodelist.getNode("/what/object").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    
    #Where
    self.assertEqual("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", nodelist.getNode("/where/projdef").data())
    self.assertEqual(240, nodelist.getNode("/where/xsize").data())
    self.assertEqual(240, nodelist.getNode("/where/ysize").data())
    self.assertAlmostEqual(2000.0, nodelist.getNode("/where/xscale").data(), 4)
    self.assertAlmostEqual(2000.0, nodelist.getNode("/where/yscale").data(), 4)

    self.assertAlmostEqual(9.1714, nodelist.getNode("/where/LL_lon").data(), 4)
    self.assertAlmostEqual(54.1539, nodelist.getNode("/where/LL_lat").data(), 4)
    self.assertAlmostEqual(8.73067, nodelist.getNode("/where/UL_lon").data(), 4)
    self.assertAlmostEqual(58.45867, nodelist.getNode("/where/UL_lat").data(), 4)
    self.assertAlmostEqual(16.9781, nodelist.getNode("/where/UR_lon").data(), 4)
    self.assertAlmostEqual(58.45867, nodelist.getNode("/where/UR_lat").data(), 4)
    self.assertAlmostEqual(16.5374, nodelist.getNode("/where/LR_lon").data(), 4)
    self.assertAlmostEqual(54.1539, nodelist.getNode("/where/LR_lat").data(), 4)

    #dataset1
    self.assertEqual("DBZH", nodelist.getNode("/dataset1/data1/what/quantity").data())
    self.assertEqual("110000", nodelist.getNode("/dataset1/what/starttime").data())
    self.assertEqual("20110101", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEqual("110005", nodelist.getNode("/dataset1/what/endtime").data())
    self.assertEqual("20110101", nodelist.getNode("/dataset1/what/enddate").data())
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEqual(255.0, nodelist.getNode("/dataset1/data1/what/nodata").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/undetect").data(), 4)
    
    self.assertEqual(numpy.uint8, nodelist.getNode("/dataset1/data1/data").data().dtype)
    self.assertEqual("IMAGE", nodelist.getNode("/dataset1/data1/data/CLASS").data())
    self.assertEqual("1.2", nodelist.getNode("/dataset1/data1/data/IMAGE_VERSION").data())

  def test_save_cartesian_attribute_visibility(self):
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
    image.starttime = "110000"
    image.startdate = "20110101"
    image.endtime = "110005"
    image.enddate = "20110101"

    param = _cartesianparam.new()    
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    image.addParameter(param)

    param.addAttribute("how/something", 1.0)
    image.addAttribute("how/else", 2.0)
    
    ios = _raveio.new()
    ios.object = image
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/how/something").data(), 4)
    self.assertTrue("/how/something" not in nodelist.getNodeNames())
    self.assertAlmostEqual(2.0, nodelist.getNode("/how/else").data(), 4)
    self.assertTrue("/dataset1/data1/how/else" not in nodelist.getNodeNames())


  def test_save_cartesian_volume(self):
    cvol = _cartesianvolume.new()
    cvol.time = "100000"
    cvol.date = "20091010"
    cvol.objectType = _rave.Rave_ObjectType_CVOL
    cvol.source = "PLC:123"
    cvol.xscale = 2000.0
    cvol.yscale = 2000.0
    cvol.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    cvol.projection = projection

    image = _cartesian.new()
    image.product = _rave.Rave_ProductType_CAPPI
    
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    image.addParameter(param)
    
    cvol.addImage(image)

    image = _cartesian.new()
    image.product = _rave.Rave_ProductType_CAPPI
    image.starttime = "110000"
    image.startdate = "20110101"
    image.endtime = "110005"
    image.enddate = "20110101"

    param = _cartesianparam.new()
    param.quantity = "MMH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    image.addParameter(param)

    qfield1 = _ravefield.new()
    qfield1.addAttribute("what/sthis", "a quality field")
    qfield1.setData(numpy.zeros((240,240), numpy.uint8))
    qfield2 = _ravefield.new()
    qfield2.addAttribute("what/sthat", "another quality field")
    qfield2.setData(numpy.zeros((240,240), numpy.uint8))
    
    image.addQualityField(qfield1)
    image.addQualityField(qfield2)

    cvol.addImage(image)

    ios = _raveio.new()
    ios.object = cvol
    ios.filename = self.TEMPORARY_FILE
    ios.save()
    
    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("ODIM_H5/V2_2", nodelist.getNode("/Conventions").data())
    # What
    self.assertEqual("100000", nodelist.getNode("/what/time").data())
    self.assertEqual("20091010", nodelist.getNode("/what/date").data())
    self.assertEqual("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEqual("CVOL", nodelist.getNode("/what/object").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    
    #Where
    self.assertEqual("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", nodelist.getNode("/where/projdef").data())
    self.assertEqual(240, nodelist.getNode("/where/xsize").data())
    self.assertEqual(240, nodelist.getNode("/where/ysize").data())
    self.assertAlmostEqual(2000.0, nodelist.getNode("/where/xscale").data(), 4)
    self.assertAlmostEqual(2000.0, nodelist.getNode("/where/yscale").data(), 4)

    self.assertAlmostEqual(9.1714, nodelist.getNode("/where/LL_lon").data(), 4)
    self.assertAlmostEqual(54.1539, nodelist.getNode("/where/LL_lat").data(), 4)
    self.assertAlmostEqual(8.73067, nodelist.getNode("/where/UL_lon").data(), 4)
    self.assertAlmostEqual(58.45867, nodelist.getNode("/where/UL_lat").data(), 4)
    self.assertAlmostEqual(16.9781, nodelist.getNode("/where/UR_lon").data(), 4)
    self.assertAlmostEqual(58.45867, nodelist.getNode("/where/UR_lat").data(), 4)
    self.assertAlmostEqual(16.5374, nodelist.getNode("/where/LR_lon").data(), 4)
    self.assertAlmostEqual(54.1539, nodelist.getNode("/where/LR_lat").data(), 4)

    #dataset1
    self.assertEqual("100000", nodelist.getNode("/dataset1/what/starttime").data())
    self.assertEqual("20091010", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEqual("100000", nodelist.getNode("/dataset1/what/endtime").data())
    self.assertEqual("20091010", nodelist.getNode("/dataset1/what/enddate").data())

    self.assertEqual("DBZH", nodelist.getNode("/dataset1/data1/what/quantity").data())
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEqual(255.0, nodelist.getNode("/dataset1/data1/what/nodata").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/undetect").data(), 4)
    
    self.assertEqual(numpy.uint8, nodelist.getNode("/dataset1/data1/data").data().dtype)
    self.assertEqual("IMAGE", nodelist.getNode("/dataset1/data1/data/CLASS").data())
    self.assertEqual("1.2", nodelist.getNode("/dataset1/data1/data/IMAGE_VERSION").data())

    #dataset2
    self.assertEqual("110000", nodelist.getNode("/dataset2/what/starttime").data())
    self.assertEqual("20110101", nodelist.getNode("/dataset2/what/startdate").data())
    self.assertEqual("110005", nodelist.getNode("/dataset2/what/endtime").data())
    self.assertEqual("20110101", nodelist.getNode("/dataset2/what/enddate").data())

    self.assertEqual("MMH", nodelist.getNode("/dataset2/data1/what/quantity").data())
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset2/data1/what/gain").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset2/data1/what/offset").data(), 4)
    self.assertAlmostEqual(255.0, nodelist.getNode("/dataset2/data1/what/nodata").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset2/data1/what/undetect").data(), 4)
    
    self.assertEqual(numpy.uint8, nodelist.getNode("/dataset2/data1/data").data().dtype)
    self.assertEqual("IMAGE", nodelist.getNode("/dataset2/data1/data/CLASS").data())
    self.assertEqual("1.2", nodelist.getNode("/dataset2/data1/data/IMAGE_VERSION").data())

    #quality information
    self.assertEqual("a quality field", nodelist.getNode("/dataset2/quality1/what/sthis").data())
    d = nodelist.getNode("/dataset2/quality1/data").data()
    self.assertTrue(d is not None)
    self.assertEqual(240, numpy.shape(d)[0])
    self.assertEqual(240, numpy.shape(d)[1])


    self.assertEqual("another quality field", nodelist.getNode("/dataset2/quality2/what/sthat").data())
    d = nodelist.getNode("/dataset2/quality2/data").data()
    self.assertTrue(d is not None)
    self.assertEqual(240, numpy.shape(d)[0])
    self.assertEqual(240, numpy.shape(d)[1])
  
  def test_load_cartesian_volume(self):
    obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_0, obj.version)
    self.assertEqual(_raveio.RaveIO_ODIM_H5rad_Version_2_0, obj.h5radversion)
    self.assertEqual(_rave.Rave_ObjectType_CVOL, obj.objectType)
    
    cvol = obj.object
    self.assertEqual(_rave.Rave_ObjectType_CVOL, cvol.objectType)
    self.assertEqual("100000", cvol.time)
    self.assertEqual("20091010", cvol.date)
    self.assertEqual("PLC:123", cvol.source)
    self.assertEqual("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", cvol.projection.definition)
    self.assertAlmostEqual(-240000.0, cvol.areaextent[0], 4)
    self.assertAlmostEqual(-240000.0, cvol.areaextent[1], 4)
    self.assertAlmostEqual(238000.0, cvol.areaextent[2], 4)  # Since AE should be projected(UR) - xscale
    self.assertAlmostEqual(238000.0, cvol.areaextent[3], 4)  # Since AE should be projected(UR) - yscale
    self.assertAlmostEqual(2000.0, cvol.xscale, 4)
    self.assertAlmostEqual(2000.0, cvol.yscale, 4)

    self.assertEqual(1, cvol.getNumberOfImages())

    image = cvol.getImage(0)
    self.assertEqual(_rave.Rave_ProductType_CAPPI, image.product)
    self.assertEqual(240, image.xsize)
    self.assertEqual(240, image.ysize)

    param = image.getParameter("DBZH")
    self.assertEqual(numpy.uint8, param.getData().dtype)
    self.assertEqual("DBZH", param.quantity)
    self.assertAlmostEqual(1.0, param.gain, 4)
    self.assertAlmostEqual(0.0, param.offset, 4)
    self.assertAlmostEqual(255.0, param.nodata, 4)
    self.assertAlmostEqual(0.0, param.undetect, 4)
    self.assertEqual(240, param.xsize)
    self.assertEqual(240, param.ysize)

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
    
    self.assertEqual("ODIM_H5/V2_2", nodelist.getNode("/Conventions").data())

    # What
    self.assertEqual("100000", nodelist.getNode("/what/time").data())
    self.assertEqual("20091010", nodelist.getNode("/what/date").data())
    self.assertEqual("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEqual("IMAGE", nodelist.getNode("/what/object").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    
    #Where
    self.assertEqual("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", nodelist.getNode("/where/projdef").data())
    self.assertEqual(240, nodelist.getNode("/where/xsize").data())
    self.assertEqual(240, nodelist.getNode("/where/ysize").data())
    self.assertAlmostEqual(2000.0, nodelist.getNode("/where/xscale").data(), 4)
    self.assertAlmostEqual(2000.0, nodelist.getNode("/where/yscale").data(), 4)

    self.assertAlmostEqual(9.1714, nodelist.getNode("/where/LL_lon").data(), 4)
    self.assertAlmostEqual(54.1539, nodelist.getNode("/where/LL_lat").data(), 4)
    self.assertAlmostEqual(8.7327, nodelist.getNode("/where/UL_lon").data(), 4)
    self.assertAlmostEqual(58.4408, nodelist.getNode("/where/UL_lat").data(), 4)
    self.assertAlmostEqual(16.9418, nodelist.getNode("/where/UR_lon").data(), 4)
    self.assertAlmostEqual(58.4419, nodelist.getNode("/where/UR_lat").data(), 4)
    self.assertAlmostEqual(16.5068, nodelist.getNode("/where/LR_lon").data(), 4)
    self.assertAlmostEqual(54.1549, nodelist.getNode("/where/LR_lat").data(), 4)

    #dataset1
    self.assertEqual("DBZH", nodelist.getNode("/dataset1/data1/what/quantity").data())
    self.assertEqual("100000", nodelist.getNode("/dataset1/what/starttime").data())
    self.assertEqual("20091010", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEqual("100000", nodelist.getNode("/dataset1/what/endtime").data())
    self.assertEqual("20091010", nodelist.getNode("/dataset1/what/enddate").data())
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEqual(255.0, nodelist.getNode("/dataset1/data1/what/nodata").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/undetect").data(), 4)
    
    self.assertEqual(numpy.uint8, nodelist.getNode("/dataset1/data1/data").data().dtype)
    self.assertEqual("IMAGE", nodelist.getNode("/dataset1/data1/data/CLASS").data())
    self.assertEqual("1.2", nodelist.getNode("/dataset1/data1/data/IMAGE_VERSION").data())

  def test_load_cartesian_image2(self):
    obj = _raveio.open(self.FIXTURE_CARTESIAN_IMAGE)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_1, obj.version)
    self.assertEqual(_raveio.RaveIO_ODIM_H5rad_Version_2_1, obj.h5radversion)
    self.assertEqual(_rave.Rave_ObjectType_IMAGE, obj.objectType)
    
    image = obj.object
    self.assertEqual(_rave.Rave_ObjectType_IMAGE, image.objectType)
    self.assertEqual(_rave.Rave_ProductType_CAPPI, image.product)
    self.assertEqual("100000", image.time)
    self.assertEqual("20100101", image.date)
    self.assertEqual("PLC:123", image.source)
    self.assertEqual("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", image.projection.definition)
    self.assertAlmostEqual(-240000.0, image.areaextent[0], 4)
    self.assertAlmostEqual(-240000.0, image.areaextent[1], 4)
    self.assertAlmostEqual(240000.0, image.areaextent[2], 4)  # Since AE should be projected(UR) - xscale
    self.assertAlmostEqual(240000.0, image.areaextent[3], 4)  # Since AE should be projected(UR) - yscale
    self.assertAlmostEqual(2000.0, image.xscale, 4)
    self.assertAlmostEqual(2000.0, image.yscale, 4)
    self.assertEqual(240, image.xsize)
    self.assertEqual(240, image.ysize)

    param = image.getParameter("DBZH")
    self.assertEqual("DBZH", param.quantity)
    self.assertAlmostEqual(1.0, param.gain, 4)
    self.assertAlmostEqual(0.0, param.offset, 4)
    self.assertAlmostEqual(255.0, param.nodata, 4)
    self.assertAlmostEqual(0.0, param.undetect, 4)
    self.assertEqual(numpy.uint8, param.getData().dtype)

    self.assertEqual(2, image.getNumberOfQualityFields())
    qf = image.getQualityField(0)
    qf2 = image.getQualityField(1)
    self.assertEqual("a quality field", qf.getAttribute("what/sthis"))
    qfd = qf.getData()
    self.assertEqual(240, numpy.shape(qfd)[0])
    self.assertEqual(240, numpy.shape(qfd)[1])
    self.assertEqual("another quality field", qf2.getAttribute("what/sthat"))
    qf2d = qf2.getData()
    self.assertEqual(240, numpy.shape(qf2d)[0])
    self.assertEqual(240, numpy.shape(qf2d)[1])
 
  def test_load_cartesian_volume2(self):
    obj = _raveio.open(self.FIXTURE_CARTESIAN_VOLUME)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_1, obj.version)
    self.assertEqual(_raveio.RaveIO_ODIM_H5rad_Version_2_1, obj.h5radversion)
    self.assertEqual(_rave.Rave_ObjectType_CVOL, obj.objectType)
    
    cvol = obj.object
    self.assertEqual("100000", cvol.time)
    self.assertEqual("20091010", cvol.date)
    self.assertEqual(_rave.Rave_ObjectType_CVOL, cvol.objectType)
    self.assertEqual("PLC:123", cvol.source)
    self.assertAlmostEqual(2000.0, cvol.xscale, 4)
    self.assertAlmostEqual(2000.0, cvol.yscale, 4)
    self.assertEqual("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84", cvol.projection.definition)
    self.assertAlmostEqual(-240000.0, cvol.areaextent[0], 4)
    self.assertAlmostEqual(-240000.0, cvol.areaextent[1], 4)
    self.assertAlmostEqual(240000.0, cvol.areaextent[2], 4)  # Since AE should be projected(UR) - xscale
    self.assertAlmostEqual(240000.0, cvol.areaextent[3], 4)  # Since AE should be projected(UR) - yscale

    self.assertEqual(2, cvol.getNumberOfImages())

    image = cvol.getImage(0)
    self.assertEqual(_rave.Rave_ProductType_CAPPI, image.product)
    self.assertEqual(240, image.xsize)
    self.assertEqual(240, image.ysize)
    
    param = image.getParameter("DBZH")
    self.assertEqual(numpy.uint8, param.getData().dtype)
    self.assertAlmostEqual(1.0, param.gain, 4)
    self.assertAlmostEqual(0.0, param.offset, 4)
    self.assertAlmostEqual(255.0, param.nodata, 4)
    self.assertAlmostEqual(0.0, param.undetect, 4)
    self.assertEqual(240, numpy.shape(param.getData())[0])
    self.assertEqual(240, numpy.shape(param.getData())[1])

    image = cvol.getImage(1)
    self.assertEqual(_rave.Rave_ProductType_CAPPI, image.product)
    self.assertEqual(240, image.xsize)
    self.assertEqual(240, image.ysize)
    self.assertEqual("110000", image.starttime)
    self.assertEqual("20110101", image.startdate)
    self.assertEqual("110005", image.endtime)
    self.assertEqual("20110101", image.enddate)

    param = image.getParameter("MMH")
    self.assertAlmostEqual(1.0, param.gain, 4)
    self.assertAlmostEqual(0.0, param.offset, 4)
    self.assertAlmostEqual(255.0, param.nodata, 4)
    self.assertAlmostEqual(0.0, param.undetect, 4)
    self.assertEqual(numpy.uint8, param.getData().dtype)
    self.assertEqual(240, numpy.shape(param.getData())[0])
    self.assertEqual(240, numpy.shape(param.getData())[1])
    
    self.assertEqual(2, image.getNumberOfQualityFields())
    qf = image.getQualityField(0)
    qf2 = image.getQualityField(1)
    self.assertEqual("a quality field", qf.getAttribute("what/sthis"))
    qfd = qf.getData()
    self.assertEqual(240, numpy.shape(qfd)[0])
    self.assertEqual(240, numpy.shape(qfd)[1])
    self.assertEqual("another quality field", qf2.getAttribute("what/sthat"))
    qf2d = qf2.getData()
    self.assertEqual(240, numpy.shape(qf2d)[0])
    self.assertEqual(240, numpy.shape(qf2d)[1])


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

    qfield = _ravefield.new()
    qfield.addAttribute("what/sthis", "a quality field")
    qfield.setData(numpy.zeros((100,120), numpy.uint8))
    mmhParam.addQualityField(qfield)

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
    
    self.assertEqual("ODIM_H5/V2_2", nodelist.getNode("/Conventions").data())
    # What
    self.assertEqual("100000", nodelist.getNode("/what/time").data())
    self.assertEqual("20091010", nodelist.getNode("/what/date").data())
    self.assertEqual("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEqual("PVOL", nodelist.getNode("/what/object").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    
    #Where
    self.assertAlmostEqual(12.0, nodelist.getNode("/where/lon").data(), 4)
    self.assertAlmostEqual(60.0, nodelist.getNode("/where/lat").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/where/height").data(), 4)

    #
    # dataset1 (scan1)
    #
    self.assertEqual("100001", nodelist.getNode("/dataset1/what/starttime").data())
    self.assertEqual("20091010", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEqual("100001", nodelist.getNode("/dataset1/what/endtime").data())
    self.assertEqual("20091010", nodelist.getNode("/dataset1/what/enddate").data())
    self.assertEqual("SCAN", nodelist.getNode("/dataset1/what/product").data())
    
    # dataset1/where
    self.assertAlmostEqual(0.1, nodelist.getNode("/dataset1/where/elangle").data(), 4)
    self.assertEqual(2, nodelist.getNode("/dataset1/where/a1gate").data())
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/where/rstart").data(), 4)
    self.assertAlmostEqual(5000.0, nodelist.getNode("/dataset1/where/rscale").data(), 4)
    self.assertEqual(120, nodelist.getNode("/dataset1/where/nbins").data())
    self.assertEqual(100, nodelist.getNode("/dataset1/where/nrays").data())
    
    # Verify that both DBZH and MMH has been stored properly.
    d1field = nodelist.getNode("/dataset1/data1/what/quantity").data()
    d2field = nodelist.getNode("/dataset1/data2/what/quantity").data()
    dbzhname = "/dataset1/data1"
    mmhname = "/dataset1/data2"
    if d1field == "MMH":
      dbzhname = "/dataset1/data2"
      mmhname = "/dataset1/data1"
    
    # dbzh field
    self.assertEqual("DBZH", nodelist.getNode(dbzhname + "/what/quantity").data())
    self.assertAlmostEqual(1.0, nodelist.getNode(dbzhname + "/what/gain").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode(dbzhname + "/what/offset").data(), 4)
    self.assertAlmostEqual(10.0, nodelist.getNode(dbzhname + "/what/nodata").data(), 4)
    self.assertAlmostEqual(11.0, nodelist.getNode(dbzhname + "/what/undetect").data(), 4)
    
    # 
    self.assertEqual(numpy.uint8, nodelist.getNode(dbzhname + "/data").data().dtype)
    self.assertEqual("IMAGE", nodelist.getNode(dbzhname + "/data/CLASS").data())
    self.assertEqual("1.2", nodelist.getNode(dbzhname + "/data/IMAGE_VERSION").data())

    # mmh field
    self.assertEqual("MMH", nodelist.getNode(mmhname + "/what/quantity").data())
    self.assertAlmostEqual(10.0, nodelist.getNode(mmhname + "/what/gain").data(), 4)
    self.assertAlmostEqual(20.0, nodelist.getNode(mmhname + "/what/offset").data(), 4)
    self.assertAlmostEqual(12.0, nodelist.getNode(mmhname + "/what/nodata").data(), 4)
    self.assertAlmostEqual(13.0, nodelist.getNode(mmhname + "/what/undetect").data(), 4)
    
    # dataset1/data2/data
    self.assertEqual(numpy.int16, nodelist.getNode(mmhname + "/data").data().dtype)

    # quality field for mmh
    self.assertEqual("a quality field", nodelist.getNode(mmhname + "/quality1/what/sthis").data())
    self.assertTrue(nodelist.getNode(mmhname + "/quality1/data").data() is not None)
    
    #
    # dataset2 (scan2)
    #
    self.assertEqual("100002", nodelist.getNode("/dataset2/what/starttime").data())
    self.assertEqual("20091010", nodelist.getNode("/dataset2/what/startdate").data())
    self.assertEqual("100002", nodelist.getNode("/dataset2/what/endtime").data())
    self.assertEqual("20091010", nodelist.getNode("/dataset2/what/enddate").data())
    self.assertEqual("SCAN", nodelist.getNode("/dataset2/what/product").data())
    
    # dataset2/where
    self.assertAlmostEqual(0.5, nodelist.getNode("/dataset2/where/elangle").data(), 4)
    self.assertEqual(1, nodelist.getNode("/dataset2/where/a1gate").data())
    self.assertAlmostEqual(1000.0, nodelist.getNode("/dataset2/where/rstart").data(), 4)
    self.assertAlmostEqual(2000.0, nodelist.getNode("/dataset2/where/rscale").data(), 4)
    self.assertEqual(120, nodelist.getNode("/dataset2/where/nbins").data())
    self.assertEqual(100, nodelist.getNode("/dataset2/where/nrays").data())
    
    # dataset2/data1/what
    self.assertEqual("MMM", nodelist.getNode("/dataset2/data1/what/quantity").data())
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset2/data1/what/gain").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset2/data1/what/offset").data(), 4)
    self.assertAlmostEqual(255.0, nodelist.getNode("/dataset2/data1/what/nodata").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset2/data1/what/undetect").data(), 4)
    
    # dataset2/data1/data
    self.assertEqual(numpy.uint8, nodelist.getNode("/dataset2/data1/data").data().dtype)
    self.assertEqual("IMAGE", nodelist.getNode("/dataset2/data1/data/CLASS").data())
    self.assertEqual("1.2", nodelist.getNode("/dataset2/data1/data/IMAGE_VERSION").data())

  def test_save_polar_volume_beamwidths(self):
    obj = _polarvolume.new()
    obj.time = "100000"
    obj.date = "20091010"
    obj.source = "PLC:123"
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    obj.beamwidth = 2.0 * math.pi/180.0
    
    scan1 = _polarscan.new()
    scan1.beamwidth = 3.0 * math.pi/180
    scan2 = _polarscan.new()
    scan2.beamwidth = 4.0 * math.pi/180
    scan3 = _polarscan.new()
    scan4 = _polarscan.new()

    obj.addScan(scan1)
    obj.addScan(scan2)
    obj.addScan(scan3)
    obj.addScan(scan4)
    
    ios = _raveio.new()
    ios.object = obj
    ios.filename = self.TEMPORARY_FILE
    ios.save()
    
    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertAlmostEqual(2.0, nodelist.getNode("/how/beamwidth").data())
    self.assertAlmostEqual(3.0, nodelist.getNode("/dataset1/how/beamwidth").data())
    self.assertAlmostEqual(4.0, nodelist.getNode("/dataset2/how/beamwidth").data())

    nodenames = nodelist.getNodeNames()
    self.assertTrue("/dataset3/how/beamwidth" not in nodenames)
    self.assertTrue("/dataset4/how/beamwidth" not in nodenames)

  # (RT: Ticket 8)
  def test_loadCartesian_differentXYSize(self):
    src = _cartesian.new()
    src.time = "100000"
    src.date = "20091010"
    src.objectType = _rave.Rave_ObjectType_IMAGE
    src.product = _rave.Rave_ProductType_COMP
    src.source = "PLC:123"
    src.xscale = 2000.0
    src.yscale = 2000.0
    src.areaextent = (-240000.0, -240000.0, 238000.0, 238000.0)
    projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    src.projection = projection

    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    param.setData(numpy.zeros((100,90),numpy.int16))

    src.addParameter(param)    

    ios = _raveio.new()
    ios.object = src
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertEqual(_rave.Rave_ObjectType_IMAGE, obj.object.objectType);

  def test_load_scan(self):
    obj = _raveio.open(self.FIXTURE_SCAN)
    self.assertNotEqual(-1, str(type(obj.object)).find("PolarScanCore"))
    scan = obj.object

    self.assertAlmostEqual(40.0, scan.elangle*180.0/math.pi, 4)

    self.assertEqual("20100702", scan.date)
    self.assertEqual("113200", scan.time)
    self.assertEqual("WMO:02570,RAD:SE48,PLC:Vilebo", scan.source)
    self.assertAlmostEqual(222, scan.height, 4)
    self.assertAlmostEqual(58.106, scan.latitude*180.0/math.pi, 4)
    self.assertAlmostEqual(15.94, scan.longitude*180.0/math.pi, 4)
    
    p1 = scan.getParameter("DBZH")
    self.assertAlmostEqual(0.4, p1.gain, 4)
    self.assertAlmostEqual(-30.0, p1.offset, 4)
    
    p2 = scan.getParameter("VRADH")
    self.assertAlmostEqual(0.375, p2.gain, 4)
    self.assertAlmostEqual(-48.0, p2.offset, 4)
  
  def test_write_scan(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    scan = vol.getScan(0)
    
    obj = _raveio.new()
    obj.object = scan
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("120000", nodelist.getNode("/what/time").data())
    self.assertEqual("20090501", nodelist.getNode("/what/date").data())
    self.assertEqual("WMO:02606,RAD:SE50", nodelist.getNode("/what/source").data())
    self.assertEqual("SCAN", nodelist.getNode("/what/object").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    self.assertAlmostEqual(209.0, nodelist.getNode("/where/height").data(), 4)
    self.assertAlmostEqual(12.8544, nodelist.getNode("/where/lon").data(), 4)
    self.assertAlmostEqual(56.3675, nodelist.getNode("/where/lat").data(), 4)
    
    self.assertEqual("20090501", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEqual("120021", nodelist.getNode("/dataset1/what/starttime").data())
    self.assertEqual("20090501", nodelist.getNode("/dataset1/what/enddate").data())
    self.assertEqual("120051", nodelist.getNode("/dataset1/what/endtime").data())
    self.assertEqual("SCAN", nodelist.getNode("/dataset1/what/product").data())

    self.assertEqual(0, nodelist.getNode("/dataset1/where/a1gate").data())
    self.assertAlmostEqual(0.5, nodelist.getNode("/dataset1/where/elangle").data())
    self.assertEqual(120, nodelist.getNode("/dataset1/where/nbins").data())
    self.assertEqual(420, nodelist.getNode("/dataset1/where/nrays").data())
    self.assertAlmostEqual(2000.0, nodelist.getNode("/dataset1/where/rscale").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/where/rstart").data(), 4)

  def test_write_scan_with_quality(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    scan = vol.getScan(0)
    
    field = _ravefield.new()
    field.addAttribute("what/strvalue", "a string")
    field.addAttribute("where/lonvalue", 123)
    field.addAttribute("how/flovalue", 1.25)
    field.setData(numpy.zeros((10,10), numpy.uint8))
    scan.addQualityField(field)
    
    p1 = scan.getParameter("DBZH")
    p1field = _ravefield.new()
    p1field.addAttribute("what/pstrvalue", "str")
    p1field.addAttribute("where/plonvalue", 321)
    p1field.addAttribute("how/pflovalue", 23.0)
    p1field.setData(numpy.zeros((10,10), numpy.uint8))
    p1.addQualityField(p1field)
    
    obj = _raveio.new()
    obj.object = scan
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("a string", nodelist.getNode("/dataset1/quality1/what/strvalue").data())
    self.assertEqual(123, nodelist.getNode("/dataset1/quality1/where/lonvalue").data())
    self.assertAlmostEqual(1.25, nodelist.getNode("/dataset1/quality1/how/flovalue").data(), 4)

    data = nodelist.getNode("/dataset1/quality1/data").data()
    self.assertEqual(data.shape[0], 10)
    self.assertEqual(data.shape[1], 10)

    self.assertEqual("str", nodelist.getNode("/dataset1/data1/quality1/what/pstrvalue").data())
    self.assertEqual(321, nodelist.getNode("/dataset1/data1/quality1/where/plonvalue").data())
    self.assertAlmostEqual(23.0, nodelist.getNode("/dataset1/data1/quality1/how/pflovalue").data(), 4)

    data = nodelist.getNode("/dataset1/data1/quality1/data").data()
    self.assertEqual(data.shape[0], 10)
    self.assertEqual(data.shape[1], 10)

  def test_read_write_cartesian_image(self):
    image = _cartesian.new()
    image.product = _rave.Rave_ProductType_CAPPI
    image.source = "PLC:1234"
    image.date = "20100810"
    image.time = "085500"
    image.xscale = 1000.0
    image.yscale = 1000.0
    image.areaextent = (-240000.0, -240000.0, 238000.0, 238000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")

    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    image.addParameter(param)
    
    obj = _raveio.new()
    obj.object = image
    obj.filename = self.TEMPORARY_FILE
    obj.save()
    
    obj = _raveio.open(self.TEMPORARY_FILE)
    obj.filename = self.TEMPORARY_FILE2
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("DBZH", nodelist.getNode("/dataset1/data1/what/quantity").data())
    self.assertEqual("IMAGE", nodelist.getNode("/what/object").data())
  
  def test_read_write_scan(self):
    obj = _raveio.open(self.FIXTURE_SCAN)
    obj.filename = self.TEMPORARY_FILE2
    obj.save()
    
    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("SCAN", nodelist.getNode("/what/object").data())

  def test_save_filename_scan(self):
    obj = _raveio.open(self.FIXTURE_SCAN)
    obj.filename = self.TEMPORARY_FILE
    obj.save()
    
    obj = _raveio.open(self.TEMPORARY_FILE)
    obj.save(self.TEMPORARY_FILE2)
    
    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("SCAN", nodelist.getNode("/what/object").data())

  def test_save_nofilename_scan(self):
    obj = _raveio.open(self.FIXTURE_SCAN)
    obj.filename = self.TEMPORARY_FILE
    obj.save()
    
    obj = _raveio.open(self.TEMPORARY_FILE)
    obj.filename = self.TEMPORARY_FILE2
    obj.save()
    
    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("SCAN", nodelist.getNode("/what/object").data())
  
  def test_save_scan_from_volume_check_metadata(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    scan = vol.getScan(0)
    
    obj.object = scan
    obj.filename = self.TEMPORARY_FILE2
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()
    
    nodenames = list(nodelist.getNodeNames().keys());
    VALID_NAMES=["/Conventions", "/what","/what/date","/what/object","/what/source","/what/time",
                 "/what/version","/where","/where/height","/where/lat","/where/lon","/how","/how/beamwidth","/dataset1",
                 "/dataset1/data1","/dataset1/data1/data","/dataset1/data1/data/CLASS","/dataset1/data1/data/IMAGE_VERSION",
                 "/dataset1/data1/what","/dataset1/data1/what/gain","/dataset1/data1/what/nodata","/dataset1/data1/what/offset",
                 "/dataset1/data1/what/quantity","/dataset1/data1/what/undetect","/dataset1/data2","/dataset1/data2/data",
                 "/dataset1/data2/data/CLASS","/dataset1/data2/data/IMAGE_VERSION","/dataset1/data2/what","/dataset1/data2/what/gain",
                 "/dataset1/data2/what/nodata","/dataset1/data2/what/offset","/dataset1/data2/what/quantity",
                 "/dataset1/data2/what/undetect","/dataset1/what","/dataset1/what/enddate","/dataset1/what/endtime",
                 "/dataset1/what/product","/dataset1/what/startdate","/dataset1/what/starttime","/dataset1/where",
                 "/dataset1/where/a1gate","/dataset1/where/elangle","/dataset1/where/nbins","/dataset1/where/nrays",
                 "/dataset1/where/rscale","/dataset1/where/rstart"]
      
    for name in VALID_NAMES:
      self.assertTrue(name in nodenames)
      nodenames.remove(name)
      
    self.assertEqual("20090501", nodelist.getNode("/what/date").data())
    self.assertEqual("SCAN", nodelist.getNode("/what/object").data())
    self.assertEqual("WMO:02606,RAD:SE50", nodelist.getNode("/what/source").data())
    self.assertEqual("120000", nodelist.getNode("/what/time").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    self.assertAlmostEqual(209, nodelist.getNode("/where/height").data(), 4)
    self.assertAlmostEqual(56.3675, nodelist.getNode("/where/lat").data(), 4)
    self.assertAlmostEqual(12.8544, nodelist.getNode("/where/lon").data(), 4)

    self.assertEqual("20090501", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEqual("120021", nodelist.getNode("/dataset1/what/starttime").data())
    self.assertEqual("SCAN", nodelist.getNode("/dataset1/what/product").data())

    self.assertEqual("20090501", nodelist.getNode("/dataset1/what/enddate").data())
    self.assertEqual("120051", nodelist.getNode("/dataset1/what/endtime").data())

    self.assertAlmostEqual(0, nodelist.getNode("/dataset1/where/a1gate").data(), 4)
    self.assertAlmostEqual(0.5, nodelist.getNode("/dataset1/where/elangle").data(), 4)
    self.assertEqual(120, nodelist.getNode("/dataset1/where/nbins").data())
    self.assertEqual(420, nodelist.getNode("/dataset1/where/nrays").data())
    self.assertAlmostEqual(2000, nodelist.getNode("/dataset1/where/rscale").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/where/rstart").data(), 4)

    self.assertAlmostEqual(0.4, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)
    self.assertAlmostEqual(255, nodelist.getNode("/dataset1/data1/what/nodata").data(), 4)
    self.assertAlmostEqual(-30, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertEqual("DBZH", nodelist.getNode("/dataset1/data1/what/quantity").data())
    self.assertAlmostEqual(0, nodelist.getNode("/dataset1/data1/what/undetect").data(), 4)

    self.assertAlmostEqual(0.1875, nodelist.getNode("/dataset1/data2/what/gain").data(), 4)
    self.assertAlmostEqual(255, nodelist.getNode("/dataset1/data2/what/nodata").data(), 4)
    self.assertAlmostEqual(-24, nodelist.getNode("/dataset1/data2/what/offset").data(), 4)
    self.assertEqual("VRADH", nodelist.getNode("/dataset1/data2/what/quantity").data())
    self.assertAlmostEqual(0, nodelist.getNode("/dataset1/data2/what/undetect").data(), 4)
  
  def test_read_arrays_from_scan(self):
    expected = [0.0109863, 1.01624, 2.02148, 2.99927, 4.0155, 5.03174, 6.00403, \
                7.00928, 8.02002, 8.9978, 10.0085, 11.0248, 11.9971, 12.9968, \
                14.0021, 15.0128, 16.0236, 16.9958, 18.0066, 19.0063, 20.0171, \
                21.0004, 22.0111, 23.0164, 24.0271, 24.9994, 26.0156, 27.0264, \
                27.9987, 29.0039, 30.0092, 31.0144, 32.0197, 32.9974, 34.0082, \
                35.0189, 36.0242, 36.9965, 38.0017, 39.007, 40.0177, 41.0284, \
                42.0007, 43.006, 44.0167, 44.9945, 45.9998, 47.0105, 48.0212, \
                49.032, 49.9988, 51.004, 52.0203, 53.0035, 54.0198, 55.0195, \
                56.0138, 57.0245, 58.0188, 59.0021, 60.0128, 61.0236, 61.9958, \
                63.0231, 64.0118, 64.9951, 66.0168, 66.9946, 67.9999, 68.9996, \
                70.0214, 71.0046, 72.0209, 72.9987, 74.0094, 75.0037, 75.9979, \
                77.0087, 78.0249, 78.9972, 80.0079, 81.0187, 81.9965, 83.0072, \
                84.0179, 84.9957, 86.0065, 87.0007, 88.0115, 89.0277, 90.011, \
                91.0162, 91.9995, 92.9993, 94.0155, 95.0208, 95.9985, 96.9983, \
                98.0145, 99.0308, 100.003, 101.008, 102.019, 103.024, 104.002, \
                105.013, 106.018, 107.023, 108.023, 109.034, 110.006, 111.022, \
                111.995, 113.027, 114, 115.01, 115.999, 117.004, 118.015, 119.02, \
                120.026, 120.998, 122.009, 123.008, 123.997, 124.997, 126.002, \
                127.013, 128.024, 128.996, 130.007, 131.001, 132.012, 133.028, \
                133.995, 135.005, 136.016, 137, 138.005, 139.01, 140.015, 141.031, \
                142.004, 143.015, 144.02, 145.025, 146.003, 147.008, 148.019, \
                148.997, 150.013, 151.007, 151.996, 153.001, 154.012, 155.028, 156, \
                157.006, 158.016, 159.027, 159.999, 161.005, 162.021, 162.999, \
                164.004, 165.009, 166.025, 167.003, 168.014, 169.019, 170.024, \
                170.997, 172.007, 172.996, 174.012, 175.023, 175.995, 177.028, 178, \
                179.006, 180.016, 181.027, 182, 183.005, 184.016, 185.032, 186.01, \
                187.015, 187.998, 188.998, 190.009, 191.025, 192.003, 193.008, \
                194.013, 195.013, 196.024, 197.001, 198.023, 199.001, 200.017, \
                200.001, 201.017, 202.017, 203.027, 204.027, 204.999, 206.016, \
                207.026, 208.004, 209.004, 210.004, 211.02, 211.998, 213.003, \
                214.008, 215.019, 216.019, 217.035, 218.007, 219.007, 220.018, \
                221.028, 222.001, 223.017, 224.017, 224.995, 226.005, 227.005, \
                228.016, 229.026, 229.999, 231.01, 232.015, 233.02, 233.998, \
                235.009, 235.997, 236.997, 238.002, 239.008, 240.013, 241.024, \
                242.001, 243.012, 244.023, 245.001, 246.006, 247.006, 248.022, \
                249.005, 250.016, 251.027, 252.021, 252.999, 254.015, 255.026, \
                256.003, 256.998, 258.014, 259.025, 260.008, 261.03, 262.002, \
                262.996, 264.001, 265.007, 265.995, 267.001, 268.006, 269.011, \
                270.022, 271, 272.021, 273.027, 273.999, 275.004, 276.021, 276.998, \
                278.009, 279.009, 280.014, 281.03, 282.003, 283.008, 284.019, \
                284.996, 286.007, 287.018, 288.023, 289.001, 290.017, 291.028, \
                291.995, 293, 294.016, 295.027, 295.999, 296.999, 298.01, 299.026, \
                300.004, 301.009, 302.02, 303.03, 303.997, 305.002, 306.013, \
                307.024, 307.996, 309.007, 310.018, 311.028, 312.001, 313, 314.011, \
                315.027, 316, 317.005, 318.01, 319.026, 320.004, 321.021, 321.998, \
                323.004, 324.014, 325.014, 326.019, 327.003, 328.019, 328.997, \
                330.007, 331.002, 332.012, 333.023, 334.001, 335.006, 335.995, \
                337.022, 338.027, 339.005, 340.016, 341.032, 342.004, 343.004, \
                344.015, 344.998, 346.02, 346.998, 348.008, 349.003, 350.008, \
                351.008, 351.996, 353.013, 354.018, 355.023, 355.995, 357.012, \
                357.995, 359]
    obj = _raveio.open(self.FIXTURE_SCAN_WITH_ARRAYS)
    scan = obj.object
    attr = scan.getAttribute("how/startazA")
    self.assertTrue(isinstance(attr, numpy.ndarray))
    self.assertEqual(len(attr), len(expected))
    for i in range(len(expected)):
      self.assertAlmostEqual(attr[i], expected[i], 2)
  
  def test_write_scan_with_array(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    scan = vol.getScan(0)

    scan.addAttribute("how/alongarray", numpy.arange(10).astype(numpy.int32))
    scan.addAttribute("how/adoublearray", numpy.arange(10).astype(numpy.float32))
    
    obj = _raveio.new()
    obj.object = scan
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    ldata = nodelist.getNode("/how/alongarray").data()
    ddata = nodelist.getNode("/how/adoublearray").data()
    
    self.assertEqual(1, ldata[1])
    self.assertEqual(5, ldata[5])
    self.assertAlmostEqual(1.0, ddata[1], 2)
    self.assertAlmostEqual(5.0, ddata[5], 2)

  def test_write_scanparam_with_array(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    scan = vol.getScan(0)
    param = scan.getParameter("DBZH")

    param.addAttribute("how/alongarray", numpy.arange(10).astype(numpy.int32))
    param.addAttribute("how/adoublearray", numpy.arange(10).astype(numpy.float32))
    
    obj = _raveio.new()
    obj.object = scan
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    ldata = nodelist.getNode("/dataset1/data1/how/alongarray").data()
    ddata = nodelist.getNode("/dataset1/data1/how/adoublearray").data()
    
    self.assertEqual(1, ldata[1])
    self.assertEqual(5, ldata[5])
    self.assertAlmostEqual(1.0, ddata[1], 2)
    self.assertAlmostEqual(5.0, ddata[5], 2)

  def test_read_scanparam_with_array(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    scan = vol.getScan(0)
    param = scan.getParameter("DBZH")

    param.addAttribute("how/alongarray", numpy.arange(10).astype(numpy.int32))
    param.addAttribute("how/adoublearray", numpy.arange(10).astype(numpy.float32))
    
    obj = _raveio.new()
    obj.object = scan
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    obj = _raveio.open(self.TEMPORARY_FILE)
    scan = obj.object
    param = scan.getParameter("DBZH")    

    ldata = param.getAttribute("how/alongarray")
    ddata = param.getAttribute("how/adoublearray")
    
    self.assertEqual(10, len(ldata))
    self.assertTrue(isinstance(ldata, numpy.ndarray))
    self.assertEqual(1, ldata[1])
    self.assertEqual(5, ldata[5])
    self.assertEqual(10, len(ddata))
    self.assertTrue(isinstance(ddata, numpy.ndarray))
    self.assertAlmostEqual(1.0, ddata[1], 2)
    self.assertAlmostEqual(5.0, ddata[5], 2)

  def test_write_volume_with_array(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object

    vol.addAttribute("how/alongarray", numpy.arange(10).astype(numpy.int32))
    vol.addAttribute("how/adoublearray", numpy.arange(10).astype(numpy.float32))
    
    obj = _raveio.new()
    obj.object = vol
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    ldata = nodelist.getNode("/how/alongarray").data()
    ddata = nodelist.getNode("/how/adoublearray").data()
    
    self.assertEqual(1, ldata[1])
    self.assertEqual(5, ldata[5])
    self.assertAlmostEqual(1.0, ddata[1], 2)
    self.assertAlmostEqual(5.0, ddata[5], 2)

  def test_read_volume_with_array(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object

    vol.addAttribute("how/alongarray", numpy.arange(10).astype(numpy.int32))
    vol.addAttribute("how/adoublearray", numpy.arange(10).astype(numpy.float32))
    
    obj = _raveio.new()
    obj.object = vol
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    obj = _raveio.open(self.TEMPORARY_FILE)
    vol = obj.object
    
    ldata = vol.getAttribute("how/alongarray")
    ddata = vol.getAttribute("how/adoublearray")
    
    self.assertEqual(10, len(ldata))
    self.assertTrue(isinstance(ldata, numpy.ndarray))
    self.assertEqual(1, ldata[1])
    self.assertEqual(5, ldata[5])
    self.assertEqual(10, len(ddata))
    self.assertTrue(isinstance(ddata, numpy.ndarray))
    self.assertAlmostEqual(1.0, ddata[1], 2)
    self.assertAlmostEqual(5.0, ddata[5], 2)

  def test_write_cartesian_with_array(self):
    obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
    cvol = obj.object
    img = cvol.getImage(0)

    img.addAttribute("how/alongarray", numpy.arange(10).astype(numpy.int32))
    img.addAttribute("how/adoublearray", numpy.arange(10).astype(numpy.float32))
    
    obj = _raveio.new()
    obj.object = img
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    ldata = nodelist.getNode("/how/alongarray").data()
    ddata = nodelist.getNode("/how/adoublearray").data()
    
    self.assertEqual(1, ldata[1])
    self.assertEqual(5, ldata[5])
    self.assertAlmostEqual(1.0, ddata[1], 2)
    self.assertAlmostEqual(5.0, ddata[5], 2)

  def test_read_cartesian_with_array(self):
    obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
    cvol = obj.object
    img = cvol.getImage(0)

    img.addAttribute("how/alongarray", numpy.arange(10).astype(numpy.int32))
    img.addAttribute("how/adoublearray", numpy.arange(10).astype(numpy.float32))
    
    obj = _raveio.new()
    obj.object = img
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    obj = _raveio.open(self.TEMPORARY_FILE)
    img = obj.object
    
    ldata = img.getAttribute("how/alongarray")
    ddata = img.getAttribute("how/adoublearray")
    
    self.assertEqual(10, len(ldata))
    self.assertTrue(isinstance(ldata, numpy.ndarray))
    self.assertEqual(1, ldata[1])
    self.assertEqual(5, ldata[5])
    self.assertEqual(10, len(ddata))
    self.assertTrue(isinstance(ddata, numpy.ndarray))
    self.assertAlmostEqual(1.0, ddata[1], 2)
    self.assertAlmostEqual(5.0, ddata[5], 2)

  def test_write_cartesianvolume_with_array(self):
    obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
    cvol = obj.object

    cvol.addAttribute("how/alongarray", numpy.arange(10).astype(numpy.int32))
    cvol.addAttribute("how/adoublearray", numpy.arange(10).astype(numpy.float32))
    
    obj = _raveio.new()
    obj.object = cvol
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    ldata = nodelist.getNode("/how/alongarray").data()
    ddata = nodelist.getNode("/how/adoublearray").data()
    
    self.assertEqual(1, ldata[1])
    self.assertEqual(5, ldata[5])
    self.assertAlmostEqual(1.0, ddata[1], 2)
    self.assertAlmostEqual(5.0, ddata[5], 2)
   
  def test_read_cartesianvolume_with_array(self):
    obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
    cvol = obj.object

    cvol.addAttribute("how/alongarray", numpy.arange(10).astype(numpy.int32))
    cvol.addAttribute("how/adoublearray", numpy.arange(10).astype(numpy.float32))
    
    obj = _raveio.new()
    obj.object = cvol
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    obj = _raveio.open(self.TEMPORARY_FILE)
    cvol = obj.object
    
    ldata = cvol.getAttribute("how/alongarray")
    ddata = cvol.getAttribute("how/adoublearray")
    
    self.assertEqual(10, len(ldata))
    self.assertTrue(isinstance(ldata, numpy.ndarray))
    self.assertEqual(1, ldata[1])
    self.assertEqual(5, ldata[5])
    self.assertEqual(10, len(ddata))
    self.assertTrue(isinstance(ddata, numpy.ndarray))
    self.assertAlmostEqual(1.0, ddata[1], 2)
    self.assertAlmostEqual(5.0, ddata[5], 2)
  
  def test_read_vp(self):
    # Read VP
    vp = _raveio.open(self.FIXTURE_VP).object
    self.assertEqual("PLC:1234", vp.source)
    self.assertEqual("20100101", vp.date)
    self.assertEqual("111500", vp.time)
    self.assertAlmostEqual(10.0, vp.longitude * 180.0 / math.pi, 4)
    self.assertAlmostEqual(15.0, vp.latitude * 180.0 / math.pi, 4)
    self.assertAlmostEqual(200.0, vp.height, 4)
    self.assertEqual(10, vp.getLevels())
    self.assertAlmostEqual(5.0, vp.interval, 4)
    self.assertAlmostEqual(10.0, vp.minheight, 4)
    self.assertAlmostEqual(20.0, vp.maxheight, 4)
    
    field = vp.getField("ff")
    self.assertEqual("ff", field.getAttribute("what/quantity"))
    data = field.getData()
    self.assertEqual(10, numpy.shape(data)[0])
    self.assertEqual(1, numpy.shape(data)[1])
  
  def test_write_vp(self):
    vp = _verticalprofile.new()
    vp.date="20100101"
    vp.time="120000"
    vp.source="PLC:1234"
    vp.longitude = 10.0 * math.pi / 180.0
    vp.latitude = 15.0 * math.pi / 180.0
    vp.setLevels(10)
    vp.height = 100.0
    vp.interval = 5.0
    vp.minheight = 10.0
    vp.maxheight = 20.0
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((10,1), numpy.uint8))
    f1.addAttribute("what/quantity", "ff")
    vp.addField(f1)
    f2 = _ravefield.new()
    f2.setData(numpy.zeros((10,1), numpy.uint8))
    f2.addAttribute("what/quantity", "ff_dev")
    vp.addField(f2)

    obj = _raveio.new()
    obj.object = vp
    obj.filename = self.TEMPORARY_FILE2
    obj.save()
    
    # Verify written data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("20100101", nodelist.getNode("/what/date").data())
    self.assertEqual("VP", nodelist.getNode("/what/object").data())
    self.assertEqual("PLC:1234", nodelist.getNode("/what/source").data())
    self.assertEqual("120000", nodelist.getNode("/what/time").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    
    self.assertAlmostEqual(100.0, nodelist.getNode("/where/height").data(), 4)
    self.assertAlmostEqual(15.0, nodelist.getNode("/where/lat").data(), 4)
    self.assertAlmostEqual(10.0, nodelist.getNode("/where/lon").data(), 4)

    self.assertEqual(10, nodelist.getNode("/where/levels").data())
    self.assertAlmostEqual(5.0, nodelist.getNode("/where/interval").data(), 4)
    self.assertAlmostEqual(10.0, nodelist.getNode("/where/minheight").data(), 4)
    self.assertAlmostEqual(20.0, nodelist.getNode("/where/maxheight").data(), 4)
    
    f1 = nodelist.getNode("/dataset1/data1/what/quantity").data()
    f1data = nodelist.getNode("/dataset1/data1/data").data()
    f2 = nodelist.getNode("/dataset1/data2/what/quantity").data()
    f2data = nodelist.getNode("/dataset1/data2/data").data() 

    if f1 == "ff":
      self.assertEqual("ff_dev", f2)
    elif f1 == "ff_dev":
      self.assertEqual("ff", f2)
  
    self.assertEqual(10, numpy.shape(f1data)[0])
    self.assertEqual(1, numpy.shape(f1data)[1])
    self.assertEqual(10, numpy.shape(f2data)[0])
    self.assertEqual(1, numpy.shape(f2data)[1])

  def test_write_vp_dev_bird(self):
    vp = _verticalprofile.new()
    vp.date="20100101"
    vp.time="120000"
    vp.source="PLC:1234"
    vp.longitude = 10.0 * math.pi / 180.0
    vp.latitude = 15.0 * math.pi / 180.0
    vp.setLevels(10)
    vp.height = 100.0
    vp.interval = 5.0
    vp.minheight = 10.0
    vp.maxheight = 20.0
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((10,1), numpy.uint8))
    f1.addAttribute("what/quantity", "dev_bird")
    vp.addField(f1)
    f2 = _ravefield.new()
    f2.setData(numpy.zeros((10,1), numpy.uint8))
    f2.addAttribute("what/quantity", "ff_dev")
    vp.addField(f2)

    obj = _raveio.new()
    obj.object = vp
    obj.filename = self.TEMPORARY_FILE2
    obj.save()
    
    # Verify written data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("20100101", nodelist.getNode("/what/date").data())
    self.assertEqual("VP", nodelist.getNode("/what/object").data())
    self.assertEqual("PLC:1234", nodelist.getNode("/what/source").data())
    self.assertEqual("120000", nodelist.getNode("/what/time").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    
    self.assertAlmostEqual(100.0, nodelist.getNode("/where/height").data(), 4)
    self.assertAlmostEqual(15.0, nodelist.getNode("/where/lat").data(), 4)
    self.assertAlmostEqual(10.0, nodelist.getNode("/where/lon").data(), 4)

    self.assertEqual(10, nodelist.getNode("/where/levels").data())
    self.assertAlmostEqual(5.0, nodelist.getNode("/where/interval").data(), 4)
    self.assertAlmostEqual(10.0, nodelist.getNode("/where/minheight").data(), 4)
    self.assertAlmostEqual(20.0, nodelist.getNode("/where/maxheight").data(), 4)
    
    f1 = nodelist.getNode("/dataset1/data1/what/quantity").data()
    f1data = nodelist.getNode("/dataset1/data1/data").data()
    f2 = nodelist.getNode("/dataset1/data2/what/quantity").data()
    f2data = nodelist.getNode("/dataset1/data2/data").data() 

    if f1 == "dev_bird":
      self.assertEqual("ff_dev", f2)
    elif f1 == "ff_dev":
      self.assertEqual("dev_bird", f2)
  
    self.assertEqual(10, numpy.shape(f1data)[0])
    self.assertEqual(1, numpy.shape(f1data)[1])
    self.assertEqual(10, numpy.shape(f2data)[0])
    self.assertEqual(1, numpy.shape(f2data)[1])
    
  def test_read_vp_new_version(self):
    # Read the new version of VP
    vp = _raveio.open(self.FIXTURE_VP_NEW_VERSION).object
    self.assertEqual("NOD:selek,WMO:02430,RAD:SE45,PLC:Leksand", vp.source)
    self.assertEqual("20170901", vp.date)
    self.assertEqual("20170901", vp.startdate)
    self.assertEqual("000000", vp.time)
    self.assertEqual("000315", vp.starttime)
    self.assertAlmostEqual(14.8775997162, vp.longitude * 180.0 / math.pi, 4)
    self.assertAlmostEqual(60.7229995728, vp.latitude * 180.0 / math.pi, 4)
    self.assertAlmostEqual(457.0, vp.height, 4)
    self.assertEqual(60, vp.getLevels())
    self.assertAlmostEqual(200.0, vp.interval, 4)
    self.assertAlmostEqual(0.0, vp.minheight, 4)
    self.assertAlmostEqual(12000.0, vp.maxheight, 4)
    
    field = vp.getField("HGHT")
    self.assertEqual("HGHT", field.getAttribute("what/quantity"))
    data = field.getData()
    self.assertEqual(60, numpy.shape(data)[0])
    self.assertEqual(1, numpy.shape(data)[1])
    
  def test_write_vp_new_version(self):
    vp = _verticalprofile.new()
    vp.date="20100101"
    vp.startdate="20100101"
    vp.enddate="20100101"
    vp.time="120000"
    vp.starttime="120202"
    vp.endtime="120405"
    vp.source="PLC:Leksand"
    vp.product= "VP"
    vp.longitude = 10.0 * math.pi / 180.0
    vp.latitude = 15.0 * math.pi / 180.0
    vp.setLevels(10)
    vp.height = 100.0
    vp.interval = 5.0
    vp.minheight = 10.0
    vp.maxheight = 20.0
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((10,1), numpy.uint8))
    f1.addAttribute("what/quantity", "UWND")
    vp.addField(f1)
    f2 = _ravefield.new()
    f2.setData(numpy.zeros((10,1), numpy.uint8))
    f2.addAttribute("what/quantity", "VWND")
    vp.addField(f2)

    obj = _raveio.new()
    obj.object = vp
    obj.filename = self.TEMPORARY_FILE2
    obj.save()
    
    # Verify written data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertEqual("20100101", nodelist.getNode("/what/date").data())
    self.assertEqual("20100101", nodelist.getNode("/dataset1/what/startdate").data())
    self.assertEqual("20100101", nodelist.getNode("/dataset1/what/enddate").data())
    self.assertEqual("120202", nodelist.getNode("/dataset1/what/starttime").data())
    self.assertEqual("120405", nodelist.getNode("/dataset1/what/endtime").data())
    self.assertEqual("VP", nodelist.getNode("/what/object").data())
    self.assertEqual("VP", nodelist.getNode("/dataset1/what/product").data())
    self.assertEqual("PLC:Leksand", nodelist.getNode("/what/source").data())
    self.assertEqual("120000", nodelist.getNode("/what/time").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    
    self.assertAlmostEqual(100.0, nodelist.getNode("/where/height").data(), 4)
    self.assertAlmostEqual(15.0, nodelist.getNode("/where/lat").data(), 4)
    self.assertAlmostEqual(10.0, nodelist.getNode("/where/lon").data(), 4)

    self.assertEqual(10, nodelist.getNode("/where/levels").data())
    self.assertAlmostEqual(5.0, nodelist.getNode("/where/interval").data(), 4)
    self.assertAlmostEqual(10.0, nodelist.getNode("/where/minheight").data(), 4)
    self.assertAlmostEqual(20.0, nodelist.getNode("/where/maxheight").data(), 4)
    
    f1 = nodelist.getNode("/dataset1/data1/what/quantity").data()
    f1data = nodelist.getNode("/dataset1/data1/data").data()
    f2 = nodelist.getNode("/dataset1/data2/what/quantity").data()
    f2data = nodelist.getNode("/dataset1/data2/data").data() 

    if f1 == "UWND":
      self.assertEqual("VWND", f2)
    elif f1 == "VWND":
      self.assertEqual("UWND", f2)
  
    self.assertEqual(10, numpy.shape(f1data)[0])
    self.assertEqual(1, numpy.shape(f1data)[1])
    self.assertEqual(10, numpy.shape(f2data)[0])
    self.assertEqual(1, numpy.shape(f2data)[1])

  def test_read_vp_new_version_with_extra_how_attribs(self):
    # Read the new version of VP having 2 extra how attributes
    vp = _raveio.open(self.FIXTURE_VP_NEW_VERSION_EXTRA_HOW).object
    self.assertEqual("WMO:02430,RAD:SE45,PLC:Leksand,NOD:selek,ORG:82,CTY:643,CMT:Swedish radar", vp.source)
    self.assertEqual("20191003", vp.date)
    self.assertEqual("20191003", vp.startdate)
    self.assertEqual("060500", vp.time)
    self.assertEqual("060752", vp.starttime)
    self.assertAlmostEqual(14.877571105957031, vp.longitude * 180.0 / math.pi, 4)
    self.assertAlmostEqual(60.72304153442384, vp.latitude * 180.0 / math.pi, 4)
    self.assertAlmostEqual(457.0, vp.height, 4)
    self.assertEqual(60, vp.getLevels())
    self.assertAlmostEqual(200.0, vp.interval, 4)
    self.assertAlmostEqual(0.0, vp.minheight, 4)
    self.assertAlmostEqual(12000.0, vp.maxheight, 4)

    field = vp.getField("UWND")
    self.assertEqual("UWND", field.getAttribute("what/quantity"))
    data = field.getData()
    self.assertEqual(60, numpy.shape(data)[0])
    self.assertEqual(1, numpy.shape(data)[1])

    # Verify the two extra attribs under /how
    nodelist = _pyhl.read_nodelist(self.FIXTURE_VP_NEW_VERSION_EXTRA_HOW)
    nodelist.selectAll()
    nodelist.fetch()

    angles = nodelist.getNode("/how/angles").data()
    task = nodelist.getNode("/how/task").data()
    self.assertEqual("4.0,8.0,14.0", angles)
    self.assertEqual("lek_zdr", task)
  
  def testReadBadlyFormattedODIM(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/what")
    self.addGroupNode(nodelist, "/where")
    self.addGroupNode(nodelist, "/dataset1")
    self.addGroupNode(nodelist, "/dataset1/what")
    self.addGroupNode(nodelist, "/dataset1/data1")
    self.addGroupNode(nodelist, "/dataset1/data1/what")
    
    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_2")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "SCAN")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.2")
    
    self.addAttributeNode(nodelist, "/where/height", "double", 100.0)
    self.addAttributeNode(nodelist, "/where/lon", "double", 13.5)
    self.addAttributeNode(nodelist, "/where/lat", "double", 61.0)
    
    self.addAttributeNode(nodelist, "/dataset1/what/startdate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset1/what/starttime", "string", "-") #BAD FORMAT
    self.addAttributeNode(nodelist, "/dataset1/what/enddate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset1/what/endtime", "string", "-") #BAD FORMAT
    self.addAttributeNode(nodelist, "/dataset1/what/product", "string", "SCAN")
    
    self.addAttributeNode(nodelist, "/dataset1/data1/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "DBZH")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    
    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,10), dset)
    
    nodelist.write(self.TEMPORARY_FILE, 6)

    try:
      obj = _raveio.open(self.TEMPORARY_FILE)
      self.fail("Expected IOError")
    except IOError:
      pass

  def testReadAndConvertV21_SCAN(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/what")
    self.addGroupNode(nodelist, "/where")
    self.addGroupNode(nodelist, "/how")
    
    self.addGroupNode(nodelist, "/dataset1")
    self.addGroupNode(nodelist, "/dataset1/what")
    self.addGroupNode(nodelist, "/dataset1/data1")
    self.addGroupNode(nodelist, "/dataset1/data1/how")
    self.addGroupNode(nodelist, "/dataset1/data1/what")
    
    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_1")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "SCAN")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.1")
    
    self.addAttributeNode(nodelist, "/where/height", "double", 100.0)
    self.addAttributeNode(nodelist, "/where/lon", "double", 13.5)
    self.addAttributeNode(nodelist, "/where/lat", "double", 61.0)

    self.addAttributeNode(nodelist, "/how/TXloss", "double", 1.1)
    self.addAttributeNode(nodelist, "/how/injectloss", "double", 1.2)  # ??
    self.addAttributeNode(nodelist, "/how/RXloss", "double", 1.3)
    self.addAttributeNode(nodelist, "/how/radomeloss", "double", 1.4)
    self.addAttributeNode(nodelist, "/how/antgain", "double", 1.5)
    self.addAttributeNode(nodelist, "/how/beamw", "double", 1.6) #??
    self.addAttributeNode(nodelist, "/how/radconst", "double", 1.7) #??
    self.addAttributeNode(nodelist, "/how/NEZ", "double", 1.8)
    self.addAttributeNode(nodelist, "/how/zcal", "double", 1.9)
    self.addAttributeNode(nodelist, "/how/nsample", "double", 2.0) #??
     
    self.addAttributeNode(nodelist, "/dataset1/what/startdate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset1/what/starttime", "string", "101010")
    self.addAttributeNode(nodelist, "/dataset1/what/enddate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset1/what/endtime", "string", "101011")
    self.addAttributeNode(nodelist, "/dataset1/what/product", "string", "SCAN")

    self.addAttributeNode(nodelist, "/dataset1/data1/how/TXloss", "double", 3.1)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/injectloss", "double", 3.2)  # ??
    self.addAttributeNode(nodelist, "/dataset1/data1/how/RXloss", "double", 3.3)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/radomeloss", "double", 3.4)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/antgain", "double", 3.5)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/beamw", "double", 3.6) #??
    self.addAttributeNode(nodelist, "/dataset1/data1/how/radconst", "double", 3.7) #??
    self.addAttributeNode(nodelist, "/dataset1/data1/how/NEZ", "double", 3.8)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/zcal", "double", 3.9)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nsample", "double", 4.0) #??

    self.addAttributeNode(nodelist, "/dataset1/data1/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "DBZH")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    
    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,10), dset)
    
    nodelist.write(self.TEMPORARY_FILE, 6)

    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertAlmostEqual(1.1, obj.object.getAttribute("how/TXlossH"), 2)
    self.assertAlmostEqual(1.2, obj.object.getAttribute("how/injectlossH"), 2)
    self.assertAlmostEqual(1.3, obj.object.getAttribute("how/RXlossH"), 2)
    self.assertAlmostEqual(1.4, obj.object.getAttribute("how/radomelossH"), 2)
    self.assertAlmostEqual(1.5, obj.object.getAttribute("how/antgainH"), 2)
    self.assertAlmostEqual(1.6, obj.object.getAttribute("how/beamwH"), 2)
    self.assertAlmostEqual(1.7, obj.object.getAttribute("how/radconstH"), 2)
    self.assertAlmostEqual(1.8, obj.object.getAttribute("how/NEZH"), 2)
    self.assertAlmostEqual(1.9, obj.object.getAttribute("how/zcalH"), 2)
    self.assertAlmostEqual(2.0, obj.object.getAttribute("how/nsampleH"), 2)
    
    self.assertAlmostEqual(3.1, obj.object.getParameter("DBZH").getAttribute("how/TXlossH"), 2)
    self.assertAlmostEqual(3.2, obj.object.getParameter("DBZH").getAttribute("how/injectlossH"), 2)
    self.assertAlmostEqual(3.3, obj.object.getParameter("DBZH").getAttribute("how/RXlossH"), 2)
    self.assertAlmostEqual(3.4, obj.object.getParameter("DBZH").getAttribute("how/radomelossH"), 2)
    self.assertAlmostEqual(3.5, obj.object.getParameter("DBZH").getAttribute("how/antgainH"), 2)
    self.assertAlmostEqual(3.6, obj.object.getParameter("DBZH").getAttribute("how/beamwH"), 2)
    self.assertAlmostEqual(3.7, obj.object.getParameter("DBZH").getAttribute("how/radconstH"), 2)
    self.assertAlmostEqual(3.8, obj.object.getParameter("DBZH").getAttribute("how/NEZH"), 2)
    self.assertAlmostEqual(3.9, obj.object.getParameter("DBZH").getAttribute("how/zcalH"), 2)
    self.assertAlmostEqual(4.0, obj.object.getParameter("DBZH").getAttribute("how/nsampleH"), 2)

  def testReadAndConvertV21_PVOL(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/what")
    self.addGroupNode(nodelist, "/where")
    self.addGroupNode(nodelist, "/how")
    
    self.addGroupNode(nodelist, "/dataset1")
    self.addGroupNode(nodelist, "/dataset1/what")
    self.addGroupNode(nodelist, "/dataset1/data1")
    self.addGroupNode(nodelist, "/dataset1/data1/how")
    self.addGroupNode(nodelist, "/dataset1/data1/what")
    self.addGroupNode(nodelist, "/dataset1/data2")
    self.addGroupNode(nodelist, "/dataset1/data2/how")
    self.addGroupNode(nodelist, "/dataset1/data2/what")
    self.addGroupNode(nodelist, "/dataset2")
    self.addGroupNode(nodelist, "/dataset2/what")
    self.addGroupNode(nodelist, "/dataset2/data1")
    self.addGroupNode(nodelist, "/dataset2/data1/how")
    self.addGroupNode(nodelist, "/dataset2/data1/what")
    
    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_1")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "PVOL")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.1")
    
    self.addAttributeNode(nodelist, "/where/height", "double", 100.0)
    self.addAttributeNode(nodelist, "/where/lon", "double", 13.5)
    self.addAttributeNode(nodelist, "/where/lat", "double", 61.0)

    self.addAttributeNode(nodelist, "/how/TXloss", "double", 1.1)
    self.addAttributeNode(nodelist, "/how/injectloss", "double", 1.2)  # ??
    self.addAttributeNode(nodelist, "/how/RXloss", "double", 1.3)
    self.addAttributeNode(nodelist, "/how/radomeloss", "double", 1.4)
    self.addAttributeNode(nodelist, "/how/antgain", "double", 1.5)
    self.addAttributeNode(nodelist, "/how/beamw", "double", 1.6) #??
    self.addAttributeNode(nodelist, "/how/radconst", "double", 1.7) #??
    self.addAttributeNode(nodelist, "/how/NEZ", "double", 1.8)
    self.addAttributeNode(nodelist, "/how/zcal", "double", 1.9)
    self.addAttributeNode(nodelist, "/how/nsample", "double", 2.0) #??
     
    self.addAttributeNode(nodelist, "/dataset1/what/startdate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset1/what/starttime", "string", "101010")
    self.addAttributeNode(nodelist, "/dataset1/what/enddate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset1/what/endtime", "string", "101011")
    self.addAttributeNode(nodelist, "/dataset1/what/product", "string", "SCAN")

    self.addAttributeNode(nodelist, "/dataset1/data1/how/TXloss", "double", 3.1)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/injectloss", "double", 3.2)  # ??
    self.addAttributeNode(nodelist, "/dataset1/data1/how/RXloss", "double", 3.3)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/radomeloss", "double", 3.4)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/antgain", "double", 3.5)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/beamw", "double", 3.6) #??
    self.addAttributeNode(nodelist, "/dataset1/data1/how/radconst", "double", 3.7) #??
    self.addAttributeNode(nodelist, "/dataset1/data1/how/NEZ", "double", 3.8)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/zcal", "double", 3.9)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nsample", "double", 4.0) #??

    self.addAttributeNode(nodelist, "/dataset1/data1/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "DBZH")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    
    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,10), dset)

    self.addAttributeNode(nodelist, "/dataset1/data2/how/TXloss", "double", 5.1)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/injectloss", "double", 5.2)  # ??
    self.addAttributeNode(nodelist, "/dataset1/data2/how/RXloss", "double", 5.3)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/radomeloss", "double", 5.4)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/antgain", "double", 5.5)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/beamw", "double", 5.6) #??
    self.addAttributeNode(nodelist, "/dataset1/data2/how/radconst", "double", 5.7) #??
    self.addAttributeNode(nodelist, "/dataset1/data2/how/NEZ", "double", 5.8)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/zcal", "double", 5.9)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/nsample", "double", 6.0) #??

    self.addAttributeNode(nodelist, "/dataset1/data2/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/quantity", "string", "VRAD")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    
    self.addDatasetNode(nodelist, "/dataset1/data2/data", "uchar", (10,10), dset)
    
    self.addAttributeNode(nodelist, "/dataset2/what/startdate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset2/what/starttime", "string", "101010")
    self.addAttributeNode(nodelist, "/dataset2/what/enddate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset2/what/endtime", "string", "101011")
    self.addAttributeNode(nodelist, "/dataset2/what/product", "string", "SCAN")

    self.addAttributeNode(nodelist, "/dataset2/data1/how/TXloss", "double", 7.1)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/injectloss", "double", 7.2)  # ??
    self.addAttributeNode(nodelist, "/dataset2/data1/how/RXloss", "double", 7.3)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/radomeloss", "double", 7.4)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/antgain", "double", 7.5)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/beamw", "double", 7.6) #??
    self.addAttributeNode(nodelist, "/dataset2/data1/how/radconst", "double", 7.7) #??
    self.addAttributeNode(nodelist, "/dataset2/data1/how/NEZ", "double", 7.8)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/zcal", "double", 7.9)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/nsample", "double", 8.0) #??

    self.addAttributeNode(nodelist, "/dataset2/data1/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset2/data1/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset2/data1/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset2/data1/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset2/data1/what/quantity", "string", "CCOR")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    
    self.addDatasetNode(nodelist, "/dataset2/data1/data", "uchar", (10,10), dset)

    nodelist.write(self.TEMPORARY_FILE, 6)

    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertAlmostEqual(1.1, obj.object.getAttribute("how/TXlossH"), 2)
    self.assertAlmostEqual(1.2, obj.object.getAttribute("how/injectlossH"), 2)
    self.assertAlmostEqual(1.3, obj.object.getAttribute("how/RXlossH"), 2)
    self.assertAlmostEqual(1.4, obj.object.getAttribute("how/radomelossH"), 2)
    self.assertAlmostEqual(1.5, obj.object.getAttribute("how/antgainH"), 2)
    self.assertAlmostEqual(1.6, obj.object.getAttribute("how/beamwH"), 2)
    self.assertAlmostEqual(1.7, obj.object.getAttribute("how/radconstH"), 2)
    self.assertAlmostEqual(1.8, obj.object.getAttribute("how/NEZH"), 2)
    self.assertAlmostEqual(1.9, obj.object.getAttribute("how/zcalH"), 2)
    self.assertAlmostEqual(2.0, obj.object.getAttribute("how/nsampleH"), 2)
    
    self.assertAlmostEqual(3.1, obj.object.getScan(0).getParameter("DBZH").getAttribute("how/TXlossH"), 2)
    self.assertAlmostEqual(3.2, obj.object.getScan(0).getParameter("DBZH").getAttribute("how/injectlossH"), 2)
    self.assertAlmostEqual(3.3, obj.object.getScan(0).getParameter("DBZH").getAttribute("how/RXlossH"), 2)
    self.assertAlmostEqual(3.4, obj.object.getScan(0).getParameter("DBZH").getAttribute("how/radomelossH"), 2)
    self.assertAlmostEqual(3.5, obj.object.getScan(0).getParameter("DBZH").getAttribute("how/antgainH"), 2)
    self.assertAlmostEqual(3.6, obj.object.getScan(0).getParameter("DBZH").getAttribute("how/beamwH"), 2)
    self.assertAlmostEqual(3.7, obj.object.getScan(0).getParameter("DBZH").getAttribute("how/radconstH"), 2)
    self.assertAlmostEqual(3.8, obj.object.getScan(0).getParameter("DBZH").getAttribute("how/NEZH"), 2)
    self.assertAlmostEqual(3.9, obj.object.getScan(0).getParameter("DBZH").getAttribute("how/zcalH"), 2)
    self.assertAlmostEqual(4.0, obj.object.getScan(0).getParameter("DBZH").getAttribute("how/nsampleH"), 2)

    self.assertAlmostEqual(5.1, obj.object.getScan(0).getParameter("VRADH").getAttribute("how/TXlossH"), 2)
    self.assertAlmostEqual(5.2, obj.object.getScan(0).getParameter("VRADH").getAttribute("how/injectlossH"), 2)
    self.assertAlmostEqual(5.3, obj.object.getScan(0).getParameter("VRADH").getAttribute("how/RXlossH"), 2)
    self.assertAlmostEqual(5.4, obj.object.getScan(0).getParameter("VRADH").getAttribute("how/radomelossH"), 2)
    self.assertAlmostEqual(5.5, obj.object.getScan(0).getParameter("VRADH").getAttribute("how/antgainH"), 2)
    self.assertAlmostEqual(5.6, obj.object.getScan(0).getParameter("VRADH").getAttribute("how/beamwH"), 2)
    self.assertAlmostEqual(5.7, obj.object.getScan(0).getParameter("VRADH").getAttribute("how/radconstH"), 2)
    self.assertAlmostEqual(5.8, obj.object.getScan(0).getParameter("VRADH").getAttribute("how/NEZH"), 2)
    self.assertAlmostEqual(5.9, obj.object.getScan(0).getParameter("VRADH").getAttribute("how/zcalH"), 2)
    self.assertAlmostEqual(6.0, obj.object.getScan(0).getParameter("VRADH").getAttribute("how/nsampleH"), 2)

    self.assertAlmostEqual(7.1, obj.object.getScan(1).getParameter("CCORH").getAttribute("how/TXlossH"), 2)
    self.assertAlmostEqual(7.2, obj.object.getScan(1).getParameter("CCORH").getAttribute("how/injectlossH"), 2)
    self.assertAlmostEqual(7.3, obj.object.getScan(1).getParameter("CCORH").getAttribute("how/RXlossH"), 2)
    self.assertAlmostEqual(7.4, obj.object.getScan(1).getParameter("CCORH").getAttribute("how/radomelossH"), 2)
    self.assertAlmostEqual(7.5, obj.object.getScan(1).getParameter("CCORH").getAttribute("how/antgainH"), 2)
    self.assertAlmostEqual(7.6, obj.object.getScan(1).getParameter("CCORH").getAttribute("how/beamwH"), 2)
    self.assertAlmostEqual(7.7, obj.object.getScan(1).getParameter("CCORH").getAttribute("how/radconstH"), 2)
    self.assertAlmostEqual(7.8, obj.object.getScan(1).getParameter("CCORH").getAttribute("how/NEZH"), 2)
    self.assertAlmostEqual(7.9, obj.object.getScan(1).getParameter("CCORH").getAttribute("how/zcalH"), 2)
    self.assertAlmostEqual(8.0, obj.object.getScan(1).getParameter("CCORH").getAttribute("how/nsampleH"), 2)

  def testReadAndConvertV21_COMP(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/what")
    self.addGroupNode(nodelist, "/where")
    self.addGroupNode(nodelist, "/how")
    
    self.addGroupNode(nodelist, "/dataset1")
    self.addGroupNode(nodelist, "/dataset1/what")
    self.addGroupNode(nodelist, "/dataset1/data1")
    self.addGroupNode(nodelist, "/dataset1/data1/how")
    self.addGroupNode(nodelist, "/dataset1/data1/what")
    self.addGroupNode(nodelist, "/dataset1/data2")
    self.addGroupNode(nodelist, "/dataset1/data2/how")
    self.addGroupNode(nodelist, "/dataset1/data2/what")
    self.addGroupNode(nodelist, "/dataset2")
    self.addGroupNode(nodelist, "/dataset2/what")
    self.addGroupNode(nodelist, "/dataset2/data1")
    self.addGroupNode(nodelist, "/dataset2/data1/how")
    self.addGroupNode(nodelist, "/dataset2/data1/what")

    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_1")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "CVOL")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.1")

    self.addAttributeNode(nodelist, "/where/LL_lat", "double", 54.1539)
    self.addAttributeNode(nodelist, "/where/LL_lon", "double", 9.1714)
    self.addAttributeNode(nodelist, "/where/LR_lat", "double", 54.1539)
    self.addAttributeNode(nodelist, "/where/LR_lon", "double", 16.5374)
    self.addAttributeNode(nodelist, "/where/UL_lat", "double", 58.4587)
    self.addAttributeNode(nodelist, "/where/UL_lon", "double", 8.73067)
    self.addAttributeNode(nodelist, "/where/UR_lat", "double", 58.4587)
    self.addAttributeNode(nodelist, "/where/UR_lon", "double", 16.9781)
    self.addAttributeNode(nodelist, "/where/projdef", "string", "+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    self.addAttributeNode(nodelist, "/where/xscale", "double", 2000.0)
    self.addAttributeNode(nodelist, "/where/xsize", "int", 240)
    self.addAttributeNode(nodelist, "/where/yscale", "double", 2000.0)
    self.addAttributeNode(nodelist, "/where/ysize", "int", 240)

    self.addAttributeNode(nodelist, "/how/TXloss", "double", 1.1)
    self.addAttributeNode(nodelist, "/how/injectloss", "double", 1.2)  # ??
    self.addAttributeNode(nodelist, "/how/RXloss", "double", 1.3)
    self.addAttributeNode(nodelist, "/how/radomeloss", "double", 1.4)
    self.addAttributeNode(nodelist, "/how/antgain", "double", 1.5)
    self.addAttributeNode(nodelist, "/how/beamw", "double", 1.6) #??
    self.addAttributeNode(nodelist, "/how/radconst", "double", 1.7) #??
    self.addAttributeNode(nodelist, "/how/NEZ", "double", 1.8)
    self.addAttributeNode(nodelist, "/how/zcal", "double", 1.9)
    self.addAttributeNode(nodelist, "/how/nsample", "double", 2.0) #??
     
    self.addAttributeNode(nodelist, "/dataset1/what/startdate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset1/what/starttime", "string", "101010")
    self.addAttributeNode(nodelist, "/dataset1/what/enddate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset1/what/endtime", "string", "101011")
    self.addAttributeNode(nodelist, "/dataset1/what/product", "string", "CAPPI")

    self.addAttributeNode(nodelist, "/dataset1/data1/how/TXloss", "double", 3.1)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/injectloss", "double", 3.2)  # ??
    self.addAttributeNode(nodelist, "/dataset1/data1/how/RXloss", "double", 3.3)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/radomeloss", "double", 3.4)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/antgain", "double", 3.5)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/beamw", "double", 3.6) #??
    self.addAttributeNode(nodelist, "/dataset1/data1/how/radconst", "double", 3.7) #??
    self.addAttributeNode(nodelist, "/dataset1/data1/how/NEZ", "double", 3.8)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/zcal", "double", 3.9)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nsample", "double", 4.0) #??

    self.addAttributeNode(nodelist, "/dataset1/data1/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "DBZH")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    
    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,10), dset)

    self.addAttributeNode(nodelist, "/dataset1/data2/how/TXloss", "double", 5.1)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/injectloss", "double", 5.2)  # ??
    self.addAttributeNode(nodelist, "/dataset1/data2/how/RXloss", "double", 5.3)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/radomeloss", "double", 5.4)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/antgain", "double", 5.5)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/beamw", "double", 5.6) #??
    self.addAttributeNode(nodelist, "/dataset1/data2/how/radconst", "double", 5.7) #??
    self.addAttributeNode(nodelist, "/dataset1/data2/how/NEZ", "double", 5.8)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/zcal", "double", 5.9)
    self.addAttributeNode(nodelist, "/dataset1/data2/how/nsample", "double", 6.0) #??

    self.addAttributeNode(nodelist, "/dataset1/data2/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/quantity", "string", "SQI")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    
    self.addDatasetNode(nodelist, "/dataset1/data2/data", "uchar", (10,10), dset)

    self.addAttributeNode(nodelist, "/dataset2/what/startdate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset2/what/starttime", "string", "101010")
    self.addAttributeNode(nodelist, "/dataset2/what/enddate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset2/what/endtime", "string", "101011")
    self.addAttributeNode(nodelist, "/dataset2/what/product", "string", "SCAN")

    self.addAttributeNode(nodelist, "/dataset2/data1/how/TXloss", "double", 7.1)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/injectloss", "double", 7.2)  # ??
    self.addAttributeNode(nodelist, "/dataset2/data1/how/RXloss", "double", 7.3)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/radomeloss", "double", 7.4)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/antgain", "double", 7.5)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/beamw", "double", 7.6) #??
    self.addAttributeNode(nodelist, "/dataset2/data1/how/radconst", "double", 7.7) #??
    self.addAttributeNode(nodelist, "/dataset2/data1/how/NEZ", "double", 7.8)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/zcal", "double", 7.9)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/nsample", "double", 8.0) #??

    self.addAttributeNode(nodelist, "/dataset2/data1/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset2/data1/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset2/data1/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset2/data1/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset2/data1/what/quantity", "string", "CCOR")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    
    self.addDatasetNode(nodelist, "/dataset2/data1/data", "uchar", (10,10), dset)

    nodelist.write(self.TEMPORARY_FILE, 6)

    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertAlmostEqual(1.1, obj.object.getAttribute("how/TXlossH"), 2)
    self.assertAlmostEqual(1.2, obj.object.getAttribute("how/injectlossH"), 2)
    self.assertAlmostEqual(1.3, obj.object.getAttribute("how/RXlossH"), 2)
    self.assertAlmostEqual(1.4, obj.object.getAttribute("how/radomelossH"), 2)
    self.assertAlmostEqual(1.5, obj.object.getAttribute("how/antgainH"), 2)
    self.assertAlmostEqual(1.6, obj.object.getAttribute("how/beamwH"), 2)
    self.assertAlmostEqual(1.7, obj.object.getAttribute("how/radconstH"), 2)
    self.assertAlmostEqual(1.8, obj.object.getAttribute("how/NEZH"), 2)
    self.assertAlmostEqual(1.9, obj.object.getAttribute("how/zcalH"), 2)
    self.assertAlmostEqual(2.0, obj.object.getAttribute("how/nsampleH"), 2)
    
    self.assertAlmostEqual(3.1, obj.object.getImage(0).getParameter("DBZH").getAttribute("how/TXlossH"), 2)
    self.assertAlmostEqual(3.2, obj.object.getImage(0).getParameter("DBZH").getAttribute("how/injectlossH"), 2)
    self.assertAlmostEqual(3.3, obj.object.getImage(0).getParameter("DBZH").getAttribute("how/RXlossH"), 2)
    self.assertAlmostEqual(3.4, obj.object.getImage(0).getParameter("DBZH").getAttribute("how/radomelossH"), 2)
    self.assertAlmostEqual(3.5, obj.object.getImage(0).getParameter("DBZH").getAttribute("how/antgainH"), 2)
    self.assertAlmostEqual(3.6, obj.object.getImage(0).getParameter("DBZH").getAttribute("how/beamwH"), 2)
    self.assertAlmostEqual(3.7, obj.object.getImage(0).getParameter("DBZH").getAttribute("how/radconstH"), 2)
    self.assertAlmostEqual(3.8, obj.object.getImage(0).getParameter("DBZH").getAttribute("how/NEZH"), 2)
    self.assertAlmostEqual(3.9, obj.object.getImage(0).getParameter("DBZH").getAttribute("how/zcalH"), 2)
    self.assertAlmostEqual(4.0, obj.object.getImage(0).getParameter("DBZH").getAttribute("how/nsampleH"), 2)

    self.assertAlmostEqual(5.1, obj.object.getImage(0).getParameter("SQIH").getAttribute("how/TXlossH"), 2)
    self.assertAlmostEqual(5.2, obj.object.getImage(0).getParameter("SQIH").getAttribute("how/injectlossH"), 2)
    self.assertAlmostEqual(5.3, obj.object.getImage(0).getParameter("SQIH").getAttribute("how/RXlossH"), 2)
    self.assertAlmostEqual(5.4, obj.object.getImage(0).getParameter("SQIH").getAttribute("how/radomelossH"), 2)
    self.assertAlmostEqual(5.5, obj.object.getImage(0).getParameter("SQIH").getAttribute("how/antgainH"), 2)
    self.assertAlmostEqual(5.6, obj.object.getImage(0).getParameter("SQIH").getAttribute("how/beamwH"), 2)
    self.assertAlmostEqual(5.7, obj.object.getImage(0).getParameter("SQIH").getAttribute("how/radconstH"), 2)
    self.assertAlmostEqual(5.8, obj.object.getImage(0).getParameter("SQIH").getAttribute("how/NEZH"), 2)
    self.assertAlmostEqual(5.9, obj.object.getImage(0).getParameter("SQIH").getAttribute("how/zcalH"), 2)
    self.assertAlmostEqual(6.0, obj.object.getImage(0).getParameter("SQIH").getAttribute("how/nsampleH"), 2)

    self.assertAlmostEqual(7.1, obj.object.getImage(1).getParameter("CCORH").getAttribute("how/TXlossH"), 2)
    self.assertAlmostEqual(7.2, obj.object.getImage(1).getParameter("CCORH").getAttribute("how/injectlossH"), 2)
    self.assertAlmostEqual(7.3, obj.object.getImage(1).getParameter("CCORH").getAttribute("how/RXlossH"), 2)
    self.assertAlmostEqual(7.4, obj.object.getImage(1).getParameter("CCORH").getAttribute("how/radomelossH"), 2)
    self.assertAlmostEqual(7.5, obj.object.getImage(1).getParameter("CCORH").getAttribute("how/antgainH"), 2)
    self.assertAlmostEqual(7.6, obj.object.getImage(1).getParameter("CCORH").getAttribute("how/beamwH"), 2)
    self.assertAlmostEqual(7.7, obj.object.getImage(1).getParameter("CCORH").getAttribute("how/radconstH"), 2)
    self.assertAlmostEqual(7.8, obj.object.getImage(1).getParameter("CCORH").getAttribute("how/NEZH"), 2)
    self.assertAlmostEqual(7.9, obj.object.getImage(1).getParameter("CCORH").getAttribute("how/zcalH"), 2)
    self.assertAlmostEqual(8.0, obj.object.getImage(1).getParameter("CCORH").getAttribute("how/nsampleH"), 2)

  def testBufrTableDir(self):
    obj = _raveio.new()
    self.assertEqual(None, obj.bufr_table_dir)
    obj.bufr_table_dir = "/tmp"
    self.assertEqual("/tmp", obj.bufr_table_dir)
    obj.bufr_table_dir = None
    self.assertEqual(None, obj.bufr_table_dir)
  
  def testReadBufr(self):
    if not _raveio.supports(_raveio.RaveIO_ODIM_FileFormat_BUFR):
      return
    result = _raveio.open(self.FIXTURE_BUFR_PVOL)
    
    self.assertEqual(_raveio.RaveIO_ODIM_FileFormat_BUFR, result.file_format);
    
    volume = result.object
    self.assertAlmostEqual(1.8347, volume.longitude * 180.0 / math.pi, 4)
    self.assertAlmostEqual(50.1358, volume.latitude * 180.0 / math.pi, 4)
    self.assertAlmostEqual(70.0, volume.height, 4)
    self.assertEqual("20090615", volume.date)
    self.assertEqual("032142", volume.time)
    self.assertEqual("WMO:07005", volume.source)
    self.assertEqual(3, volume.getNumberOfScans())
    
    scan = volume.getScan(0)
    self.assertAlmostEqual(0.4, scan.elangle * 180.0 / math.pi, 4)
    self.assertEqual(256, scan.nbins)
    self.assertEqual(720, scan.nrays)
    self.assertAlmostEqual(900.0, scan.rscale, 4)
    self.assertEqual(0, scan.a1gate)
    # beamwidth !? !?
    self.assertAlmostEqual(1.8347, scan.longitude * 180.0 / math.pi, 4)
    self.assertAlmostEqual(50.1358, scan.latitude * 180.0 / math.pi, 4)
    self.assertAlmostEqual(70.0, scan.height, 4)
    self.assertEqual("20090615", scan.startdate)
    self.assertEqual("031642", scan.starttime)
    self.assertEqual("20090615", scan.enddate)
    self.assertEqual("032142", scan.endtime)
    self.assertEqual("WMO:07005", scan.source)
    
    param = scan.getParameter("DBZH")
    self.assertEqual(256, param.nbins)
    self.assertEqual(720, param.nrays)
    self.assertEqual("DBZH", param.quantity)
    self.assertAlmostEqual(1.0, param.gain, 4)
    self.assertAlmostEqual(0.0, param.offset, 4)
    self.assertTrue(param.nodata > 1e30)
    self.assertTrue(param.undetect < -1e30)
    self.assertEqual(_rave.RaveDataType_DOUBLE, param.datatype)

    
  def testReadBufrOdim22(self):
    import _rave
    if not _raveio.supports(_raveio.RaveIO_ODIM_FileFormat_BUFR):
      return

    result = _raveio.open(self.FIXTURE_BUFR_2_2)
    
    self.assertEqual(_raveio.RaveIO_ODIM_FileFormat_BUFR, result.file_format);
    self.assertAlmostEqual(-4.43, result.object.longitude * 180.0 / math.pi, 6)
    self.assertAlmostEqual(48.460830, result.object.latitude * 180.0 / math.pi, 6)
    self.assertAlmostEqual(100.0, result.object.height, 4)
    self.assertEqual("WMO:07108", result.object.source)
    self.assertEqual("20140630", result.object.date)
    self.assertEqual("115801", result.object.time)
    self.assertAlmostEqual(-71.0, result.object.getAttribute("how/radconstH"), 4)
    self.assertAlmostEqual(-115.0, result.object.getAttribute("how/mindetect"), 4)
    self.assertAlmostEqual(-60.2, result.object.getAttribute("how/NI"), 4)
    self.assertAlmostEqual(-4.43, result.object.longitude * 180.0/math.pi, 4)
    self.assertEqual(1, result.object.getNumberOfScans())
    scan = result.object.getScan(0)
    self.assertAlmostEqual(1.8, scan.elangle*180.0/math.pi,4)
    self.assertAlmostEqual(1000.0, scan.rscale, 4)
    self.assertAlmostEqual(500.0, scan.rstart, 4)
    self.assertEqual(0, scan.a1gate)
    self.assertEqual(3, len(scan.getParameterNames()))
    dbzh = scan.getParameter("DBZH")
    th = scan.getParameter("TH")
    vrad = scan.getParameter("VRAD")
    self.assertTrue(dbzh is not None)
    self.assertTrue(th is not None)
    self.assertTrue(vrad is not None)
        
  def testReadBufrComposite(self):
    if not _raveio.supports(_raveio.RaveIO_ODIM_FileFormat_BUFR):
      return
    try:
      _raveio.open(self.FIXTURE_BUFR_COMPO)
      self.fail("Expected IOError")
    except IOError:
      pass

  def addGroupNode(self, nodelist, name):
    node = _pyhl.node(_pyhl.GROUP_ID, name)
    nodelist.addNode(node)
    
  def addAttributeNode(self, nodelist, name, type, value):
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, name)
    node.setScalarValue(-1,value,type,-1)
    nodelist.addNode(node)

  def addDatasetNode(self, nodelist, name, type, dims, value):
    node = _pyhl.node(_pyhl.DATASET_ID, name)
    node.setArrayValue(-1, dims, value, type, -1)
    nodelist.addNode(node)

  def rad2deg(self, coord):
    return (coord[0]*180.0/math.pi, coord[1]*180.0/math.pi)

#  def testWriteCF_Cartesian(self):
#    import _rave
#    _rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
#    obj = _raveio.open("swegmaps_2000_20171211_1300.h5").object.getImage(0)
#    rio = _raveio.new()
#    rio.object = obj
#    rio.file_format=_raveio.RaveIO_FileFormat_CF;
#    rio.save("test_slask_output.nc")
#     
#   def XtestWriteCF_Cartesian(self):
#     import _rave
#     _rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
#     obj = _raveio.open("swegmaps_2000_20171211_1300.h5").object.getImage(0)
#     #obj = _raveio.open("swecomposite_gmap.h5").object.getImage(0)
#     rio = _raveio.new()
#     rio.object = self.copy_image(obj, 100.0)
#     #rio.object.source = "NOD:swegmaps_2000"
#     rio.file_format=_raveio.RaveIO_FileFormat_CF;
#     rio.save("test_swegmaps.nc")
#     
#   def XtestWriteCF_CompositeVolume(self):
#     import _rave
#     _rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
#     vol = _raveio.open("swegmaps_2000_20171211_1300.h5").object
#     obj = vol.getImage(0)
#     obj.addAttribute("what/prodpar", 0)
#     for i in range(20):
#       vol.addImage(self.copy_image(obj, (i+1)*200))
#     rio = _raveio.new()
#     rio.object = vol
#     rio.file_format=_raveio.RaveIO_FileFormat_CF;
#     rio.save("test_swegmaps_vol.nc")
# 
#   def testWriteCF_LargeCompositeVolume(self):
#     import _rave
#     _rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
#     vol = _raveio.open("swegmaps_2000_201712161600_0_12000.h5").object
#     rio = _raveio.new()
#     rio.object = vol
#     rio.compression_level = 0
#     rio.file_format=_raveio.RaveIO_FileFormat_CF;
#     rio.save("test_swegmaps_large_vol.nc")
# 
#     rio.compression_level = 6
#     rio.file_format=_raveio.RaveIO_FileFormat_CF;
#     rio.save("test_swegmaps_large_vol_with_compression.nc")
# 
#   def copy_image(self, obj, height):
#     other = obj.clone()
#     other.addAttribute("what/prodpar", height)
#     return other
    
    
