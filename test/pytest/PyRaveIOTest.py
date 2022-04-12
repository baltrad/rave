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
  FIXTURE_HUV_WITH_0_86_BW="fixtures/scan_sehuv_1.5_20110126T184600Z.h5"
  FIXTURE_SEHEM_SCAN_0_5="fixtures/sehem_scan_20200414T160000Z.h5"
  FIXTURE_SEHEM_PVOL="fixtures/sehem_qcvol_20200507T064500Z.h5"

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
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84 +nadgrids=@null")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "BALTRAD"

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

    self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())
    # What
    self.assertEqual("100000", nodelist.getNode("/what/time").data())
    self.assertEqual("20100101", nodelist.getNode("/what/date").data())
    self.assertEqual("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEqual("IMAGE", nodelist.getNode("/what/object").data())
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())

    #Where
    self.assertEqual("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84 +nadgrids=@null", nodelist.getNode("/where/projdef").data())
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

    self.assertEqual("BALTRAD" , nodelist.getNode("/dataset1/what/prodname").data())

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

  def test_save_cartesian_with_default_prodname(self):
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
    image.prodname = None

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

    self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())
    self.assertEqual("BALTRAD cartesian" , nodelist.getNode("/dataset1/what/prodname").data())

    #Just verify that we can read it as well
    obj = _raveio.open(self.TEMPORARY_FILE).object
    self.assertEqual("BALTRAD cartesian", obj.prodname)

  def test_save_cartesian_24(self):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123,WIGOS:0-123-1-123456"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "My Product"

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

    self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())
    self.assertEqual("My Product" , nodelist.getNode("/dataset1/what/prodname").data())
    self.assertEqual("PLC:123,WIGOS:0-123-1-123456", nodelist.getNode("/what/source").data())
    #Just verify that we can read it as well
    obj = _raveio.open(self.TEMPORARY_FILE).object
    self.assertEqual("My Product", obj.prodname)

  def test_save_cartesian_24_strict(self):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123,WIGOS:0-123-1-123456"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "My Product"

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
    ios.strict = True
    try:
        ios.save()
        self.fail("Expected IOError")
    except IOError:
        pass

  def test_save_cartesian_24_strict_success(self):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123,WIGOS:0-123-1-123456,ORG:82"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "My Product"
    image.addAttribute("how/simulated", True)  # Must set simulated to be accepted.
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
    ios.strict = True
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())
    self.assertEqual("My Product" , nodelist.getNode("/dataset1/what/prodname").data())
    self.assertEqual("PLC:123,WIGOS:0-123-1-123456,ORG:82", nodelist.getNode("/what/source").data())
    #Just verify that we can read it as well
    obj = _raveio.open(self.TEMPORARY_FILE).object
    self.assertEqual("My Product", obj.prodname)

  def test_save_cartesian_24_strict_source(self):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123,WIGOS:0-123-1-123456"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "My Product"
    image.addAttribute("how/simulated", True)  # Must set simulated to be accepted.
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
    ios.strict = True
    try:
        ios.save()
        self.fail("Expected IOError")
    except IOError:
        pass
    ios.object.source = ios.object.source + ",ORG:82"
    ios.save()

  def test_save_cartesian_23(self):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123,WIGOS:0-123-1-123456"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "My Product"

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
    ios.version = _rave.RaveIO_ODIM_Version_2_3
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("ODIM_H5/V2_3", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.3", nodelist.getNode("/what/version").data())
    self.assertEqual("My Product" , nodelist.getNode("/dataset1/what/prodname").data())
    self.assertEqual("PLC:123,WIGOS:0-123-1-123456", nodelist.getNode("/what/source").data())
    #Just verify that we can read it as well
    obj = _raveio.open(self.TEMPORARY_FILE).object
    self.assertEqual("My Product", obj.prodname)

  def test_save_cartesian_22(self):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123,WIGOS:0-123-1-123456"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "My Product"

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
    ios.version = _raveio.RaveIO_ODIM_Version_2_2
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("ODIM_H5/V2_2", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    self.assertFalse("/dataset1/what/prodname" in nodelist.getNodeNames())
    self.assertEqual("PLC:123", nodelist.getNode("/what/source").data())

    #Just verify that we can read it as well
    obj = _raveio.open(self.TEMPORARY_FILE).object
    self.assertEqual(None, obj.prodname)

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
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84 +nadgrids=@null")
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

    self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())
    # What
    self.assertEqual("100000", nodelist.getNode("/what/time").data())
    self.assertEqual("20100101", nodelist.getNode("/what/date").data())
    self.assertEqual("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEqual("IMAGE", nodelist.getNode("/what/object").data())
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())

    #Where
    self.assertEqual("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84 +nadgrids=@null", nodelist.getNode("/where/projdef").data())
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

  def test_save_cartesian_bessel_legacy_mode(self):
    if not _rave.isLegacyProjEnabled():
        # No meaning to try legacy mode if it has been disabled on compile time
        return

    os.environ["RAVE_USE_CARTESIAN_LEGACY_EXTENT"]="yes"
    try:
      image = _cartesian.new()
      image.time = "100000"
      image.date = "20100101"
      image.objectType = _rave.Rave_ObjectType_IMAGE
      image.source = "PLC:123"
      image.xscale = 2000.0
      image.yscale = 2000.0
      image.areaextent = (-745231.940399, -3995729.577395, 1066768.059601, -1747729.577395)
      image.projection = _projection.new("x","y","+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +datum=WGS84")
      image.product = _rave.Rave_ProductType_CAPPI
      image.prodname = None

      param = _cartesianparam.new()
      param.quantity = "DBZH"
      param.gain = 1.0
      param.offset = 0.0
      param.nodata = 255.0
      param.undetect = 0.0
      data = numpy.zeros((906,1124),numpy.uint8)
      param.setData(data)

      image.addParameter(param)

      ios = _raveio.new()
      ios.object = image
      ios.filename = self.TEMPORARY_FILE #self.TEMPORARY_FILE
      ios.save()

      # Verify result
      nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
      nodelist.selectAll()
      nodelist.fetch()


      LL_lat = 52.3548
      LL_lon = 3.43531
      LR_lat = 51.7426
      LR_lon = 28.948
      UL_lat = 71.9126
      UL_lon = -9.0934
      UR_lat = 70.5325
      UR_lon = 45.3988

      self.assertAlmostEqual(LL_lat, nodelist.getNode("/where/LL_lat").data(), 4)
      self.assertAlmostEqual(LL_lon, nodelist.getNode("/where/LL_lon").data(), 4)
      self.assertAlmostEqual(LR_lat, nodelist.getNode("/where/LR_lat").data(), 4)
      self.assertAlmostEqual(LR_lon, nodelist.getNode("/where/LR_lon").data(), 4)
      self.assertAlmostEqual(UL_lat, nodelist.getNode("/where/UL_lat").data(), 4)
      self.assertAlmostEqual(UL_lon, nodelist.getNode("/where/UL_lon").data(), 4)
      self.assertAlmostEqual(UR_lat, nodelist.getNode("/where/UR_lat").data(), 4)
      self.assertAlmostEqual(UR_lon, nodelist.getNode("/where/UR_lon").data(), 4)
    finally:
      if "RAVE_USE_CARTESIAN_LEGACY_EXTENT" in os.environ:
          del os.environ["RAVE_USE_CARTESIAN_LEGACY_EXTENT"]

  def test_save_cartesian_bessel(self):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-745244.228412, -3995795.462356, 1066755.771588, -1747795.462356)
    image.projection = _projection.new("x","y","+proj=stere +ellps=bessel +lat_0=90 +lon_0=14 +lat_ts=60 +towgs84=0,0,0")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = None

    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((906,1124),numpy.uint8)
    param.setData(data)

    image.addParameter(param)

    ios = _raveio.new()
    ios.object = image
    ios.filename = self.TEMPORARY_FILE #self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    LL_lat = 52.3548
    LL_lon = 3.43531
    LR_lat = 51.7427
    LR_lon = 28.9476
    UL_lat = 71.9123
    UL_lon = -9.09296
    UR_lat = 70.5324
    UR_lon = 45.3975

    self.assertAlmostEqual(LL_lat, nodelist.getNode("/where/LL_lat").data(), 4)
    self.assertAlmostEqual(LL_lon, nodelist.getNode("/where/LL_lon").data(), 4)
    self.assertAlmostEqual(LR_lat, nodelist.getNode("/where/LR_lat").data(), 4)
    self.assertAlmostEqual(LR_lon, nodelist.getNode("/where/LR_lon").data(), 4)
    self.assertAlmostEqual(UL_lat, nodelist.getNode("/where/UL_lat").data(), 4)
    self.assertAlmostEqual(UL_lon, nodelist.getNode("/where/UL_lon").data(), 4)
    self.assertAlmostEqual(UR_lat, nodelist.getNode("/where/UR_lat").data(), 4)
    self.assertAlmostEqual(UR_lon, nodelist.getNode("/where/UR_lon").data(), 4)

  def test_write_read_cartesian_withHowSubgroups(self):
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
    image.addAttribute("how/attrib", "value")
    image.addAttribute("how/subgroup1/attrib", "value 1")
    image.addAttribute("how/subgroup1/subgroup2/attrib", "value 1 2")

    qfield1 = _ravefield.new()
    qfield1.addAttribute("how/attrib", "qfield1 value")
    qfield1.addAttribute("how/subgroup1/attrib", "qfield1 value 1")
    qfield1.addAttribute("how/subgroup1/subgroup2/attrib", "qfield1 value 1 2")
    qfield1.setData(numpy.zeros((240,240), numpy.uint8))
    image.addQualityField(qfield1)

    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)

    param.addAttribute("how/attrib", "param value")
    param.addAttribute("how/subgroup1/attrib", "param value 1")
    param.addAttribute("how/subgroup1/subgroup2/attrib", "param value 1 2")

    qfield2 = _ravefield.new()
    qfield2.addAttribute("how/attrib", "qfield2 value")
    qfield2.addAttribute("how/subgroup1/attrib", "qfield2 value 1")
    qfield2.addAttribute("how/subgroup1/subgroup2/attrib", "qfield2 value 1 2")
    qfield2.setData(numpy.zeros((240,240), numpy.uint8))

    param.addQualityField(qfield2)

    image.addParameter(param)

    ios = _raveio.new()
    ios.object = image
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("value", nodelist.getNode("/how/attrib").data())
    self.assertEqual("value 1", nodelist.getNode("/how/subgroup1/attrib").data())
    self.assertEqual("value 1 2", nodelist.getNode("/how/subgroup1/subgroup2/attrib").data())

    self.assertEqual("qfield1 value", nodelist.getNode("/dataset1/quality1/how/attrib").data())
    self.assertEqual("qfield1 value 1", nodelist.getNode("/dataset1/quality1/how/subgroup1/attrib").data())
    self.assertEqual("qfield1 value 1 2", nodelist.getNode("/dataset1/quality1/how/subgroup1/subgroup2/attrib").data())

    self.assertEqual("param value", nodelist.getNode("/dataset1/data1/how/attrib").data())
    self.assertEqual("param value 1", nodelist.getNode("/dataset1/data1/how/subgroup1/attrib").data())
    self.assertEqual("param value 1 2", nodelist.getNode("/dataset1/data1/how/subgroup1/subgroup2/attrib").data())

    self.assertEqual("qfield2 value", nodelist.getNode("/dataset1/data1/quality1/how/attrib").data())
    self.assertEqual("qfield2 value 1", nodelist.getNode("/dataset1/data1/quality1/how/subgroup1/attrib").data())
    self.assertEqual("qfield2 value 1 2", nodelist.getNode("/dataset1/data1/quality1/how/subgroup1/subgroup2/attrib").data())

    newcartesian = _raveio.open(self.TEMPORARY_FILE).object
    self.assertEqual("value", newcartesian.getAttribute("how/attrib"))
    self.assertEqual("value 1", newcartesian.getAttribute("how/subgroup1/attrib"))
    self.assertEqual("value 1 2", newcartesian.getAttribute("how/subgroup1/subgroup2/attrib"))

    self.assertEqual("qfield1 value", newcartesian.getQualityField(0).getAttribute("how/attrib"))
    self.assertEqual("qfield1 value 1", newcartesian.getQualityField(0).getAttribute("how/subgroup1/attrib"))
    self.assertEqual("qfield1 value 1 2", newcartesian.getQualityField(0).getAttribute("how/subgroup1/subgroup2/attrib"))

    self.assertEqual("param value", newcartesian.getParameter("DBZH").getAttribute("how/attrib"))
    self.assertEqual("param value 1", newcartesian.getParameter("DBZH").getAttribute("how/subgroup1/attrib"))
    self.assertEqual("param value 1 2", newcartesian.getParameter("DBZH").getAttribute("how/subgroup1/subgroup2/attrib"))

    self.assertEqual("qfield2 value", newcartesian.getParameter("DBZH").getQualityField(0).getAttribute("how/attrib"))
    self.assertEqual("qfield2 value 1", newcartesian.getParameter("DBZH").getQualityField(0).getAttribute("how/subgroup1/attrib"))
    self.assertEqual("qfield2 value 1 2", newcartesian.getParameter("DBZH").getQualityField(0).getAttribute("how/subgroup1/subgroup2/attrib"))

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
    projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84 +nadgrids=@null")
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

    self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())
    # What
    self.assertEqual("100000", nodelist.getNode("/what/time").data())
    self.assertEqual("20091010", nodelist.getNode("/what/date").data())
    self.assertEqual("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEqual("CVOL", nodelist.getNode("/what/object").data())
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())

    #Where
    self.assertEqual("+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84 +nadgrids=@null", nodelist.getNode("/where/projdef").data())
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

  def test_save_cartesian_volume_strict_failure(self):
    cvol = _cartesianvolume.new()
    cvol.time = "100000"
    cvol.date = "20091010"
    cvol.objectType = _rave.Rave_ObjectType_CVOL
    cvol.source = "PLC:123"
    cvol.xscale = 2000.0
    cvol.yscale = 2000.0
    cvol.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84 +nadgrids=@null")
    cvol.projection = projection

    image = _cartesian.new()
    image.product = _rave.Rave_ProductType_CAPPI
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    image.addParameter(param)

    cvol.addImage(image)

    image = _cartesian.new()
    image.product = _rave.Rave_ProductType_CAPPI
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    image.addParameter(param)
    
    rio = _raveio.new()
    rio.object = cvol
    rio.filename = self.TEMPORARY_FILE
    rio.strict = True
    try:
        rio.save()
        self.fail("Expected IOError")
    except IOError:
        pass
    self.assertTrue(len(rio.error_message) > 0)

  def test_save_cartesian_volume_strict_success(self):
    cvol = _cartesianvolume.new()
    cvol.time = "100000"
    cvol.date = "20091010"
    cvol.objectType = _rave.Rave_ObjectType_CVOL
    cvol.source = "PLC:123"
    cvol.xscale = 2000.0
    cvol.yscale = 2000.0
    cvol.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    cvol.addAttribute("how/simulated", True)
    
    projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84 +nadgrids=@null")
    cvol.projection = projection

    image = _cartesian.new()
    image.product = _rave.Rave_ProductType_CAPPI
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    image.addParameter(param)

    cvol.addImage(image)

    image = _cartesian.new()
    image.product = _rave.Rave_ProductType_CAPPI
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.setData(numpy.zeros((240,240),numpy.uint8))
    image.addParameter(param)
    
    rio = _raveio.new()
    rio.object = cvol
    rio.filename = self.TEMPORARY_FILE
    rio.strict = True
    try:
        rio.save()
        self.fail("Expected IOError")
    except IOError:
        pass
    rio.object.source = rio.object.source+",ORG:82"
    rio.save()

  def test_write_read_cartesian_volume_howSubgroups(self):
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
    cvol.addAttribute("how/attr", "value")
    cvol.addAttribute("how/subgroup1/attr", "value 1")
    cvol.addAttribute("how/subgroup1/subgroup2/attr", "value 1 2")

    image = _cartesian.new()
    image.product = _rave.Rave_ProductType_CAPPI
    image.addAttribute("how/attr", "image value")
    image.addAttribute("how/subgroup1/attr", "image value 1")
    image.addAttribute("how/subgroup1/subgroup2/attr", "image value 1 2")

    qfield1 = _ravefield.new()
    qfield1.addAttribute("how/attr", "qfield1 value")
    qfield1.addAttribute("how/subgroup1/attr", "qfield1 value 1")
    qfield1.addAttribute("how/subgroup1/subgroup2/attr", "qfield1 value 1 2")
    qfield1.setData(numpy.zeros((240,240), numpy.uint8))
    image.addQualityField(qfield1)

    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)

    param.addAttribute("how/attr", "param value")
    param.addAttribute("how/subgroup1/attr", "param value 1")
    param.addAttribute("how/subgroup1/subgroup2/attr", "param value 1 2")

    qfield2 = _ravefield.new()
    qfield2.addAttribute("how/attr", "qfield2 value")
    qfield2.addAttribute("how/subgroup1/attr", "qfield2 value 1")
    qfield2.addAttribute("how/subgroup1/subgroup2/attr", "qfield2 value 1 2")
    qfield2.setData(numpy.zeros((240,240), numpy.uint8))
    param.addQualityField(qfield2)

    image.addParameter(param)

    cvol.addImage(image)

    ios = _raveio.new()
    ios.object = cvol
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertEqual("value", nodelist.getNode("/how/attr").data())
    self.assertEqual("value 1", nodelist.getNode("/how/subgroup1/attr").data())
    self.assertEqual("value 1 2", nodelist.getNode("/how/subgroup1/subgroup2/attr").data())

    self.assertEqual("image value", nodelist.getNode("/dataset1/how/attr").data())
    self.assertEqual("image value 1", nodelist.getNode("/dataset1/how/subgroup1/attr").data())
    self.assertEqual("image value 1 2", nodelist.getNode("/dataset1/how/subgroup1/subgroup2/attr").data())

    self.assertEqual("qfield1 value", nodelist.getNode("/dataset1/quality1/how/attr").data())
    self.assertEqual("qfield1 value 1", nodelist.getNode("/dataset1/quality1/how/subgroup1/attr").data())
    self.assertEqual("qfield1 value 1 2", nodelist.getNode("/dataset1/quality1/how/subgroup1/subgroup2/attr").data())

    self.assertEqual("param value", nodelist.getNode("/dataset1/data1/how/attr").data())
    self.assertEqual("param value 1", nodelist.getNode("/dataset1/data1/how/subgroup1/attr").data())
    self.assertEqual("param value 1 2", nodelist.getNode("/dataset1/data1/how/subgroup1/subgroup2/attr").data())

    self.assertEqual("qfield2 value", nodelist.getNode("/dataset1/data1/quality1/how/attr").data())
    self.assertEqual("qfield2 value 1", nodelist.getNode("/dataset1/data1/quality1/how/subgroup1/attr").data())
    self.assertEqual("qfield2 value 1 2", nodelist.getNode("/dataset1/data1/quality1/how/subgroup1/subgroup2/attr").data())

    newvol = _raveio.open(self.TEMPORARY_FILE).object
    self.assertEqual("value", newvol.getAttribute("how/attr"))
    self.assertEqual("value 1", newvol.getAttribute("how/subgroup1/attr"))
    self.assertEqual("value 1 2", newvol.getAttribute("how/subgroup1/subgroup2/attr"))

    self.assertEqual("image value", newvol.getImage(0).getAttribute("how/attr"))
    self.assertEqual("image value 1", newvol.getImage(0).getAttribute("how/subgroup1/attr"))
    self.assertEqual("image value 1 2", newvol.getImage(0).getAttribute("how/subgroup1/subgroup2/attr"))

    self.assertEqual("qfield1 value", newvol.getImage(0).getQualityField(0).getAttribute("how/attr"))
    self.assertEqual("qfield1 value 1", newvol.getImage(0).getQualityField(0).getAttribute("how/subgroup1/attr"))
    self.assertEqual("qfield1 value 1 2", newvol.getImage(0).getQualityField(0).getAttribute("how/subgroup1/subgroup2/attr"))

    self.assertEqual("param value", newvol.getImage(0).getParameter("DBZH").getAttribute("how/attr"))
    self.assertEqual("param value 1", newvol.getImage(0).getParameter("DBZH").getAttribute("how/subgroup1/attr"))
    self.assertEqual("param value 1 2", newvol.getImage(0).getParameter("DBZH").getAttribute("how/subgroup1/subgroup2/attr"))

    self.assertEqual("qfield2 value", newvol.getImage(0).getParameter("DBZH").getQualityField(0).getAttribute("how/attr"))
    self.assertEqual("qfield2 value 1", newvol.getImage(0).getParameter("DBZH").getQualityField(0).getAttribute("how/subgroup1/attr"))
    self.assertEqual("qfield2 value 1 2", newvol.getImage(0).getParameter("DBZH").getQualityField(0).getAttribute("how/subgroup1/subgroup2/attr"))

  def test_load_cartesian_volume(self):
    try:
        _projection.setDefaultLonLatProjDef("+proj=longlat +ellps=WGS84")
        obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
        self.assertEqual(_raveio.RaveIO_ODIM_Version_2_0, obj.read_version)
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
    finally:
        _projection.setDefaultLonLatProjDef("+proj=longlat +ellps=WGS84 +datum=WGS84") # Reset to not cause problems with other test cases

  def test_load_cartesian_volume_20_save_24(self):
    obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, obj.version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_0, obj.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_H5rad_Version_2_0, obj.h5radversion)
    self.assertEqual(_rave.Rave_ObjectType_CVOL, obj.objectType)

    obj.object.source = "%s,WIGOS:0-123-1-123456"%obj.object.source
    obj.object.getImage(0).prodname="NISSE"
    obj.object.zscale = 123.0
    obj.object.zstart = 432.0
    rio = _raveio.new()
    rio.object = obj.object
    rio.save(self.TEMPORARY_FILE)

    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertEqual(1, nodelist.getNode("/where/zsize").data())

    nrio = _raveio.open(self.TEMPORARY_FILE)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, nrio.version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, nrio.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_H5rad_Version_2_4, nrio.h5radversion)
    self.assertEqual(obj.object.source, nrio.object.source)
    self.assertEqual("NISSE", nrio.object.getImage(0).prodname)

  def test_load_cartesian_volume_20_save_23(self):
    obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, obj.version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_0, obj.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_H5rad_Version_2_0, obj.h5radversion)
    self.assertEqual(_rave.Rave_ObjectType_CVOL, obj.objectType)

    obj.object.source = "%s,WIGOS:0-123-1-123456"%obj.object.source
    obj.object.getImage(0).prodname="NISSE"
    obj.object.zscale = 123.0
    obj.object.zstart = 432.0
    rio = _raveio.new()
    rio.object = obj.object
    rio.version = _raveio.RaveIO_ODIM_Version_2_3
    rio.save(self.TEMPORARY_FILE)

    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertAlmostEqual(123.0, nodelist.getNode("/where/zscale").data(), 4)
    self.assertAlmostEqual(432.0, nodelist.getNode("/where/zstart").data(), 4)
    self.assertEqual(1, nodelist.getNode("/where/zsize").data())

    nrio = _raveio.open(self.TEMPORARY_FILE)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, nrio.version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_3, nrio.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_H5rad_Version_2_3, nrio.h5radversion)
    self.assertEqual(obj.object.source, nrio.object.source)
    self.assertEqual("NISSE", nrio.object.getImage(0).prodname)
    self.assertAlmostEqual(123.0, nrio.object.zscale, 4)
    self.assertAlmostEqual(432.0, nrio.object.zstart, 4)

  def test_load_cartesian_volume_20_save_22(self):
    obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, obj.version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_0, obj.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_H5rad_Version_2_0, obj.h5radversion)
    self.assertEqual(_rave.Rave_ObjectType_CVOL, obj.objectType)
    originalstr = obj.object.source
    obj.object.source = "%s,WIGOS:0-123-1-123456"%originalstr
    obj.object.getImage(0).prodname="NISSE"

    rio = _raveio.new()
    rio.object = obj.object
    rio.version = _raveio.RaveIO_ODIM_Version_2_2
    rio.save(self.TEMPORARY_FILE)

    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertFalse("/where/zscale" in nodelist.getNodeNames())
    self.assertFalse("/where/zstart" in nodelist.getNodeNames())
    self.assertFalse("/where/zsize" in nodelist.getNodeNames())

    nrio = _raveio.open(self.TEMPORARY_FILE)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, nrio.version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_2, nrio.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_H5rad_Version_2_2, nrio.h5radversion)
    self.assertEqual(originalstr, nrio.object.source)
    self.assertEqual(None, nrio.object.getImage(0).prodname)
    self.assertAlmostEqual(0.0, nrio.object.zscale, 4)
    self.assertAlmostEqual(0.0, nrio.object.zstart, 4)

  def test_load_cartesian_volume_save_cartesian_image(self):
    try:
        _projection.setDefaultLonLatProjDef("+proj=longlat +ellps=WGS84")
        obj = _raveio.open(self.FIXTURE_CVOL_CAPPI)
        image = obj.object.getImage(0)
        ios = _raveio.new()
        ios.object = image
        ios.filename = self.TEMPORARY_FILE
        ios.save()    

        nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
        nodelist.selectAll()
        nodelist.fetch()

        self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())

        # What
        self.assertEqual("100000", nodelist.getNode("/what/time").data())
        self.assertEqual("20091010", nodelist.getNode("/what/date").data())
        self.assertEqual("PLC:123", nodelist.getNode("/what/source").data())
        self.assertEqual("IMAGE", nodelist.getNode("/what/object").data())
        self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())

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
    finally:
        _projection.setDefaultLonLatProjDef("+proj=longlat +ellps=WGS84 +datum=WGS84") # Reset to not cause problems with other test cases

  def test_load_cartesian_image2(self):
    try:
        _projection.setDefaultLonLatProjDef("+proj=longlat +ellps=WGS84")
        obj = _raveio.open(self.FIXTURE_CARTESIAN_IMAGE)
        self.assertEqual(_raveio.RaveIO_ODIM_Version_2_1, obj.read_version)
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
    finally:
        _projection.setDefaultLonLatProjDef("+proj=longlat +ellps=WGS84 +datum=WGS84") # Reset to not cause problems with other test cases

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
    try:
        _projection.setDefaultLonLatProjDef("+proj=longlat +ellps=WGS84")

        obj = _raveio.open(self.FIXTURE_CARTESIAN_VOLUME)
        self.assertEqual(_raveio.RaveIO_ODIM_Version_2_1, obj.read_version)
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
    finally:
        _projection.setDefaultLonLatProjDef("+proj=longlat +ellps=WGS84 +datum=WGS84") # Reset to not cause problems with other test cases


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
    scan2.rstart = 1.0
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
    scan2.addAttribute("how/scan_index", 3)
    obj.addScan(scan2)

    ios = _raveio.new()
    ios.object = obj
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())
    # What
    self.assertEqual("100000", nodelist.getNode("/what/time").data())
    self.assertEqual("20091010", nodelist.getNode("/what/date").data())
    self.assertEqual("PLC:123", nodelist.getNode("/what/source").data())
    self.assertEqual("PVOL", nodelist.getNode("/what/object").data())
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())

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

    # This scan_index was manually set. This was added manually so they must know what they are doing.
    self.assertEqual(3, nodelist.getNode("/dataset2/how/scan_index").data())

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


  def test_save_read_polar_volume_with_howSubgroups(self):
    obj = _polarvolume.new()
    obj.time = "100000"
    obj.date = "20091010"
    obj.source = "PLC:123"
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0

    obj.addAttribute("how/attrib", "value")
    obj.addAttribute("how/grp1/attrib", "value 1")
    obj.addAttribute("how/grp1/grp2/attrib", "value 1 2")

    scan1 = _polarscan.new()
    scan1.elangle = 0.1 * math.pi / 180.0
    scan1.a1gate = 2
    scan1.rstart = 0.0
    scan1.rscale = 5000.0
    scan1.time = "100001"
    scan1.date = "20091010"
    scan1.addAttribute("how/attrib", "scan value")
    scan1.addAttribute("how/grp1/attrib", "scan value 1")
    scan1.addAttribute("how/grp1/grp2/attrib", "scan value 1 2")

    dbzhParam = _polarscanparam.new()
    dbzhParam.nodata = 10.0
    dbzhParam.undetect = 11.0
    dbzhParam.quantity = "DBZH"
    dbzhParam.gain = 1.0
    dbzhParam.offset = 0.0
    data = numpy.zeros((100, 120), numpy.uint8)
    dbzhParam.setData(data)
    dbzhParam.addAttribute("how/attrib", "param value")
    dbzhParam.addAttribute("how/grp1/attrib", "param value 1")
    dbzhParam.addAttribute("how/grp1/grp2/attrib", "param value 1 2")

    scan1.addParameter(dbzhParam)

    qfield = _ravefield.new()
    qfield.addAttribute("what/sthis", "a quality field")
    qfield.setData(numpy.zeros((100,120), numpy.uint8))
    qfield.addAttribute("how/attrib", "field value")
    qfield.addAttribute("how/grp1/attrib", "field value 1")
    qfield.addAttribute("how/grp1/grp2/attrib", "field value 1 2")

    dbzhParam.addQualityField(qfield)

    obj.addScan(scan1)

    ios = _raveio.new()
    ios.object = obj
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertEqual("value", nodelist.getNode("/how/attrib").data())
    self.assertEqual("value 1", nodelist.getNode("/how/grp1/attrib").data())
    self.assertEqual("value 1 2", nodelist.getNode("/how/grp1/grp2/attrib").data())
    self.assertEqual("scan value", nodelist.getNode("/dataset1/how/attrib").data())
    self.assertEqual("scan value 1", nodelist.getNode("/dataset1/how/grp1/attrib").data())
    self.assertEqual("scan value 1 2", nodelist.getNode("/dataset1/how/grp1/grp2/attrib").data())
    self.assertEqual("param value", nodelist.getNode("/dataset1/data1/how/attrib").data())
    self.assertEqual("param value 1", nodelist.getNode("/dataset1/data1/how/grp1/attrib").data())
    self.assertEqual("param value 1 2", nodelist.getNode("/dataset1/data1/how/grp1/grp2/attrib").data())
    self.assertEqual("field value", nodelist.getNode("/dataset1/data1/quality1/how/attrib").data())
    self.assertEqual("field value 1", nodelist.getNode("/dataset1/data1/quality1/how/grp1/attrib").data())
    self.assertEqual("field value 1 2", nodelist.getNode("/dataset1/data1/quality1/how/grp1/grp2/attrib").data())

    savedvol = _raveio.open(self.TEMPORARY_FILE).object
    self.assertEqual("value", savedvol.getAttribute("how/attrib"))
    self.assertEqual("value 1", savedvol.getAttribute("how/grp1/attrib"))
    self.assertEqual("value 1 2", savedvol.getAttribute("how/grp1/grp2/attrib"))
    self.assertEqual("scan value", savedvol.getScan(0).getAttribute("how/attrib"))
    self.assertEqual("scan value 1", savedvol.getScan(0).getAttribute("how/grp1/attrib"))
    self.assertEqual("scan value 1 2", savedvol.getScan(0).getAttribute("how/grp1/grp2/attrib"))
    self.assertEqual("param value", savedvol.getScan(0).getParameter("DBZH").getAttribute("how/attrib"))
    self.assertEqual("param value 1", savedvol.getScan(0).getParameter("DBZH").getAttribute("how/grp1/attrib"))
    self.assertEqual("param value 1 2", savedvol.getScan(0).getParameter("DBZH").getAttribute("how/grp1/grp2/attrib"))
    self.assertEqual("field value", savedvol.getScan(0).getParameter("DBZH").getQualityField(0).getAttribute("how/attrib"))
    self.assertEqual("field value 1", savedvol.getScan(0).getParameter("DBZH").getQualityField(0).getAttribute("how/grp1/attrib"))
    self.assertEqual("field value 1 2", savedvol.getScan(0).getParameter("DBZH").getQualityField(0).getAttribute("how/grp1/grp2/attrib"))

  def test_save_polar_volume_beamwidths(self):
    obj = _polarvolume.new()
    obj.time = "100000"
    obj.date = "20091010"
    obj.source = "PLC:123"
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    obj.beamwH = 2.0 * math.pi/180.0
    obj.beamwV = 3.0 * math.pi/180.0

    scan1 = _polarscan.new()
    scan1.beamwH = 4.0 * math.pi/180
    scan2 = _polarscan.new()
    scan2.beamwV = 5.0 * math.pi/180
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

    self.assertAlmostEqual(2.0, nodelist.getNode("/how/beamwH").data())
    self.assertAlmostEqual(2.0, nodelist.getNode("/how/beamwidth").data())
    self.assertAlmostEqual(3.0, nodelist.getNode("/how/beamwV").data())
    self.assertAlmostEqual(4.0, nodelist.getNode("/dataset1/how/beamwH").data())
    self.assertAlmostEqual(4.0, nodelist.getNode("/dataset1/how/beamwidth").data())
    self.assertAlmostEqual(5.0, nodelist.getNode("/dataset2/how/beamwV").data())

    nodenames = nodelist.getNodeNames()
    self.assertTrue("/dataset3/how/beamwidth" not in nodenames)
    self.assertTrue("/dataset4/how/beamwidth" not in nodenames)

  def test_save_polar_scan_beamwH(self):
    obj = _polarscan.new()
    obj.time = "100000"
    obj.date = "20091010"
    obj.source = "PLC:123"
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    obj.beamwH = 1.0 * math.pi/180.0
    obj.beamwV = 2.0 * math.pi/180.0

    ios = _raveio.new()
    ios.object = obj
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    ios = _raveio.new()
    ios.object = obj
    ios.filename = self.TEMPORARY_FILE
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertAlmostEqual(1.0, nodelist.getNode("/how/beamwH").data())
    self.assertAlmostEqual(1.0, nodelist.getNode("/how/beamwidth").data())
    self.assertAlmostEqual(2.0, nodelist.getNode("/how/beamwV").data())

  def test_loadScan_withBeamwidth(self):
    obj = _raveio.open(self.FIXTURE_HUV_WITH_0_86_BW).object

    self.assertAlmostEqual(0.86, obj.beamwH*180.0/math.pi, 3)
    self.assertAlmostEqual(1.0, obj.beamwV*180.0/math.pi, 3)

  def test_save_polar_scan_strict_source(self):
    obj = _polarscan.new()
    obj.time = "100000"
    obj.date = "20091010"
    obj.source = "PLC:123"
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0
    obj.beamwH = 1.0 * math.pi/180.0
    obj.beamwV = 2.0 * math.pi/180.0
    obj.addAttribute("how/simulated", True)
    obj.addAttribute("how/wavelength", 1.1)
    obj.addAttribute("how/pulsewidth", 1.1)
    obj.addAttribute("how/RXlossH", 1.1)
    obj.addAttribute("how/antgainH", 1.1)
    obj.addAttribute("how/beamwH", 1.1)
    obj.addAttribute("how/radconstH", 1.1)
    obj.addAttribute("how/NI", 1.1)
    obj.addAttribute("how/startazA", numpy.arange(0.0,360.0,1.0))
    obj.addAttribute("how/stopazA", numpy.arange(0.0,360.0,1.0))    
    ios = _raveio.new()
    ios.object = obj
    ios.filename = self.TEMPORARY_FILE
    ios.strict=True
    try:
        ios.save()
        self.fail("Expected IOError")
    except IOError:
        pass
    self.assertTrue(len(ios.error_message) > 0)
    ios.object.source = "PLC:123,NOD:seang"
    ios.save()

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

    self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())
    self.assertEqual("120000", nodelist.getNode("/what/time").data())
    self.assertEqual("20090501", nodelist.getNode("/what/date").data())
    self.assertEqual("WMO:02606,RAD:SE50", nodelist.getNode("/what/source").data())
    self.assertEqual("SCAN", nodelist.getNode("/what/object").data())
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())
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

  def test_write_read_scan_withHowSubgroups(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    scan = vol.getScan(0)

    scan.addAttribute("how/attrib", "hello 0")
    scan.addAttribute("how/group1/attrib", "hello 1")
    scan.addAttribute("how/group1/group11/attrib", "hello 11")
    scan.addAttribute("how/group1/group11/attrib2", "hello 112")

    scan.getParameter("DBZH").addAttribute("how/attrib", "phello 0")
    scan.getParameter("DBZH").addAttribute("how/group1/attrib", "phello 1")
    scan.getParameter("DBZH").addAttribute("how/group1/group11/attrib", "phello 11")
    scan.getParameter("DBZH").addAttribute("how/group1/group11/attrib2", "phello 112")

    obj = _raveio.new()
    obj.object = scan
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("hello 0", nodelist.getNode("/how/attrib").data())
    self.assertEqual("hello 1", nodelist.getNode("/how/group1/attrib").data())
    self.assertEqual("hello 11", nodelist.getNode("/how/group1/group11/attrib").data())
    self.assertEqual("hello 112", nodelist.getNode("/how/group1/group11/attrib2").data())

    self.assertEqual("phello 0", nodelist.getNode("/dataset1/data1/how/attrib").data())
    self.assertEqual("phello 1", nodelist.getNode("/dataset1/data1/how/group1/attrib").data())
    self.assertEqual("phello 11", nodelist.getNode("/dataset1/data1/how/group1/group11/attrib").data())
    self.assertEqual("phello 112", nodelist.getNode("/dataset1/data1/how/group1/group11/attrib2").data())

    nobj = _raveio.open(self.TEMPORARY_FILE).object
    self.assertEqual("hello 0", nobj.getAttribute("how/attrib"))
    self.assertEqual("hello 1", nobj.getAttribute("how/group1/attrib"))
    self.assertEqual("hello 11", nobj.getAttribute("how/group1/group11/attrib"))
    self.assertEqual("hello 112", nobj.getAttribute("how/group1/group11/attrib2"))

    self.assertEqual("phello 0", nobj.getParameter("DBZH").getAttribute("how/attrib"))
    self.assertEqual("phello 1", nobj.getParameter("DBZH").getAttribute("how/group1/attrib"))
    self.assertEqual("phello 11", nobj.getParameter("DBZH").getAttribute("how/group1/group11/attrib"))
    self.assertEqual("phello 112", nobj.getParameter("DBZH").getAttribute("how/group1/group11/attrib2"))


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
                 "/what/version","/where","/where/height","/where/lat","/where/lon","/how","/how/beamwH","/how/beamwidth","/how/beamwV","/dataset1",
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

    self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())
    self.assertEqual("20090501", nodelist.getNode("/what/date").data())
    self.assertEqual("SCAN", nodelist.getNode("/what/object").data())
    self.assertEqual("WMO:02606,RAD:SE50", nodelist.getNode("/what/source").data())
    self.assertEqual("120000", nodelist.getNode("/what/time").data())
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())
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

  def test_save_scan_rstart_version_2_4(self):
    obj = _raveio.open(self.FIXTURE_VOLUME).object.getScan(0)
    obj.rstart = 0.5
    rio = _raveio.new()
    rio.object = obj

    rio.save(self.TEMPORARY_FILE)
    
    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertAlmostEqual(500.0, nodelist.getNode("/dataset1/where/rstart").data(), 4)

  def test_save_scan_rstart_version_2_3(self):
    obj = _raveio.open(self.FIXTURE_VOLUME).object.getScan(0)
    obj.rstart = 0.5
    rio = _raveio.new()
    rio.object = obj
    rio.version = _rave.RaveIO_ODIM_Version_2_3

    rio.save(self.TEMPORARY_FILE)
    
    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertAlmostEqual(0.5, nodelist.getNode("/dataset1/where/rstart").data(), 4)

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

  def test_write_volume_scan_count_scan_index(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object

    self.assertFalse(vol.hasAttribute("how/scan_count"))

    obj = _raveio.new()
    obj.object = vol
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual(10, nodelist.getNode("/how/scan_count").data())

  def test_write_volume_strict_2_3_success(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object

    obj = _raveio.new()
    obj.object = vol
    obj.filename = self.TEMPORARY_FILE
    obj.version = _rave.RaveIO_ODIM_Version_2_3
    obj.strict = True

    obj.save()

  def test_write_volume_strict_2_4_failure(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object

    obj = _raveio.new()
    obj.object = vol
    obj.filename = self.TEMPORARY_FILE
    obj.strict = True
    try:
        obj.save()
        self.fail("Expected IOError")
    except IOError as e:
        pass

  def test_write_volume_strict_2_4_success(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    nrscans = vol.getNumberOfScans()
    for i in range(nrscans):
        scan = vol.getScan(i)
        scan.addAttribute("how/simulated", True)
        scan.addAttribute("how/wavelength", 1.1)
        scan.addAttribute("how/pulsewidth", 1.1)
        scan.addAttribute("how/RXlossH", 1.1)
        scan.addAttribute("how/antgainH", 1.1)
        scan.addAttribute("how/beamwH", 1.1)
        scan.addAttribute("how/radconstH", 1.1)
        scan.addAttribute("how/NI", 1.1)
        scan.addAttribute("how/scan_index", i+1)
        scan.addAttribute("how/startazA", numpy.arange(0.0,360.0,1.0))
        scan.addAttribute("how/stopazA", numpy.arange(0.0,360.0,1.0))
    vol.source = vol.source + ",NOD:seang"
    obj = _raveio.new()
    obj.object = vol
    obj.filename = self.TEMPORARY_FILE
    obj.strict = True
    obj.save()

  def test_write_volume_strict_2_4_source(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    nrscans = vol.getNumberOfScans()
    for i in range(nrscans):
        scan = vol.getScan(i)
        scan.addAttribute("how/simulated", True)
        scan.addAttribute("how/wavelength", 1.1)
        scan.addAttribute("how/pulsewidth", 1.1)
        scan.addAttribute("how/RXlossH", 1.1)
        scan.addAttribute("how/antgainH", 1.1)
        scan.addAttribute("how/beamwH", 1.1)
        scan.addAttribute("how/radconstH", 1.1)
        scan.addAttribute("how/NI", 1.1)
        scan.addAttribute("how/scan_index", i+1)
        scan.addAttribute("how/startazA", numpy.arange(0.0,360.0,1.0))
        scan.addAttribute("how/stopazA", numpy.arange(0.0,360.0,1.0))
    vol.source = "WMO:00000"
    obj = _raveio.new()
    obj.object = vol
    obj.filename = self.TEMPORARY_FILE
    obj.strict = True
    try:
        obj.save()
        self.fail("Expected IOError")
    except IOError:
        pass
    obj.object.source="NOD:seang,WMO:00000"
    obj.save()

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
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())

    self.assertAlmostEqual(100.0, nodelist.getNode("/where/height").data(), 4)
    self.assertAlmostEqual(15.0, nodelist.getNode("/where/lat").data(), 4)
    self.assertAlmostEqual(10.0, nodelist.getNode("/where/lon").data(), 4)

    self.assertEqual("BALTRAD vp", nodelist.getNode("/dataset1/what/prodname").data())

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

  def test_write_read_vp_with_howSubgroups(self):
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
    vp.prodname = "With how subgroups"
    vp.addAttribute("how/attrib", "value")
    vp.addAttribute("how/subgroup1/attrib", "value 1")
    vp.addAttribute("how/subgroup1/subgroup2/attrib", "value 1 2")

    f1 = _ravefield.new()
    f1.setData(numpy.zeros((10,1), numpy.uint8))
    f1.addAttribute("what/quantity", "ff")
    f1.addAttribute("how/attrib", "ff value")
    f1.addAttribute("how/subgroup1/attrib", "ff value 1")
    f1.addAttribute("how/subgroup1/subgroup2/attrib", "ff value 1 2")

    vp.addField(f1)

    obj = _raveio.new()
    obj.object = vp
    obj.filename = self.TEMPORARY_FILE2
    obj.save()

    # Verify written data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("value", nodelist.getNode("/how/attrib").data())
    self.assertEqual("value 1", nodelist.getNode("/how/subgroup1/attrib").data())
    self.assertEqual("value 1 2", nodelist.getNode("/how/subgroup1/subgroup2/attrib").data())
    self.assertEqual("ff value", nodelist.getNode("/dataset1/data1/how/attrib").data())
    self.assertEqual("ff value 1", nodelist.getNode("/dataset1/data1/how/subgroup1/attrib").data())
    self.assertEqual("ff value 1 2", nodelist.getNode("/dataset1/data1/how/subgroup1/subgroup2/attrib").data())

    self.assertEqual("With how subgroups", nodelist.getNode("/dataset1/what/prodname").data())

    savedvp = _raveio.open(self.TEMPORARY_FILE2).object
    self.assertEqual("value", savedvp.getAttribute("how/attrib"))
    self.assertEqual("value 1", savedvp.getAttribute("how/subgroup1/attrib"))
    self.assertEqual("value 1 2", savedvp.getAttribute("how/subgroup1/subgroup2/attrib"))

    self.assertEqual("ff value", savedvp.getField("ff").getAttribute("how/attrib"))
    self.assertEqual("ff value 1", savedvp.getField("ff").getAttribute("how/subgroup1/attrib"))
    self.assertEqual("ff value 1 2", savedvp.getField("ff").getAttribute("how/subgroup1/subgroup2/attrib"))

  def test_write_read_vp_odim_23(self):
    vp = _verticalprofile.new()
    vp.date="20100101"
    vp.time="120000"
    vp.source="PLC:1234,WIGOS:0-123-1-123456"
    vp.longitude = 10.0 * math.pi / 180.0
    vp.latitude = 15.0 * math.pi / 180.0
    vp.setLevels(10)
    vp.prodname = "VP 23"

    f1 = _ravefield.new()
    f1.setData(numpy.zeros((10,1), numpy.uint8))
    f1.addAttribute("what/quantity", "ff")

    vp.addField(f1)

    obj = _raveio.new()
    obj.object = vp
    obj.version = _rave.RaveIO_ODIM_Version_2_3
    obj.save(self.TEMPORARY_FILE2)


    # Verify written data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("ODIM_H5/V2_3", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.3", nodelist.getNode("/what/version").data())

    self.assertEqual("BALTRAD", nodelist.getNode("/how/software").data())
    self.assertEqual("PLC:1234,WIGOS:0-123-1-123456", nodelist.getNode("/what/source").data())
    self.assertEqual("VP 23", nodelist.getNode("/dataset1/what/prodname").data())

  def test_write_read_vp_odim_22(self):
    vp = _verticalprofile.new()
    vp.date="20100101"
    vp.time="120000"
    vp.source="PLC:1234,WIGOS:0-123-1-123456"
    vp.longitude = 10.0 * math.pi / 180.0
    vp.latitude = 15.0 * math.pi / 180.0
    vp.setLevels(10)
    vp.prodname = "VP 23"

    f1 = _ravefield.new()
    f1.setData(numpy.zeros((10,1), numpy.uint8))
    f1.addAttribute("what/quantity", "ff")

    vp.addField(f1)

    obj = _raveio.new()
    obj.object = vp
    obj.version = _raveio.RaveIO_ODIM_Version_2_2
    obj.save(self.TEMPORARY_FILE2)

    # Verify written data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("ODIM_H5/V2_2", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())

    self.assertEqual("BALTRAD", nodelist.getNode("/how/software").data())
    self.assertEqual("PLC:1234", nodelist.getNode("/what/source").data())
    self.assertFalse("/dataset1/what/prodname" in nodelist.getNodeNames())

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
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())

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

  def getVpFieldName(self, nodelist, quantity):
    ctr = 1
    nodenames = nodelist.getNodeNames().keys()
    while True:
      nodename = "/dataset1/data%d/what/quantity"%ctr
      if nodename in nodenames:
        node = nodelist.getNode(nodename)
        if node.data() == quantity:
          return "/dataset1/data%d"%ctr
      if "/dataset1/data%d"%ctr not in nodenames:
          break
      ctr = ctr + 1
    return None

  def test_read_vp_write_2_3(self):
    # Read the new version of VP
    vp = _raveio.open(self.FIXTURE_VP_NEW_VERSION).object
    
    rio = _raveio.new()
    rio.object = vp
    rio.version = _raveio.RaveIO_ODIM_Version_2_3
    rio.save(self.TEMPORARY_FILE)
    
    # Verify written data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    vpfield = self.getVpFieldName(nodelist, "HGHT")
    self.assertTrue(vpfield is not None)
    self.assertAlmostEqual(1.0, nodelist.getNode("%s/what/gain"%vpfield).data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("%s/what/offset"%vpfield).data(), 4)

    vpfield = self.getVpFieldName(nodelist, "n")
    self.assertTrue(vpfield is not None)
    self.assertAlmostEqual(1.0, nodelist.getNode("%s/what/gain"%vpfield).data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("%s/what/offset"%vpfield).data(), 4)
    
  def test_read_vp_write_2_4(self):
    # Read the new version of VP
    vp = _raveio.open(self.FIXTURE_VP_NEW_VERSION).object
    
    rio = _raveio.new()
    rio.object = vp
    rio.version = _raveio.RaveIO_ODIM_Version_2_4
    rio.save(self.TEMPORARY_FILE)
    
    # Verify written data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    vpfield = self.getVpFieldName(nodelist, "HGHT")
    self.assertTrue(vpfield is not None)
    self.assertAlmostEqual(1000.0, nodelist.getNode("%s/what/gain"%vpfield).data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("%s/what/offset"%vpfield).data(), 4)

    vpfield = self.getVpFieldName(nodelist, "n")
    self.assertTrue(vpfield is not None)
    self.assertAlmostEqual(1.0, nodelist.getNode("%s/what/gain"%vpfield).data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("%s/what/offset"%vpfield).data(), 4)

  def test_write_2_4_read(self):
    # Read the new version of VP
    vp = _raveio.open(self.FIXTURE_VP_NEW_VERSION).object
    
    rio = _raveio.new()
    rio.object = vp
    rio.version = _raveio.RaveIO_ODIM_Version_2_4
    rio.save(self.TEMPORARY_FILE)
    
    # Verify written data
    vp = _raveio.open(self.TEMPORARY_FILE).object
    self.assertAlmostEqual(1.0, vp.getHGHT().getAttribute("what/gain"), 4)
    self.assertAlmostEqual(0.0, vp.getHGHT().getAttribute("what/offset"), 4)

    self.assertAlmostEqual(1.0, vp.getField("n").getAttribute("what/gain"), 4)
    self.assertAlmostEqual(0.0, vp.getField("n").getAttribute("what/offset"), 4)

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
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())

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

  def test_write_vp_strict_failure(self):
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

    obj = _raveio.new()
    obj.object = vp
    obj.filename = self.TEMPORARY_FILE2
    obj.strict = True
    try:
        obj.save()
        self.fail("Expected IOError")
    except IOError:
        pass

  def test_write_vp_strict_success_fields(self):
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
    f1.addAttribute("how/simulated",False)
    vp.addField(f1)
    f2 = _ravefield.new()
    f2.setData(numpy.zeros((10,1), numpy.uint8))
    f2.addAttribute("what/quantity", "VWND")
    f2.addAttribute("how/simulated",False)
    vp.addField(f2)

    obj = _raveio.new()
    obj.object = vp
    obj.filename = self.TEMPORARY_FILE2
    obj.strict = True
    try:
        obj.save()
        self.fail("Expected IOError")
    except IOError:
      pass
  
    obj.object.source = obj.object.source + ",NOD:selek"
    obj.save()

  def test_write_vp_strict_success(self):
    vp = _verticalprofile.new()
    vp.date="20100101"
    vp.startdate="20100101"
    vp.enddate="20100101"
    vp.time="120000"
    vp.starttime="120202"
    vp.endtime="120405"
    vp.source="PLC:Leksand,NOD:selek"
    vp.product= "VP"
    vp.longitude = 10.0 * math.pi / 180.0
    vp.latitude = 15.0 * math.pi / 180.0
    vp.setLevels(10)
    vp.height = 100.0
    vp.interval = 5.0
    vp.minheight = 10.0
    vp.maxheight = 20.0
    vp.addAttribute("how/simulated",False)
    f1 = _ravefield.new()
    f1.setData(numpy.zeros((10,1), numpy.uint8))
    f1.addAttribute("what/quantity", "UWND")
    vp.addField(f1)

    obj = _raveio.new()
    obj.object = vp
    obj.filename = self.TEMPORARY_FILE2
    obj.strict = True
    obj.save()

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
    self.assertAlmostEqual(1.6, obj.object.beamwH*180.0/math.pi, 2)
    #self.assertAlmostEqual(1.6, obj.object.getAttribute("how/beamwH"), 2)
    #self.assertAlmostEqual(1.6, obj.object.getAttribute("how/beamwidth"), 2)
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
    self.assertAlmostEqual(3.6, obj.object.getParameter("DBZH").getAttribute("how/beamwidth"), 2)
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
    self.assertAlmostEqual(1.6, obj.object.beamwH*180.0/math.pi)
    #self.assertAlmostEqual(1.6, obj.object.getAttribute("how/beamwH"), 2)
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
    self.assertAlmostEqual(3.6, obj.object.getScan(0).getParameter("DBZH").getAttribute("how/beamwidth"), 2)
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
    self.assertAlmostEqual(5.6, obj.object.getScan(0).getParameter("VRADH").getAttribute("how/beamwidth"), 2)
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
    self.assertAlmostEqual(7.6, obj.object.getScan(1).getParameter("CCORH").getAttribute("how/beamwidth"), 2)
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
  """
  """
  def testReadAndConvertV24_how_attributes_SCAN(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/what")
    self.addGroupNode(nodelist, "/where")
    self.addGroupNode(nodelist, "/how")

    self.addGroupNode(nodelist, "/dataset1")
    self.addGroupNode(nodelist, "/dataset1/what")
    self.addGroupNode(nodelist, "/dataset1/data1")
    self.addGroupNode(nodelist, "/dataset1/data1/how")
    self.addGroupNode(nodelist, "/dataset1/data1/what")
    self.addGroupNode(nodelist, "/dataset1/quality1")
    self.addGroupNode(nodelist, "/dataset1/quality1/how")
    self.addGroupNode(nodelist, "/dataset1/data1/quality1")
    self.addGroupNode(nodelist, "/dataset1/data1/quality1/how")

    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_4")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "SCAN")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.3")

    self.addAttributeNode(nodelist, "/where/height", "double", 100.0)
    self.addAttributeNode(nodelist, "/where/lon", "double", 13.5)
    self.addAttributeNode(nodelist, "/where/lat", "double", 61.0)

    self.addAttributeNode(nodelist, "/how/gasattn", "double", 0.0221)
    self.addAttributeNode(nodelist, "/how/nomTXpower", "double", 85.6832)
    self.addAttributeNode(nodelist, "/how/nsampleH", "double", 2.2)
    self.addAttributeNode(nodelist, "/how/nsampleV", "double", 2.2)
    self.addAttributeNode(nodelist, "/how/melting_layer_top_A", "double", numpy.array([2200.0, 2300.0, 2400.0]))
    self.addAttributeNode(nodelist, "/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 2100.0, 2200.0]))
    self.addAttributeNode(nodelist, "/how/peakpwr", "double", 85.7310)
    self.addAttributeNode(nodelist, "/how/avgpwr", "double", 85.7194)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/gasattn", "double", 0.0231)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nomTXpower", "double", 85.6832)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nsampleH", "double", 23.2)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nsampleV", "double", 23.2)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_top_A", "double", numpy.array([2200.0, 2300.0, 2400.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 2100.0, 2200.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/peakpwr", "double", 85.7322)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/avgpwr", "double", 85.7206)
    
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/nsampleH", "double", 1.1)
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/nsampleV", "double", 1.1)
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/melting_layer_top_A", "double", numpy.array([2200.0, 2300.0, 2400.0]))
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 2100.0, 2200.0]))
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/peakpwr", "double", 85.6949)
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/avgpwr", "double", 85.7206)

    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/nsampleH", "double", 2.1)
    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/nsampleV", "double", 2.1)
    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/melting_layer_top_A", "double", numpy.array([2200.0, 2300.0, 2400.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 2100.0, 2200.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/peakpwr", "double", 85.7183)
    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/avgpwr", "double", 85.6949)

    self.addAttributeNode(nodelist, "/dataset1/what/startdate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset1/what/starttime", "string", "101010")
    self.addAttributeNode(nodelist, "/dataset1/what/enddate", "string", "20100101")
    self.addAttributeNode(nodelist, "/dataset1/what/endtime", "string", "101011")
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

    obj = _raveio.open(self.TEMPORARY_FILE).object
    self.assertAlmostEqual(22.1, obj.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, obj.getAttribute("how/nomTXpower"), 4)
    self.assertAlmostEqual(2.2, obj.getAttribute("how/nsampleH"), 4)
    self.assertAlmostEqual(2.2, obj.getAttribute("how/nsampleV"), 4)
    self.assertTrue(numpy.allclose(obj.getAttribute("how/melting_layer_top_A"), numpy.array([2200.0, 2300.0, 2400.0]), atol=1e-1))
    self.assertTrue(numpy.allclose(obj.getAttribute("how/melting_layer_bottom_A"), numpy.array([2000.0, 2100.0, 2200.0]), atol=1e-1))
    self.assertAlmostEqual(374.1967, obj.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.1986, obj.getAttribute("how/avgpwr"), 4)
    
    param = obj.getParameter("DBZH")
    self.assertAlmostEqual(23.1, param.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, param.getAttribute("how/nomTXpower"), 4)
    self.assertAlmostEqual(23.2, param.getAttribute("how/nsampleH"), 4)
    self.assertAlmostEqual(23.2, param.getAttribute("how/nsampleV"), 4)
    self.assertTrue(numpy.allclose(param.getAttribute("how/melting_layer_top_A"), numpy.array([2200.0, 2300.0, 2400.0]), atol=1e-1))
    self.assertTrue(numpy.allclose(param.getAttribute("how/melting_layer_bottom_A"), numpy.array([2000.0, 2100.0, 2200.0]), atol=1e-1))
    self.assertAlmostEqual(374.3001, param.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.3017, param.getAttribute("how/avgpwr"), 4)
    
    qfield = obj.getQualityField(0)
    self.assertAlmostEqual(1.1, qfield.getAttribute("how/nsampleH"), 4)
    self.assertAlmostEqual(1.1, qfield.getAttribute("how/nsampleV"), 4)
    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_top_A"), numpy.array([2200.0, 2300.0, 2400.0]), atol=1e-1))
    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_bottom_A"), numpy.array([2000.0, 2100.0, 2200.0]), atol=1e-1))
    self.assertAlmostEqual(371.0992, qfield.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.3017, qfield.getAttribute("how/avgpwr"), 4)

    qfield = param.getQualityField(0)
    self.assertAlmostEqual(2.1, qfield.getAttribute("how/nsampleH"), 4)
    self.assertAlmostEqual(2.1, qfield.getAttribute("how/nsampleV"), 4)
    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_top_A"), numpy.array([2200.0, 2300.0, 2400.0]), atol=1e-1))
    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_bottom_A"), numpy.array([2000.0, 2100.0, 2200.0]), atol=1e-1))
    
    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_top"), numpy.array([2.2, 2.3, 2.4]), 1e-1)) # Also verify if using old name
    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_bottom"), numpy.array([2.0, 2.1, 2.2]), 1e-1))
    self.assertAlmostEqual(373.1041, qfield.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(371.0992, qfield.getAttribute("how/avgpwr"), 4)
    
  def testReadAndConvertV24_how_attributes_VOLUME(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/what")
    self.addGroupNode(nodelist, "/where")
    self.addGroupNode(nodelist, "/how")

    self.addGroupNode(nodelist, "/dataset1")
    self.addGroupNode(nodelist, "/dataset1/what")
    self.addGroupNode(nodelist, "/dataset1/how")
    self.addGroupNode(nodelist, "/dataset1/data1")
    self.addGroupNode(nodelist, "/dataset1/data1/how")
    self.addGroupNode(nodelist, "/dataset1/data1/what")
    self.addGroupNode(nodelist, "/dataset1/quality1")
    self.addGroupNode(nodelist, "/dataset1/quality1/how")

    self.addGroupNode(nodelist, "/dataset1/data1/quality1")
    self.addGroupNode(nodelist, "/dataset1/data1/quality1/how")

    self.addGroupNode(nodelist, "/dataset2")
    self.addGroupNode(nodelist, "/dataset2/what")
    self.addGroupNode(nodelist, "/dataset2/data1")
    self.addGroupNode(nodelist, "/dataset2/data1/how")
    self.addGroupNode(nodelist, "/dataset2/data1/what")

    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_4")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "PVOL")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.3")

    self.addAttributeNode(nodelist, "/where/height", "double", 100.0)
    self.addAttributeNode(nodelist, "/where/lon", "double", 13.5)
    self.addAttributeNode(nodelist, "/where/lat", "double", 61.0)

    self.addAttributeNode(nodelist, "/how/gasattn", "double", 0.0221)
    self.addAttributeNode(nodelist, "/how/nomTXpower", "double", 85.6832)
    self.addAttributeNode(nodelist, "/how/nsampleH", "double", 2.2)
    self.addAttributeNode(nodelist, "/how/nsampleV", "double", 2.2)
    self.addAttributeNode(nodelist, "/how/melting_layer_top_A", "double", numpy.array([2200.0, 2300.0, 2400.0]))
    self.addAttributeNode(nodelist, "/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 2100.0, 2200.0]))
    self.addAttributeNode(nodelist, "/how/peakpwr", "double", 85.7310)
    self.addAttributeNode(nodelist, "/how/avgpwr", "double", 85.7194)

    self.addAttributeNode(nodelist, "/dataset1/how/gasattn", "double", 0.0211)
    self.addAttributeNode(nodelist, "/dataset1/how/nomTXpower", "double", 85.6949)
    self.addAttributeNode(nodelist, "/dataset1/how/nsampleH", "double", 23.1)
    self.addAttributeNode(nodelist, "/dataset1/how/nsampleV", "double", 23.1)
    self.addAttributeNode(nodelist, "/dataset1/how/melting_layer_top_A", "double", numpy.array([2200.0, 2300.0, 2400.0]))
    self.addAttributeNode(nodelist, "/dataset1/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 2100.0, 2200.0]))
    self.addAttributeNode(nodelist, "/dataset1/how/peakpwr", "double", 85.7426)
    self.addAttributeNode(nodelist, "/dataset1/how/avgpwr", "double", 85.7310)
    
    self.addAttributeNode(nodelist, "/dataset1/data1/how/gasattn", "double", 0.0231)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nomTXpower", "double", 85.6832)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nsampleH", "double", 23.2)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nsampleV", "double", 23.2)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_top_A", "double", numpy.array([2200.0, 2300.0, 2400.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 2100.0, 2200.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/peakpwr", "double", 85.7322)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/avgpwr", "double", 85.7206)
    
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/nsampleH", "double", 1.1)
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/nsampleV", "double", 1.1)
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/melting_layer_top_A", "double", numpy.array([2200.0, 2300.0, 2400.0]))
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 2100.0, 2200.0]))
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/peakpwr", "double", 85.7183)
    self.addAttributeNode(nodelist, "/dataset1/quality1/how/avgpwr", "double", 85.6949)

    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/nsampleH", "double", 2.2)
    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/nsampleV", "double", 2.2)
    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/melting_layer_top_A", "double", numpy.array([2200.0, 2300.0, 2400.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 2100.0, 2200.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/peakpwr", "double", 85.6949)
    self.addAttributeNode(nodelist, "/dataset1/data1/quality1/how/avgpwr", "double", 85.6961)
    
    self.addAttributeNode(nodelist, "/dataset1/what/product", "string", "SCAN")
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "DBZH")

    self.addAttributeNode(nodelist, "/dataset2/data1/how/gasattn", "double", 23.1)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/nomTXpower", "double", 370.1)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/nsampleH", "double", 23.2)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/nsampleV", "double", 23.2)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/melting_layer_top_A", "double", numpy.array([2200.0, 2300.0, 2400.0]))
    self.addAttributeNode(nodelist, "/dataset2/data1/how/melting_layer_bottom_A", "double",numpy.array([2000.0, 2100.0, 2200.0]))
    self.addAttributeNode(nodelist, "/dataset2/data1/how/peakpwr", "double", 374.3)
    self.addAttributeNode(nodelist, "/dataset2/data1/how/avgpwr", "double", 373.3)
    self.addAttributeNode(nodelist, "/dataset2/what/product", "string", "SCAN")
    self.addAttributeNode(nodelist, "/dataset2/data1/what/quantity", "string", "DBZH")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,10), dset)

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    self.addDatasetNode(nodelist, "/dataset2/data1/data", "uchar", (10,10), dset)

    nodelist.write(self.TEMPORARY_FILE, 6)
    os.sync(); os.sync()

    obj = _raveio.open(self.TEMPORARY_FILE).object
    self.assertAlmostEqual(22.1, obj.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, obj.getAttribute("how/nomTXpower"), 4)
    self.assertAlmostEqual(2.2, obj.getAttribute("how/nsampleH"), 4)
    self.assertAlmostEqual(2.2, obj.getAttribute("how/nsampleV"), 4)
    self.assertTrue(numpy.allclose(obj.getAttribute("how/melting_layer_top_A"), numpy.array([2200.0, 2300.0, 2400.0]), atol=1e-1))
    self.assertTrue(numpy.allclose(obj.getAttribute("how/melting_layer_bottom_A"), numpy.array([2000.0, 2100.0, 2200.0]), atol=1e-1))
    self.assertAlmostEqual(374.1967, obj.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.1986, obj.getAttribute("how/avgpwr"), 4)
  
    scan1 = obj.getScan(0)
    self.assertAlmostEqual(21.1, scan1.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(371.0992, scan1.getAttribute("how/nomTXpower"), 4)
    self.assertAlmostEqual(23.1, scan1.getAttribute("how/nsampleH"), 4)
    self.assertAlmostEqual(23.1, scan1.getAttribute("how/nsampleV"), 4)
    self.assertTrue(numpy.allclose(scan1.getAttribute("how/melting_layer_top_A"), numpy.array([2200.0, 2300.0, 2400.0]), atol=1e-1))
    self.assertTrue(numpy.allclose(scan1.getAttribute("how/melting_layer_bottom_A"), numpy.array([2000.0, 2100.0, 2200.0]), atol=1e-1))
    self.assertAlmostEqual(375.1976, scan1.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(374.1967, scan1.getAttribute("how/avgpwr"), 4)
    
    qfield = scan1.getQualityField(0)
    self.assertAlmostEqual(1.1, qfield.getAttribute("how/nsampleH"), 4)
    self.assertAlmostEqual(1.1, qfield.getAttribute("how/nsampleV"), 4)
    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_top_A"), numpy.array([2200.0, 2300.0, 2400.0]), atol=1e-1))
    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_bottom_A"), numpy.array([2000.0, 2100.0, 2200.0]), atol=1e-1))
    self.assertAlmostEqual(373.1041, qfield.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(371.0992, qfield.getAttribute("how/avgpwr"), 4)
    
    param1 = scan1.getParameter("DBZH")
    self.assertAlmostEqual(23.1, param1.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, param1.getAttribute("how/nomTXpower"), 4)
    self.assertAlmostEqual(23.2, param1.getAttribute("how/nsampleH"), 4)
    self.assertAlmostEqual(23.2, param1.getAttribute("how/nsampleV"), 4)
    self.assertTrue(numpy.allclose(param1.getAttribute("how/melting_layer_top_A"), numpy.array([2200.0, 2300.0, 2400.0]), atol=1e-1))
    self.assertTrue(numpy.allclose(param1.getAttribute("how/melting_layer_bottom_A"), numpy.array([2000.0, 2100.0, 2200.0]), atol=1e-1))
    self.assertAlmostEqual(374.3001, param1.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.3017, param1.getAttribute("how/avgpwr"), 4)

    qfield = param1.getQualityField(0)
    self.assertAlmostEqual(2.2, qfield.getAttribute("how/nsampleH"), 4)
    self.assertAlmostEqual(2.2, qfield.getAttribute("how/nsampleV"), 4)
    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_top_A"), numpy.array([2200.0, 2300.0, 2400.0]), atol=1e-1))
    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_bottom_A"), numpy.array([2000.0, 2100.0, 2200.0]), atol=1e-1))

    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_top"), numpy.array([2.2, 2.3, 2.4]), 1e-1)) # Also verify if using old name
    self.assertTrue(numpy.allclose(qfield.getAttribute("how/melting_layer_bottom"), numpy.array([2.0, 2.1, 2.2]), 1e-1))
    
    self.assertAlmostEqual(371.0992, qfield.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(371.2017, qfield.getAttribute("how/avgpwr"), 4)

  def test_write_scan_with_converted_howAttributes(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object
    
    scan = vol.getScan(0)
    scan.addAttribute("how/gasattn", 22.1)
    scan.addAttribute("how/nomTXpower", 370.1008)
    scan.addAttribute("how/nsampleH", 2.2)
    scan.addAttribute("how/nsampleV", 2.3)
    scan.addAttribute("how/melting_layer_top", numpy.array([2200.0, 2300.0, 2400.0]))
    scan.addAttribute("how/melting_layer_bottom", numpy.array([2000.0, 2100.0, 2200.0]))
    scan.addAttribute("how/peakpwr", 374.1967)
    scan.addAttribute("how/avgpwr", 373.1986)

    scan.removeParametersExcept(["DBZH"])
    param = scan.getParameter("DBZH")
    param.addAttribute("how/gasattn", 22.1)
    param.addAttribute("how/nomTXpower", 370.1008)
    param.addAttribute("how/nsampleH", 2.2)
    param.addAttribute("how/nsampleV", 2.3)
    param.addAttribute("how/melting_layer_top", numpy.array([2200.0, 2300.0, 2400.0]))
    param.addAttribute("how/melting_layer_bottom", numpy.array([2000.0, 2100.0, 2200.0]))
    param.addAttribute("how/peakpwr", 374.1967)
    param.addAttribute("how/avgpwr", 373.1986)

    field = _ravefield.new()
    field.setData(scan.getParameter("DBZH").getData())
    field.addAttribute("how/gasattn", 22.1)
    field.addAttribute("how/nomTXpower", 370.1008)
    field.addAttribute("how/nsampleH", 2.2)
    field.addAttribute("how/nsampleV", 2.3)
    field.addAttribute("how/melting_layer_top", numpy.array([2200.0, 2300.0, 2400.0]))
    field.addAttribute("how/melting_layer_bottom", numpy.array([2000.0, 2100.0, 2200.0]))
    field.addAttribute("how/peakpwr", 374.1967)
    field.addAttribute("how/avgpwr", 373.1986)
      
    scan.addQualityField(field)
  
    field = _ravefield.new()
    field.setData(scan.getParameter("DBZH").getData())
    field.addAttribute("how/gasattn", 22.1)
    field.addAttribute("how/nomTXpower", 370.1008)
    field.addAttribute("how/nsampleH", 2.2)
    field.addAttribute("how/nsampleV", 2.3)
    field.addAttribute("how/melting_layer_top", numpy.array([2200.0, 2300.0, 2400.0]))
    field.addAttribute("how/melting_layer_bottom", numpy.array([2000.0, 2100.0, 2200.0]))
    field.addAttribute("how/peakpwr", 374.1967)
    field.addAttribute("how/avgpwr", 373.1986)
    
    param.addQualityField(field)
    
    obj = _raveio.new()
    obj.object = scan
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    
    self.assertAlmostEqual(0.0221, nodelist.getNode("/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/how/nomTXpower").data(), 4)
    self.assertAlmostEqual(2.2, nodelist.getNode("/how/nsampleH").data(), 4)
    self.assertAlmostEqual(2.3, nodelist.getNode("/how/nsampleV").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([2200000.0, 2300000.0, 2400000.0]), nodelist.getNode("/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000000.0, 2100000.0, 2200000.0]), nodelist.getNode("/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/how/avgpwr").data(), 4)

    self.assertAlmostEqual(0.0221, nodelist.getNode("/dataset1/data1/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/dataset1/data1/how/nomTXpower").data(), 4)
    self.assertAlmostEqual(2.2, nodelist.getNode("/dataset1/data1/how/nsampleH").data(), 4)
    self.assertAlmostEqual(2.3, nodelist.getNode("/dataset1/data1/how/nsampleV").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([2200000.0, 2300000.0, 2400000.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000000.0, 2100000.0, 2200000.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/dataset1/data1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/dataset1/data1/how/avgpwr").data(), 4)
 
    self.assertAlmostEqual(0.0221, nodelist.getNode("/dataset1/quality1/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/dataset1/quality1/how/nomTXpower").data(), 4)
    self.assertAlmostEqual(2.2, nodelist.getNode("/dataset1/quality1/how/nsampleH").data(), 4)
    self.assertAlmostEqual(2.3, nodelist.getNode("/dataset1/quality1/how/nsampleV").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([2200000.0, 2300000.0, 2400000.0]), nodelist.getNode("/dataset1/quality1/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000000.0, 2100000.0, 2200000.0]), nodelist.getNode("/dataset1/quality1/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/dataset1/quality1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/dataset1/quality1/how/avgpwr").data(), 4)
 
    self.assertAlmostEqual(0.0221, nodelist.getNode("/dataset1/data1/quality1/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/dataset1/data1/quality1/how/nomTXpower").data(), 4)
    self.assertAlmostEqual(2.2, nodelist.getNode("/dataset1/data1/quality1/how/nsampleH").data(), 4)
    self.assertAlmostEqual(2.3, nodelist.getNode("/dataset1/data1/quality1/how/nsampleV").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([2200000.0, 2300000.0, 2400000.0]), nodelist.getNode("/dataset1/data1/quality1/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000000.0, 2100000.0, 2200000.0]), nodelist.getNode("/dataset1/data1/quality1/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/dataset1/data1/quality1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/dataset1/data1/quality1/how/avgpwr").data(), 4)

  def test_write_volume_with_converted_howAttributes(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object

    vol.addAttribute("how/gasattn", 23.1)
    vol.addAttribute("how/nomTXpower", 371.1008)
    vol.addAttribute("how/nsampleH", 2.3)
    vol.addAttribute("how/nsampleV", 2.4)
    vol.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 4.0]))
    vol.addAttribute("how/melting_layer_bottom", numpy.array([4.0, 5.0, 7.0]))
    vol.addAttribute("how/peakpwr", 374.1911)
    vol.addAttribute("how/avgpwr", 373.1999)
    vol.addAttribute("how/pulsewidth", 123.1)
    vol.addAttribute("how/wavelength", 99.9)
    vol.addAttribute("how/not_handled", 1.3)
    
    scan = vol.getScan(0)
    scan.addAttribute("how/gasattn", 22.1)
    scan.addAttribute("how/nomTXpower", 370.1008)
    scan.addAttribute("how/nsampleH", 2.2)
    scan.addAttribute("how/nsampleV", 2.3)
    scan.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    scan.addAttribute("how/melting_layer_bottom", numpy.array([4.0, 5.0, 6.0]))
    scan.addAttribute("how/peakpwr", 374.1967)
    scan.addAttribute("how/avgpwr", 373.1986)
    scan.addAttribute("how/pulsewidth", 123.0)
    scan.addAttribute("how/not_handled", 1.2)
    
    scan.removeParametersExcept(["DBZH"])
    
    param = scan.getParameter("DBZH")
    param.addAttribute("how/gasattn", 22.1)
    param.addAttribute("how/nomTXpower", 370.1008)
    param.addAttribute("how/nsampleH", 2.2)
    param.addAttribute("how/nsampleV", 2.3)
    param.addAttribute("how/melting_layer_top_A", numpy.array([1.0, 2.0, 3.0]))
    param.addAttribute("how/melting_layer_bottom_A", numpy.array([4.0, 5.0, 6.0]))
    param.addAttribute("how/melting_layer_top", 1.0)
    param.addAttribute("how/melting_layer_bottom", 2.0)
    param.addAttribute("how/peakpwr", 374.1967)
    param.addAttribute("how/avgpwr", 373.1986)

    field = _ravefield.new()
    field.setData(scan.getParameter("DBZH").getData())
    field.addAttribute("how/gasattn", 22.1)
    field.addAttribute("how/nomTXpower", 370.1008)
    field.addAttribute("how/nsampleH", 2.2)
    field.addAttribute("how/nsampleV", 2.3)
    field.addAttribute("how/melting_layer_top_A", numpy.array([1.0, 2.0, 3.0]))
    field.addAttribute("how/melting_layer_bottom_A", numpy.array([4.0, 5.0, 6.0]))
    field.addAttribute("how/melting_layer_top", 1.0)
    field.addAttribute("how/melting_layer_bottom", 2.0)
    field.addAttribute("how/peakpwr", 374.1967)
    field.addAttribute("how/avgpwr", 373.1986)
    scan.addQualityField(field)

    param.addQualityField(field)
    
    obj = _raveio.new()
    obj.object = vol
    obj.filename = self.TEMPORARY_FILE
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertAlmostEqual(0.0231, nodelist.getNode("/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6949, nodelist.getNode("/how/nomTXpower").data(), 4)
    self.assertAlmostEqual(2.3, nodelist.getNode("/how/nsampleH").data(), 4)
    self.assertAlmostEqual(2.4, nodelist.getNode("/how/nsampleV").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 4000.0]), nodelist.getNode("/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([4000.0, 5000.0, 7000.0]), nodelist.getNode("/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7309, nodelist.getNode("/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/how/avgpwr").data(), 4)
    self.assertAlmostEqual(0.000123, nodelist.getNode("/how/pulsewidth").data(), 4)
    self.assertAlmostEqual(300092550.5506, nodelist.getNode("/how/frequency").data(), 4)
    
    self.assertAlmostEqual(1.3, nodelist.getNode("/how/not_handled").data(), 4)

    self.assertAlmostEqual(0.0221, nodelist.getNode("/dataset1/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/dataset1/how/nomTXpower").data(), 4)
    self.assertAlmostEqual(2.2, nodelist.getNode("/dataset1/how/nsampleH").data(), 4)
    self.assertAlmostEqual(2.3, nodelist.getNode("/dataset1/how/nsampleV").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), nodelist.getNode("/dataset1/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([4000.0, 5000.0, 6000.0]), nodelist.getNode("/dataset1/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/dataset1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/dataset1/how/avgpwr").data(), 4)
    self.assertAlmostEqual(0.000123, nodelist.getNode("/dataset1/how/pulsewidth").data(), 4)
    self.assertAlmostEqual(1.2, nodelist.getNode("/dataset1/how/not_handled").data(), 4)
    
    self.assertAlmostEqual(0.0221, nodelist.getNode("/dataset1/data1/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/dataset1/data1/how/nomTXpower").data(), 4)
    self.assertAlmostEqual(2.2, nodelist.getNode("/dataset1/data1/how/nsampleH").data(), 4)
    self.assertAlmostEqual(2.3, nodelist.getNode("/dataset1/data1/how/nsampleV").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_top_A").data(), 4))
    self.assertTrue(numpy.allclose(numpy.array([4.0, 5.0, 6.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_bottom_A").data(), 4))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/dataset1/data1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/dataset1/data1/how/avgpwr").data(), 4)

    self.assertAlmostEqual(0.0221, nodelist.getNode("/dataset1/quality1/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/dataset1/quality1/how/nomTXpower").data(), 4)
    self.assertAlmostEqual(2.2, nodelist.getNode("/dataset1/quality1/how/nsampleH").data(), 4)
    self.assertAlmostEqual(2.3, nodelist.getNode("/dataset1/quality1/how/nsampleV").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/quality1/how/melting_layer_top_A").data(), 4))
    self.assertTrue(numpy.allclose(numpy.array([4.0, 5.0, 6.0]), nodelist.getNode("/dataset1/quality1/how/melting_layer_bottom_A").data(), 4))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/dataset1/quality1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/dataset1/quality1/how/avgpwr").data(), 4)

    self.assertAlmostEqual(0.0221, nodelist.getNode("/dataset1/data1/quality1/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/dataset1/data1/quality1/how/nomTXpower").data(), 4)
    self.assertAlmostEqual(2.2, nodelist.getNode("/dataset1/data1/quality1/how/nsampleH").data(), 4)
    self.assertAlmostEqual(2.3, nodelist.getNode("/dataset1/data1/quality1/how/nsampleV").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/data1/quality1/how/melting_layer_top_A").data(), 4))
    self.assertTrue(numpy.allclose(numpy.array([4.0, 5.0, 6.0]), nodelist.getNode("/dataset1/data1/quality1/how/melting_layer_bottom_A").data(), 4))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/dataset1/data1/quality1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/dataset1/data1/quality1/how/avgpwr").data(), 4)

  def test_write_volume_with_2_3_howAttributes(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.object

    vol.addAttribute("how/gasattn", 23.1)
    vol.addAttribute("how/nomTXpower", 371.1008)
    vol.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 4.0]))
    vol.addAttribute("how/melting_layer_bottom", numpy.array([4.0, 5.0, 7.0]))
    vol.addAttribute("how/peakpwr", 374.1911)
    vol.addAttribute("how/avgpwr", 373.1999)
    vol.addAttribute("how/pulsewidth", 123.1)
    vol.addAttribute("how/wavelength", 99.9)
    vol.addAttribute("how/not_handled", 1.3)
    
    scan = vol.getScan(0)
    scan.addAttribute("how/gasattn", 22.1)
    scan.addAttribute("how/nomTXpower", 370.1008)
    scan.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    scan.addAttribute("how/melting_layer_bottom", numpy.array([4.0, 5.0, 6.0]))
    scan.addAttribute("how/peakpwr", 374.1967)
    scan.addAttribute("how/avgpwr", 373.1986)
    scan.addAttribute("how/pulsewidth", 123.0)
    scan.addAttribute("how/not_handled", 1.2)
    
    scan.removeParametersExcept(["DBZH"])
    
    param = scan.getParameter("DBZH")
    param.addAttribute("how/gasattn", 22.1)
    param.addAttribute("how/nomTXpower", 370.1008)
    param.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    param.addAttribute("how/melting_layer_bottom", numpy.array([4.0, 5.0, 6.0]))
    param.addAttribute("how/peakpwr", 374.1967)
    param.addAttribute("how/avgpwr", 373.1986)

    field = _ravefield.new()
    field.setData(scan.getParameter("DBZH").getData())
    field.addAttribute("how/gasattn", 22.1)
    field.addAttribute("how/nomTXpower", 370.1008)
    field.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    field.addAttribute("how/melting_layer_bottom", numpy.array([4.0, 5.0, 6.0]))
    field.addAttribute("how/peakpwr", 374.1967)
    field.addAttribute("how/avgpwr", 373.1986)
    scan.addQualityField(field)

    param.addQualityField(field)
    
    obj = _raveio.new()
    obj.object = vol
    obj.filename = self.TEMPORARY_FILE
    obj.version = _rave.RaveIO_ODIM_Version_2_3
    obj.save()

    # Verify data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertAlmostEqual(23.1, nodelist.getNode("/how/gasattn").data(), 4)
    self.assertAlmostEqual(371.1008, nodelist.getNode("/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 4.0]), nodelist.getNode("/how/melting_layer_top").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([4.0, 5.0, 7.0]), nodelist.getNode("/how/melting_layer_bottom").data(), atol=1e-1))
    self.assertAlmostEqual(374.1911, nodelist.getNode("/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1999, nodelist.getNode("/how/avgpwr").data(), 4)
    self.assertAlmostEqual(123.1, nodelist.getNode("/how/pulsewidth").data(), 4)
    self.assertAlmostEqual(99.9, nodelist.getNode("/how/wavelength").data(), 4)
    self.assertAlmostEqual(1.3, nodelist.getNode("/how/not_handled").data(), 4)

    self.assertAlmostEqual(22.1, nodelist.getNode("/dataset1/how/gasattn").data(), 4)
    self.assertAlmostEqual(370.1008, nodelist.getNode("/dataset1/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/how/melting_layer_top").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([4.0, 5.0, 6.0]), nodelist.getNode("/dataset1/how/melting_layer_bottom").data(), atol=1e-1))
    self.assertAlmostEqual(374.1967, nodelist.getNode("/dataset1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1986, nodelist.getNode("/dataset1/how/avgpwr").data(), 4)
    self.assertAlmostEqual(123.0, nodelist.getNode("/dataset1/how/pulsewidth").data(), 4)
    self.assertAlmostEqual(1.2, nodelist.getNode("/dataset1/how/not_handled").data(), 4)
    
    self.assertAlmostEqual(22.1, nodelist.getNode("/dataset1/data1/how/gasattn").data(), 4)
    self.assertAlmostEqual(370.1008, nodelist.getNode("/dataset1/data1/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_top").data(), 4))
    self.assertTrue(numpy.allclose(numpy.array([4.0, 5.0, 6.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_bottom").data(), 4))
    self.assertAlmostEqual(374.1967, nodelist.getNode("/dataset1/data1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1986, nodelist.getNode("/dataset1/data1/how/avgpwr").data(), 4)

    self.assertAlmostEqual(22.1, nodelist.getNode("/dataset1/quality1/how/gasattn").data(), 4)
    self.assertAlmostEqual(370.1008, nodelist.getNode("/dataset1/quality1/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/quality1/how/melting_layer_top").data(), 4))
    self.assertTrue(numpy.allclose(numpy.array([4.0, 5.0, 6.0]), nodelist.getNode("/dataset1/quality1/how/melting_layer_bottom").data(), 4))
    self.assertAlmostEqual(374.1967, nodelist.getNode("/dataset1/quality1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1986, nodelist.getNode("/dataset1/quality1/how/avgpwr").data(), 4)

    self.assertAlmostEqual(22.1, nodelist.getNode("/dataset1/data1/quality1/how/gasattn").data(), 4)
    self.assertAlmostEqual(370.1008, nodelist.getNode("/dataset1/data1/quality1/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/data1/quality1/how/melting_layer_top").data(), 4))
    self.assertTrue(numpy.allclose(numpy.array([4.0, 5.0, 6.0]), nodelist.getNode("/dataset1/data1/quality1/how/melting_layer_bottom").data(), 4))
    self.assertAlmostEqual(374.1967, nodelist.getNode("/dataset1/data1/quality1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1986, nodelist.getNode("/dataset1/data1/quality1/how/avgpwr").data(), 4)

  def test_write_vp_convertedHow_2_3(self):
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

    vp.addAttribute("how/gasattn", 22.1)
    vp.addAttribute("how/nomTXpower", 370.1008)
    vp.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    vp.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    vp.addAttribute("how/peakpwr", 374.1967)
    vp.addAttribute("how/avgpwr", 373.1986)

    f1 = _ravefield.new()
    f1.setData(numpy.zeros((10,1), numpy.uint8))
    f1.addAttribute("what/quantity", "ff")
    f1.addAttribute("how/gasattn", 22.1)
    f1.addAttribute("how/nomTXpower", 370.1008)
    f1.addAttribute("how/nsampleH", 2.2)
    f1.addAttribute("how/nsampleV", 2.3)
    f1.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    f1.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    f1.addAttribute("how/peakpwr", 374.1967)
    f1.addAttribute("how/avgpwr", 373.1986)
    
    vp.addField(f1)

    obj = _raveio.new()
    obj.object = vp
    obj.filename = self.TEMPORARY_FILE2
    obj.version = _raveio.RaveIO_ODIM_Version_2_3
    obj.save()

    # Verify written data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertAlmostEqual(22.1, nodelist.getNode("/how/gasattn").data(), 4)
    self.assertAlmostEqual(370.1008, nodelist.getNode("/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/how/melting_layer_top").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), nodelist.getNode("/how/melting_layer_bottom").data(), atol=1e-1))
    self.assertAlmostEqual(374.1967, nodelist.getNode("/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1986, nodelist.getNode("/how/avgpwr").data(), 4)

    
    self.assertAlmostEqual(22.1, nodelist.getNode("/dataset1/data1/how/gasattn").data(), 4)
    self.assertAlmostEqual(370.1008, nodelist.getNode("/dataset1/data1/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_top").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_bottom").data(), atol=1e-1))
    self.assertAlmostEqual(374.1967, nodelist.getNode("/dataset1/data1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1986, nodelist.getNode("/dataset1/data1/how/avgpwr").data(), 4)

  def test_write_vp_convertedHow_2_4(self):
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

    vp.addAttribute("how/gasattn", 22.1)
    vp.addAttribute("how/nomTXpower", 370.1008)
    vp.addAttribute("how/nsampleH", 2.2)
    vp.addAttribute("how/nsampleV", 2.3)
    vp.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    vp.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    vp.addAttribute("how/peakpwr", 374.1967)
    vp.addAttribute("how/avgpwr", 373.1986)

    f1 = _ravefield.new()
    f1.setData(numpy.zeros((10,1), numpy.uint8))
    f1.addAttribute("what/quantity", "ff")
    f1.addAttribute("how/gasattn", 22.1)
    f1.addAttribute("how/nomTXpower", 370.1008)
    f1.addAttribute("how/nsampleH", 2.2)
    f1.addAttribute("how/nsampleV", 2.3)
    f1.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    f1.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    f1.addAttribute("how/peakpwr", 374.1967)
    f1.addAttribute("how/avgpwr", 373.1986)
    
    vp.addField(f1)

    obj = _raveio.new()
    obj.object = vp
    obj.filename = self.TEMPORARY_FILE2
    obj.version = _raveio.RaveIO_ODIM_Version_2_4
    obj.save()

    # Verify written data
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE2)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertAlmostEqual(0.0221, nodelist.getNode("/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/how/nomTXpower").data(), 4)
    self.assertAlmostEqual(2.2, nodelist.getNode("/how/nsampleH").data(), 4)
    self.assertAlmostEqual(2.3, nodelist.getNode("/how/nsampleV").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), nodelist.getNode("/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000.0, 3000.0, 4000.0]), nodelist.getNode("/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/how/avgpwr").data(), 4)

    
    self.assertAlmostEqual(0.0221, nodelist.getNode("/dataset1/data1/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/dataset1/data1/how/nomTXpower").data(), 4)
    self.assertAlmostEqual(2.2, nodelist.getNode("/dataset1/data1/how/nsampleH").data(), 4)
    self.assertAlmostEqual(2.3, nodelist.getNode("/dataset1/data1/how/nsampleV").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000.0, 3000.0, 4000.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/dataset1/data1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/dataset1/data1/how/avgpwr").data(), 4)

  def test_read_vp_convertedHow_2_3(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/what")
    self.addGroupNode(nodelist, "/where")
    self.addGroupNode(nodelist, "/how")

    self.addGroupNode(nodelist, "/dataset1")
    self.addGroupNode(nodelist, "/dataset1/what")
    self.addGroupNode(nodelist, "/dataset1/how")
    self.addGroupNode(nodelist, "/dataset1/data1")
    self.addGroupNode(nodelist, "/dataset1/data1/how")
    self.addGroupNode(nodelist, "/dataset1/data1/what")

    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_3")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "VP")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.3")

    self.addAttributeNode(nodelist, "/where/height", "double", 100.0)
    self.addAttributeNode(nodelist, "/where/lon", "double", 13.5)
    self.addAttributeNode(nodelist, "/where/lat", "double", 61.0)
    self.addAttributeNode(nodelist, "/where/interval", "double", 5)
    self.addAttributeNode(nodelist, "/where/levels", "int", 10)
    self.addAttributeNode(nodelist, "/where/minheight", "double", 10.0)
    self.addAttributeNode(nodelist, "/where/maxheight", "double", 20.0)

    self.addAttributeNode(nodelist, "/how/gasattn", "double", 22.1)
    self.addAttributeNode(nodelist, "/how/nomTXpower", "double", 370.1008)
    self.addAttributeNode(nodelist, "/how/melting_layer_top", "double", numpy.array([1.0, 2.0, 3.0]))
    self.addAttributeNode(nodelist, "/how/melting_layer_bottom", "double", numpy.array([2.0, 3.0, 4.0]))
    self.addAttributeNode(nodelist, "/how/peakpwr", "double", 374.1967)
    self.addAttributeNode(nodelist, "/how/avgpwr", "double", 373.1986)
    
    self.addAttributeNode(nodelist, "/dataset1/data1/how/gasattn", "double", 22.1)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nomTXpower", "double", 370.1008)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_top", "double", numpy.array([1.0, 2.0, 3.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_bottom", "double", numpy.array([2.0, 3.0, 4.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/peakpwr", "double", 374.19670)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/avgpwr", "double", 373.1986)
    
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "ff")
    dset = numpy.arange(10)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,1)).astype(numpy.uint8)    
    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,1), dset)
    
    nodelist.write(self.TEMPORARY_FILE, 6)
    
    rio = _raveio.open(self.TEMPORARY_FILE)

    vp = rio.object    

    self.assertAlmostEqual(22.1, vp.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, vp.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_top").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_bottom").data(), atol=1e-11))
    self.assertAlmostEqual(374.1967, vp.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.1986, vp.getAttribute("how/avgpwr"), 4)

    param = vp.getField("ff")
    self.assertAlmostEqual(22.1, param.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, param.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_top").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_bottom").data(), atol=1e-1))
    self.assertAlmostEqual(374.1967, param.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.1986, param.getAttribute("how/avgpwr"), 4)

  def test_read_vp_convertedHow_2_4(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/what")
    self.addGroupNode(nodelist, "/where")
    self.addGroupNode(nodelist, "/how")

    self.addGroupNode(nodelist, "/dataset1")
    self.addGroupNode(nodelist, "/dataset1/what")
    self.addGroupNode(nodelist, "/dataset1/how")
    self.addGroupNode(nodelist, "/dataset1/data1")
    self.addGroupNode(nodelist, "/dataset1/data1/how")
    self.addGroupNode(nodelist, "/dataset1/data1/what")

    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_4")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "VP")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.4")

    self.addAttributeNode(nodelist, "/where/height", "double", 100.0)
    self.addAttributeNode(nodelist, "/where/lon", "double", 13.5)
    self.addAttributeNode(nodelist, "/where/lat", "double", 61.0)
    self.addAttributeNode(nodelist, "/where/interval", "double", 5)
    self.addAttributeNode(nodelist, "/where/levels", "int", 10)
    self.addAttributeNode(nodelist, "/where/minheight", "double", 10.0)
    self.addAttributeNode(nodelist, "/where/maxheight", "double", 20.0)

    self.addAttributeNode(nodelist, "/how/gasattn", "double", 22.1)
    self.addAttributeNode(nodelist, "/how/nomTXpower", "double", 85.6832)
    self.addAttributeNode(nodelist, "/how/melting_layer_top_A", "double", numpy.array([1000.0, 2000.0, 3000.0]))
    self.addAttributeNode(nodelist, "/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 3000.0, 4000.0]))
    self.addAttributeNode(nodelist, "/how/peakpwr", "double", 85.7310)
    self.addAttributeNode(nodelist, "/how/avgpwr", "double", 85.7194)
   
    self.addAttributeNode(nodelist, "/dataset1/data1/how/gasattn", "double", 22.1)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nomTXpower", "double", 85.6832)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_top_A", "double", numpy.array([1000.0, 2000.0, 3000.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 3000.0, 4000.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/peakpwr", "double", 85.7310)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/avgpwr", "double", 85.7194)
    
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "ff")
    dset = numpy.arange(10)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,1)).astype(numpy.uint8)    
    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,1), dset)
    
    nodelist.write(self.TEMPORARY_FILE, 6)
    
    rio = _raveio.open(self.TEMPORARY_FILE)

    vp = rio.object    

    self.assertAlmostEqual(22100.0, vp.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, vp.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), vp.getAttribute("how/melting_layer_top"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), vp.getAttribute("how/melting_layer_bottom"), atol=1e-1))
    self.assertAlmostEqual(374.1967, vp.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.1986, vp.getAttribute("how/avgpwr"), 4)

    param = vp.getField("ff")
    self.assertAlmostEqual(22100.0, param.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, param.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), param.getAttribute("how/melting_layer_top"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), param.getAttribute("how/melting_layer_bottom"), atol=1e-1))
    self.assertAlmostEqual(374.1967, param.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.1986, param.getAttribute("how/avgpwr"), 4)

  def test_save_cartesian_23_howAttributes(self):
    image = _cartesian.new() #XYZ
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123,WIGOS:0-123-1-123456,ORG:82"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "My Product"
    image.addAttribute("how/gasattn", 22.1)
    image.addAttribute("how/nomTXpower", 370.1008)
    image.addAttribute("how/TXpower", numpy.asarray([370.1008,371.1008,372.1008]))
    image.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    image.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    image.addAttribute("how/peakpwr", 374.1967)
    image.addAttribute("how/avgpwr", 373.1986)
    
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    param.addAttribute("how/gasattn", 22.1)
    param.addAttribute("how/nomTXpower", 370.1008)
    param.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    param.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    param.addAttribute("how/peakpwr", 374.1967)
    param.addAttribute("how/avgpwr", 373.1986)

    image.addParameter(param)

    ios = _raveio.new()
    ios.object = image
    ios.filename = self.TEMPORARY_FILE
    ios.version = _raveio.RaveIO_ODIM_Version_2_3
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertAlmostEqual(22.1, nodelist.getNode("/how/gasattn").data(), 4)
    self.assertAlmostEqual(370.1008, nodelist.getNode("/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008,371.1008,372.1008]), nodelist.getNode("/how/TXpower").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/how/melting_layer_top").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), nodelist.getNode("/how/melting_layer_bottom").data(), atol=1e-1))
    self.assertAlmostEqual(374.1967, nodelist.getNode("/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1986, nodelist.getNode("/how/avgpwr").data(), 4)    

    self.assertAlmostEqual(22.1, nodelist.getNode("/dataset1/data1/how/gasattn").data(), 4)
    self.assertAlmostEqual(370.1008, nodelist.getNode("/dataset1/data1/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_top").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_bottom").data(), atol=1e-1))
    self.assertAlmostEqual(374.1967, nodelist.getNode("/dataset1/data1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1986, nodelist.getNode("/dataset1/data1/how/avgpwr").data(), 4)    

  def test_save_cartesian_24_howAttributes(self):
    image = _cartesian.new() #XYZ
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123,WIGOS:0-123-1-123456,ORG:82"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "My Product"
    image.addAttribute("how/gasattn", 22.1)
    image.addAttribute("how/nomTXpower", 370.1008)
    image.addAttribute("how/TXpower", numpy.asarray([370.1008,371.1008,372.1008]))
    image.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    image.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    image.addAttribute("how/peakpwr", 374.1967)
    image.addAttribute("how/avgpwr", 373.1986)
    
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    param.addAttribute("how/gasattn", 22.1)
    param.addAttribute("how/nomTXpower", 370.1008)
    param.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    param.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    param.addAttribute("how/peakpwr", 374.1967)
    param.addAttribute("how/avgpwr", 373.1986)

    image.addParameter(param)

    ios = _raveio.new()
    ios.object = image
    ios.filename = self.TEMPORARY_FILE
    ios.version = _raveio.RaveIO_ODIM_Version_2_4
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertAlmostEqual(0.0221, nodelist.getNode("/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([85.6832,85.6949,85.7066]), nodelist.getNode("/how/TXpower").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), nodelist.getNode("/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000.0, 3000.0, 4000.0]), nodelist.getNode("/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/how/avgpwr").data(), 4)    

    self.assertAlmostEqual(0.0221, nodelist.getNode("/dataset1/data1/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/dataset1/data1/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000.0, 3000.0, 4000.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/dataset1/data1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/dataset1/data1/how/avgpwr").data(), 4)    
  """
  """ 
  def test_save_cartesian_volume_23_howAttributes(self):
    vol = _cartesianvolume.new()
    vol.time = "100000"
    vol.date = "20100101"
    vol.objectType = _rave.Rave_ObjectType_CVOL
    vol.source = "PLC:123,WIGOS:0-123-1-123456,ORG:82"
    vol.addAttribute("how/gasattn", 22.1)
    vol.addAttribute("how/nomTXpower", 370.1008)
    vol.addAttribute("how/TXpower", numpy.asarray([370.1008,371.1008,372.1008]))
    vol.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    vol.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    vol.addAttribute("how/peakpwr", 374.1967)
    vol.addAttribute("how/avgpwr", 373.1986)
    
    image = _cartesian.new() #XYZ
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123,WIGOS:0-123-1-123456,ORG:82"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "My Product"
    image.addAttribute("how/gasattn", 22.1)
    image.addAttribute("how/nomTXpower", 370.1008)
    image.addAttribute("how/TXpower", numpy.asarray([370.1008,371.1008,372.1008]))
    image.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    image.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    image.addAttribute("how/peakpwr", 374.1967)
    image.addAttribute("how/avgpwr", 373.1986)
    
    vol.addImage(image)
    
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    param.addAttribute("how/gasattn", 22.1)
    param.addAttribute("how/nomTXpower", 370.1008)
    param.addAttribute("how/TXpower", numpy.asarray([370.1008,371.1008,372.1008]))
    param.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    param.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    param.addAttribute("how/peakpwr", 374.1967)
    param.addAttribute("how/avgpwr", 373.1986)

    image.addParameter(param)

    ios = _raveio.new()
    ios.object = vol
    ios.filename = self.TEMPORARY_FILE
    ios.version = _raveio.RaveIO_ODIM_Version_2_3
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    vol.addAttribute("how/gasattn", 22.1)
    vol.addAttribute("how/nomTXpower", 370.1008)
    vol.addAttribute("how/TXpower", numpy.asarray([370.1008,371.1008,372.1008]))
    vol.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    vol.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    vol.addAttribute("how/peakpwr", 374.1967)
    vol.addAttribute("how/avgpwr", 373.1986)

    self.assertAlmostEqual(22.1, nodelist.getNode("/how/gasattn").data(), 4)
    self.assertAlmostEqual(370.1008, nodelist.getNode("/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008,371.1008,372.1008]), nodelist.getNode("/how/TXpower").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/how/melting_layer_top").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), nodelist.getNode("/how/melting_layer_bottom").data(), atol=1e-1))
    self.assertAlmostEqual(374.1967, nodelist.getNode("/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1986, nodelist.getNode("/how/avgpwr").data(), 4)    

    self.assertAlmostEqual(22.1, nodelist.getNode("/dataset1/how/gasattn").data(), 4)
    self.assertAlmostEqual(370.1008, nodelist.getNode("/dataset1/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008,371.1008,372.1008]), nodelist.getNode("/dataset1/how/TXpower").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/how/melting_layer_top").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), nodelist.getNode("/dataset1/how/melting_layer_bottom").data(), atol=1e-1))
    self.assertAlmostEqual(374.1967, nodelist.getNode("/dataset1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1986, nodelist.getNode("/dataset1/how/avgpwr").data(), 4)    

    self.assertAlmostEqual(22.1, nodelist.getNode("/dataset1/data1/how/gasattn").data(), 4)
    self.assertAlmostEqual(370.1008, nodelist.getNode("/dataset1/data1/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008,371.1008,372.1008]), nodelist.getNode("/dataset1/data1/how/TXpower").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_top").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_bottom").data(), atol=1e-1))
    self.assertAlmostEqual(374.1967, nodelist.getNode("/dataset1/data1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(373.1986, nodelist.getNode("/dataset1/data1/how/avgpwr").data(), 4)    

  def test_save_cartesian_volume_24_howAttributes(self):
    vol = _cartesianvolume.new()
    vol.time = "100000"
    vol.date = "20100101"
    vol.objectType = _rave.Rave_ObjectType_CVOL
    vol.source = "PLC:123,WIGOS:0-123-1-123456,ORG:82"
    vol.addAttribute("how/gasattn", 22.1)
    vol.addAttribute("how/nomTXpower", 370.1008)
    vol.addAttribute("how/TXpower", numpy.asarray([370.1008,371.1008,372.1008]))
    vol.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    vol.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    vol.addAttribute("how/peakpwr", 374.1967)
    vol.addAttribute("how/avgpwr", 373.1986)
    
    image = _cartesian.new() #XYZ
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123,WIGOS:0-123-1-123456,ORG:82"
    image.xscale = 2000.0
    image.yscale = 2000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "My Product"
    image.addAttribute("how/gasattn", 22.1)
    image.addAttribute("how/nomTXpower", 370.1008)
    image.addAttribute("how/TXpower", numpy.asarray([370.1008,371.1008,372.1008]))
    image.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    image.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    image.addAttribute("how/peakpwr", 374.1967)
    image.addAttribute("how/avgpwr", 373.1986)
    
    vol.addImage(image)
    
    param = _cartesianparam.new()
    param.quantity = "DBZH"
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    data = numpy.zeros((240,240),numpy.uint8)
    param.setData(data)
    param.addAttribute("how/gasattn", 22.1)
    param.addAttribute("how/nomTXpower", 370.1008)
    param.addAttribute("how/TXpower", numpy.asarray([370.1008,371.1008,372.1008]))
    param.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]))
    param.addAttribute("how/melting_layer_bottom", numpy.array([2.0, 3.0, 4.0]))
    param.addAttribute("how/peakpwr", 374.1967)
    param.addAttribute("how/avgpwr", 373.1986)

    image.addParameter(param)

    ios = _raveio.new()
    ios.object = vol
    ios.filename = self.TEMPORARY_FILE
    ios.version = _raveio.RaveIO_ODIM_Version_2_4
    ios.save()

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertAlmostEqual(0.0221, nodelist.getNode("/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([85.6832,85.6949,85.7066]), nodelist.getNode("/how/TXpower").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), nodelist.getNode("/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000.0, 3000.0, 4000.0]), nodelist.getNode("/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/how/avgpwr").data(), 4)    

    self.assertAlmostEqual(0.0221, nodelist.getNode("/dataset1/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/dataset1/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([85.6832,85.6949,85.7066]), nodelist.getNode("/dataset1/how/TXpower").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), nodelist.getNode("/dataset1/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000.0, 3000.0, 4000.0]), nodelist.getNode("/dataset1/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/dataset1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/dataset1/how/avgpwr").data(), 4)    

    self.assertAlmostEqual(0.0221, nodelist.getNode("/dataset1/data1/how/gasattn").data(), 4)
    self.assertAlmostEqual(85.6832, nodelist.getNode("/dataset1/data1/how/nomTXpower").data(), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([85.6832,85.6949,85.7066]), nodelist.getNode("/dataset1/data1/how/TXpower").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_top_A").data(), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000.0, 3000.0, 4000.0]), nodelist.getNode("/dataset1/data1/how/melting_layer_bottom_A").data(), atol=1e-1))
    self.assertAlmostEqual(85.7310, nodelist.getNode("/dataset1/data1/how/peakpwr").data(), 4)
    self.assertAlmostEqual(85.7194, nodelist.getNode("/dataset1/data1/how/avgpwr").data(), 4)    

  def testReadAndConvertV24_IMAGE(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/what")
    self.addGroupNode(nodelist, "/where")
    self.addGroupNode(nodelist, "/how")

    self.addGroupNode(nodelist, "/dataset1")
    self.addGroupNode(nodelist, "/dataset1/what")
    self.addGroupNode(nodelist, "/dataset1/how")
    self.addGroupNode(nodelist, "/dataset1/data1")
    self.addGroupNode(nodelist, "/dataset1/data1/how")
    self.addGroupNode(nodelist, "/dataset1/data1/what")
    self.addGroupNode(nodelist, "/dataset1/data2")
    self.addGroupNode(nodelist, "/dataset1/data2/what")

    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_4")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "IMAGE")
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

    self.addAttributeNode(nodelist, "/how/gasattn", "double", 0.0231)
    self.addAttributeNode(nodelist, "/how/nomTXpower", "double", 85.6832)
    self.addAttributeNode(nodelist, "/how/TXpower", "double", numpy.asarray([85.6832,85.6949,85.7066]))
    self.addAttributeNode(nodelist, "/how/melting_layer_top_A", "double", numpy.array([1000.0, 2000.0, 3000.0]))
    self.addAttributeNode(nodelist, "/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 3000.0, 4000.0]))
    self.addAttributeNode(nodelist, "/how/peakpwr", "double", 85.7322)
    self.addAttributeNode(nodelist, "/how/avgpwr", "double", 85.7206)

    self.addAttributeNode(nodelist, "/dataset1/data1/how/gasattn", "double", 0.0231)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nomTXpower", "double", 85.6832)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/TXpower", "double", numpy.asarray([85.6832,85.6949,85.7066]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_top_A", "double", numpy.array([1000.0, 2000.0, 3000.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 3000.0, 4000.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/peakpwr", "double", 85.7322)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/avgpwr", "double", 85.7206)

    self.addAttributeNode(nodelist, "/dataset1/data2/what/offset", "double", 1000.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/gain", "double", 2000.0)

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    
    dset2 = numpy.arange(100)
    dset2=numpy.array(dset2.astype(numpy.uint8),numpy.uint8)
    dset2=numpy.reshape(dset2,(10,10)).astype(numpy.uint8)

    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,10), dset)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "DBZH")

    self.addDatasetNode(nodelist, "/dataset1/data2/data", "uchar", (10,10), dset2)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/quantity", "string", "HGHT")

    nodelist.write(self.TEMPORARY_FILE, 6)
    
    rio = _raveio.open(self.TEMPORARY_FILE)
    image = rio.object
    
    self.assertAlmostEqual(23.1, image.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, image.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008, 371.0992, 372.1003]), image.getAttribute("how/TXpower"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), image.getAttribute("how/melting_layer_top_A"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000.0, 3000.0, 4000.0]), image.getAttribute("how/melting_layer_bottom_A"), atol=1e-1))
    self.assertAlmostEqual(374.3001, image.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.3017, image.getAttribute("how/avgpwr"), 4)

    param = image.getParameter("DBZH")
    self.assertAlmostEqual(23.1, param.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, param.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008, 371.0992, 372.1003]), param.getAttribute("how/TXpower"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), param.getAttribute("how/melting_layer_top_A"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2000.0, 3000.0, 4000.0]), param.getAttribute("how/melting_layer_bottom_A"), atol=1e-1))
    self.assertAlmostEqual(374.3001, param.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.3017, param.getAttribute("how/avgpwr"), 4)
    
    param = image.getParameter("HGHT")
    self.assertAlmostEqual(1.0, param.offset, 4)
    self.assertAlmostEqual(2.0, param.gain, 4)

  def testReadV24_COMP(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/what")
    self.addGroupNode(nodelist, "/where")
    self.addGroupNode(nodelist, "/how")

    self.addGroupNode(nodelist, "/dataset1")
    self.addGroupNode(nodelist, "/dataset1/what")
    self.addGroupNode(nodelist, "/dataset1/how")
    self.addGroupNode(nodelist, "/dataset1/data1")
    self.addGroupNode(nodelist, "/dataset1/data1/how")
    self.addGroupNode(nodelist, "/dataset1/data1/what")
    self.addGroupNode(nodelist, "/dataset1/data2")
    self.addGroupNode(nodelist, "/dataset1/data2/what")

    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_3")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "CVOL")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.3")

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

    self.addAttributeNode(nodelist, "/how/gasattn", "double", 23.1)
    self.addAttributeNode(nodelist, "/how/nomTXpower", "double", 370.1008)
    self.addAttributeNode(nodelist, "/how/TXpower", "double", numpy.asarray([370.1008, 371.0992, 372.1003]))
    self.addAttributeNode(nodelist, "/how/melting_layer_top", "double", numpy.array([1.0, 2.0, 3.0]))
    self.addAttributeNode(nodelist, "/how/melting_layer_bottom", "double", numpy.array([2.0, 3.0, 4.0]))
    self.addAttributeNode(nodelist, "/how/peakpwr", "double", 374.3001)
    self.addAttributeNode(nodelist, "/how/avgpwr", "double", 373.3017)

    self.addAttributeNode(nodelist, "/dataset1/how/gasattn", "double", 23.1)
    self.addAttributeNode(nodelist, "/dataset1/how/nomTXpower", "double", 370.1008)
    self.addAttributeNode(nodelist, "/dataset1/how/TXpower", "double", numpy.asarray([370.1008, 371.0992, 372.1003]))
    self.addAttributeNode(nodelist, "/dataset1/how/melting_layer_top", "double", numpy.array([1.0, 2.0, 3.0]))
    self.addAttributeNode(nodelist, "/dataset1/how/melting_layer_bottom", "double", numpy.array([2.0, 3.0, 4.0]))
    self.addAttributeNode(nodelist, "/dataset1/how/peakpwr", "double", 374.3001)
    self.addAttributeNode(nodelist, "/dataset1/how/avgpwr", "double", 373.3017)
    
    self.addAttributeNode(nodelist, "/dataset1/data1/how/gasattn", "double", 23.1)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nomTXpower", "double", 370.1008)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/TXpower", "double", numpy.asarray([370.1008, 371.0992, 372.1003]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_top", "double", numpy.array([1.0, 2.0, 3.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_bottom", "double", numpy.array([2.0, 3.0, 4.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/peakpwr", "double", 374.3001)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/avgpwr", "double", 373.3017)

    self.addAttributeNode(nodelist, "/dataset1/data2/what/offset", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/gain", "double", 2.0)

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)

    dset2 = numpy.arange(100)
    dset2=numpy.array(dset2.astype(numpy.uint8),numpy.uint8)
    dset2=numpy.reshape(dset2,(10,10)).astype(numpy.uint8)

    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,10), dset)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "DBZH")

    self.addDatasetNode(nodelist, "/dataset1/data2/data", "uchar", (10,10), dset2)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/quantity", "string", "HGHT")

    nodelist.write(self.TEMPORARY_FILE, 6)
    
    rio = _raveio.open(self.TEMPORARY_FILE)
    cvol = rio.object
    
    self.assertAlmostEqual(23.1, cvol.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, cvol.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008, 371.0992, 372.1003]), cvol.getAttribute("how/TXpower"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), cvol.getAttribute("how/melting_layer_top"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), cvol.getAttribute("how/melting_layer_bottom"), atol=1e-1))
    self.assertAlmostEqual(374.3001, cvol.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.3017, cvol.getAttribute("how/avgpwr"), 4)
    
    img = cvol.getImage(0)
    self.assertAlmostEqual(23.1, img.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, img.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008, 371.0992, 372.1003]), img.getAttribute("how/TXpower"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), img.getAttribute("how/melting_layer_top"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), img.getAttribute("how/melting_layer_bottom"), atol=1e-1))
    self.assertAlmostEqual(374.3001, img.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.3017, img.getAttribute("how/avgpwr"), 4)
    
    param = img.getParameter("DBZH")
    self.assertAlmostEqual(23.1, param.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, param.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008, 371.0992, 372.1003]), param.getAttribute("how/TXpower"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), param.getAttribute("how/melting_layer_top"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), param.getAttribute("how/melting_layer_bottom"), atol=1e-1))
    self.assertAlmostEqual(374.3001, param.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.3017, param.getAttribute("how/avgpwr"), 4)

    param = img.getParameter("HGHT")
    self.assertAlmostEqual(1.0, param.offset, 4)
    self.assertAlmostEqual(2.0, param.gain, 4)

  def testReadAndConvertV24_COMP(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/what")
    self.addGroupNode(nodelist, "/where")
    self.addGroupNode(nodelist, "/how")

    self.addGroupNode(nodelist, "/dataset1")
    self.addGroupNode(nodelist, "/dataset1/what")
    self.addGroupNode(nodelist, "/dataset1/how")
    self.addGroupNode(nodelist, "/dataset1/data1")
    self.addGroupNode(nodelist, "/dataset1/data1/how")
    self.addGroupNode(nodelist, "/dataset1/data1/what")
    self.addGroupNode(nodelist, "/dataset1/data2")
    self.addGroupNode(nodelist, "/dataset1/data2/what")

    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_4")
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

    self.addAttributeNode(nodelist, "/how/gasattn", "double", 0.0231)
    self.addAttributeNode(nodelist, "/how/nomTXpower", "double", 85.6832)
    self.addAttributeNode(nodelist, "/how/TXpower", "double", numpy.asarray([85.6832,85.6949,85.7066]))
    self.addAttributeNode(nodelist, "/how/melting_layer_top_A", "double", numpy.array([1000.0, 2000.0, 3000.0]))
    self.addAttributeNode(nodelist, "/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 3000.0, 4000.0]))
    self.addAttributeNode(nodelist, "/how/peakpwr", "double", 85.7322)
    self.addAttributeNode(nodelist, "/how/avgpwr", "double", 85.7206)


    self.addAttributeNode(nodelist, "/dataset1/how/gasattn", "double", 0.0231)
    self.addAttributeNode(nodelist, "/dataset1/how/nomTXpower", "double", 85.6832)
    self.addAttributeNode(nodelist, "/dataset1/how/TXpower", "double", numpy.asarray([85.6832,85.6949,85.7066]))
    self.addAttributeNode(nodelist, "/dataset1/how/melting_layer_top_A", "double", numpy.array([1000.0, 2000.0, 3000.0]))
    self.addAttributeNode(nodelist, "/dataset1/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 3000.0, 4000.0]))
    self.addAttributeNode(nodelist, "/dataset1/how/peakpwr", "double", 85.7322)
    self.addAttributeNode(nodelist, "/dataset1/how/avgpwr", "double", 85.7206)
    
    self.addAttributeNode(nodelist, "/dataset1/data1/how/gasattn", "double", 0.0231)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/nomTXpower", "double", 85.6832)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/TXpower", "double", numpy.asarray([85.6832,85.6949,85.7066]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_top_A", "double", numpy.array([1000.0, 2000.0, 3000.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/melting_layer_bottom_A", "double", numpy.array([2000.0, 3000.0, 4000.0]))
    self.addAttributeNode(nodelist, "/dataset1/data1/how/peakpwr", "double", 85.7322)
    self.addAttributeNode(nodelist, "/dataset1/data1/how/avgpwr", "double", 85.7206)

    self.addAttributeNode(nodelist, "/dataset1/data2/what/offset", "double", 1000.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/gain", "double", 2000.0)

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)

    dset2 = numpy.arange(100)
    dset2=numpy.array(dset2.astype(numpy.uint8),numpy.uint8)
    dset2=numpy.reshape(dset2,(10,10)).astype(numpy.uint8)

    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,10), dset)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "DBZH")

    self.addDatasetNode(nodelist, "/dataset1/data2/data", "uchar", (10,10), dset2)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/quantity", "string", "HGHT")

    nodelist.write(self.TEMPORARY_FILE, 6)
    
    rio = _raveio.open(self.TEMPORARY_FILE)
    cvol = rio.object
    
    self.assertAlmostEqual(23.1, cvol.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, cvol.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008, 371.0992, 372.1003]), cvol.getAttribute("how/TXpower"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), cvol.getAttribute("how/melting_layer_top"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), cvol.getAttribute("how/melting_layer_bottom"), atol=1e-1))
    self.assertAlmostEqual(374.3001, cvol.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.3017, cvol.getAttribute("how/avgpwr"), 4)
    
    img = cvol.getImage(0)
    self.assertAlmostEqual(23.1, img.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, img.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008, 371.0992, 372.1003]), img.getAttribute("how/TXpower"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), img.getAttribute("how/melting_layer_top"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), img.getAttribute("how/melting_layer_bottom"), atol=1e-1))
    self.assertAlmostEqual(374.3001, img.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.3017, img.getAttribute("how/avgpwr"), 4)
    
    param = img.getParameter("DBZH")
    self.assertAlmostEqual(23.1, param.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(370.1008, param.getAttribute("how/nomTXpower"), 4)
    self.assertTrue(numpy.allclose(numpy.asarray([370.1008, 371.0992, 372.1003]), param.getAttribute("how/TXpower"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), param.getAttribute("how/melting_layer_top"), atol=1e-1))
    self.assertTrue(numpy.allclose(numpy.array([2.0, 3.0, 4.0]), param.getAttribute("how/melting_layer_bottom"), atol=1e-1))
    self.assertAlmostEqual(374.3001, param.getAttribute("how/peakpwr"), 4)
    self.assertAlmostEqual(373.3017, param.getAttribute("how/avgpwr"), 4)

    param = img.getParameter("HGHT")
    self.assertAlmostEqual(1.0, param.offset, 4)
    self.assertAlmostEqual(2.0, param.gain, 4)

  def test_read_22_write_23_scan(self):
    rio = _raveio.open(self.FIXTURE_SEHEM_SCAN_0_5)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_2, rio.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, rio.version)
    rio.object.removeAttribute("how/software")
    rio.object.source = "%s,WIGOS:0-123-1-123456"%rio.object.source
    rio.version=_raveio.RaveIO_ODIM_Version_2_3
    rio.save(self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    #print(str(nodelist.getNodeNames()))
    self.assertEqual("ODIM_H5/V2_3", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.3", nodelist.getNode("/what/version").data())

    self.assertTrue("BALTRAD", nodelist.getNode("/how/software").data())
    self.assertTrue("WIGOS:0-123-1-123456" in nodelist.getNode("/what/source").data())

  def test_read_22_write_24_scan(self):
    rio = _raveio.open(self.FIXTURE_SEHEM_SCAN_0_5)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_2, rio.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, rio.version)
    rio.object.removeAttribute("how/software")
    rio.object.source = "%s,WIGOS:0-123-1-123456"%rio.object.source
    rio.save(self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    #print(str(nodelist.getNodeNames()))
    self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())

    self.assertTrue("BALTRAD", nodelist.getNode("/how/software").data())
    self.assertTrue("WIGOS:0-123-1-123456" in nodelist.getNode("/what/source").data())

  def test_read_22_write_24_scan_strict(self):
    rio = _raveio.open(self.FIXTURE_SEHEM_SCAN_0_5)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_2, rio.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, rio.version)
    rio.object.removeAttribute("how/software")
    rio.object.source = "%s,WIGOS:0-123-1-123456"%rio.object.source
    rio.strict = True
    try:
        rio.save(self.TEMPORARY_FILE)
        self.fail("Expected IOError")
    except IOError:
        pass

  def test_read_22_write_22_scan(self):
    rio = _raveio.open(self.FIXTURE_SEHEM_SCAN_0_5)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_2, rio.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, rio.version)
    rio.object.removeAttribute("how/software") #Exists in fixture. Remove to ensure that we get it added.
    rio.object.source = "%s,WIGOS:0-123-1-123456"%rio.object.source
    rio.version = _raveio.RaveIO_ODIM_Version_2_2
    rio.save(self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    #print(str(nodelist.getNodeNames()))
    self.assertEqual("ODIM_H5/V2_2", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    self.assertTrue("BALTRAD", nodelist.getNode("/how/software").data())
    self.assertFalse("WIGOS:0-123-1-123456" in nodelist.getNode("/what/source").data())

  def test_read_22_write_22_source_wo_wigos_1_scan(self):
    rio = _raveio.open(self.FIXTURE_SEHEM_SCAN_0_5)
    rio.object.source = "WIGOS:0-123-1-123456,WMO:02588,RAD:SE47,PLC:Hemse(Ase),NOD:sehem,ORG:82,CTY:643,CMT:Swedish radar"
    rio.version = _raveio.RaveIO_ODIM_Version_2_2
    rio.save(self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertEqual("WMO:02588,RAD:SE47,PLC:Hemse(Ase),NOD:sehem,ORG:82,CTY:643,CMT:Swedish radar", nodelist.getNode("/what/source").data())

  def test_read_22_write_22_source_wo_wigos_2_scan(self):
    rio = _raveio.open(self.FIXTURE_SEHEM_SCAN_0_5)
    rio.object.source = "WMO:02588,WIGOS:0-123-1-123456,RAD:SE47,PLC:Hemse(Ase),NOD:sehem,ORG:82,CTY:643,CMT:Swedish radar"
    rio.version = _raveio.RaveIO_ODIM_Version_2_2
    rio.save(self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertEqual("WMO:02588,RAD:SE47,PLC:Hemse(Ase),NOD:sehem,ORG:82,CTY:643,CMT:Swedish radar", nodelist.getNode("/what/source").data())

  def test_read_22_write_22_source_wo_wigos_3_scan(self):
    rio = _raveio.open(self.FIXTURE_SEHEM_SCAN_0_5)
    rio.object.source = "WMO:02588,RAD:SE47,PLC:Hemse(Ase),NOD:sehem,ORG:82,CTY:643,CMT:Swedish radar,WIGOS:0-123-1-123456"
    rio.version = _raveio.RaveIO_ODIM_Version_2_2
    rio.save(self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertEqual("WMO:02588,RAD:SE47,PLC:Hemse(Ase),NOD:sehem,ORG:82,CTY:643,CMT:Swedish radar", nodelist.getNode("/what/source").data())

  #
  def test_read_22_write_24_volume(self):
    rio = _raveio.open(self.FIXTURE_SEHEM_PVOL)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_2, rio.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, rio.version)
    rio.object.removeAttribute("how/software")
    rio.object.source = "%s,WIGOS:0-123-1-123456"%rio.object.source
    rio.save(self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual("ODIM_H5/V2_4", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.4", nodelist.getNode("/what/version").data())
    self.assertTrue("BALTRAD", nodelist.getNode("/how/software").data())
    self.assertTrue("WIGOS:0-123-1-123456" in nodelist.getNode("/what/source").data())
  
  def create_cartesian_image_with_param(self, quantity):
    image = _cartesian.new()
    image.time = "100000"
    image.date = "20100101"
    image.objectType = _rave.Rave_ObjectType_IMAGE
    image.source = "PLC:123"
    image.xscale = 20000.0
    image.yscale = 20000.0
    image.areaextent = (-240000.0, -240000.0, 240000.0, 240000.0)
    image.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84 +nadgrids=@null")
    image.product = _rave.Rave_ProductType_CAPPI
    image.prodname = "BALTRAD"

    param = _cartesianparam.new()
    param.quantity = quantity
    param.gain = 1.0
    param.offset = 0.0
    param.nodata = 255.0
    param.undetect = 0.0
    param.setData(numpy.zeros((10,10),numpy.uint8))
    image.addParameter(param)
    
    return image
  
  def create_cartesian_volume(self, images):
    cvol = _cartesianvolume.new()
    cvol.time = images[0].time
    cvol.date = images[0].date
    cvol.objectType = _rave.Rave_ObjectType_CVOL
    cvol.source = images[0].source
    cvol.xscale = images[0].xscale
    cvol.yscale = images[0].yscale
    cvol.areaextent = images[0].areaextent
    projection = images[0].projection

    for image in images:
      cvol.addImage(image)
    
    return cvol
  
  def save_file_with_rio(self, obj, filename, version):
    ios = _raveio.new()
    ios.object = obj
    ios.version = version
    ios.save(filename)
  
  def test_write_cartesian_image_HGHT_2_3(self):
    image = self.create_cartesian_image_with_param("HGHT")
    
    self.save_file_with_rio(image, self.TEMPORARY_FILE, _raveio.RaveIO_ODIM_Version_2_3)
    
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)
    
  def test_write_cartesian_image_HGHT_2_4(self):
    image = self.create_cartesian_image_with_param("HGHT")
    
    self.save_file_with_rio(image, self.TEMPORARY_FILE, _raveio.RaveIO_ODIM_Version_2_4)
    
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEqual(1000.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)


  def test_write_cartesian_image_MESH_2_3(self):
    image = self.create_cartesian_image_with_param("MESH")
    
    self.save_file_with_rio(image, self.TEMPORARY_FILE, _raveio.RaveIO_ODIM_Version_2_3)
    
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)
    
  def test_write_cartesian_image_MESH_2_4(self):
    image = self.create_cartesian_image_with_param("MESH")
    
    self.save_file_with_rio(image, self.TEMPORARY_FILE, _raveio.RaveIO_ODIM_Version_2_4)
    
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEqual(10.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)

  def test_write_cartesian_image_DBZH_2_4(self):
    image = self.create_cartesian_image_with_param("DBZH")
    
    self.save_file_with_rio(image, self.TEMPORARY_FILE, _raveio.RaveIO_ODIM_Version_2_4)
    
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)
    
  def test_write_cartesian_volume_HGHT_MESH_2_4(self):
    image1 = self.create_cartesian_image_with_param("HGHT")
    image2 = self.create_cartesian_image_with_param("MESH")
    image3 = self.create_cartesian_image_with_param("DBZH")
    
    volume = self.create_cartesian_volume([image1, image2, image3])
    
    self.save_file_with_rio(volume, self.TEMPORARY_FILE, _raveio.RaveIO_ODIM_Version_2_4)
    
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEqual(1000.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset2/data1/what/offset").data(), 4)
    self.assertAlmostEqual(10.0, nodelist.getNode("/dataset2/data1/what/gain").data(), 4)
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset3/data1/what/offset").data(), 4)
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset3/data1/what/gain").data(), 4)
    
  def create_scan_with_param(self, elangle, quantity):
    scan = _polarscan.new()
    scan.elangle = elangle * math.pi / 180.0
    scan.a1gate = 2
    scan.rstart = 0.0
    scan.rscale = 5000.0
    scan.time = "100001"
    scan.date = "20091010"
    param = _polarscanparam.new()
    param.nodata = 10.0
    param.undetect = 11.0
    param.quantity = quantity
    param.gain = 1.0
    param.offset = 0.0
    data = numpy.zeros((10, 10), numpy.uint8)
    param.setData(data)
    scan.addParameter(param)
    return scan

  def create_polar_volume(self, scans):
    obj = _polarvolume.new()
    obj.time = "100000"
    obj.date = "20091010"
    obj.source = "PLC:123"
    obj.longitude = 12.0 * math.pi/180.0
    obj.latitude = 60.0 * math.pi/180.0
    obj.height = 0.0

    for scan in scans:
      obj.addScan(scan)
    
    return obj

  def test_write_scan_HGHT_2_3(self):
    scan = self.create_scan_with_param(0.1, "HGHT")
    self.save_file_with_rio(scan, self.TEMPORARY_FILE, _raveio.RaveIO_ODIM_Version_2_3)
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEqual(1.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)

  def test_write_scan_HGHT_2_4(self):
    scan = self.create_scan_with_param(0.1, "HGHT")
    self.save_file_with_rio(scan, self.TEMPORARY_FILE, _raveio.RaveIO_ODIM_Version_2_4)
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    self.assertAlmostEqual(0.0, nodelist.getNode("/dataset1/data1/what/offset").data(), 4)
    self.assertAlmostEqual(1000.0, nodelist.getNode("/dataset1/data1/what/gain").data(), 4)

  def test_read_scan_HGHT_2_3(self):
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

    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_3")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "SCAN")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.3")

    self.addAttributeNode(nodelist, "/where/height", "double", 100.0)
    self.addAttributeNode(nodelist, "/where/lon", "double", 13.5)
    self.addAttributeNode(nodelist, "/where/lat", "double", 61.0)

    self.addAttributeNode(nodelist, "/dataset1/data1/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "HGHT")

    self.addAttributeNode(nodelist, "/dataset1/data2/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/quantity", "string", "DBZH")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)

    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,10), dset)
    self.addDatasetNode(nodelist, "/dataset1/data2/data", "uchar", (10,10), dset)

    nodelist.write(self.TEMPORARY_FILE, 6)

    obj = _raveio.open(self.TEMPORARY_FILE).object
    self.assertAlmostEqual(1.0, obj.getParameter("HGHT").gain, 4)
    self.assertAlmostEqual(0.0, obj.getParameter("HGHT").offset, 4)
    self.assertAlmostEqual(1.0, obj.getParameter("DBZH").gain, 4)
    self.assertAlmostEqual(0.0, obj.getParameter("DBZH").offset, 4)

  
  # WORKING BELOW
  def test_read_scan_HGHT_2_4(self):
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

    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_4")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "SCAN")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.4")

    self.addAttributeNode(nodelist, "/where/height", "double", 100.0)
    self.addAttributeNode(nodelist, "/where/lon", "double", 13.5)
    self.addAttributeNode(nodelist, "/where/lat", "double", 61.0)

    self.addAttributeNode(nodelist, "/dataset1/data1/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "HGHT")

    self.addAttributeNode(nodelist, "/dataset1/data2/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/quantity", "string", "DBZH")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)

    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,10), dset)
    self.addDatasetNode(nodelist, "/dataset1/data2/data", "uchar", (10,10), dset)

    nodelist.write(self.TEMPORARY_FILE, 6)

    obj = _raveio.open(self.TEMPORARY_FILE).object
    self.assertAlmostEqual(0.001, obj.getParameter("HGHT").gain, 4)
    self.assertAlmostEqual(0.0, obj.getParameter("HGHT").offset, 4)
    self.assertAlmostEqual(1.0, obj.getParameter("DBZH").gain, 4)
    self.assertAlmostEqual(0.0, obj.getParameter("DBZH").offset, 4)

  def test_read_volume_HGHT_2_4(self):
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

    self.addAttributeNode(nodelist, "/Conventions", "string", "ODIM_H5/V2_4")
    self.addAttributeNode(nodelist, "/what/date", "string", "20100101")
    self.addAttributeNode(nodelist, "/what/time", "string", "101500")
    self.addAttributeNode(nodelist, "/what/source", "string", "PLC:123")
    self.addAttributeNode(nodelist, "/what/object", "string", "PVOL")
    self.addAttributeNode(nodelist, "/what/version", "string", "H5rad 2.3")

    self.addAttributeNode(nodelist, "/where/height", "double", 100.0)
    self.addAttributeNode(nodelist, "/where/lon", "double", 13.5)
    self.addAttributeNode(nodelist, "/where/lat", "double", 61.0)

    self.addAttributeNode(nodelist, "/dataset1/data1/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data1/what/quantity", "string", "HGHT")

    self.addAttributeNode(nodelist, "/dataset1/data2/what/gain", "double", 1.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/offset", "double", 0.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/nodata", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/undetect", "double", 255.0)
    self.addAttributeNode(nodelist, "/dataset1/data2/what/quantity", "string", "DBZH")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)

    self.addDatasetNode(nodelist, "/dataset1/data1/data", "uchar", (10,10), dset)
    self.addDatasetNode(nodelist, "/dataset1/data2/data", "uchar", (10,10), dset)

    nodelist.write(self.TEMPORARY_FILE, 6)

    obj = _raveio.open(self.TEMPORARY_FILE).object.getScan(0)
    self.assertAlmostEqual(0.001, obj.getParameter("HGHT").gain, 4)
    self.assertAlmostEqual(0.0, obj.getParameter("HGHT").offset, 4)
    self.assertAlmostEqual(1.0, obj.getParameter("DBZH").gain, 4)
    self.assertAlmostEqual(0.0, obj.getParameter("DBZH").offset, 4)

  #
  def test_read_22_write_23_volume(self):
    rio = _raveio.open(self.FIXTURE_SEHEM_PVOL)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_2, rio.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, rio.version)
    rio.object.removeAttribute("how/software")
    rio.object.source = "%s,WIGOS:0-123-1-123456"%rio.object.source
    rio.version = _rave.RaveIO_ODIM_Version_2_3
    rio.save(self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    #print(str(nodelist.getNodeNames()))
    self.assertEqual("ODIM_H5/V2_3", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.3", nodelist.getNode("/what/version").data())
    self.assertTrue("BALTRAD", nodelist.getNode("/how/software").data())
    self.assertTrue("WIGOS:0-123-1-123456" in nodelist.getNode("/what/source").data())

  
  def test_read_22_write_22_volume(self):
    rio = _raveio.open(self.FIXTURE_SEHEM_PVOL)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_2, rio.read_version)
    self.assertEqual(_raveio.RaveIO_ODIM_Version_2_4, rio.version)
    rio.object.removeAttribute("how/software")
    rio.object.source = "%s,WIGOS:0-123-1-123456"%rio.object.source
    rio.version = _raveio.RaveIO_ODIM_Version_2_2
    rio.save(self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()
    #print(str(nodelist.getNodeNames()))
    self.assertEqual("ODIM_H5/V2_2", nodelist.getNode("/Conventions").data())
    self.assertEqual("H5rad 2.2", nodelist.getNode("/what/version").data())
    self.assertTrue("BALTRAD", nodelist.getNode("/how/software").data())
    self.assertFalse("WIGOS:0-123-1-123456" in nodelist.getNode("/what/source").data())

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

  def test_read_volume_with_lazyio(self):
    obj = _raveio.open(self.FIXTURE_VOLUME, True)
    vol = obj.object

    self.assertEqual(10, vol.getNumberOfScans())
    for i in range(10):
      self.assertEqual(2, len(vol.getScan(i).getParameterNames()))
      self.assertEqual("DBZH", vol.getScan(i).getParameterNames()[0])
      self.assertEqual("VRADH", vol.getScan(i).getParameterNames()[1])

    data = vol.getScan(0).getParameter("DBZH").getData()
    self.assertTrue(data is not None)
    self.assertEqual(58, data[0][2])

  def test_read_scan_with_lazyio_shift(self):
    scan = _raveio.open(self.FIXTURE_SEHEM_SCAN_0_5, True).object
    scan.shiftData(-1)
    original = list(scan.getParameter("DBZH").getData()[0:10,0])
    self.assertTrue(original == [114,117,115,116,116,112,114,112,114, 117])

  def test_read_scan_with_lazyio_shift_2(self):
    scan = _raveio.open(self.FIXTURE_SEHEM_SCAN_0_5, True).object
    field = _ravefield.new()
    field.setData(scan.getParameter("DBZH").getData())
    field.addAttribute("how/task", "se.nisses.test")
    scan.addQualityField(field)
    rio = _raveio.new()
    rio.object = scan
    rio.save(self.TEMPORARY_FILE)

    scan = _raveio.open(self.TEMPORARY_FILE, True).object
    scan.shiftData(-1)
    original = list(scan.getParameter("DBZH").getData()[0:10,0])
    qoriginal = list(scan.getQualityField(0).getData()[0:10,0])
    self.assertTrue(original == [114,117,115,116,116,112,114,112,114, 117])
    self.assertTrue(qoriginal == [114,117,115,116,116,112,114,112,114, 117])

  def test_read_scan_with_lazyio_shift_after_dataaccess(self):
    scan = _raveio.open(self.FIXTURE_SEHEM_SCAN_0_5, True).object
    scan.getParameter("DBZH").getData()
    scan.shiftData(-1)
    original = list(scan.getParameter("DBZH").getData()[0:10,0])
    self.assertTrue(original == [114,117,115,116,116,112,114,112,114, 117])

  def test_read_cartesian_with_lazyio(self):
    obj = _raveio.open(self.FIXTURE_CARTESIAN_IMAGE)
    obj.object.getParameter("DBZH").setValue((1,1),10)
    obj.save(self.TEMPORARY_FILE)

    obj = _raveio.open(self.TEMPORARY_FILE, True)
    self.assertAlmostEqual(10.0, obj.object.getParameter("DBZH").getValue((1,1))[1], 4)

  def test_read_cartesian_volume_with_lazyio(self):
    obj = _raveio.open(self.FIXTURE_CARTESIAN_VOLUME, False)
    obj.object.getImage(0).getParameter("DBZH").setValue((1,1),10)
    obj.save(self.TEMPORARY_FILE)

    obj = _raveio.open(self.TEMPORARY_FILE, True)
    self.assertAlmostEqual(10.0, obj.object.getImage(0).getParameter("DBZH").getValue((1,1))[1], 4)

  def test_read_vp_with_lazyio(self):
    obj = _raveio.open(self.FIXTURE_VP_NEW_VERSION, True)
    field = obj.object.getField("HGHT")
    self.assertEqual("HGHT", field.getAttribute("what/quantity"))
    self.assertAlmostEqual(100.0, field.getData()[0], 4)

  def test_clone_volume_with_lazyio(self):
    obj = _raveio.open(self.FIXTURE_VOLUME, True)
    newvol = obj.object.clone()
    data = newvol.getScan(0).getParameter("DBZH").getData()
    self.assertTrue(data is not None)
    self.assertEqual(58, data[0][2])

  def test_clone_cartesian_with_lazyio(self):
    obj = _raveio.open(self.FIXTURE_CARTESIAN_IMAGE)
    obj.object.getParameter("DBZH").setValue((1,1),10)
    obj.save(self.TEMPORARY_FILE)

    obj = _raveio.open(self.TEMPORARY_FILE, True)
    newobj = obj.object.clone()
    self.assertAlmostEqual(10.0, newobj.getParameter("DBZH").getValue((1,1))[1], 4)

  def test_clone_cartesian_volume_with_lazyio(self):
    obj = _raveio.open(self.FIXTURE_CARTESIAN_VOLUME, False)
    obj.object.getImage(0).getParameter("DBZH").setValue((1,1),10)
    obj.save(self.TEMPORARY_FILE)

    obj = _raveio.open(self.TEMPORARY_FILE, True).object.clone()
    self.assertAlmostEqual(10.0, obj.getImage(0).getParameter("DBZH").getValue((1,1))[1], 4)

  def test_clone_vp_with_lazyio(self):
    obj = _raveio.open(self.FIXTURE_VP_NEW_VERSION, True).object.clone()
    field = obj.getField("HGHT")
    self.assertEqual("HGHT", field.getAttribute("what/quantity"))
    self.assertAlmostEqual(100.0, field.getData()[0], 4)

  def addGroupNode(self, nodelist, name):
    node = _pyhl.node(_pyhl.GROUP_ID, name)
    nodelist.addNode(node)

  def addAttributeNode(self, nodelist, name, type, value):
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, name)
    if isinstance(value, numpy.ndarray):
      node.setArrayValue(-1,value.shape,value,type,-1)
    else:
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


