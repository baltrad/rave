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
import _rave
import string
import numpy
import _pyhl
import math

class PyRaveIOTest(unittest.TestCase):
  FIXTURE_VOLUME="fixture_ODIM_H5_pvol_ang_20090501T1200Z.h5"
  FIXTURE_IMAGE="fixture_old_pcappi-dbz-500.ang-gnom-2000.h5"
  
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
    self.assertEquals(20, vol.getNumberOfScans())
    self.assertAlmostEquals(56.3675, vol.latitude*180.0/math.pi, 4)
    self.assertAlmostEquals(12.8544, vol.longitude*180.0/math.pi, 4)
    self.assertAlmostEquals(209, vol.height, 4)
    
    # Verify the scans
    scan = vol.getScan(0)
    self.assertAlmostEquals(0.4, scan.gain, 4)
    self.assertAlmostEquals(-30.0, scan.offset, 4)
    self.assertAlmostEquals(255.0, scan.nodata, 4)
    self.assertAlmostEquals(0.0, scan.undetect, 4)
    self.assertEquals("DBZH", scan.quantity)
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
    
    scan = vol.getScan(1)
    self.assertAlmostEquals(0.1875, scan.gain, 4)
    self.assertAlmostEquals(-24.0, scan.offset, 4)
    self.assertAlmostEquals(255.0, scan.nodata, 4)
    self.assertAlmostEquals(0.0, scan.undetect, 4)
    self.assertEquals("VRAD", scan.quantity)
    self.assertAlmostEquals(0.5, scan.elangle*180.0/math.pi, 4)

    scan = vol.getScan(18)
    self.assertAlmostEquals(0.4, scan.gain, 4)
    self.assertAlmostEquals(-30.0, scan.offset, 4)
    self.assertAlmostEquals(255.0, scan.nodata, 4)
    self.assertAlmostEquals(0.0, scan.undetect, 4)
    self.assertEquals("DBZH", scan.quantity)
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

    scan = vol.getScan(19)
    self.assertAlmostEquals(0.375, scan.gain, 4)
    self.assertAlmostEquals(-48.0, scan.offset, 4)
    self.assertAlmostEquals(255.0, scan.nodata, 4)
    self.assertAlmostEquals(0.0, scan.undetect, 4)
    self.assertEquals("VRAD", scan.quantity)
    self.assertAlmostEquals(40.0, scan.elangle*180.0/math.pi, 4)   

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
    obj.projection = _projection.new("x","y","+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544 +datum=WGS84")
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
    
  def addGroupNode(self, nodelist, name):
    node = _pyhl.node(_pyhl.GROUP_ID, name)
    nodelist.addNode(node)
    
  def addAttributeNode(self, nodelist, name, type, value):
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, name)
    node.setScalarValue(-1,value,type,-1)
    nodelist.addNode(node)
