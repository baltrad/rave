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
import _rave
import _raveio
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

  def testNewRaveIO(self):
    obj = _raveio.new()
    israveio = string.find(`type(obj)`, "RaveIOCore")
    self.assertNotEqual(-1, israveio)

  def testOpen(self):
    obj = _raveio.new()
    obj.open(self.FIXTURE_VOLUME)
    self.assertEquals(True, obj.isOpen())
  
  def testOpen_2(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    self.assertEquals(True, obj.isOpen())
    obj.close()
    self.assertEquals(False, obj.isOpen())
    obj.open(self.FIXTURE_VOLUME)
    self.assertEquals(True, obj.isOpen())

  def testOpen_noSuchFile(self):
    try:
      _raveio.open("No_Such_File_Fixture.h5")
      self.fail("Expected IOError")
    except IOError, e:
      pass

  def testIsOpen(self):
    obj = _raveio.new()
    self.assertEquals(False, obj.isOpen())
    obj.open(self.FIXTURE_VOLUME)
    self.assertEquals(True, obj.isOpen())
    obj.close()
    self.assertEquals(False, obj.isOpen())
  
  def testGetObjectType(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    self.assertEquals(_rave.RaveIO_ObjectType_PVOL, obj.getObjectType())

  def testGetObjectType_notReckognizedObjectType(self):
    # Create fixture
    nl = _pyhl.nodelist()
    self.addAttributeNode(nl, "/Conventions", "string", "ODIM_H5/V2_0")     
    self.addGroupNode(nl, "/what")
    self.addAttributeNode(nl, "/what/object", "string", "PYX")
    nl.write(self.TEMPORARY_FILE)
    nl = None
    
    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertEquals(_rave.RaveIO_ObjectType_UNDEFINED, obj.getObjectType())
 
  def testIsSupported_pvol(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    self.assertEquals(True, obj.isSupported())

  def testIsSupported_notReckognizedObjectType(self):
    # Create fixture
    nl = _pyhl.nodelist()
    self.addAttributeNode(nl, "/Conventions", "string", "ODIM_H5/V2_0")    
    self.addGroupNode(nl, "/what")
    self.addAttributeNode(nl, "/what/object", "string", "PYX")
    nl.write(self.TEMPORARY_FILE)
    nl = None
    
    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertEquals(False, obj.isSupported())

  def testIsSupported_missingConventions(self):
    # Create fixture
    nl = _pyhl.nodelist()
    self.addGroupNode(nl, "/what")
    self.addAttributeNode(nl, "/what/object", "string", "PVOL")
    nl.write(self.TEMPORARY_FILE)
    nl = None

    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertEquals(False, obj.isSupported())
    
  def testIsSupported_unsupportedConventions(self):
    # Create fixture
    nl = _pyhl.nodelist()
    self.addAttributeNode(nl, "/Conventions", "string", "ODIM_H5/V3_0")
    self.addGroupNode(nl, "/what")
    self.addAttributeNode(nl, "/what/object", "string", "PVOL")
    nl.write(self.TEMPORARY_FILE)
    nl = None

    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertEquals(False, obj.isSupported())

  def testGetOdimVersion(self):
    # Create fixture
    nl = _pyhl.nodelist()
    self.addAttributeNode(nl, "/Conventions", "string", "ODIM_H5/V2_0")
    nl.write(self.TEMPORARY_FILE)
    nl = None
    
    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertEquals(_rave.RaveIO_ODIM_Version_2_0, obj.getOdimVersion())
    
  def testGetOdimVersion_missingConventions(self):
    # Create fixture
    nl = _pyhl.nodelist()
    self.addAttributeNode(nl, "/smurf", "string", "ODIM_H5/V2_0")
    nl.write(self.TEMPORARY_FILE)
    nl = None
    
    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertEquals(_rave.RaveIO_ODIM_Version_UNDEFINED, obj.getOdimVersion())

  def testGetOdimVersion_undefinedConventions(self):
    # Create fixture
    nl = _pyhl.nodelist()
    self.addAttributeNode(nl, "/Conventions", "string", "GLGL")
    nl.write(self.TEMPORARY_FILE)
    nl = None
    
    obj = _raveio.open(self.TEMPORARY_FILE)
    self.assertEquals(_rave.RaveIO_ODIM_Version_UNDEFINED, obj.getOdimVersion())
  
  def testLoad_volume(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.load()
    result = string.find(`type(vol)`, "PolarVolumeCore")
    self.assertNotEqual(-1, result)     

  def testLoad_volume_checkData(self):
    obj = _raveio.open(self.FIXTURE_VOLUME)
    vol = obj.load()
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
       
  def addGroupNode(self, nodelist, name):
    node = _pyhl.node(_pyhl.GROUP_ID, name)
    nodelist.addNode(node)
    
  def addAttributeNode(self, nodelist, name, type, value):
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, name)
    node.setScalarValue(-1,value,type,-1)
    nodelist.addNode(node)
