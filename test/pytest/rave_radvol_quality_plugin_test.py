'''
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the radvol quality plugins

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2014-11-16
'''
import unittest
import os
import math
import string
import rave_radvol_quality_plugin
import _raveio

# Since all of the radvol tests more or less behave the same, we create a generic test case that can be used for
# all types of radvols. However, we inherit from object and let the different test case inherit the actual TestCase. Otherwise
# these tests will be run without any quality handler.
class radvol_tests(object):
  VOLUME_FIXTURE = "fixtures/pvol_sehud_20090501T120000Z.h5"
  SCAN_FIXTURE = "fixtures/scan_sehud_0.5_20110126T184500Z.h5"

  def get_quality_flag(self):
    raise Exception, "Must return quality flag"
  
  def get_quality_plugin(self):
    raise Exception, "Must return quality plugin"
  
  def test_getQualityFields(self):
    result = self.get_quality_plugin().getQualityFields()
    self.assertEquals(1, len(result))
    self.assertEquals(self.get_quality_flag(), result[0]) 
  
  def test_process_scan(self):
    obj = _raveio.open(self.SCAN_FIXTURE).object
    result = self.get_quality_plugin().process(obj)
    self.assertEquals(1, result.getNumberOfQualityFields())
    field = result.getQualityField(0)
    self.assertEquals(self.get_quality_flag(), field.getAttribute("how/task"))

  def test_process_scan_reprocess_true(self):
    obj = _raveio.open(self.SCAN_FIXTURE).object
    result = self.get_quality_plugin().process(obj)
    self.assertEquals(1, result.getNumberOfQualityFields())
    field = result.getQualityField(0)
    self.assertEquals(self.get_quality_flag(), field.getAttribute("how/task"))
    
    result = self.classUnderTest.process(obj, reprocess_quality_flag=True)
    self.assertEquals(1, result.getNumberOfQualityFields())
    field2 = result.getQualityField(0)
    self.assertEquals(self.get_quality_flag(), field2.getAttribute("how/task"))
    self.assertTrue(field != field2)
    
  def test_process_scan_reprocess_false(self):
    obj = _raveio.open(self.SCAN_FIXTURE).object
    result = self.get_quality_plugin().process(obj)
    self.assertEquals(1, result.getNumberOfQualityFields())
    field = result.getQualityField(0)
    self.assertEquals(self.get_quality_flag(), field.getAttribute("how/task"))
    
    result = self.get_quality_plugin().process(obj, reprocess_quality_flag=False)
    self.assertEquals(1, result.getNumberOfQualityFields())
    field2 = result.getQualityField(0)
    self.assertEquals(self.get_quality_flag(), field2.getAttribute("how/task"))
    self.assertTrue(field == field2)  

  def test_process_pvol(self):
    obj = _raveio.open(self.VOLUME_FIXTURE).object
    result = self.get_quality_plugin().process(obj)
    
    for i in range(result.getNumberOfScans()):
      scan = result.getScan(i)
      self.assertEquals(1, scan.getNumberOfQualityFields())
      field = scan.getQualityField(0)
      self.assertEquals(self.get_quality_flag(), field.getAttribute("how/task"))

  def test_process_pvol_reprocess_true(self):
    obj = _raveio.open(self.VOLUME_FIXTURE).object
    result = self.get_quality_plugin().process(obj)
    
    fields = []
    for i in range(result.getNumberOfScans()):
      scan = result.getScan(i)
      self.assertEquals(1, scan.getNumberOfQualityFields())
      field = scan.getQualityField(0)
      self.assertEquals(self.get_quality_flag(), field.getAttribute("how/task"))
      fields.append(field)
    
    fields2 = []
    result = self.get_quality_plugin().process(obj, reprocess_quality_flag = True)
    for i in range(result.getNumberOfScans()):
      scan = result.getScan(i)
      self.assertEquals(1, scan.getNumberOfQualityFields())
      field = scan.getQualityField(0)
      self.assertEquals(self.get_quality_flag(), field.getAttribute("how/task"))
      fields2.append(field)
      
    self.assertEquals(len(fields), len(fields2))
    for i in range(len(fields)):
      self.assertTrue(fields[i] != fields2[i])

    
  def test_process_pvol_reprocess_false(self):
    obj = _raveio.open(self.VOLUME_FIXTURE).object
    result = self.get_quality_plugin().process(obj)
    
    fields = []
    for i in range(result.getNumberOfScans()):
      scan = result.getScan(i)
      self.assertEquals(1, scan.getNumberOfQualityFields())
      field = scan.getQualityField(0)
      self.assertEquals(self.get_quality_flag(), field.getAttribute("how/task"))
      fields.append(field)
    
    fields2 = []
    result = self.get_quality_plugin().process(obj, reprocess_quality_flag = False)
    for i in range(result.getNumberOfScans()):
      scan = result.getScan(i)
      self.assertEquals(1, scan.getNumberOfQualityFields())
      field = scan.getQualityField(0)
      self.assertEquals(self.get_quality_flag(), field.getAttribute("how/task"))
      fields2.append(field)
      
    self.assertEquals(len(fields), len(fields2))
    for i in range(len(fields)):
      self.assertTrue(fields[i] == fields2[i])


class radvol_att_plugin_test(unittest.TestCase, radvol_tests):
  def setUp(self):
    self.classUnderTest = rave_radvol_quality_plugin.radvol_att_plugin() 

  def tearDown(self):
    self.classUnderTest = None

  def get_quality_flag(self):
    return "pl.imgw.radvolqc.att"
  
  def get_quality_plugin(self):
    return self.classUnderTest


class radvol_broad_plugin_test(unittest.TestCase, radvol_tests):
  def setUp(self):
    self.classUnderTest = rave_radvol_quality_plugin.radvol_broad_plugin() 

  def tearDown(self):
    self.classUnderTest = None

  def get_quality_flag(self):
    return "pl.imgw.radvolqc.broad"
  
  def get_quality_plugin(self):
    return self.classUnderTest


class radvol_nmet_plugin_test(unittest.TestCase, radvol_tests):
  def setUp(self):
    self.classUnderTest = rave_radvol_quality_plugin.radvol_nmet_plugin() 

  def tearDown(self):
    self.classUnderTest = None

  def get_quality_flag(self):
    return "pl.imgw.radvolqc.nmet"
  
  def get_quality_plugin(self):
    return self.classUnderTest
  
class radvol_speck_plugin_test(unittest.TestCase, radvol_tests):
  def setUp(self):
    self.classUnderTest = rave_radvol_quality_plugin.radvol_speck_plugin() 

  def tearDown(self):
    self.classUnderTest = None

  def get_quality_flag(self):
    return "pl.imgw.radvolqc.speck"
  
  def get_quality_plugin(self):
    return self.classUnderTest  

class radvol_spike_plugin_test(unittest.TestCase, radvol_tests):
  def setUp(self):
    self.classUnderTest = rave_radvol_quality_plugin.radvol_spike_plugin() 

  def tearDown(self):
    self.classUnderTest = None

  def get_quality_flag(self):
    return "pl.imgw.radvolqc.spike"
  
  def get_quality_plugin(self):
    return self.classUnderTest  
  