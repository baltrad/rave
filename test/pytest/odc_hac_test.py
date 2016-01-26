'''
Copyright (C) 2014 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the odc_hac

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2015-04-08
'''
import unittest
import os
import math
import string
import _odc_hac, odc_hac, rave_zdiff_quality_plugin
import _raveio, _ravefield
import _polarscanparam,_polarvolume
import numpy

class odc_hac_test(unittest.TestCase):
  VOLUME_FIXTURE = "fixtures/pvol_seang_20090501T120000Z.h5"
  SCAN_FIXTURE = "fixtures/scan_sehud_0.5_20110126T184500Z.h5"
  classUnderTest = None
    
  def setUp(self):
    pass 

  def tearDown(self):
    pass
  
  def test_py_odc_hac_zdiffScan(self):
    scan = self.create_scan()
    odc_hac.zdiffScan(scan)
    qfield = scan.getQualityFieldByHowTask("eu.opera.odc.zdiff")
    self.assertTrue(numpy.amax(qfield.getData()) == 255)
    self.assertAlmostEqual(0.0, qfield.getValue(0,0)[1], 4)  # (1.0 - (40.0/40.0)) / (1.0/255.0)
    self.assertAlmostEqual(25.0, qfield.getValue(1,0)[1], 4)  # (1.0 - (39.6 - 3.6)/40.0)/(1.0/255.0)

  def test_py_odc_hac_zdiffPvol(self):
    scan1 = self.create_scan(3.0, 4.0, 53.0, 40.0)
    scan2 = self.create_scan(4.0, 3.0, 40.0, 53.0)
    scan2.elangle = 10.0 * math.pi / 180.0
    
    pvol = _polarvolume.new()
    pvol.addScan(scan1)
    pvol.addScan(scan2)
    
    odc_hac.zdiffPvol(pvol)
    qfield1 = pvol.getScan(0).getQualityFieldByHowTask("eu.opera.odc.zdiff")
    qfield2 = pvol.getScan(1).getQualityFieldByHowTask("eu.opera.odc.zdiff")
    self.assertTrue(numpy.amax(qfield1.getData()) == 255)
    self.assertTrue(numpy.amax(qfield2.getData()) == 255)
    self.assertAlmostEqual(0.0, qfield1.getValue(0,0)[1], 4)  # (1.0 - (40.0/40.0)) / (1.0/255.0)
    self.assertAlmostEqual(25.0, qfield1.getValue(1,0)[1], 4)  # (1.0 - (39.6 - 3.6)/40.0)/(1.0/255.0)
    self.assertAlmostEqual(25.0, qfield2.getValue(0,0)[1], 4)  # (1.0 - (40.0/40.0)) / (1.0/255.0)
    self.assertAlmostEqual(0.0, qfield2.getValue(1,0)[1], 4)  # (1.0 - (39.6 - 3.6)/40.0)/(1.0/255.0)

  def test_py_odc_hac_zdiffScan_noTH(self):
    scan = _raveio.open(self.SCAN_FIXTURE).object
    param_dbzh = scan.getParameter("DBZH")
    odc_hac.zdiffScan(scan)
    try:
      qfield = scan.getQualityFieldByHowTask("eu.opera.odc.zdiff")
      self.fail("Expected NameError")
    except NameError, e:
      pass

  def test_py_odc_zdiff_scan(self):
    scan = self.create_scan(3.0, 4.0, 53.0, 40.0)
    odc_hac.zdiff(scan)
    qfield = scan.getQualityFieldByHowTask("eu.opera.odc.zdiff")
    self.assertTrue(numpy.amax(qfield.getData()) == 255)
    self.assertAlmostEqual(0.0, qfield.getValue(0,0)[1], 4)  # (1.0 - (40.0/40.0)) / (1.0/255.0)
    self.assertAlmostEqual(25.0, qfield.getValue(1,0)[1], 4)  # (1.0 - (39.6 - 3.6)/40.0)/(1.0/255.0)    

  def test_py_odc_zdiff_pvol(self):
    scan1 = self.create_scan(3.0, 4.0, 53.0, 40.0)
    scan2 = self.create_scan(4.0, 3.0, 40.0, 53.0)
    scan2.elangle = 10.0 * math.pi / 180.0
    
    pvol = _polarvolume.new()
    pvol.addScan(scan1)
    pvol.addScan(scan2)
    
    odc_hac.zdiff(pvol)
    
    qfield1 = pvol.getScan(0).getQualityFieldByHowTask("eu.opera.odc.zdiff")
    qfield2 = pvol.getScan(1).getQualityFieldByHowTask("eu.opera.odc.zdiff")
    self.assertTrue(numpy.amax(qfield1.getData()) == 255)
    self.assertTrue(numpy.amax(qfield2.getData()) == 255)
    self.assertAlmostEqual(0.0, qfield1.getValue(0,0)[1], 4)  # (1.0 - (40.0/40.0)) / (1.0/255.0)
    self.assertAlmostEqual(25.0, qfield1.getValue(1,0)[1], 4)  # (1.0 - (39.6 - 3.6)/40.0)/(1.0/255.0)
    self.assertAlmostEqual(25.0, qfield2.getValue(0,0)[1], 4)  # (1.0 - (40.0/40.0)) / (1.0/255.0)
    self.assertAlmostEqual(0.0, qfield2.getValue(1,0)[1], 4)  # (1.0 - (39.6 - 3.6)/40.0)/(1.0/255.0)

  def test_odc_hac_zdiff(self):
    scan = self.create_scan(3.0, 4.0, 53.0, 40.0)
    qind = _ravefield.new()
    qind.setData(numpy.zeros((scan.nrays,scan.nbins), numpy.uint8))
    qind.addAttribute("how/task", "eu.opera.odc.zdiff")
    qind.addAttribute("how/task_args", 40.0)
    qind.addAttribute("what/gain", 1/255.0)
    qind.addAttribute("what/offset", 0.0)
    scan.addQualityField(qind)    
    ret = _odc_hac.zdiff(scan, 40.0)
    self.assertTrue(numpy.amax(qind.getData()) == 255)
    self.assertAlmostEqual(0.0, qind.getValue(0,0)[1], 4)  # (1.0 - (40.0/40.0)) / (1.0/255.0)
    self.assertAlmostEqual(25.0, qind.getValue(1,0)[1], 4)  # (1.0 - (39.6 - 3.6)/40.0)/(1.0/255.0)
  
  def test_rave_zdiff_quality_plugin_process_scan(self):
    scan = self.create_scan(3.0, 4.0, 53.0, 40.0)
    
    qp = rave_zdiff_quality_plugin.rave_zdiff_quality_plugin()
    processed, qfield = qp.process(scan)
    
    self.assertEquals(qfield, ["eu.opera.odc.zdiff"], "Wrong qfield returned from process")
    
    qfield = processed.getQualityFieldByHowTask("eu.opera.odc.zdiff")
    self.assertTrue(numpy.amax(qfield.getData()) == 255)
    self.assertAlmostEqual(0.0, qfield.getValue(0,0)[1], 4)  # (1.0 - (40.0/40.0)) / (1.0/255.0)
    self.assertAlmostEqual(25.0, qfield.getValue(1,0)[1], 4)  # (1.0 - (39.6 - 3.6)/40.0)/(1.0/255.0)    

  def test_rave_zdiff_quality_plugin_process_pvol(self):
    scan1 = self.create_scan(3.0, 4.0, 53.0, 40.0)
    scan2 = self.create_scan(4.0, 3.0, 40.0, 53.0)
    scan2.elangle = 10.0 * math.pi / 180.0
    
    pvol = _polarvolume.new()
    pvol.addScan(scan1)
    pvol.addScan(scan2)
        
    qp = rave_zdiff_quality_plugin.rave_zdiff_quality_plugin()
    processed, qfield = qp.process(pvol)
    
    self.assertEquals(qfield, ["eu.opera.odc.zdiff"], "Wrong qfield returned from process")
    
    qfield1 = processed.getScan(0).getQualityFieldByHowTask("eu.opera.odc.zdiff")
    qfield2 = processed.getScan(1).getQualityFieldByHowTask("eu.opera.odc.zdiff")
    self.assertTrue(numpy.amax(qfield1.getData()) == 255)
    self.assertTrue(numpy.amax(qfield2.getData()) == 255)
    self.assertAlmostEqual(0.0, qfield1.getValue(0,0)[1], 4)  # (1.0 - (40.0/40.0)) / (1.0/255.0)
    self.assertAlmostEqual(25.0, qfield1.getValue(1,0)[1], 4)  # (1.0 - (39.6 - 3.6)/40.0)/(1.0/255.0)
    self.assertAlmostEqual(25.0, qfield2.getValue(0,0)[1], 4)  # (1.0 - (40.0/40.0)) / (1.0/255.0)
    self.assertAlmostEqual(0.0, qfield2.getValue(1,0)[1], 4)  # (1.0 - (39.6 - 3.6)/40.0)/(1.0/255.0)
  
  def test_rave_zdiff_quality_plugin_getQualityFields(self):
    qp = rave_zdiff_quality_plugin.rave_zdiff_quality_plugin()
    fields = qp.getQualityFields()
    self.assertEquals(1, len(fields))
    self.assertEquals("eu.opera.odc.zdiff", fields[0])
    
  def create_scan(self, dbzh0_0=3.0, dbzh0_1=4.0, th0_0=53.0, th0_1=40.0):
    scan = _raveio.open(self.SCAN_FIXTURE).object
    param_dbzh = scan.getParameter("DBZH")
    param_dbzh.setValue((0,0), (dbzh0_0 - param_dbzh.offset)/param_dbzh.gain)
    param_dbzh.setValue((0,1), (dbzh0_1 - param_dbzh.offset)/param_dbzh.gain)
    param_th = _polarscanparam.new()
    data = param_dbzh.getData()
    param_th.setData(data)
    param_th.quantity = "TH"
    param_th.gain = param_dbzh.gain
    param_th.offset = param_dbzh.offset
    param_th.setValue((0,0), (th0_0 - param_th.offset)/param_th.gain)
    param_th.setValue((0,1), (th0_1 - param_th.offset)/param_th.gain)
    scan.addParameter(param_th)
    return scan

    