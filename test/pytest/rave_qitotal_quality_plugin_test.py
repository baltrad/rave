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

Tests the qitotal quality plugin

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2014-03-22
'''
import unittest
import os
import math
import string
import numpy
import rave_qitotal_quality_plugin
import _raveio, _ravefield, _rave

rave_qitotal_quality_plugin.QITOTAL_DTYPE = _rave.RaveDataType_DOUBLE
rave_qitotal_quality_plugin.QITOTAL_GAIN = 1.0
rave_qitotal_quality_plugin.QITOTAL_OFFSET = 0.0 
rave_qitotal_quality_plugin.QITOTAL_METHOD = "additive"


class rave_qitotal_quality_plugin_test(unittest.TestCase):
  VOLUME_FIXTURE = "fixtures/pvol_sekir_20090501T120000Z.h5"
  SCAN_FIXTURE = "fixtures/scan_sehuv_0.5_20110126T184500Z.h5"
  OPTIONS_FIXTURE = "fixtures/test_qitotal_options.xml"
  
  classUnderTest = None
  
  def setUp(self):
    self.classUnderTest = rave_qitotal_quality_plugin.rave_qitotal_quality_plugin()
    self.classUnderTest._qitotal_option_file = self.OPTIONS_FIXTURE
    
  def tearDown(self):
    self.classUnderTest = None
    
  def test_getQualityFields(self):
    result = self.classUnderTest.getQualityFields()
    self.assertEqual(1, len(result))
    self.assertEqual("pl.imgw.quality.qi_total", result[0])

  def test_process_scan(self):
    scan = _raveio.open(self.SCAN_FIXTURE).object
    qf1data = numpy.zeros((scan.nrays, scan.nbins), numpy.int8)
    qf1data = qf1data + 1
    qf1 = _ravefield.new()
    qf1.setData(qf1data)
    qf1.addAttribute("how/task", "se.smhi.test.1")
    
    qf2data = numpy.zeros((scan.nrays, scan.nbins), numpy.int8)
    qf2data = qf2data + 2
    qf2 = _ravefield.new()
    qf2.setData(qf2data)
    qf2.addAttribute("how/task", "se.smhi.test.2")
    
    scan.addQualityField(qf1)
    scan.addQualityField(qf2)
    
    result, qfield = self.classUnderTest.process(scan)
    self.assertEqual(qfield, ["pl.imgw.quality.qi_total"], "Wrong qfield returned from process")
    
    field = result.getQualityFieldByHowTask("pl.imgw.quality.qi_total")
    self.assertTrue(field != None)
    self.assertAlmostEqual(1.7, field.getValue(0,0)[1], 4)
    self.assertAlmostEqual(1.7, field.getValue(10,10)[1], 4)

  def test_process_scan_reprocess_false(self):
    scan = _raveio.open(self.SCAN_FIXTURE).object
    qf1data = numpy.zeros((scan.nrays, scan.nbins), numpy.int8)
    qf1data = qf1data + 1
    qf1 = _ravefield.new()
    qf1.setData(qf1data)
    qf1.addAttribute("how/task", "se.smhi.test.1")
    
    qf2data = numpy.zeros((scan.nrays, scan.nbins), numpy.int8)
    qf2data = qf2data + 2
    qf2 = _ravefield.new()
    qf2.setData(qf2data)
    qf2.addAttribute("how/task", "se.smhi.test.2")
    
    scan.addQualityField(qf1)
    scan.addQualityField(qf2)
    
    result, _ = self.classUnderTest.process(scan)
    qfield = result.getQualityFieldByHowTask("pl.imgw.quality.qi_total")
    result, _ = self.classUnderTest.process(scan, reprocess_quality_flag=False)
    qfield2 = result.getQualityFieldByHowTask("pl.imgw.quality.qi_total")
    self.assertTrue(qfield != qfield2)

  def test_process_scan_quality_control_mode(self):
    scan = _raveio.open(self.SCAN_FIXTURE).object
    qf1data = numpy.zeros((scan.nrays, scan.nbins), numpy.int8)
    qf1data = qf1data + 1
    qf1 = _ravefield.new()
    qf1.setData(qf1data)
    qf1.addAttribute("how/task", "se.smhi.test.1")
    
    qf2data = numpy.zeros((scan.nrays, scan.nbins), numpy.int8)
    qf2data = qf2data + 2
    qf2 = _ravefield.new()
    qf2.setData(qf2data)
    qf2.addAttribute("how/task", "se.smhi.test.2")
    
    scan.addQualityField(qf1)
    scan.addQualityField(qf2)
    
    result, _ = self.classUnderTest.process(scan, True, quality_control_mode="analyze")
    qfield = result.getQualityFieldByHowTask("pl.imgw.quality.qi_total")
    result, _ = self.classUnderTest.process(scan, reprocess_quality_flag=False)
    qfield2 = result.getQualityFieldByHowTask("pl.imgw.quality.qi_total")
    self.assertTrue(qfield != qfield2)

  def test_process_volume(self):
    volume = _raveio.open(self.VOLUME_FIXTURE).object
    scan1 = volume.getScan(0)
    scan1.addQualityField(self.createQualityField(scan1.nrays, scan1.nbins, 1, "se.smhi.test.1"))
    scan1.addQualityField(self.createQualityField(scan1.nrays, scan1.nbins, 2, "se.smhi.test.2"))
    scan1.addQualityField(self.createQualityField(scan1.nrays, scan1.nbins, 3, "se.smhi.test.3"))
    # Result = 1.0 * 0.3 + 2.0 * 0.2 + 3.0 * 0.5 = 2.2

    scan2 = volume.getScan(1)
    scan2.addQualityField(self.createQualityField(scan2.nrays, scan2.nbins, 4, "se.smhi.test.1"))
    scan2.addQualityField(self.createQualityField(scan2.nrays, scan2.nbins, 5, "se.smhi.test.2"))
    scan2.addQualityField(self.createQualityField(scan2.nrays, scan2.nbins, 6, "se.smhi.test.3"))
    # Result = 4.0 * 0.3 + 5.0 * 0.2 + 6.0 * 0.5 = 5.2

    scan3 = volume.getScan(2)
    scan3.addQualityField(self.createQualityField(scan3.nrays, scan3.nbins, 7, "se.smhi.test.1"))
    scan3.addQualityField(self.createQualityField(scan3.nrays, scan3.nbins, 8, "se.smhi.test.2"))
    # Result = 7.0 * 0.3/0.5 + 8.0 * 0.2/0.5 = 7.4
    
    nscans = volume.getNumberOfScans()
    
    result, _ = self.classUnderTest.process(volume)
    
    field = result.getScan(0).getQualityFieldByHowTask("pl.imgw.quality.qi_total")
    self.assertTrue(field != None)
    self.assertAlmostEqual(2.2, field.getValue(0,0)[1], 4)
    self.assertAlmostEqual(2.2, field.getValue(10,10)[1], 4)
    
    field = result.getScan(1).getQualityFieldByHowTask("pl.imgw.quality.qi_total")
    self.assertTrue(field != None)
    self.assertAlmostEqual(5.2, field.getValue(0,0)[1], 4)
    self.assertAlmostEqual(5.2, field.getValue(10,10)[1], 4)

    field = result.getScan(2).getQualityFieldByHowTask("pl.imgw.quality.qi_total")
    self.assertTrue(field != None)
    self.assertAlmostEqual(7.4, field.getValue(0,0)[1], 4)
    self.assertAlmostEqual(7.4, field.getValue(10,10)[1], 4)
    
    for i in range(3, nscans):
      self.assertEqual(None, result.getScan(i).findQualityFieldByHowTask("pl.imgw.quality.qi_total"))

  def createQualityField(self, nrays, nbins, zv, howtask):
    qfdata = numpy.zeros((nrays, nbins), numpy.int8)
    qfdata = qfdata + zv
    qf = _ravefield.new()
    qf.setData(qfdata)
    qf.addAttribute("how/task", howtask)
    return qf
    