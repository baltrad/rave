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

PyComputePIA module

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2025-11-04
'''
import unittest
import os
import string
import numpy
import math
import _polarscan, _polarscanparam, _raveio
import compute_pia
from compute_pia import PIAHitschfeldBordan
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE, QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY

## Example ray from Ängelholm radar, first azimuth
## 20250911 060000 elangle = 0.5

EXAMPLE_RAY=[-32767, -32767, -32767, -32767, -32767,   -861,   -804,   -796,
         -549,   -560,   -451,   -227,   -180,   -347,   -338,   -446,
         -465,   -136,     -1,     40,    -25,   -166,   -439,   -819,
        -1172,  -1189,  -1225,  -1095,  -1081,  -1019,  -1003,   -793,
         -601,   -353,   -494,   -529,   -338,   -766,  -1021,   -863,
         -855,   -720,  -1168,  -1058,  -1048,   -988,  -1016,   -847,
         -756,   -834,   -863,   -825,   -766,   -753,   -763,   -615,
         -638,   -740,   -739,   -394,   -300,   -351,   -245,   -106,
          130,    191,    302,    237,    395,    414,    593,    591,
          946,   1307,   1277,   1328,   1251,   1296,   1391,   1360,
         1281,   1669,   2067,   1913,   1647,   1521,   1430,   1389,
         1490,   1549,   1407,   1188,   1302,   1122,    878,    840,
          711,    739,    745,    735,    728,    545,    488,    636,
          680,    774,    857,    813,    732,    573,    547,    433,
          328,    504,    545,    549,    566,    627,    606,    733,
          855,    716,    658,    668,    729,    640,    643,    693,
          738,    704,    722,    659,    619,    479,    413,    471,
          438,    253,    308,    298,    338,    -14,    117,     62,
           17,    -68,   -184,   -239, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767,    147,     87, -32767,    289,    281,
          208,    387,    273,    230,    278,    348,    324,    209,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767,
       -32767, -32767, -32767, -32767, -32767, -32767, -32767, -32767]

EXAMPLE_RAY_FULL=numpy.full((480),4000.0)

class PyComputePIATest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass
  
  def createScan(self, rscale, quantity, data, offset, gain):
    scan = _polarscan.new()
    scan.rscale=rscale
    param = _polarscanparam.new()
    param.quantity=quantity
    param.setData(data)
    param.nodata=-32768.0
    param.undetect=-32767.0
    param.offset=offset
    param.gain=gain
    scan.addParameter(param)
    return scan

  def test_createPIAField(self):
    scan = self.createScan(2000.0, "DBZH", numpy.array(EXAMPLE_RAY).reshape(1,480), 0.0, 0.01)
    resultPIA, resultDatatype, dr = PIAHitschfeldBordan("DBZH").createPIAField(scan)
    lastvalue=resultPIA[0][0]
    for x in resultPIA[0]:
      self.assertTrue(x >= lastvalue)
    self.assertFalse(numpy.isnan(resultPIA).any())    

  def test_createPIAField_high_values(self):
    scan = self.createScan(2000.0, "DBZH", numpy.array(EXAMPLE_RAY).reshape(1,480), 0.0, 0.01)
    resultPIA, resultDatatype, dr = PIAHitschfeldBordan("DBZH").createPIAField(scan)

    lastvalue=resultPIA[0][0]
    for x in resultPIA[0]:
      self.assertTrue(x >= lastvalue)
    self.assertFalse(numpy.isnan(resultPIA).any())    
    self.assertFalse((resultPIA > 10.0).any())    

  def test_createPIAField_high_values_PIAMax_9(self):
    scan = self.createScan(2000.0, "DBZH", numpy.array(EXAMPLE_RAY).reshape(1,480), 0.0, 0.01)
    resultPIA, resultDatatype, dr = PIAHitschfeldBordan("DBZH", PIAMax=9.0).createPIAField(scan)

    lastvalue=resultPIA[0][0]
    for x in resultPIA[0]:
      self.assertTrue(x >= lastvalue)
    self.assertFalse(numpy.isnan(resultPIA).any())    
    self.assertFalse((resultPIA > 9.0).any())    

  def test_process(self):
    original_data = numpy.array(EXAMPLE_RAY).reshape(1,480)
    scan = self.createScan(2000.0, "DBZH", original_data, 0.0, 0.01)
    PIAHitschfeldBordan("DBZH").process(scan, True, QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY)

    self.assertTrue(scan.hasParameter("PIA"))

    pia_param = scan.getParameter("PIA")
    pia_raw_data = pia_param.getData()
    pia_data = pia_raw_data * pia_param.gain + pia_param.offset
    dbzh_param = scan.getParameter("DBZH")
    dbzh_raw_data = dbzh_param.getData()
    dbzh_data = numpy.where((dbzh_raw_data==dbzh_param.nodata)|(dbzh_raw_data==dbzh_param.undetect), dbzh_raw_data, dbzh_raw_data * dbzh_param.gain + dbzh_param.offset)
    original_data = original_data * 0.01
    expected_data = numpy.where((dbzh_raw_data==dbzh_param.nodata)|(dbzh_raw_data==dbzh_param.undetect), dbzh_raw_data, original_data + pia_data)
    numpy.testing.assert_array_almost_equal(dbzh_data, expected_data, 5)

    qfield = scan.findQualityFieldByHowTask("se.smhi.qc.hitschfeld-bordan")
    self.assertTrue(qfield is not None)
    numpy.testing.assert_array_almost_equal(pia_raw_data, qfield.getData(), 5)

  def test_process_QUALITY_CONTROL_MODE_ANALYZE(self):
    original_data = numpy.array(EXAMPLE_RAY).reshape(1,480)
    scan = self.createScan(2000.0, "DBZH", original_data, 0.0, 0.01)
    PIAHitschfeldBordan("DBZH").process(scan, True, QUALITY_CONTROL_MODE_ANALYZE)

    self.assertTrue(scan.hasParameter("PIA"))

    pia_raw_data = scan.getParameter("PIA").getData()
    dbzh_raw_data = scan.getParameter("DBZH").getData()
    numpy.testing.assert_array_almost_equal(dbzh_raw_data, original_data, 5)
    qfield = scan.findQualityFieldByHowTask("se.smhi.qc.hitschfeld-bordan")
    numpy.testing.assert_array_almost_equal(pia_raw_data, qfield.getData(), 5)
