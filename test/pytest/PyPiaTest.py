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
import numpy as np
import math
import _polarscan, _polarscanparam, _raveio
import _pia
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE, QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY

import compute_pia
from compute_pia import PIAHitschfeldBordan

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

class PyPiaTest(unittest.TestCase):
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
    pia = _pia.new()
    scan = self.createScan(2000.0, "DBZH", numpy.array(EXAMPLE_RAY).reshape(1,480), 0.0, 0.01)
    resultPIA = pia.calculatePIA(scan, "DBZH").getData()
    lastvalue=resultPIA[0][0]
    for x in resultPIA[0]:
      self.assertTrue(x >= lastvalue)
      lastvalue = x
    self.assertFalse(numpy.isnan(resultPIA).any())    

  def test_createPIAField_high_values(self):
    pia = _pia.new()
    scan = self.createScan(2000.0, "DBZH", numpy.array(EXAMPLE_RAY).reshape(1,480), 0.0, 0.01)
    resultPIA = pia.calculatePIA(scan, "DBZH").getData()

    lastvalue=resultPIA[0][0]
    for x in resultPIA[0]:
      self.assertTrue(x >= lastvalue)
      lastvalue = x
    self.assertFalse(numpy.isnan(resultPIA).any())    
    self.assertFalse((resultPIA > 10.0).any())    

  def test_createPIAField_high_values_PIAMax_9(self):
    pia = _pia.new()
    pia.max_pia = 9.0
    scan = self.createScan(2000.0, "DBZH", numpy.array(EXAMPLE_RAY).reshape(1,480), 0.0, 0.01)
    resultPIA = pia.calculatePIA(scan, "DBZH").getData()


    lastvalue=resultPIA[0][0]
    for x in resultPIA[0]:
      self.assertTrue(x >= lastvalue)
    self.assertFalse(numpy.isnan(resultPIA).any())    
    self.assertFalse((resultPIA > 9.0).any())    

  def create_expected_pia(self, dbzhparam, rscale, piamax):
    dr = rscale / 1000.0  # m -> km

    kacumsum_factor = 0.2 * dr  * np.log(10) / 1.344
    kacumsum_limit = 1 / kacumsum_factor   # To avoid NaN when calculating PIA (i.e. not any calls to log10 with negative values)

    raw_data = dbzhparam.getData()
    dbzh_data = raw_data * dbzhparam.gain + dbzhparam.offset  # Don't care about including undetect and nodata, it will be handled later on.
    datatypes = np.where(raw_data==dbzhparam.undetect, 0, 2).astype(np.uint8)   # Datatypes, 0 = Undetect, 1 = Nodata and 2 = Data
    datatypes[raw_data==dbzhparam.nodata] = 1

    ka = np.where(datatypes==2, ((10**(0.1 * (dbzh_data))) / 7.34e5) ** (1 / 1.344), 0)    # Apparent specific attenuation [dB/km]
    kacumsum = np.cumsum(ka, 1)
    kacumsum[kacumsum > kacumsum_limit] = kacumsum_limit-(1e-06)  # Ensure that PIA calculate does not explode

    PIA = -10 * 1.344 * np.log10(1 - kacumsum_factor * kacumsum)

    PIA[PIA > piamax] = piamax

    return PIA

  def test_create_parameter(self):
    pia = _pia.new()
    original_data = numpy.array(EXAMPLE_RAY).reshape(1,480)
    scan = self.createScan(2000.0, "DBZH", original_data, 0.0, 0.01)
    param = pia.createPIAParameter(scan, "DBZH")
    self.assertEqual("PIA", param.quantity)
    pia_raw_data = param.getData()
    self.assertEqual(np.float64, pia_raw_data.dtype)
    pia_data = pia_raw_data * param.gain + param.offset

    expected_pia = self.create_expected_pia(scan.getParameter("DBZH"), scan.rscale, 10.0)
    numpy.testing.assert_array_almost_equal(pia_data, expected_pia, 2)

  def test_process(self):
    pia = _pia.new()
    original_data = numpy.array(EXAMPLE_RAY).reshape(1,480)
    scan = self.createScan(2000.0, "DBZH", original_data, 0.0, 0.01)

    pia.process(scan, "DBZH", True, True, False)
    numpy.testing.assert_array_almost_equal(original_data, scan.getParameter("DBZH").getData(), 5)  # We should not allow modification of DBZH in this case

    self.assertTrue(scan.hasParameter("PIA"))
    param = scan.getParameter("PIA")
    self.assertEqual("PIA", param.quantity)
    self.assertAlmostEqual(0.0, param.offset, 4)
    self.assertAlmostEqual(0.01, param.gain, 4)

    qfield = scan.getQualityFieldByHowTask("se.smhi.qc.hitschfeld-bordan")
    self.assertEqual("se.smhi.qc.hitschfeld-bordan", qfield.getAttribute("how/task"))
    self.assertEqual("param_name=PIA c_ZK=7.3e+05 d_ZK=1.344 PIAmax=10 dr=2", qfield.getAttribute("how/task_args"))
    numpy.testing.assert_array_almost_equal(param.getData(), qfield.getData(), 5)


  def test_process_apply(self):
    pia = _pia.new()
    original_data = numpy.array(EXAMPLE_RAY).reshape(1,480)
    scan = self.createScan(2000.0, "DBZH", original_data, 0.0, 0.01)

    pia.process(scan, "DBZH", True, True, True)

    self.assertTrue(scan.hasParameter("PIA"))
    param = scan.getParameter("PIA")
    self.assertEqual("PIA", param.quantity)
    self.assertAlmostEqual(0.0, param.offset, 4)
    self.assertAlmostEqual(0.01, param.gain, 4)

    qfield = scan.getQualityFieldByHowTask("se.smhi.qc.hitschfeld-bordan")  # We verified PIA calc in test_create_parameter so we just assume that it is correct

    self.assertEqual("se.smhi.qc.hitschfeld-bordan", qfield.getAttribute("how/task"))
    self.assertEqual("param_name=PIA c_ZK=7.3e+05 d_ZK=1.344 PIAmax=10 dr=2", qfield.getAttribute("how/task_args"))
    numpy.testing.assert_array_almost_equal(param.getData(), qfield.getData(), 5)

    original = original_data*0.01
    adjusted = original + qfield.getData()*0.01
    expected = numpy.round(numpy.where((original_data==-32768.0)|(original_data==-32767.0), original_data, adjusted/0.01)).astype(numpy.int16)

    dbzh_data = scan.getParameter("DBZH").getData()

    actual = numpy.where((dbzh_data==-32768.0)|(dbzh_data==-32767.0), dbzh_data, dbzh_data).astype(numpy.int16)

    numpy.testing.assert_array_almost_equal(expected, actual, 0)
