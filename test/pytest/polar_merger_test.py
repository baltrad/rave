'''
Copyright (C) 2016 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the polar merger module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2016-10-17
'''
import unittest
import _raveio, _polarvolume, _polarscan
import polar_merger
import string
import numpy
from numpy import array, reshape, uint8

class polar_merger_test(unittest.TestCase):
  FIXTURE_1="fixtures/scan_sehuv_0.5_20110126T184500Z.h5"  
  FIXTURE_2="fixtures/scan_sehuv_1.0_20110126T184600Z.h5"
  FIXTURE_3="fixtures/scan_sehuv_1.5_20110126T184600Z.h5"
  FIXTURE_PVOL="fixtures/selul_pvol_201503300845.h5"
  def setUp(self):
    pass

  def tearDown(self):
    pass
  
  def test_merge(self):
    scan_dbzh = _raveio.open(self.FIXTURE_1).object
    scan_vrad = _raveio.open(self.FIXTURE_1).object
    p = scan_vrad.removeParameter("DBZH")
    p.quantity="VRAD"
    scan_vrad.addParameter(p)
    
    classUnderTest = polar_merger.polar_merger()
    result = classUnderTest.merge([scan_dbzh,scan_vrad])
    self.assertTrue("VRAD" in result.getParameterNames())
    self.assertTrue("DBZH" in result.getParameterNames())

  def test_merge_different_sizes(self):
    scan_dbzh = _raveio.open(self.FIXTURE_1).object
    scan_vrad = _raveio.open(self.FIXTURE_1).object
    p = scan_vrad.removeParameter("DBZH")
    p.quantity="VRAD"
    p.setData(numpy.zeros((10,10), numpy.uint8))
    scan_vrad.addParameter(p)
    
    
    classUnderTest = polar_merger.polar_merger()
    try:
      result = classUnderTest.merge([scan_dbzh,scan_vrad])
      self.fail("Expected an exception")
    except AttributeError,e:
      pass
    
  def test_merge_with_volume(self):
    pvol = _raveio.open(self.FIXTURE_PVOL).object
    scan = pvol.getScanClosestToElevation(0.5, 0).clone()
    self.assertEquals(2, len(scan.getParameterNames()))
    self.assertTrue("DBZH" in scan.getParameterNames())
    self.assertTrue("VRAD" in scan.getParameterNames())
    p1 = scan.getParameter("DBZH").clone()
    p1.quantity = "TH"
    scan.addParameter(p1)
    scan.removeParameter("DBZH")
    scan.removeParameter("VRAD")
    
    classUnderTest = polar_merger.polar_merger()
    result = classUnderTest.merge([pvol, scan])
    rs = pvol.getScanClosestToElevation(0.5, 0)
    self.assertEquals(3, len(rs.getParameterNames()))
    self.assertTrue("DBZH" in rs.getParameterNames())
    self.assertTrue("VRAD" in rs.getParameterNames())
    self.assertTrue("TH" in rs.getParameterNames())
    
    rs = pvol.getScanClosestToElevation(1.0, 0)
    self.assertEquals(2, len(rs.getParameterNames()))
    self.assertTrue("DBZH" in rs.getParameterNames())
    self.assertTrue("VRAD" in rs.getParameterNames())
    

    