'''
Copyright (C) 2010- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.

'''
## Tests the utility functions in rave_util
##

## @file
## @author Anders Henja, SMHI
## @date 2014-05-06
import unittest
import os
import math
import string
import numpy
import rave_util
import _polarscan, _polarvolume, _cartesian

class rave_util_test(unittest.TestCase):
  classUnderTest = None
  
  def setUp(self):
    pass
    
  def tearDown(self):
    pass
    
  def test_is_polar_malfunc_pv(self):
    pv = _polarvolume.new()
    pv.addAttribute("how/malfunc", 'False')
    ps1 = _polarscan.new()
    ps1.addAttribute("how/malfunc", 'False')
    ps1.elangle = 1.0
    ps2 = _polarscan.new()
    ps2.addAttribute("how/malfunc", 'True')
    ps2.elangle = 2.0
    pv.addScan(ps1)
    pv.addScan(ps2)
    
    self.assertEqual(True, rave_util.is_polar_malfunc(pv))
    
  def test_is_polar_malfunc_pv2(self):
    pv = _polarvolume.new()
    pv.addAttribute("how/malfunc", 'False')
    ps1 = _polarscan.new()
    ps1.addAttribute("how/malfunc", 'False')
    ps1.elangle = 1.0
    ps2 = _polarscan.new()
    ps2.elangle = 2.0
    pv.addScan(ps1)
    pv.addScan(ps2)
    
    self.assertEqual(False, rave_util.is_polar_malfunc(pv))
    
  def test_is_polar_malfunc_pv3(self):
    pv = _polarvolume.new()
    ps1 = _polarscan.new()
    ps1.addAttribute("how/malfunc", 'False')
    ps1.elangle = 1.0
    ps2 = _polarscan.new()
    ps1.addAttribute("how/malfunc", 'True')
    ps2.elangle = 2.0
    pv.addScan(ps1)
    pv.addScan(ps2)
    
    self.assertEqual(True, rave_util.is_polar_malfunc(pv))
    
  def test_is_polar_malfunc_ps(self):
    ps = _polarscan.new()
    ps.addAttribute("how/malfunc", 'False')
    self.assertEqual(False, rave_util.is_polar_malfunc(ps))

  def test_is_polar_malfunc_ps2(self):
    ps = _polarscan.new()
    ps.addAttribute("how/malfunc", 'yes')
    self.assertEqual(True, rave_util.is_polar_malfunc(ps))

  def test_is_polar_malfunc_ps3(self):
    ps = _polarscan.new()
    self.assertEqual(False, rave_util.is_polar_malfunc(ps))
    
  def test_remove_malfunc_from_volume(self):
    pv = _polarvolume.new()
    pv.addAttribute("how/malfunc", 'False')
    ps1 = _polarscan.new()
    ps1.addAttribute("how/malfunc", 'False')
    ps1.elangle = 1.0
    ps2 = _polarscan.new()
    ps2.elangle = 2.0
    ps2.addAttribute("how/malfunc", 'False')
    pv.addScan(ps1)
    pv.addScan(ps2)
    
    result = rave_util.remove_malfunc_from_volume(pv)
    self.assertTrue(ps1 == pv.getScan(0))
    self.assertTrue(ps2 == pv.getScan(1))

  def test_remove_malfunc_from_volume2(self):
    pv = _polarvolume.new()
    pv.addAttribute("how/malfunc", 'True')
    ps1 = _polarscan.new()
    ps1.addAttribute("how/malfunc", 'False')
    ps1.elangle = 1.0
    ps2 = _polarscan.new()
    ps2.elangle = 2.0
    ps2.addAttribute("how/malfunc", 'False')
    pv.addScan(ps1)
    pv.addScan(ps2)
    
    result = rave_util.remove_malfunc_from_volume(pv)
    self.assertTrue(result == None)

  def test_remove_malfunc_from_volume3(self):
    pv = _polarvolume.new()
    pv.addAttribute("how/malfunc", 'False')
    ps1 = _polarscan.new()
    ps1.addAttribute("how/malfunc", 'False')
    ps1.elangle = 1.0
    ps2 = _polarscan.new()
    ps2.addAttribute("how/malfunc", 'True')
    ps2.elangle = 2.0
    ps3 = _polarscan.new()
    ps3.addAttribute("how/malfunc", 'False')
    ps3.elangle = 3.0
    pv.addScan(ps1)
    pv.addScan(ps2)
    pv.addScan(ps3)
    
    result = rave_util.remove_malfunc_from_volume(pv)
    self.assertEqual(2, result.getNumberOfScans())
    self.assertTrue(ps1 == pv.getScan(0))
    self.assertTrue(ps3 == pv.getScan(1))
    
  def test_remove_malfunc_from_volume__all_malfunc(self):
    pv = _polarvolume.new()
    pv.addAttribute("how/malfunc", 'False')
    ps1 = _polarscan.new()
    ps1.addAttribute("how/malfunc", 'True')
    ps1.elangle = 1.0
    ps2 = _polarscan.new()
    ps2.addAttribute("how/malfunc", 'True')
    ps2.elangle = 2.0
    ps3 = _polarscan.new()
    ps3.addAttribute("how/malfunc", 'True')
    ps3.elangle = 3.0
    pv.addScan(ps1)
    pv.addScan(ps2)
    pv.addScan(ps3)
    
    result = rave_util.remove_malfunc_from_volume(pv)
    self.assertEqual(0, result.getNumberOfScans())
  
  def test_remove_malfunc(self):
    pv = _polarvolume.new()
    pv.addAttribute("how/malfunc", 'False')
    ps1 = _polarscan.new()
    ps1.addAttribute("how/malfunc", 'False')
    ps1.elangle = 1.0
    ps2 = _polarscan.new()
    ps2.addAttribute("how/malfunc", 'True')
    ps2.elangle = 2.0
    ps3 = _polarscan.new()
    ps3.addAttribute("how/malfunc", 'False')
    ps3.elangle = 3.0
    pv.addScan(ps1)
    pv.addScan(ps2)
    pv.addScan(ps3)
    
    result = rave_util.remove_malfunc(pv)
    self.assertTrue(result == pv)
    self.assertEqual(2, result.getNumberOfScans())
    self.assertTrue(ps1 == pv.getScan(0))
    self.assertTrue(ps3 == pv.getScan(1))

  def test_remove_malfunc_2(self):
    ps = _polarscan.new()
    ps.addAttribute("how/malfunc", 'False')
    result = rave_util.remove_malfunc(ps)
    self.assertTrue(result == ps)
  
  def test_remove_malfunc_3(self):
    ps = _polarscan.new()
    ps.addAttribute("how/malfunc", 'True')
    result = rave_util.remove_malfunc(ps)
    self.assertTrue(result is None)
    
  def test_remove_malfunc_4(self):
    c = _cartesian.new()
    c.addAttribute("how/malfunc", 'True')
    result = rave_util.remove_malfunc(c)
    self.assertTrue(result == c)
    
  def test_remove_malfunc__all_malfunc(self):
    pv = _polarvolume.new()
    pv.addAttribute("how/malfunc", 'False')
    ps1 = _polarscan.new()
    ps1.addAttribute("how/malfunc", 'True')
    ps1.elangle = 1.0
    ps2 = _polarscan.new()
    ps2.addAttribute("how/malfunc", 'True')
    ps2.elangle = 2.0
    ps3 = _polarscan.new()
    ps3.addAttribute("how/malfunc", 'True')
    ps3.elangle = 3.0
    pv.addScan(ps1)
    pv.addScan(ps2)
    pv.addScan(ps3)
    
    result = rave_util.remove_malfunc(pv)
    self.assertTrue(result == None)
