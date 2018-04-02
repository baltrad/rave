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
import unittest, os, datetime, math, string
import rave_dom_db
from rave_dom import observation
from gadjust import obsmatcher
import mock
import _raveio, _rave
from gadjust.grapoint import grapoint
from gadjust import gra
import numpy

##
# Tests that the gra works as expected.
#
class gadjust_gra_test(unittest.TestCase):
  TEMP_STAT_FILE = "gadjust-test.tmp.stat"
  def setUp(self):
    self.classUnderTest = None
    try:
      if os.path.isfile(self.TEMP_STAT_FILE):
        os.unlink(self.TEMP_STAT_FILE)
    except:
      pass
    self.old_settings=None
    try:
      self.old_settings=numpy.seterr(all='ignore')
    except:
      pass

  def tearDown(self):
    self.classUnderTest = None
    try:
      if os.path.isfile(self.TEMP_STAT_FILE):
        os.unlink(self.TEMP_STAT_FILE)
    except:
      pass
    if self.old_settings is not None:
      numpy.seterr(**self.old_settings)    

  def test_generate(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    gra.generate(points, "20131010","101500", self.TEMP_STAT_FILE)

    #20131010 101500 False 3 0 nan T 0.000000 1.000000 -2.000000 1.000000 0.666667 0.471405
    fp = open(self.TEMP_STAT_FILE, 'r')
    lines=fp.readlines()
    tokens = string.split(lines[0].rstrip().lstrip(), " ")
    fp.close()
    self.assertEquals(13, len(tokens))
    self.assertEquals("20131010", tokens[0])
    self.assertEquals("101500", tokens[1])
    self.assertEquals("False", tokens[2])
    self.assertEquals(3, int(tokens[3]))
    self.assertEquals(0, int(tokens[4]))
    self.assertEquals("T", tokens[6])
    self.assertAlmostEquals(0.0, float(tokens[7]), 4)
    self.assertAlmostEquals(1.0, float(tokens[8]), 4)
    self.assertAlmostEquals(-2.0, float(tokens[9]), 4)
    self.assertAlmostEquals(1.0, float(tokens[10]), 4)
    self.assertAlmostEquals(0.6667, float(tokens[11]), 4)
    self.assertAlmostEquals(0.4714, float(tokens[12]), 4)
  
  def test_generate_2(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    significant, npoints, loss, r, sig, corr_coeff, a, b, c, m, dev = gra.generate(points, "20131010","101500", self.TEMP_STAT_FILE)
    
    self.assertEquals('False', significant)
    self.assertEquals(3, npoints)
    self.assertEquals(0, loss)
    self.assertFalse(`type(r)`.find("numpy")>=0)
    self.assertTrue(math.isnan(r))
    
    self.assertFalse(`type(corr_coeff)`.find("numpy")>=0)
    self.assertAlmostEquals(0.0, corr_coeff)
    
    self.assertFalse(`type(a)`.find("numpy")>=0)
    self.assertAlmostEquals(1.0, a)
    
    self.assertFalse(`type(b)`.find("numpy")>=0)
    self.assertAlmostEquals(-2.0, b)
    
    self.assertFalse(`type(c)`.find("numpy")>=0)
    self.assertAlmostEquals(1.0, c)
    
    self.assertFalse(`type(m)`.find("numpy")>=0)
    self.assertAlmostEquals(0.6667, m, 4)
    
    self.assertFalse(`type(dev)`.find("numpy")>=0)
    self.assertAlmostEquals(0.4714, dev, 4)
    
  def test_get_2nd_order_adjustment(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    self.classUnderTest = gra.gra(points)
    result = self.classUnderTest.get_2nd_order_adjustment()

    self.assertEquals(6, len(result))
    
    self.assertAlmostEquals(1.0, result[0], 4)  # The following 3 are from least square 2nd degree
    self.assertAlmostEquals(-2.0, result[1], 4)
    self.assertAlmostEquals(1.0, result[2], 4)
    
    self.assertAlmostEquals(0.6667, result[3], 4) # The std deviation
    self.assertAlmostEquals(0.4714, result[4], 4)
    
    self.assertEquals(0, result[5]) # This is the difference between provided points and used points, should be 0 in this case
    


  def test_least_square_nth_degree(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    self.classUnderTest = gra.gra(points)
    result = self.classUnderTest.least_square_nth_degree(2)
    self.assertEquals(3, len(result))
    self.assertAlmostEquals(1.0, result[0], 4)
    self.assertAlmostEquals(-2.0, result[1], 4)
    self.assertAlmostEquals(1.0, result[2], 4)
    
  def test_get_correlation(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    self.classUnderTest = gra.gra(points)
    self.assertAlmostEquals(0.0, self.classUnderTest.get_correlation(), 4)
    
  def test_get_correlation_2(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    self.classUnderTest = gra.gra(points)
    self.assertAlmostEquals(-1.0, self.classUnderTest.get_correlation(), 2)
    
    
  def test_get_std_deviation(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]

    self.classUnderTest = gra.gra(points)
    result = self.classUnderTest.get_std_deviation()
    self.assertEquals(2, len(result))
    self.assertAlmostEquals(0.6667, result[0], 4)
    self.assertAlmostEquals(0.4714, result[1], 4)
    
    
