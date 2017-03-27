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
    self.old_settings=numpy.seterr(all='ignore')

  def tearDown(self):
    self.classUnderTest = None
    try:
      if os.path.isfile(self.TEMP_STAT_FILE):
        os.unlink(self.TEMP_STAT_FILE)
    except:
      pass
    numpy.seterr(**self.old_settings)

  def test_generate(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    gra.generate(points, "20131010","101500", self.TEMP_STAT_FILE)

    #20131010 101500 False 3 0 nan T 0.000000 1.000000 -2.000000 1.000000 0.666667 0.471405
    fp = open(self.TEMP_STAT_FILE, 'r')
    lines=fp.readlines()
    tokens = lines[0].rstrip().lstrip().split(" ")
    fp.close()
    self.assertEqual(13, len(tokens))
    self.assertEqual("20131010", tokens[0])
    self.assertEqual("101500", tokens[1])
    self.assertEqual("False", tokens[2])
    self.assertEqual(3, int(tokens[3]))
    self.assertEqual(0, int(tokens[4]))
    self.assertEqual("T", tokens[6])
    self.assertAlmostEqual(0.0, float(tokens[7]), 4)
    self.assertAlmostEqual(1.0, float(tokens[8]), 4)
    self.assertAlmostEqual(-2.0, float(tokens[9]), 4)
    self.assertAlmostEqual(1.0, float(tokens[10]), 4)
    self.assertAlmostEqual(0.6667, float(tokens[11]), 4)
    self.assertAlmostEqual(0.4714, float(tokens[12]), 4)
  
  def test_generate_2(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    significant, npoints, loss, r, sig, corr_coeff, a, b, c, m, dev = gra.generate(points, "20131010","101500", self.TEMP_STAT_FILE)
    
    self.assertEqual('False', significant)
    self.assertEqual(3, npoints)
    self.assertEqual(0, loss)
    self.assertFalse(str(type(r)).find("numpy")>=0)
    self.assertTrue(math.isnan(r))
    
    self.assertFalse(str(type(corr_coeff)).find("numpy")>=0)
    self.assertAlmostEqual(0.0, corr_coeff)
    
    self.assertFalse(str(type(a)).find("numpy")>=0)
    self.assertAlmostEqual(1.0, a)
    
    self.assertFalse(str(type(b)).find("numpy")>=0)
    self.assertAlmostEqual(-2.0, b)
    
    self.assertFalse(str(type(c)).find("numpy")>=0)
    self.assertAlmostEqual(1.0, c)
    
    self.assertFalse(str(type(m)).find("numpy")>=0)
    self.assertAlmostEqual(0.6667, m, 4)
    
    self.assertFalse(str(type(dev)).find("numpy")>=0)
    self.assertAlmostEqual(0.4714, dev, 4)
    
  def test_get_2nd_order_adjustment(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    self.classUnderTest = gra.gra(points)
    result = self.classUnderTest.get_2nd_order_adjustment()

    self.assertEqual(6, len(result))
    
    self.assertAlmostEqual(1.0, result[0], 4)  # The following 3 are from least square 2nd degree
    self.assertAlmostEqual(-2.0, result[1], 4)
    self.assertAlmostEqual(1.0, result[2], 4)
    
    self.assertAlmostEqual(0.6667, result[3], 4) # The std deviation
    self.assertAlmostEqual(0.4714, result[4], 4)
    
    self.assertEqual(0, result[5]) # This is the difference between provided points and used points, should be 0 in this case
    


  def test_least_square_nth_degree(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    self.classUnderTest = gra.gra(points)
    result = self.classUnderTest.least_square_nth_degree(2)
    self.assertEqual(3, len(result))
    self.assertAlmostEqual(1.0, result[0], 4)
    self.assertAlmostEqual(-2.0, result[1], 4)
    self.assertAlmostEqual(1.0, result[2], 4)
    
  def test_get_correlation(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    self.classUnderTest = gra.gra(points)
    self.assertAlmostEqual(0.0, self.classUnderTest.get_correlation(), 4)
    
  def test_get_correlation_2(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]
    
    self.classUnderTest = gra.gra(points)
    self.assertAlmostEqual(-1.0, self.classUnderTest.get_correlation(), 2)
    
    
  def test_get_std_deviation(self):
    points = [grapoint(_rave.RaveValueType_DATA, 1.0, 0.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 1.0, 10.0, 20.0, "20131010", "101500", 1.0, 12),
              grapoint(_rave.RaveValueType_DATA, 1.0, 2.0, 10.0, 20.0, "20131010", "101500", 10**0.1, 12)]

    self.classUnderTest = gra.gra(points)
    result = self.classUnderTest.get_std_deviation()
    self.assertEqual(2, len(result))
    self.assertAlmostEqual(0.6667, result[0], 4)
    self.assertAlmostEqual(0.4714, result[1], 4)
    
    