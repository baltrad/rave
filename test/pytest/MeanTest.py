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

Tests the _mean module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-12-16
'''
import unittest
import _mean
import _cartesian
import _cartesianparam
import numpy
import string

class MeanTest(unittest.TestCase):
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_average(self):
    param = _cartesianparam.new()
    param.nodata = 255.0
    param.undetect = 0.0
    
    data = numpy.zeros((5,5), numpy.float64)
    for y in range(5):
      for x in range(5):
        data[y][x] = float(x+y*5)

    # add some nodata and undetect
    data[0][0] = param.nodata    # 0
    data[0][3] = param.nodata    # 3
    data[1][2] = param.nodata    # 7
    data[1][3] = param.undetect  # 8
    data[3][2] = param.undetect  # 17
    data[4][4] = param.nodata    # 24
    #print `data` #to be able to see the array when calculating result
    
    param.setData(data)
    param.quantity="DBZH"
    
    src = _cartesian.new()
    src.addParameter(param)
    
    target = _mean.average(src, 2)
    
    # Table with the result
    expected = [[param.nodata, 1, 1.5, param.nodata, 4.0],
                [5.0, 4.0, param.nodata, param.undetect, 6.5],
                [7.5, 8.0, 9.67, 12.5, 12.0],
                [12.5, 13.0, param.undetect, 14.33, 16.0],
                [17.5, 18.0, 19.67, 21.0, param.nodata]]
    
    expectedarr = numpy.array(expected, numpy.float64)
    
    actualarr = target.getParameter("DBZH").getData()
    
    # Unfortunately there is no numpy.compareAlmostEquals or similar (at least as I know).
    for y in range(5):
      for x in range(5):
        self.assertAlmostEqual(expectedarr[y][x], actualarr[y][x], 2)
    