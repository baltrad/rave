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

Tests ctfilter functionality

@file
@author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2014-03-31
'''
import unittest
import os
import _raveio
import rave_ctfilter
from numpy import *

class PyCTTest(unittest.TestCase):
    CTPATH = "fixtures"
    CTFILE = "fixtures/SAFNWC_MSG3_CT___201403250900_FES_________.h5"
    CT_AE = (-5570248.4773392612, -5567248.074173444,
             5567248.074173444, 5570248.4773392612) # From PyTROLL
    COMP = "fixtures/comp4ctfilter.h5"
    classUnderTest = None
  
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testAreaExtent(self):
        rave_ctfilter.CTPATH = self.CTPATH
        ct = rave_ctfilter.readCT(self.CTFILE)
        for i in range(4):
            self.assertAlmostEqual(ct.areaextent[i],
                                    self.CT_AE[i], places=1)

    def XtestCT(self): ### @TODO: Need to rewrite fixture composite to get new reference value since we have moved index to upper left
        rave_ctfilter.CTPATH = self.CTPATH
        rio = _raveio.open(self.COMP)
        image = rio.object.getImage(0)
        ret = rave_ctfilter.ctFilter(image)
        self.assertTrue(ret)
        filtered = image.getParameter("DBZH").getData()
        reference = rio.object.getImage(1).getParameter("DBZH").getData()
        self.assertFalse(different(filtered, reference))
        qind_f = image.getParameter("DBZH").getQualityFieldByHowTask("se.smhi.quality.ctfilter").getData()
        qind_r = rio.object.getImage(1).getParameter("DBZH").getQualityFieldByHowTask("se.smhi.quality.ctfilter").getData()
        self.assertFalse(different(qind_f, qind_r))

    def testBadQuantity(self):
        rave_ctfilter.CTPATH = self.CTPATH
        rio = _raveio.open(self.COMP)
        image = rio.object.getImage(0)
        try:
            ret = rave_ctfilter.ctFilter(image, quantity="eggs")
        except AttributeError:
            self.assertTrue(True)

    
def different(data1, data2):
    print(str(data1.shape))
    print(str(data2.shape))
    
    c = data1 == data2
    print(str(c))
    d = sum(where(equal(c, False), 1, 0).flat)
    if d > 0:
        return True
    else:
        return False 
