'''
Copyright (C) 2012 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the dealiasing functionality

@file
@author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2012-11-20
'''
import unittest
import os
import _raveio
import _dealias
import _polarscan, _polarscanparam
from numpy import *

class PyDealiasTest(unittest.TestCase):
    FIXTURE = "fixtures/var-20111127T194500Z.h5"
    DEALIASED = "fixtures/dealiased_scan.h5"
    BADINPUT = "fixtures/vp_fixture.h5"
    classUnderTest = None
  
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_notDealiased(self):
        scan = _polarscan.new()
        vrad = _polarscanparam.new()
        vrad.quantity = "VRAD"
        scan.addParameter(vrad)
        self.assertFalse(_dealias.dealiased(scan)) 

    def testDealiased(self):
        scan = _polarscan.new()
        vrad = _polarscanparam.new()
        vrad.quantity = "VRAD"
        vrad.addAttribute("how/dealiased", "True")
        scan.addParameter(vrad)
        self.assertTrue(_dealias.dealiased(scan)) 
        try:
          scan.getParameter("VRAD").getAttribute("how/task")
          self.fail("Expected AttributeError")
        except AttributeError:
          pass

    def testDealiasScan(self):
        scan = _raveio.open(self.FIXTURE).object.getScan(0)
        dscan = _raveio.open(self.DEALIASED).object
        self.assertTrue(different(scan, dscan))        
        status = _dealias.dealias(scan)
        self.assertFalse(different(scan, dscan))
        self.assertEquals("se.smhi.detector.dealias", scan.getParameter("VRAD").getAttribute("how/task"))

    # Only checks the first scan in the volume.
    def testDealiasPvol(self):
        pvol = _raveio.open(self.FIXTURE).object
        dscan = _raveio.open(self.DEALIASED).object
        self.assertTrue(different(pvol.getScan(0), dscan))        
        status = _dealias.dealias(pvol)
        for i in range(pvol.getNumberOfScans()):
          scan = pvol.getScan(i)
          if scan.hasParameter("VRAD") and scan.elangle < 2.0*math.pi/180.0: # Currently, max elev angle is 2.0
            self.assertEquals("se.smhi.detector.dealias", scan.getParameter("VRAD").getAttribute("how/task"))

        self.assertFalse(different(pvol.getScan(0), dscan))

    def testAlreadyDealiased(self):
        scan = _raveio.open(self.FIXTURE).object.getScan(0)
        status = _dealias.dealias(scan)
        self.assertTrue(status)
        status = _dealias.dealias(scan)
        self.assertFalse(status)

    def testWrongInput(self):
        vertical_profile = _raveio.open(self.BADINPUT).object
        try:
            status = _dealias.dealias(vertical_profile)
        except AttributeError:
            self.assertTrue(True)

    
def different(scan1, scan2):
    a = scan1.getParameter("VRAD").getData()
    b = scan2.getParameter("VRAD").getData()
    c = a == b
    d = sum(where(equal(c, False), 1, 0).flat)
    if d > 0:
        return True
    else:
        return False 