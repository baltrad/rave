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
import math

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
        vrad.quantity = "VRADH"
        scan.addParameter(vrad)
        self.assertFalse(_dealias.dealiased(scan)) 

    def test_notDealiased_notVRADh(self):
        scan = _polarscan.new()
        vrad = _polarscanparam.new()
        vrad.quantity = "VRAD"
        vrad.addAttribute("how/dealiased", "True")
        scan.addParameter(vrad)
        self.assertFalse(_dealias.dealiased(scan)) 

    def testDealiased(self):
        scan = _polarscan.new()
        vrad = _polarscanparam.new()
        vrad.quantity = "VRADH"
        vrad.addAttribute("how/dealiased", "True")
        scan.addParameter(vrad)
        self.assertTrue(_dealias.dealiased(scan)) 
        try:
          scan.getParameter("VRADH").getAttribute("how/task")
          self.fail("Expected AttributeError")
        except AttributeError:
          pass

    def testDealiasScan(self):
        scan = _raveio.open(self.FIXTURE).object.getScan(0)
        dscan = _raveio.open(self.DEALIASED).object
        self.assertTrue(different(scan, dscan))        
        status = _dealias.dealias(scan)
        self.assertFalse(different(scan, dscan))
        self.assertEqual("se.smhi.detector.dealias", scan.getParameter("VRADH").getAttribute("how/task"))

    def testDealiasScan_VRAD(self):
        # Really not relevant but we don't force what parameter to use
        scan = _raveio.open(self.FIXTURE).object.getScan(0)
        dscan = _raveio.open(self.DEALIASED).object
        scan.addParameter(copyParam(scan.getParameter("VRADH"), "ABC"))
        dscan.addParameter(copyParam(dscan.getParameter("VRADH"), "ABC"))

        self.assertTrue(different(scan, dscan))
        self.assertTrue(different(scan, dscan, "ABC"))
        
        status = _dealias.dealias(scan, "VRADH")
        self.assertFalse(different(scan, dscan, "VRADH"))
        self.assertTrue(different(scan, dscan, "ABC"))
        self.assertFalse(scan.getParameter("ABC").hasAttribute("how/task"))
        self.assertEqual("se.smhi.detector.dealias", scan.getParameter("VRADH").getAttribute("how/task"))

    def testDealiasScan_VRADV(self):
        # Really not relevant but we don't force what parameter to use
        scan = _raveio.open(self.FIXTURE).object.getScan(0)
        dscan = _raveio.open(self.DEALIASED).object
        scan.addParameter(copyParam(scan.getParameter("VRADH"), "VRADV"))
        dscan.addParameter(copyParam(dscan.getParameter("VRADH"), "VRADV"))

        self.assertTrue(different(scan, dscan))
        self.assertTrue(different(scan, dscan, "VRADV"))
        
        status = _dealias.dealias(scan, "VRADV")
        self.assertTrue(different(scan, dscan, "VRADH"))
        self.assertFalse(different(scan, dscan, "VRADV"))
        self.assertFalse(scan.getParameter("VRADH").hasAttribute("how/task"))
        self.assertEqual("se.smhi.detector.dealias", scan.getParameter("VRADV").getAttribute("how/task"))

    # Only checks the first scan in the volume.
    def testDealiasPvol(self):
        pvol = _raveio.open(self.FIXTURE).object
        dscan = _raveio.open(self.DEALIASED).object
        self.assertTrue(different(pvol.getScan(0), dscan))        
        status = _dealias.dealias(pvol)
        for i in range(pvol.getNumberOfScans()):
          scan = pvol.getScan(i)
          if scan.hasParameter("VRADH") and scan.elangle < 2.0*math.pi/180.0: # Currently, max elev angle is 2.0
            self.assertEqual("se.smhi.detector.dealias", scan.getParameter("VRADH").getAttribute("how/task"))

        self.assertFalse(different(pvol.getScan(0), dscan))

    # Only checks the first scan in the volume.
    def testDealiasPvol_byEMAX_default(self):
        pvol = _raveio.open(self.FIXTURE).object
        dscan = _raveio.open(self.DEALIASED).object
        self.assertTrue(different(pvol.getScan(0), dscan))        
        status = _dealias.dealias(pvol, "VRADH")
        for i in range(pvol.getNumberOfScans()):
          scan = pvol.getScan(i)
          if scan.hasParameter("VRADH"):
            if scan.elangle <= 2.0*math.pi/180.0:
              self.assertEqual("se.smhi.detector.dealias", scan.getParameter("VRADH").getAttribute("how/task"))
            else:
              self.assertFalse(scan.getParameter("VRADH").hasAttribute("how/task"))

        self.assertFalse(different(pvol.getScan(0), dscan))

    def testDealiasPvol_byEMAX_higher(self):
        pvol = _raveio.open(self.FIXTURE).object
        dscan = _raveio.open(self.DEALIASED).object
        self.assertTrue(different(pvol.getScan(0), dscan))        
        status = _dealias.dealias(pvol, "VRADH", 30.0)
        for i in range(pvol.getNumberOfScans()):
          scan = pvol.getScan(i)
          if scan.hasParameter("VRADH"):
            if scan.elangle <= 30.0*math.pi/180.0:
              self.assertEqual("se.smhi.detector.dealias", scan.getParameter("VRADH").getAttribute("how/task"))
            else:
              self.assertFalse(scan.getParameter("VRADH").hasAttribute("how/task"))

        self.assertFalse(different(pvol.getScan(0), dscan))

    # Only checks the first scan in the volume.
    def test_create_dealiased_parameter(self):
        pvol = _raveio.open(self.FIXTURE).object
        dscan = _raveio.open(self.DEALIASED).object
        self.assertTrue(different(pvol.getScan(0), dscan))
        param = _dealias.create_dealiased_parameter(pvol.getScan(0), "VRADH", "VRADDH")
        self.assertFalse(different_param(dscan.getParameter("VRADH"), param))
        dscan.getParameter("VRADH").addAttribute("how/something", "jupp")
        # verify that created param is copy and not reference
        self.assertFalse(param.hasAttribute("how/something"))
        self.assertTrue(param.hasAttribute("how/dealiased"))
        self.assertTrue(param.quantity == "VRADDH")

    def testAlreadyDealiased(self):
        scan = _raveio.open(self.FIXTURE).object.getScan(0)
        status = _dealias.dealias(scan)
        self.assertTrue(status)
        status = _dealias.dealias(scan)
        self.assertFalse(status)

    def testAlreadyDealiased_VRADH(self):
        scan = _raveio.open(self.FIXTURE).object.getScan(0)
        scan.addParameter(copyParam(scan.getParameter("VRADH"), "VRADV"))
        self.assertTrue(_dealias.dealias(scan, "VRADV"))
        self.assertFalse(_dealias.dealias(scan, "VRADV"))
        self.assertTrue(_dealias.dealias(scan, "VRADH"))
        self.assertFalse(_dealias.dealias(scan, "VRADH"))
        self.assertFalse(_dealias.dealias(scan))

    def testWrongInput(self):
        vertical_profile = _raveio.open(self.BADINPUT).object
        try:
            status = _dealias.dealias(vertical_profile)
        except AttributeError:
            self.assertTrue(True)

def copyParam(param, newquantity):
    newparam = _polarscanparam.new()
    newparam.setData(param.getData())
    newparam.quantity=newquantity
    newparam.gain = param.gain
    newparam.offset = param.offset
    newparam.undetect = param.undetect
    newparam.nodata = param.nodata
    if param.hasAttribute("how/task"):
        newparam.addAttribute("how/task", param.getAttribute("how/task"))
    if param.hasAttribute("how/dealiased"):
        newparam.addAttribute("how/dealiased", param.getAttribute("how/dealiased"))
    return newparam

def different(scan1, scan2, quantity="VRADH"):
    a = scan1.getParameter(quantity).getData()
    b = scan2.getParameter(quantity).getData()
    c = a == b
    d = sum(where(equal(c, False), 1, 0).flat)
    if d > 0:
        return True
    else:
        return False 
    
def different_param(p1, p2):
    a = p1.getData()
    b = p2.getData()
    c = a == b
    d = sum(where(equal(c, False), 1, 0).flat)
    if d > 0:
        return True
    else:
        return False 
    