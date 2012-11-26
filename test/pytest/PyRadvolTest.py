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

Tests RADVOL-QC functionality

@file
@author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2012-11-23
'''
import unittest
import os
import _raveio
import _radvol
from numpy import *

class PyRadvolTest(unittest.TestCase):
    FIXATT, FIXATTC = "fixtures/fake_att.h5", "fixtures/fake_att_cor.h5"
    FIXBROAD, FIXBROADC = "fixtures/fake_broad.h5", "fixtures/fake_broad_cor.h5"
    FIXSPECK, FIXSPECKC = "fixtures/fake_speck.h5", "fixtures/fake_speck_cor.h5"
    FIXSPIKE, FIXSPIKEC = "fixtures/fake_spike.h5", "fixtures/fake_spike_cor.h5"
    BADINPUT = "fixtures/vp_fixture.h5"
    classUnderTest = None
  
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testRadvolAttCorrection(self):
        pvol = _raveio.open(self.FIXATT).object
        status = _radvol.attCorrection(pvol)
        self.assertTrue(status)
        ref = _raveio.open(self.FIXATTC).object
        myscan = pvol.getScan(0)
        refscan = ref.getScan(0)
        qf, task_args = None, None
        try:
            qf = myscan.getQualityFieldByHowTask("pl.imgw.radvolqc.att")
            task_args =  qf.getAttribute("how/task_args")
        except:
            pass        
        self.assertNotEquals(qf, None)
        self.assertEquals(task_args, "ATT: ATT_QI1=1.0, ATT_QI0=5.0, ATT_QIUn=0.9, ATT_a= 0.0044, ATT_b= 1.17, ATT_MPa=200.0, ATT_MPb= 1.6, ATT_Refl= 4.0, ATT_Last= 1.0, ATT_Sum= 5.0")
        self.assertFalse(different(myscan, refscan))

    def testRadvolBroadAssessment(self):
        pvol = _raveio.open(self.FIXBROAD).object
        status = _radvol.broadAssessment(pvol)
        self.assertTrue(status)
        ref = _raveio.open(self.FIXBROADC).object
        myscan = pvol.getScan(0)
        refscan = ref.getScan(0)
        qf, task_args = None, None
        try:
            qf = myscan.getQualityFieldByHowTask("pl.imgw.radvolqc.broad")
            task_args =  qf.getAttribute("how/task_args")
        except:
            pass        
        self.assertNotEquals(qf, None)
        self.assertEquals(task_args, "BROAD: BROAD_LhQI1=1.1, BROAD_LhQI0=2.5, BROAD_LvQI1=1.5, BROAD_LvQI0=3.2, BROAD_Pulse=0.15")
        self.assertFalse(different(myscan, refscan))

    def testRadvolSpeckRemoval(self):
        pvol = _raveio.open(self.FIXSPECK).object
        status = _radvol.speckRemoval(pvol)
        self.assertTrue(status)
        ref = _raveio.open(self.FIXSPECKC).object
        myscan = pvol.getScan(0)
        refscan = ref.getScan(0)
        qf, task_args = None, None
        try:
            qf = myscan.getQualityFieldByHowTask("pl.imgw.radvolqc.speck")
            task_args =  qf.getAttribute("how/task_args")
        except:
            pass        
        self.assertNotEquals(qf, None)
        self.assertEquals(task_args, "SPECK: SPECK_QI=0.9, SPECK_QIUn=0.5, SPECK_AGrid=1, SPECK_ANum=2, SPECK_AStep=1, SPECK_BGrid=1, SPECK_BNum=2, SPECK_BStep=2")
        self.assertFalse(different(myscan, refscan))

    def testRadvolSpikeRemoval(self):
        pvol = _raveio.open(self.FIXSPIKE).object
        status = _radvol.spikeRemoval(pvol)
        self.assertTrue(status)
        ref = _raveio.open(self.FIXSPIKEC).object
        myscan = pvol.getScan(0)
        refscan = ref.getScan(0)
        qf, task_args = None, None
        try:
            qf = myscan.getQualityFieldByHowTask("pl.imgw.radvolqc.spike")
            task_args =  qf.getAttribute("how/task_args")
        except:
            pass        
        self.assertNotEquals(qf, None)
        self.assertEquals(task_args, "SPIKE: SPIKE_QI=0.5, SPIKE_QIUn=0.3, SPIKE_ACovFrac=0.9, SPIKE_AAzim=3, SPIKE_AVarAzim=  1000.0, SPIKE_ABeam=15, SPIKE_AVarBeam=5.0, SPIKE_AFrac=0.45, SPIKE_BDiff=10.0, SPIKE_BAzim=3, SPIKE_BFrac=0.25, SPIKE_CAlt=20.0")
        self.assertFalse(different(myscan, refscan))

    def testWrongAttInput(self):
        vertical_profile = _raveio.open(self.BADINPUT).object
        try:
            status = _radvol.attCorrection(vertical_profile)
        except AttributeError:
            self.assertTrue(True)

    
def different(scan1, scan2):
    a = scan1.getParameter("DBZH").getData()
    b = scan2.getParameter("DBZH").getData()
    c = a == b
    d = sum(where(equal(c, False), 1, 0).flat)
    if d > 0:
        return True
    else:
        return False 