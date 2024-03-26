'''
Copyright (C) 2019 The Crown (i.e. Her Majesty the Queen in Right of Canada)

This file is an add-on to RAVE.

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

dr_qc unit tests

@file
@author Daniel Michelson, Environment and Climate Change Cananda
@date 2019-04-11
'''
import os, unittest
import _rave
import _raveio
import _dr_qc
import ec_drqc
from numpy import *


class dr_qcTest(unittest.TestCase):
    FIXTURE = '../WKR_201209090100_POLPPI.h5'
    DR_DERIVE_PARAMETER_FIXTURE = '../WKR_201209090100_POLPPI_DR.h5'
    DR_SPECKLE_FILTER_FIXTURE = '../WKR_201209090100_POLPPI_DRSF.h5'

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_drCalculate(self):
        ZDR, RHOHV, zdr_offset = 0.0, 0.9, 1.0
        reference = -12.52
        result = _dr_qc.drCalculate(ZDR, RHOHV, zdr_offset)
        self.assertAlmostEqual(result, reference, 2)

    # Should fail because we remove RHOHV which is required
    def test_drDeriveParameter_noParam(self):
        zdr_offset = 0.0
        rio = _raveio.open(self.FIXTURE)
        scan = rio.object
        scan.removeParameter("RHOHV")
        try:
            _dr_qc.drDeriveParameter(scan, zdr_offset)
        except AttributeError:
            pass

    def test_drDeriveParameter(self):
        zdr_offset = 0.0
        rio = _raveio.open(self.FIXTURE)
        scan = rio.object
        _dr_qc.drDeriveParameter(scan, zdr_offset)
        #rio.save(self.DR_DERIVE_PARAMETER_FIXTURE)
        refio = _raveio.open(self.DR_DERIVE_PARAMETER_FIXTURE)
        ref_scan = refio.object
        status = different(scan, ref_scan, "DR")
        self.assertFalse(status)

    # Should fail because it doesn't contain the DR parameter
    def test_drSpeckleFilter_noParam(self):
        zdr_offset = 0.0
        param_name, param_thresh, dr_thresh = "DBZH", 35.0, -12.0
        kernely = kernelx = 3
        rio = _raveio.open(self.FIXTURE)
        scan = rio.object
        try:
            _dr_qc.drSpeckleFilter(scan, param_name, kernely, kernelx, 
                                   param_thresh, dr_thresh)
        except AttributeError:
            pass

    def test_drSpeckleFilter(self):
        zdr_offset = 0.0
        param_name, param_thresh, dr_thresh = "DBZH", 35.0, -12.0
        kernely = kernelx = 3
        rio = _raveio.open(self.DR_DERIVE_PARAMETER_FIXTURE)
        scan = rio.object
        _dr_qc.drSpeckleFilter(scan, param_name, kernely, kernelx,
                               param_thresh, dr_thresh)
        #rio.save(self.DR_SPECKLE_FILTER_FIXTURE)
        refio = _raveio.open(self.DR_SPECKLE_FILTER_FIXTURE)
        ref_scan = refio.object
        status = different(scan, ref_scan, param_name)
        self.assertFalse(status)

    def test_drQC(self):
        rio = _raveio.open(self.FIXTURE)
        scan = rio.object
        ec_drqc.drQC(scan, kernely=3, kernelx=3)
        refio = _raveio.open(self.DR_SPECKLE_FILTER_FIXTURE)
        ref_scan = refio.object
        status = different(scan, ref_scan, "DBZH")
        self.assertFalse(status)
        param = scan.getParameter("DR")
        self.assertEqual(ec_drqc.TASK, param.getAttribute("how/task"))
        self.assertEqual("param_name=DBZH zdr_offset=0.00 kernely=3 kernelx=3 param_thresh=35.0 dr_thresh=-12.0", param.getAttribute("how/task_args"))


# Helper function to determine whether two parameter arrays differ
def different(scan1, scan2, param="DBZH"):
    a = scan1.getParameter(param).getData()
    b = scan2.getParameter(param).getData()
    c = a == b
    d = sum(where(equal(c, False), 1, 0).flat)
    if d > 0:
        return True
    else:
        return False 
