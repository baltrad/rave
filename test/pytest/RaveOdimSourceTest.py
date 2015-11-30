'''
Copyright (C) 2015 The Crown (i.e. Her Majesty the Queen in Right of Canada)

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

Tests functionality for managing ODIM /what/source identifiers.

@file
@author Daniel Michelson, Environment Canada
@date 2015-11-02
'''
import unittest
import os, types
import _raveio
import odim_source
from numpy import *

class RaveOdimSourceTest(unittest.TestCase):
    FIXTURE = "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000000.h5"
    MYNOD = 'fipet'
    regular_string = 'NOD:fipet,WMO:02775,RAD:FI51,PLC:Pet\xc3\xa4j\xc3\xa4vesi'
  
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testUnicode(self):
        s = odim_source.SOURCE[self.MYNOD]
        self.assertTrue(type(s), types.UnicodeType)
        self.assertEquals(s.encode('UTF-8'), self.regular_string)

    def testODIM_Source(self):
        s = odim_source.SOURCE[self.MYNOD]
        S = odim_source.ODIM_Source(s.encode("UTF-8"))
        self.assertEquals(S.nod, self.MYNOD)
        self.assertEquals(S.rad, 'FI51')
        self.assertEquals(S.plc, u'Pet\xe4j\xe4vesi')
        self.assertEquals(S.source, s.encode("UTF-8"))        
        self.assertEquals(S.wmo, '02775')

    def testNODfromSource(self):
        rio = _raveio.open(self.FIXTURE)
        n = odim_source.NODfromSource(rio.object)
        self.assertEquals(n, 'selul')

    def testCheckSource(self):
        variants = ["WMO:02092,NOD:selul,RAD:SE41,PLC:Lule\xc3\xa5",
                    "WMO:02092,CMT:searl,RAD:SE49,PLC:Luleaa",
                    "WMO:02092"]
        rio = _raveio.open(self.FIXTURE)
        for v in variants:
            rio.object.source = v
            odim_source.CheckSource(rio.object)
            split = rio.object.source.split(",")
            self.assertEquals(len(split), 4)
            for s in split:
                self.assertTrue(s in "WMO:02092,NOD:selul,RAD:SE41,PLC:Lule\xc3\xa5")


if __name__ == "__main__":
    unittest.main()
