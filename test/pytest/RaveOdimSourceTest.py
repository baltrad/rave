# coding=utf-8
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
import _raveio
import odim_source

import sys as system

use_unicode_variant=True
if system.version_info < (3,):
    use_unicode_variant=False

SESOURCES=[("SE40", "02032", "0-20000-0-2032", "sekrn"),
           ("SE41",	"02092", "0-20000-0-2092", "sella"),
           ("SE42",	"02200", "0-20000-0-2200", "seosd"),
           ("SE43",	"02262", "0-20000-0-2262", "seoer"),
           ("SE44", "02334", "0-20000-0-2334", "sehuv"),
           ("SE45", "02430", "0-20000-0-2430", "selek"),
           ("SE52", None,    "0-21010-0-724",  "sebaa"),
           ("SE47",	"02588", "0-20000-0-2588", "sehem"),
           ("SE48", "02570", "0-20000-0-2570", "seatv"),
           ("SE49",	"02600", "0-20000-0-2600", "sevax"),
           ("SE50",	"02606", "0-20000-0-2606", "seang"),
           ("SE51",	"02666", "0-20000-0-2666", "sekaa")]


class TestSourceObject:
    def __init__(self, src):
        self.source = src

class RaveOdimSourceTest(unittest.TestCase):
    FIXTURE = "fixtures/Z_SCAN_C_ESWI_20101023180200_selul_000000.h5"
    MYNOD = 'fipet'
    regular_string = b'NOD:fipet,WMO:02775,RAD:FI51,PLC:Pet\xc3\xa4j\xc3\xa4vesi'
  
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testUnicode(self):
        s = odim_source.SOURCE[self.MYNOD]
        self.assertTrue(type(s), str)
        self.assertEqual(s.encode('UTF-8'), self.regular_string)

    def testODIM_Source(self):
        s = odim_source.SOURCE[self.MYNOD]
        S = odim_source.ODIM_Source(s.encode("UTF-8"))
        self.assertEqual(S.nod, self.MYNOD)
        self.assertEqual(S.rad, 'FI51')
        self.assertEqual(S.plc, u'Pet\xe4j\xe4vesi')
        self.assertEqual(S.source, s.encode("UTF-8"))        
        self.assertEqual(S.wmo, '02775')

    def testNODfromSource(self):
        rio = _raveio.open(self.FIXTURE)
        n = odim_source.NODfromSource(rio.object)
        self.assertEqual(n, 'sella')

    def testNODfromSource_RadWmoWigos(self):
        n = odim_source.NODfromSource(TestSourceObject("RAD:SE40,NOD:sekrn"))
        self.assertEqual(n, 'sekrn')
        n = odim_source.NODfromSource(TestSourceObject("RAD:SE40,WMO:02032"))
        self.assertEqual(n, 'sekrn')
        n = odim_source.NODfromSource(TestSourceObject("RAD:SE40,WIGOS:0-20000-0-2032"))
        self.assertEqual(n, 'sekrn')

    def testNodFromSourceWigosAndWmo(self):
        for src in SESOURCES:
            if src[1] != None:
                n = odim_source.NODfromSource(TestSourceObject("WMO:%s"%src[1]))
                self.assertEqual(n, src[3])

            if src[2] != None:
                n = odim_source.NODfromSource(TestSourceObject("WIGOS:%s"%src[2]))
                self.assertEqual(n, src[3])

    def testNODfromSource_OnlyRad(self):
        n = odim_source.NODfromSource(TestSourceObject("RAD:SE40"))
        self.assertEqual(n, 'n/a')

    def testCheckSource(self):
        if use_unicode_variant:
            variants = [b'WMO:02092,NOD:sella,RAD:SE41,PLC:Lule\xc3\xa5'.decode('utf-8'),
                        b'WMO:02092,CMT:searl,RAD:SE49,PLC:Luleaa'.decode('utf-8'),
                        b'WMO:02092'.decode('utf-8')]
        else:
            variants = ['WMO:02092,NOD:sella,RAD:SE41,PLC:Lule\xc3\xa5',
                        'WMO:02092,CMT:searl,RAD:SE49,PLC:Luleaa',
                        'WMO:02092']

        rio = _raveio.open(self.FIXTURE)
        for v in variants:
            rio.object.source = v
            odim_source.CheckSource(rio.object)
            split = rio.object.source.split(',')
            if use_unicode_variant:
                [x.encode('utf-8').decode() for x in split]
            self.assertTrue(len(split) >= 4)
            for s in split:
                if use_unicode_variant:
                    s=s.encode('utf-8').decode()
                self.assertTrue(s in "WMO:02092,NOD:sella,RAD:SE41,PLC:Luleå,WIGOS:0-20000-0-2092")

    def testCheckSourceRadWigosAndWmo(self):
        for src in SESOURCES:
            if src[0] != None:
                tso = TestSourceObject("RAD:%s"%src[0])
                odim_source.CheckSource(tso)
                self.assertEqual(tso.source, odim_source.SOURCE[src[3]])

            if src[1] != None:
                tso = TestSourceObject("WMO:%s"%src[1])
                odim_source.CheckSource(tso)
                self.assertEqual(tso.source, odim_source.SOURCE[src[3]])

            if src[2] != None:
                tso = TestSourceObject("WIGOS:%s"%src[2])
                odim_source.CheckSource(tso)
                self.assertEqual(tso.source, odim_source.SOURCE[src[3]])


if __name__ == "__main__":
    unittest.main()
