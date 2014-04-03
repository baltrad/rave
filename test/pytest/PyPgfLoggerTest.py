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

Tests PGF Queue functionality

@file
@author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2014-03-24
'''
import unittest
import os, time
import rave_defines
import rave_pgf_logger

class PyPgfLoggerTest(unittest.TestCase):
    LOGGER = os.path.join(os.getcwd(), "../../bin/rave_pgf_logger -a")
    LFILE = os.path.join(os.getcwd(), "fixtures/rave_pgf.log")
    PFILE = os.path.join(os.getcwd(), "fixtures/rave_pgf_logger.pid")
    PORT = 8088
    MSG = "Hej hopp i lingonskogen!"
    classUnderTest = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testLoggerA(self):
        os.system("%s start -P %s -p %i -L %s> /dev/null 2>&1" % (self.LOGGER, self.PFILE, 
                                                                  self.PORT, self.LFILE))
        client = rave_pgf_logger.rave_pgf_logger_client(port=self.PORT)
        client.info(self.MSG)
        time.sleep(2.0)

    def testLoggerB(self):
        fd = open(self.LFILE)
        c = fd.read()
        fd.close()
        msg = c.split("     ")[-1]
        self.assertEquals(msg[:-1], self.MSG)
        os.system("%s stop -P %s> /dev/null 2>&1" % (self.LOGGER, self.PFILE))
        os.remove(self.LFILE)
