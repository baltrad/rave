'''
Copyright (C) 2009-2010 Swedish Meteorological and Hydrological Institute, SMHI,

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

Test suite for rave

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2010-11-24
'''
import unittest

import _rave
#_rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
#A="""
from RaveTestSuite import *
from PyCompositeTest import *
from PyCartesianCompositeTest import *
#from PolarVolumeTransformTest import *
from MeanTest import *
#"""
#from PyRaveIOTest import *
#from RaveScansun import *

if __name__ == '__main__':
  unittest.main()
