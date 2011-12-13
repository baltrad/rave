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

Test suite for rave

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-10-12
'''
import unittest
from RaveTest import *
from PyPolarVolumeTest import *
from PyPolarScanTest import *
from PyPolarScanParamTest import *
from RaveModuleConstantsTest import *
from PyCartesianTest import *
from PyCartesianVolumeTest import *
from PyRaveFieldTest import *
from PyTransformTest import *
from PyProjectionTest import *
from PyRaveIOTest import *
from PyPolarNavTest import *
from PyAreaTest import *
from PyRadarDefinitionTest import *
from PyProjectionRegistryTest import *
from PyAreaRegistryTest import *
from PgfVolumePluginTest import *
from RaveScansun import *
from PyDetectionRangeTest import *
from PyRaveTest import *
from PyPooCompositeAlgorithmTest import *
from rave_pgf_volume_plugin_test import *

if __name__ == '__main__':
  unittest.main()
