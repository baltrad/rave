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
import unittest, os
A="""
from RaveTest import *
"""
from PyPolarVolumeTest import *
from PyPolarScanTest import *
from PyPolarScanParamTest import *
from RaveModuleConstantsTest import *

from PyCartesianTest import *
from PyCartesianParamTest import *
from PyCartesianVolumeTest import *
from PyVerticalProfileTest import *
from PyRaveFieldTest import *
A="""
from PyTransformTest import *

from PyVprCorrectionTest import *
"""
from PyProjectionTest import *
from PyRaveIOTest import *
from PyPolarNavTest import *
from PyAreaTest import *
from PyRadarDefinitionTest import *
from PyProjectionRegistryTest import *
A="""
from PyAcrrTest import *
from PyGraTest import *
from PyAreaRegistryTest import *
from PgfVolumePluginTest import *
"""
from RaveScansun import *
from PyDetectionRangeTest import *
A="""
from PyRaveTest import *
from PyQITotalTest import *
from PyPooCompositeAlgorithmTest import *
from rave_overshooting_quality_plugin_test import *
from rave_distance_quality_plugin_test import *
from rave_dealias_quality_plugin_test import *
from rave_radvol_quality_plugin_test import *
from qitotal_options_test import *
from rave_qitotal_quality_plugin_test import *
from rave_pgf_quality_registry_mgr_test import *
from rave_quality_chain_registry_test import *
from odc_hac_test import *
"""
from rave_hexquant_test import *
A="""
from polar_merger_test import *

from algorithm_runner_test import *

try:
  import mock
  if int(mock.__version__[0]) >= 1:
    from rave_quality_chain_plugin_test import *
    from compositing_test import *
    from rave_pgf_volume_plugin_test import *
    from rave_pgf_gra_plugin_test import *
except:
  pass

from area_registry_test import *

from PyDealiasTest import *

from PyRadvolTest import *
from PyCtfilterTest import *
from PyPgfQtoolsTest import *
from PyPgfLoggerTest import *
from rave_util_test import *
from RaveOdimSourceTest import *
from PyBitmapGeneratorTest import *

#
# Unless RAVE_TESTDB_URI has been set we don't want to run the dom db tests
#
if os.environ.get("RAVE_TESTDB_URI", "") != "":
  from rave_dom_db_test import *

from rave_wmo_flatfile_test import *
from rave_fm12_test import *
"""

# Gra adjustment tests requires mock version 1.0.1 or higher.
try:
  import mock
  if int(mock.__version__[0]) >= 1:
    from gadjust_obsmatcher_test import *
    from gadjust_gra_test import *
except:
  pass

try:
  import pygrib
  from grib_reader_test import *
except:
  pass

if __name__ == '__main__':
  unittest.main()
