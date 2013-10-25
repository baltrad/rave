'''
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the pgf quality registry manager

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2013-10-25
'''
import unittest
import os
import math
import string
from rave_pgf_quality_registry_mgr import rave_pgf_quality_registry_mgr

class rave_pgf_quality_registry_mgr_test(unittest.TestCase):
  FIXTURE_1 = "fixtures/rave_pgf_quality_registry_mgr_1.xml"
  TEMPORARY_FILE = "rave_pgf_quality_tempfile.xml"
  
  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)
  
  def test_constructor_1(self):
    classUnderTest = rave_pgf_quality_registry_mgr(self.FIXTURE_1)
    expected = """<?xml version="1.0" encoding="UTF-8"?>
<rave-pgf-composite-quality-registry>
  <quality-plugin class="rave_overshooting_quality_plugin" module="rave_overshooting_quality_plugin" name="rave-overshooting" />
  <quality-plugin class="rave_distance_quality_plugin" module="rave_distance_quality_plugin" name="distance" />
</rave-pgf-composite-quality-registry>
"""
    self.assertEquals(expected, classUnderTest.tostring())    
  
  def test_constructor_2(self):
    classUnderTest = rave_pgf_quality_registry_mgr(self.FIXTURE_1, "ISO-8859-1")
    expected = """<?xml version="1.0" encoding="ISO-8859-1"?>
<rave-pgf-composite-quality-registry>
  <quality-plugin class="rave_overshooting_quality_plugin" module="rave_overshooting_quality_plugin" name="rave-overshooting" />
  <quality-plugin class="rave_distance_quality_plugin" module="rave_distance_quality_plugin" name="distance" />
</rave-pgf-composite-quality-registry>
"""
    self.assertEquals(expected, classUnderTest.tostring())    
    
  def test_add_plugin(self):
    classUnderTest = rave_pgf_quality_registry_mgr(self.FIXTURE_1)
    classUnderTest.add_plugin("nisse", "nisses_module", "nisses_plugin")
    expected = """<?xml version="1.0" encoding="UTF-8"?>
<rave-pgf-composite-quality-registry>
  <quality-plugin class="rave_overshooting_quality_plugin" module="rave_overshooting_quality_plugin" name="rave-overshooting" />
  <quality-plugin class="rave_distance_quality_plugin" module="rave_distance_quality_plugin" name="distance" />
  <quality-plugin class="nisses_plugin" module="nisses_module" name="nisse" />
</rave-pgf-composite-quality-registry>
"""
    self.assertEquals(expected, classUnderTest.tostring())    

  def test_remove_plugin(self):
    classUnderTest = rave_pgf_quality_registry_mgr(self.FIXTURE_1)
    classUnderTest.remove_plugin("rave-overshooting")
    expected = """<?xml version="1.0" encoding="UTF-8"?>
<rave-pgf-composite-quality-registry>
  <quality-plugin class="rave_distance_quality_plugin" module="rave_distance_quality_plugin" name="distance" />
</rave-pgf-composite-quality-registry>
"""
    self.assertEquals(expected, classUnderTest.tostring())    

  def test_has_plugin(self):
    classUnderTest = rave_pgf_quality_registry_mgr(self.FIXTURE_1)

    self.assertEquals(True, classUnderTest.has_plugin("distance"))
    self.assertEquals(False, classUnderTest.has_plugin("nisses_plugin"))

  def test_save_1(self):
    classUnderTest = rave_pgf_quality_registry_mgr(self.FIXTURE_1)
    classUnderTest.save(self.TEMPORARY_FILE)
    expected = """<?xml version="1.0" encoding="UTF-8"?>
<rave-pgf-composite-quality-registry>
  <quality-plugin class="rave_overshooting_quality_plugin" module="rave_overshooting_quality_plugin" name="rave-overshooting" />
  <quality-plugin class="rave_distance_quality_plugin" module="rave_distance_quality_plugin" name="distance" />
</rave-pgf-composite-quality-registry>
"""
    self.assertEquals(expected, open(self.TEMPORARY_FILE).read())

  def test_save_2(self):
    classUnderTest = rave_pgf_quality_registry_mgr(self.FIXTURE_1)
    classUnderTest.add_plugin("nisse", "nisses_module", "nisses_plugin")
    classUnderTest.save(self.TEMPORARY_FILE)
    expected = """<?xml version="1.0" encoding="UTF-8"?>
<rave-pgf-composite-quality-registry>
  <quality-plugin class="rave_overshooting_quality_plugin" module="rave_overshooting_quality_plugin" name="rave-overshooting" />
  <quality-plugin class="rave_distance_quality_plugin" module="rave_distance_quality_plugin" name="distance" />
  <quality-plugin class="nisses_plugin" module="nisses_module" name="nisse" />
</rave-pgf-composite-quality-registry>
"""
    self.assertEquals(expected, open(self.TEMPORARY_FILE).read())
    
  def test_save_3(self):
    classUnderTest = rave_pgf_quality_registry_mgr(self.FIXTURE_1)
    classUnderTest.remove_plugin("rave-overshooting")
    classUnderTest.save(self.TEMPORARY_FILE)
    expected = """<?xml version="1.0" encoding="UTF-8"?>
<rave-pgf-composite-quality-registry>
  <quality-plugin class="rave_distance_quality_plugin" module="rave_distance_quality_plugin" name="distance" />
</rave-pgf-composite-quality-registry>
"""
    self.assertEquals(expected, open(self.TEMPORARY_FILE).read())
    