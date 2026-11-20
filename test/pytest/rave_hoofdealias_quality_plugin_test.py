'''
Copyright (C) 2026 Swedish Meteorological and Hydrological Institute, SMHI,

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


Tests the HOOF dealias quality plugin

@file
@author Peter Smerkol, Vito Švagelj, ARSO
@date 2026-11-13
'''
import unittest
import numpy
import rave_hoofdealias_quality_plugin
import _raveio

class rave_hoofdealias_quality_plugin_test(unittest.TestCase):
  VOLUME_FIXTURE = "fixtures/sehem_pvol_pn215_20171204T071500Z_0x81540b.h5"
  classUnderTest = None

  def setUp(self):
    self.classUnderTest = rave_hoofdealias_quality_plugin.hoofdealias_plugin()

  def tearDown(self):
    self.classUnderTest = None

  def test_getQualityFields(self):
    result = self.classUnderTest.getQualityFields()
    self.assertEqual(1, len(result))
    self.assertEqual("si.arso.detector.hoofdealias", result[0])

  def test_process(self):
    pvol = _raveio.open(self.VOLUME_FIXTURE).object
    before = [pvol.getScan(i).getParameter("VRADH").getData().copy() for i in range(pvol.getNumberOfScans())]

    result, qfield = self.classUnderTest.process(pvol)

    self.assertEqual(["si.arso.detector.hoofdealias"], qfield)
    for i in range(result.getNumberOfScans()):
      scan = result.getScan(i)
      self.assertTrue(scan.hasParameter("VRADH"))
      self.assertTrue(scan.hasParameter("VRADDH"))
      self.assertFalse(scan.getParameter("VRADH").hasAttribute("how/dealiased"))
      self.assertTrue(numpy.array_equal(before[i], scan.getParameter("VRADH").getData()))
      self.assertEqual("True", scan.getParameter("VRADDH").getAttribute("how/dealiased"))

  def test_process_scan(self):
    scan = _raveio.open(self.VOLUME_FIXTURE).object.getScan(0)

    result, qfield = self.classUnderTest.process(scan)

    self.assertTrue(result.hasParameter("VRADH"))
    self.assertFalse(result.hasParameter("VRADDH"))

if __name__ == "__main__":
  unittest.main()
