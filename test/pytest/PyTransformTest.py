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

Tests the transform module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-10-20
'''
import unittest
import os
import _rave
import _transform
import _radardef
import _raveio
import string
import numpy

class PyTransformTest(unittest.TestCase):
  FIXTURE_CARTESIAN_PCAPPI = "fixture_cartesian_pcappi.h5"
  FIXTURE_CARTESIAN_PPI = "fixture_cartesian_ppi.h5"
  FIXTURE_VOLUME = "fixture_ODIM_H5_pvol_ang_20090501T1200Z.h5"
  TRANSFORM_FILLGAP_FILENAME = "transform_filledGap.h5"
  
  def setUp(self):
    if os.path.isfile(self.TRANSFORM_FILLGAP_FILENAME):
      os.unlink(self.TRANSFORM_FILLGAP_FILENAME)

  def tearDown(self):
    if os.path.isfile(self.TRANSFORM_FILLGAP_FILENAME):
      os.unlink(self.TRANSFORM_FILLGAP_FILENAME)

  def test_new(self):
    obj = _transform.new()
    
    istransform = string.find(`type(obj)`, "TransformCore")
    self.assertNotEqual(-1, istransform) 

  def test_attribute_visibility(self):
    attrs = ['method']
    obj = _transform.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEquals(True, a in alist)

  def testMethod(self):
    obj = _transform.new()
    self.assertEqual(_rave.NEAREST, obj.method)
    obj.method = _rave.CUBIC
    self.assertEqual(_rave.CUBIC, obj.method)
    
  def testValidMethods(self):
    obj = _transform.new()
    meths = [_rave.NEAREST, _rave.BILINEAR, _rave.CUBIC, _rave.CRESSMAN, _rave.UNIFORM, _rave.INVERSE]
    for method in meths:
      obj.method = method
      self.assertEqual(method, obj.method)
  
  def testInvalidMethods(self):
    obj = _transform.new()
    meths = [99, 33, 22, 11]
    for method in meths:
      try:
        obj.method = method
        self.fail("Expected ValueError")
      except ValueError, e:
        pass
      self.assertEqual(_rave.NEAREST, obj.method)

  def test_ctoscan(self):
    obj = _transform.new()
    cartesian = _raveio.open(self.FIXTURE_CARTESIAN_PPI).object
    volume = _raveio.open(self.FIXTURE_VOLUME).object
    radardef = _radardef.new()
    radardef.id = volume.source
    radardef.description = "ctop test"
    radardef.longitude = volume.longitude
    radardef.latitude = volume.latitude
    radardef.height = volume.height
    
    elangles = []
    nangles = volume.getNumberOfScans()
    for i in range(nangles):
      scan = volume.getScan(i)
      elangles.append(scan.elangle)
    radardef.elangles = elangles
    scan = volume.getScan(0)
    radardef.nrays = scan.nrays
    radardef.nbins = scan.nbins
    radardef.scale = scan.rscale
    radardef.beamwidth = scan.beamwidth
    elangle = scan.elangle
    result = obj.ctoscan(cartesian, radardef, elangle, "DBZH")
    rio = _raveio.new()
    rio.object = result
    rio.save("ctop_polarscan.h5")

  def test_ctop(self):
    obj = _transform.new()
    cartesian = _raveio.open(self.FIXTURE_CARTESIAN_PCAPPI).object
    volume = _raveio.open(self.FIXTURE_VOLUME).object
    radardef = _radardef.new()
    radardef.id = volume.source
    radardef.description = "ctop test"
    radardef.longitude = volume.longitude
    radardef.latitude = volume.latitude
    radardef.height = volume.height
    
    elangles = []
    nangles = volume.getNumberOfScans()
    for i in range(nangles):
      scan = volume.getScan(i)
      elangles.append(scan.elangle)
    radardef.elangles = elangles
    scan = volume.getScan(0)
    radardef.nrays = scan.nrays
    radardef.nbins = scan.nbins
    radardef.scale = scan.rscale
    radardef.beamwidth = scan.beamwidth
    
    result = obj.ctop(cartesian, radardef, "DBZH")
    rio = _raveio.new()
    rio.object = result
    rio.save("ctop_polarvolume.h5")
    
  def test_fillGap(self):
    obj = _transform.new()
    io = _raveio.new()
    
    cartesian = _raveio.open(self.FIXTURE_CARTESIAN_PCAPPI).object
    io.object = obj.fillGap(cartesian)
    io.filename=self.TRANSFORM_FILLGAP_FILENAME
    io.save()
    
