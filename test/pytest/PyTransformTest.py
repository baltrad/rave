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
import _area,_projection
import _transform
import _radardef
import _raveio
import _cartesian
import _cartesianparam
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
  
  ##
  # A number of static tests to verify that the gap filling is working as expected
  #
  #
  # 0   1   2   3   4   5
  # 1       X
  # 2   X   ?   X
  # 3       X
  # 4
  # 5
  def testFillGap_onParameter(self):
    data = numpy.zeros((6, 6), numpy.uint8)
    data[1][2] = 1
    data[2][1] = 1
    data[3][2] = 1
    data[2][3] = 1
     
    param = _cartesianparam.new()
    param.setData(data)
    param.nodata = 255.0
    t = _transform.new()
    result = t.fillGap(param)
    
    data = result.getData() 
    self.assertEquals(1, data[2][2])

  def testFillGap_onCartesianParameters(self):
    data = numpy.zeros((6, 6), numpy.uint8)
    data[1][2] = 1
    data[2][1] = 1
    data[3][2] = 1
    data[2][3] = 1
    
    obj = _cartesian.new()
    
    param = _cartesianparam.new()
    param.setData(data)
    param.nodata = 255.0
    param.quantity = "DBZH"
    obj.addParameter(param)

    param = _cartesianparam.new()
    data[1][2] = 2
    data[2][1] = 2
    data[3][2] = 2
    data[2][3] = 2    
    param.setData(data)
    param.nodata = 255.0
    param.quantity = "TH"
    obj.addParameter(param)
    
    t = _transform.new()
    result = t.fillGap(obj)
    
    data = result.getParameter("DBZH").getData() 
    self.assertEquals(1, data[2][2])
    data = result.getParameter("TH").getData() 
    self.assertEquals(2, data[2][2])
  
  def create_cartesian_with_parameter(self, xsize, ysize, xscale, yscale, extent, projstr, dtype, value, quantity):
    obj = _cartesian.new()
    a = _area.new()
    a.xsize = xsize
    a.ysize = ysize
    a.xscale = xscale
    a.yscale = yscale
    a.extent = extent
    a.projection = _projection.new("x", "y", projstr)

    obj.init(a)    
    
    for i in range(len(quantity)):
      data = numpy.zeros((ysize,xsize), dtype)
      data = data + value[i]
      param = _cartesianparam.new()
      param.setData(data)
      param.nodata = 255.0
      param.quantity = quantity[i]
      obj.addParameter(param)
    
    return obj
  
  def test_combine_tiles(self):
    pyarea = _area.new()
    pyarea.extent = (971337.728807, 7196461.17902, 3015337.72881, 11028461.179)
    pyarea.xscale = 511000.0
    pyarea.yscale = 958000.0
    pyarea.xsize = 4
    pyarea.ysize = 4
    pyarea.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +R=6378137.0 +nadgrids=@null +no_defs")

    ul = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (971337.728807,9112461.1790100001,1993337.7288084999,11028461.179),
                                              pyarea.projection.definition,
                                              numpy.uint8, [2], ["DBZH"])
    
    ur = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (1993337.7288084999,9112461.1790100001,3015337.72881,11028461.179),
                                              pyarea.projection.definition,
                                              numpy.uint8, [3], ["DBZH"])
    
    ll = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (971337.728807,7196461.17902,1993337.7288084999,9112461.1790100001),
                                              pyarea.projection.definition,
                                              numpy.uint8, [4], ["DBZH"])
    
    lr = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (1993337.7288084999,7196461.17902,3015337.72881,9112461.1790100001),
                                              pyarea.projection.definition,
                                              numpy.uint8, [5], ["DBZH"])
    
    t = _transform.new()
    result = t.combine_tiles(pyarea, [ul,ur,ll,lr])
    param = result.getParameter("DBZH")
    self.assertEquals(4, result.xsize)
    self.assertEquals(4, result.ysize)
    self.assertEquals(511000.0, result.xscale, 4)
    self.assertEquals(958000.0, result.yscale, 4)
    self.assertEquals(971337.728807, result.areaextent[0], 4)
    self.assertEquals(7196461.17902, result.areaextent[1], 4)
    self.assertEquals(3015337.72881, result.areaextent[2], 4)
    self.assertEquals(11028461.179, result.areaextent[3], 4)
    self.assertEquals("+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +R=6378137.0 +nadgrids=@null +no_defs", result.projection.definition)
    self.assertEquals(ul.date, result.date)
    self.assertEquals(ul.time, result.time)
    self.assertEquals(ul.startdate, result.startdate)
    self.assertEquals(ul.starttime, result.starttime)
    self.assertEquals(ul.enddate, result.enddate)
    self.assertEquals(ul.endtime, result.endtime)
    self.assertEquals(ul.product, result.product)
    self.assertEquals(ul.objectType, result.objectType)
    
    self.assertEquals([[2,2,3,3],[2,2,3,3],[4,4,5,5],[4,4,5,5]], param.getData().tolist())
    
  def test_combine_tiles_with_two_parameters(self):
    pyarea = _area.new()
    pyarea.extent = (971337.728807, 7196461.17902, 3015337.72881, 11028461.179)
    pyarea.xscale = 511000.0
    pyarea.yscale = 958000.0
    pyarea.xsize = 4
    pyarea.ysize = 4
    pyarea.projection = _projection.new("x", "y", "+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +R=6378137.0 +nadgrids=@null +no_defs")

    ul = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (971337.728807,9112461.1790100001,1993337.7288084999,11028461.179),
                                              pyarea.projection.definition,
                                              numpy.uint8, [2,12], ["DBZH", "TH"])
    
    ur = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (1993337.7288084999,9112461.1790100001,3015337.72881,11028461.179),
                                              pyarea.projection.definition,
                                              numpy.uint8, [3,13], ["DBZH", "TH"])
    
    ll = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (971337.728807,7196461.17902,1993337.7288084999,9112461.1790100001),
                                              pyarea.projection.definition,
                                              numpy.uint8, [4,14], ["DBZH", "TH"])
    
    lr = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (1993337.7288084999,7196461.17902,3015337.72881,9112461.1790100001),
                                              pyarea.projection.definition,
                                              numpy.uint8, [5,15], ["DBZH", "TH"])
    
    t = _transform.new()
    result = t.combine_tiles(pyarea, [ul,ur,ll,lr])
    self.assertEquals(4, result.xsize)
    self.assertEquals(4, result.ysize)
    self.assertEquals(511000.0, result.xscale, 4)
    self.assertEquals(958000.0, result.yscale, 4)
    self.assertEquals(971337.728807, result.areaextent[0], 4)
    self.assertEquals(7196461.17902, result.areaextent[1], 4)
    self.assertEquals(3015337.72881, result.areaextent[2], 4)
    self.assertEquals(11028461.179, result.areaextent[3], 4)
    self.assertEquals("+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +R=6378137.0 +nadgrids=@null +no_defs", result.projection.definition)
    self.assertEquals(ul.date, result.date)
    self.assertEquals(ul.time, result.time)
    self.assertEquals(ul.startdate, result.startdate)
    self.assertEquals(ul.starttime, result.starttime)
    self.assertEquals(ul.enddate, result.enddate)
    self.assertEquals(ul.endtime, result.endtime)
    self.assertEquals(ul.product, result.product)
    self.assertEquals(ul.objectType, result.objectType)
    
    param = result.getParameter("DBZH")
    self.assertEquals([[2,2,3,3],[2,2,3,3],[4,4,5,5],[4,4,5,5]], param.getData().tolist())

    param = result.getParameter("TH")
    self.assertEquals([[12,12,13,13],[12,12,13,13],[14,14,15,15],[14,14,15,15]], param.getData().tolist())
    