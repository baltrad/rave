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
import _ravefield
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
    self.assertNotEqual(-1, str(type(obj)).find("TransformCore")) 

  def test_attribute_visibility(self):
    attrs = ['method']
    obj = _transform.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)

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
      except ValueError:
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
    self.assertEqual(1, data[2][2])

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
    self.assertEqual(1, data[2][2])
    data = result.getParameter("TH").getData() 
    self.assertEqual(2, data[2][2])
  
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
  
  def create_quality_field(self, xsize, ysize, dtype, value, howtask):
    obj = _ravefield.new()
    data = numpy.zeros((ysize, xsize), dtype)
    data = data + value
    obj.setData(data)
    obj.addAttribute("how/task", howtask)
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
    ul.addQualityField(self.create_quality_field(2,2,numpy.uint8, 15, "se.some.how.task.1"))
    ul.addQualityField(self.create_quality_field(2,2,numpy.uint8, 19, "se.some.how.task.2"))
    ul.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 23, "se.some.how.task.1"))
    ul.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 24, "se.some.how.task.2"))
    
    ur = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (1993337.7288084999,9112461.1790100001,3015337.72881,11028461.179),
                                              pyarea.projection.definition,
                                              numpy.uint8, [3], ["DBZH"])
    ur.addQualityField(self.create_quality_field(2,2,numpy.uint8, 16, "se.some.how.task.1"))
    ur.addQualityField(self.create_quality_field(2,2,numpy.uint8, 20, "se.some.how.task.2"))
    ur.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 25, "se.some.how.task.1"))
    ur.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 26, "se.some.how.task.2"))
    
    ll = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (971337.728807,7196461.17902,1993337.7288084999,9112461.1790100001),
                                              pyarea.projection.definition,
                                              numpy.uint8, [4], ["DBZH"])
    ll.addQualityField(self.create_quality_field(2,2,numpy.uint8, 17, "se.some.how.task.1"))
    ll.addQualityField(self.create_quality_field(2,2,numpy.uint8, 21, "se.some.how.task.2"))
    ll.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 27, "se.some.how.task.1"))
    ll.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 28, "se.some.how.task.2"))
    
    lr = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (1993337.7288084999,7196461.17902,3015337.72881,9112461.1790100001),
                                              pyarea.projection.definition,
                                              numpy.uint8, [5], ["DBZH"])
    lr.addQualityField(self.create_quality_field(2,2,numpy.uint8, 18, "se.some.how.task.1"))
    lr.addQualityField(self.create_quality_field(2,2,numpy.uint8, 22, "se.some.how.task.2"))
    lr.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 29, "se.some.how.task.1"))
    lr.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 30, "se.some.how.task.2"))
    
    t = _transform.new()
    result = t.combine_tiles(pyarea, [ul,ur,ll,lr])
    param = result.getParameter("DBZH")
    qf1 = result.getQualityFieldByHowTask("se.some.how.task.1")
    qf2 = result.getQualityFieldByHowTask("se.some.how.task.2")
    pqf1 = param.getQualityFieldByHowTask("se.some.how.task.1")
    pqf2 = param.getQualityFieldByHowTask("se.some.how.task.2")
    
    self.assertEqual(4, result.xsize)
    self.assertEqual(4, result.ysize)
    self.assertEqual(511000.0, result.xscale, 4)
    self.assertEqual(958000.0, result.yscale, 4)
    self.assertEqual(971337.728807, result.areaextent[0], 4)
    self.assertEqual(7196461.17902, result.areaextent[1], 4)
    self.assertEqual(3015337.72881, result.areaextent[2], 4)
    self.assertEqual(11028461.179, result.areaextent[3], 4)
    self.assertEqual("+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +R=6378137.0 +nadgrids=@null +no_defs", result.projection.definition)
    self.assertEqual(ul.date, result.date)
    self.assertEqual(ul.time, result.time)
    self.assertEqual(ul.startdate, result.startdate)
    self.assertEqual(ul.starttime, result.starttime)
    self.assertEqual(ul.enddate, result.enddate)
    self.assertEqual(ul.endtime, result.endtime)
    self.assertEqual(ul.product, result.product)
    self.assertEqual(ul.objectType, result.objectType)
    
    self.assertEqual([[2,2,3,3],[2,2,3,3],[4,4,5,5],[4,4,5,5]], param.getData().tolist())
    self.assertEqual([[15,15,16,16],[15,15,16,16],[17,17,18,18],[17,17,18,18]], qf1.getData().tolist())
    self.assertEqual([[19,19,20,20],[19,19,20,20],[21,21,22,22],[21,21,22,22]], qf2.getData().tolist())
    self.assertEqual([[23,23,25,25],[23,23,25,25],[27,27,29,29],[27,27,29,29]], pqf1.getData().tolist())
    self.assertEqual([[24,24,26,26],[24,24,26,26],[28,28,30,30],[28,28,30,30]], pqf2.getData().tolist())
    
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
    ul.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 22, "se.some.how.task.1"))
    ul.getParameter("TH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 32, "se.some.how.task.1"))
    
    ur = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (1993337.7288084999,9112461.1790100001,3015337.72881,11028461.179),
                                              pyarea.projection.definition,
                                              numpy.uint8, [3,13], ["DBZH", "TH"])
    ur.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 23, "se.some.how.task.1"))
    ur.getParameter("TH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 33, "se.some.how.task.1"))
    
    ll = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (971337.728807,7196461.17902,1993337.7288084999,9112461.1790100001),
                                              pyarea.projection.definition,
                                              numpy.uint8, [4,14], ["DBZH", "TH"])
    ll.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 24, "se.some.how.task.1"))
    ll.getParameter("TH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 34, "se.some.how.task.1"))
    
    lr = self.create_cartesian_with_parameter(2, 2, pyarea.xscale, pyarea.yscale, 
                                              (1993337.7288084999,7196461.17902,3015337.72881,9112461.1790100001),
                                              pyarea.projection.definition,
                                              numpy.uint8, [5,15], ["DBZH", "TH"])
    lr.getParameter("DBZH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 25, "se.some.how.task.1"))
    lr.getParameter("TH").addQualityField(self.create_quality_field(2,2,numpy.uint8, 35, "se.some.how.task.1"))
    
    t = _transform.new()
    result = t.combine_tiles(pyarea, [ul,ur,ll,lr])
    self.assertEqual(4, result.xsize)
    self.assertEqual(4, result.ysize)
    self.assertEqual(511000.0, result.xscale, 4)
    self.assertEqual(958000.0, result.yscale, 4)
    self.assertEqual(971337.728807, result.areaextent[0], 4)
    self.assertEqual(7196461.17902, result.areaextent[1], 4)
    self.assertEqual(3015337.72881, result.areaextent[2], 4)
    self.assertEqual(11028461.179, result.areaextent[3], 4)
    self.assertEqual("+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +R=6378137.0 +nadgrids=@null +no_defs", result.projection.definition)
    self.assertEqual(ul.date, result.date)
    self.assertEqual(ul.time, result.time)
    self.assertEqual(ul.startdate, result.startdate)
    self.assertEqual(ul.starttime, result.starttime)
    self.assertEqual(ul.enddate, result.enddate)
    self.assertEqual(ul.endtime, result.endtime)
    self.assertEqual(ul.product, result.product)
    self.assertEqual(ul.objectType, result.objectType)
    
    param = result.getParameter("DBZH")
    self.assertEqual([[2,2,3,3],[2,2,3,3],[4,4,5,5],[4,4,5,5]], param.getData().tolist())
    self.assertEqual([[22,22,23,23],[22,22,23,23],[24,24,25,25],[24,24,25,25]], param.getQualityFieldByHowTask("se.some.how.task.1").getData().tolist())

    param = result.getParameter("TH")
    self.assertEqual([[12,12,13,13],[12,12,13,13],[14,14,15,15],[14,14,15,15]], param.getData().tolist())
    self.assertEqual([[32,32,33,33],[32,32,33,33],[34,34,35,35],[34,34,35,35]], param.getQualityFieldByHowTask("se.some.how.task.1").getData().tolist())
    