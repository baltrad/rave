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

Tests some basic product generation.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2009-10-22
'''
import unittest
import os
import _rave
import _raveio
import string
import numpy
#from rave_loader import rave_loader
import area
import pcs
import _pyhl
import math

class PolarVolumeTransformTest(unittest.TestCase):
  VOLUMENAME = "fixture_ODIM_H5_pvol_ang_20090501T1200Z.h5"
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testCAPPI(self):
    volume = _raveio.open(self.VOLUMENAME).object #rave_loader().load_file(self.VOLUMENAME, "DBZH") 

    transformer = _rave.transform()
    transformer.method = _rave.NEAREST
    
    a = area.area("ang_240")
    cartesian = _rave.cartesian()
    cartesian.nodata = 255.0
    cartesian.undetect = 0.0
    cartesian.xscale = a.xscale
    cartesian.yscale = a.yscale
    cartesian.areaextent = a.extent
    data = numpy.zeros((a.ysize, a.xsize), numpy.uint8)
    cartesian.setData(data)
    projection = _rave.projection(a.Id, a.name, pcs.pcs(a.pcs).tostring())
    cartesian.projection = projection
    
    transformer.cappi(volume, cartesian, 1000.0)

    newdata = cartesian.getData()
    nodelist = _pyhl.nodelist()
    node = _pyhl.node(_pyhl.DATASET_ID, "/data")
    node.setArrayValue(-1, newdata.shape, newdata, "uchar", -1)
    nodelist.addNode(node)
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/data/CLASS")
    node.setScalarValue(-1, "IMAGE", "string", -1)
    nodelist.addNode(node)
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/data/IMAGE_VERSION")
    node.setScalarValue(-1, "1.2", "string", -1)
    nodelist.addNode(node)
    nodelist.write("cartesian_cappi.h5")    

  def testPPI(self):
    volume = _raveio.open(self.VOLUMENAME).object #rave_loader().load_file(self.VOLUMENAME, "DBZH") 

    transformer = _rave.transform()
    transformer.method = _rave.NEAREST
    
    a = area.area("ang_240")
    cartesian = _rave.cartesian()
    cartesian.nodata = 255.0
    cartesian.undetect = 0.0
    cartesian.xscale = a.xscale
    cartesian.yscale = a.yscale
    cartesian.areaextent = a.extent
    data = numpy.zeros((a.ysize, a.xsize), numpy.uint8)
    cartesian.setData(data)
    projection = _rave.projection(a.Id, a.name, pcs.pcs(a.pcs).tostring())
    cartesian.projection = projection
    
    scan = volume.getScan(0)
    transformer.ppi(scan, cartesian)

    newdata = cartesian.getData()
    nodelist = _pyhl.nodelist()
    node = _pyhl.node(_pyhl.DATASET_ID, "/data")
    node.setArrayValue(-1, newdata.shape, newdata, "uchar", -1)
    nodelist.addNode(node)
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/data/CLASS")
    node.setScalarValue(-1, "IMAGE", "string", -1)
    nodelist.addNode(node)
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/data/IMAGE_VERSION")
    node.setScalarValue(-1, "1.2", "string", -1)
    nodelist.addNode(node)
    nodelist.write("cartesian_ppi.h5")    

  def testPCAPPI(self):
    volume = _raveio.open(self.VOLUMENAME).object #rave_loader().load_file(self.VOLUMENAME, "DBZH") 

    transformer = _rave.transform()
    transformer.method = _rave.NEAREST
    
    a = area.area("ang_240")
    cartesian = _rave.cartesian()
    cartesian.nodata = 255.0
    cartesian.undetect = 0.0
    cartesian.xscale = a.xscale
    cartesian.yscale = a.yscale
    cartesian.areaextent = a.extent
    data = numpy.zeros((a.ysize, a.xsize), numpy.uint8)
    cartesian.setData(data)
    projection = _rave.projection(a.Id, a.name, pcs.pcs(a.pcs).tostring())
    cartesian.projection = projection
    
    transformer.pcappi(volume, cartesian, 1000.0)

    newdata = cartesian.getData()
    nodelist = _pyhl.nodelist()
    node = _pyhl.node(_pyhl.DATASET_ID, "/data")
    node.setArrayValue(-1, newdata.shape, newdata, "uchar", -1)
    nodelist.addNode(node)
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/data/CLASS")
    node.setScalarValue(-1, "IMAGE", "string", -1)
    nodelist.addNode(node)
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/data/IMAGE_VERSION")
    node.setScalarValue(-1, "1.2", "string", -1)
    nodelist.addNode(node)
    nodelist.write("cartesian_pcappi.h5")
            
if __name__ == "__main__":
    unittest.main()