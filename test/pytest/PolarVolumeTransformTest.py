'''
Created on Oct 22, 2009
Tests some basic product generation
@author: Anders Henja
'''
import unittest
import os
import _rave
import string
import numpy
from rave_loader import rave_loader
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

  def testSimple(self):
    scan1 = _rave.scan()
    scan2 = _rave.scan()
    scan3 = _rave.scan()
    scan4 = _rave.scan()
    scan5 = _rave.scan()
    
    vol = _rave.volume()
    vol.addScan(scan1)
    vol.addScan(scan2)
    vol.addScan(scan3)
    vol.addScan(scan4)
    vol.addScan(scan5)

    del scan3
    del vol

  def testCAPPI(self):
    volume = rave_loader().load_file(self.VOLUMENAME, "DBZH") 

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
    volume = rave_loader().load_file(self.VOLUMENAME, "DBZH") 

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
    volume = rave_loader().load_file(self.VOLUMENAME, "DBZH") 

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