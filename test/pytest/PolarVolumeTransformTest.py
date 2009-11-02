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

class PolarVolumeTransformTest(unittest.TestCase):
  VOLUMENAME = "fixture_ODIM_H5_pvol_ang_20090501T1200Z.h5"
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testCappi(self):
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
    
    transformer.cappi(volume, cartesian)

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
    nodelist.write("cartesian.h5")    
        
if __name__ == "__main__":
    unittest.main()