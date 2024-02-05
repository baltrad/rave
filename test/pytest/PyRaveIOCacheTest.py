'''
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the PyRaveIOCache module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2024-02-05
'''
import unittest
import os
import _iocache
import _rave
import _ravefield
import string
import numpy
import _pyhl
import math

class PyRaveIOTest(unittest.TestCase):
  TEMPORARY_FILE="ravemodule_iocachetest.h5"

  def setUp(self):
    #_rave.setDebugLevel(_rave.Debug_RAVE_SPEWDEBUG)
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def test_new(self):
    obj = _iocache.new()
    israveiocache = str(type(obj)).find("RaveIOCacheCore")
    self.assertNotEqual(-1, israveiocache)

  def test_attribute_visibility(self):
    attrs = ['compression_level', 'fcp_userblock', 'fcp_sizes', 'fcp_symk', 'fcp_istorek', 'fcp_metablocksize']
    obj = _iocache.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)

  def test_saveField(self):
    obj = _ravefield.new()
    obj.setData(numpy.zeros((10,10), numpy.uint8))
    obj.setValue(0,1,1.0)
    obj.setValue(0,2,2.0)
    obj.setValue(0,3,3.0)
    obj.setValue(0,4,4.0)
    obj.addAttribute("what/value", 123)
    obj.addAttribute("where/why", 1.23)
    obj.addAttribute("how/some", "noname")
  
    _iocache.saveField(obj, self.TEMPORARY_FILE)

    # Verify result
    nodelist = _pyhl.read_nodelist(self.TEMPORARY_FILE)
    nodelist.selectAll()
    nodelist.fetch()

    self.assertEqual(123, nodelist.getNode("/field1/what/value").data())
    self.assertAlmostEqual(1.23, nodelist.getNode("/field1/where/why").data(),4)
    self.assertEqual("noname", nodelist.getNode("/field1/how/some").data())
    df = nodelist.getNode("/field1/data").data()
    self.assertTrue((df == obj.getData()).all())

  def test_loadField(self):
    nodelist = _pyhl.nodelist()
    self.addGroupNode(nodelist, "/field1")
    self.addGroupNode(nodelist, "/field1/what")
    self.addGroupNode(nodelist, "/field1/where")
    self.addGroupNode(nodelist, "/field1/how")

    self.addAttributeNode(nodelist, "/field1/what/value", "int", 123)
    self.addAttributeNode(nodelist, "/field1/where/why", "double", 1.23)
    self.addAttributeNode(nodelist, "/field1/how/some", "string", "noname")

    dset = numpy.arange(100)
    dset=numpy.array(dset.astype(numpy.uint8),numpy.uint8)
    dset=numpy.reshape(dset,(10,10)).astype(numpy.uint8)
    dset[0][1]=1
    dset[0][2]=2
    dset[0][3]=3
    dset[0][4]=4

    self.addDatasetNode(nodelist, "/field1/data", "uchar", (10,10), dset)

    nodelist.write(self.TEMPORARY_FILE, 6)

    result = _iocache.loadField(self.TEMPORARY_FILE)
    self.assertEqual(123, result.getAttribute("what/value"))
    self.assertAlmostEqual(1.23, result.getAttribute("where/why"),4)
    self.assertEqual("noname", result.getAttribute("how/some"))
    self.assertTrue((dset == result.getData()).all())

  def addGroupNode(self, nodelist, name):
    node = _pyhl.node(_pyhl.GROUP_ID, name)
    nodelist.addNode(node)

  def addAttributeNode(self, nodelist, name, type, value):
    node = _pyhl.node(_pyhl.ATTRIBUTE_ID, name)
    if isinstance(value, numpy.ndarray):
      node.setArrayValue(-1,value.shape,value,type,-1)
    else:
      node.setScalarValue(-1,value,type,-1)
    nodelist.addNode(node)

  def addDatasetNode(self, nodelist, name, type, dims, value):
    node = _pyhl.node(_pyhl.DATASET_ID, name)
    node.setArrayValue(-1, dims, value, type, -1)
    nodelist.addNode(node)