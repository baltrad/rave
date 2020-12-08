'''
Copyright (C) 2020 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the PyLazyNodeListIO module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2020-11-06

'''
import unittest
import os
import _lazynodelistreader
import string
import numpy
import math

class PyLazyNodeListReaderTest(unittest.TestCase):
  FIXTURE_VOLUME="fixture_ODIM_H5_pvol_ang_20090501T1200Z.h5"

  TEMPORARY_FILE="ravemodule_lazyreadertest.h5"
  TEMPORARY_FILE2="ravemodule_lazyreadertest2.h5"

  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)
    if os.path.isfile(self.TEMPORARY_FILE2):
      os.unlink(self.TEMPORARY_FILE2)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)
    if os.path.isfile(self.TEMPORARY_FILE2):
      os.unlink(self.TEMPORARY_FILE2)
  
  def test_new(self):
    obj = _lazynodelistreader.new()
    islazynodelistreader = str(type(obj)).find("LazyNodeListReaderCore")
    self.assertNotEqual(-1, islazynodelistreader)
  
  def test_read(self):
    obj = _lazynodelistreader.read(self.FIXTURE_VOLUME)
    islazynodelistio = str(type(obj)).find("LazyNodeListReaderCore")
    self.assertNotEqual(-1, islazynodelistio)

  def test_getNodeNames(self):
    obj = _lazynodelistreader.read(self.FIXTURE_VOLUME)
    nodenames = obj.getNodeNames()
    self.assertEqual("/Conventions", nodenames[0])
    self.assertEqual("/dataset1", nodenames[1])
    self.assertEqual("/dataset1/data1", nodenames[2])

  def test_getNodeNames(self):
    obj = _lazynodelistreader.read(self.FIXTURE_VOLUME)
    nodenames = obj.getNodeNames()
    self.assertEqual("/Conventions", nodenames[0])
    self.assertEqual("/dataset1", nodenames[1])
    self.assertEqual("/dataset1/data1", nodenames[2])

  def test_isLoaded(self):
    obj = _lazynodelistreader.read(self.FIXTURE_VOLUME)
    self.assertFalse(obj.isLoaded("/dataset1/data1/data"))
    d = obj.getDataset("/dataset1/data1/data")
    self.assertTrue(obj.isLoaded("/dataset1/data1/data"))

  def test_exists(self):
    obj = _lazynodelistreader.read(self.FIXTURE_VOLUME)
    self.assertTrue(obj.exists("/dataset1/data1/data"))
    self.assertFalse(obj.exists("/dataset1/data77/data"))

  def test_preload_1(self):
    obj = _lazynodelistreader.read(self.FIXTURE_VOLUME)
    self.assertFalse(obj.isLoaded("/dataset1/data1/data"))
    self.assertFalse(obj.isLoaded("/dataset1/data2/data"))
    obj.preload()
    self.assertTrue(obj.isLoaded("/dataset1/data1/data"))
    self.assertTrue(obj.isLoaded("/dataset1/data2/data"))

  def test_preload_2(self):
    obj = _lazynodelistreader.read(self.FIXTURE_VOLUME)
    obj.preload("DBZH")
    self.assertTrue(obj.isLoaded("/dataset1/data1/data"))
    self.assertFalse(obj.isLoaded("/dataset1/data2/data"))
    
  def test_preload_3(self):
    obj = _lazynodelistreader.read(self.FIXTURE_VOLUME)
    obj.preload("DBZH,VRAD")
    self.assertTrue(obj.isLoaded("/dataset1/data1/data"))
    self.assertTrue(obj.isLoaded("/dataset1/data2/data"))
    
  def test_preload_4(self):
    obj = _lazynodelistreader.read(self.FIXTURE_VOLUME)
    obj.preload("KALLE,VRAD")
    self.assertFalse(obj.isLoaded("/dataset1/data1/data"))
    self.assertTrue(obj.isLoaded("/dataset1/data2/data"))

