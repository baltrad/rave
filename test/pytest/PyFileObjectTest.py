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

Tests the py rave value module.

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2025-02-14
'''
import unittest
import os
import _ravevalue, _fileobject
import _raveio, _rave, _polarscan
import string
import math
import json
import numpy

class PyFileObjectTest(unittest.TestCase):
  FIXTURE_RHI="fixtures/seatv-RHIVol-20251215-085918-2621.h5"
  FIXTURE_SCAN="fixtures/sehem_scan_20200414T160000Z.h5"

  TEMPORARY_FILE="fileobject_iotest.h5"

  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def test_new(self):
    obj = _fileobject.new()

    isobj = str(type(obj)).find("FileObjectCore")
    self.assertNotEqual(-1, isobj)
    self.assertEqual("", obj.name)

  def test_create_1(self):
    fileobj = _fileobject.new()
    fileobj.name = "THIS"
    r = fileobj.create("")
    self.assertEqual("THIS", r.name)

  def test_create_2(self):
    fileobj = _fileobject.new()
    r = fileobj.create("dataset1")
    self.assertEqual("", fileobj.name)
    self.assertEqual("dataset1", r.name)
    self.assertEqual("dataset1", fileobj.get("dataset1").name)

  def test_create_4(self):
    fileobj = _fileobject.new()
    r = fileobj.create("/dataset1/data1")
    self.assertEqual("", fileobj.name)
    self.assertEqual("dataset1", fileobj.get("dataset1").name)
    self.assertEqual("data1", fileobj.get("dataset1").get("data1").name)

  def test_create_5(self):
    fileobj = _fileobject.new()
    r = fileobj.create("/dataset1/data1/")
    self.assertEqual("", fileobj.name)
    self.assertEqual("dataset1", fileobj.get("dataset1").name)
    self.assertEqual("data1", fileobj.get("dataset1").get("data1").name)
    self.assertEqual(0, len(fileobj.get("dataset1").get("data1").groups))

  def test_create_6(self):
    fileobj = _fileobject.new()
    fileobj.create("/dataset1/data1/")
    fileobj.create("/dataset1/data2/")
    fileobj.create("/dataset2/data1/")
    fileobj.create("/dataset2/data1/q1")

    self.assertEqual(2, len(fileobj.groups))
    self.assertEqual(2, len(fileobj.get("/dataset1").groups))
    self.assertEqual(1, len(fileobj.get("/dataset2").groups))
    self.assertEqual(1, len(fileobj.get("/dataset2/data1").groups))

  def test_attrs_1(self):
    fileobj = _fileobject.new()
    c = fileobj.create("/dataset1/data1/how")
    c.attributes["task"]="somevalue"
    c.attributes["task_args"] = "NOD:123"
    self.assertEqual("somevalue", fileobj.get("/dataset1/data1/how").attributes["task"])

  def test_attrs_2(self):
    fileobj = _fileobject.new()
    c = fileobj.create("/dataset1/data1/how")
    c.attributes["task"]=[1,2,3]
    self.assertEqual([1,2,3], fileobj.get("/dataset1/data1/how").attributes["task"])

  def test_attrs_remove(self):
    fileobj = _fileobject.new()
    c = fileobj.create("/dataset1")
    c.attributes["task"]="HELLO"
    self.assertTrue("task" in c.attributes)
    del c.attributes["task"]
    self.assertFalse("task" in c.attributes)

  def test_get_byName_1(self):
    fileobj = _fileobject.new()
    fileobj.create("/dataset1/data1").attributes["a"] = "A"
    fileobj.create("/dataset1/data2").attributes["a"] = "B"
    fileobj.create("/dataset2/data1").attributes["a"] = "C"
    fileobj.create("/dataset2/data1/q1").attributes["a"] = "D"

    self.assertEqual("A", fileobj.get("/dataset1/data1").attributes["a"])
    self.assertEqual("B", fileobj.get("/dataset1/data2").attributes["a"])
    self.assertEqual("C", fileobj.get("/dataset2/data1").attributes["a"])
    self.assertEqual("D", fileobj.get("/dataset2/data1/q1").attributes["a"])

  def test_get_byIndex_1(self):
    fileobj = _fileobject.new()
    fileobj.create("/dataset1/data1")
    fileobj.create("/dataset1/data2")
    fileobj.create("/dataset2/data1")
    fileobj.create("/dataset2/data1/q1")

    self.assertEqual("dataset1", fileobj.get(0).name)
    self.assertEqual("data1", fileobj.get(0).get(0).name)
    self.assertEqual("data2", fileobj.get(0).get(1).name)
    self.assertEqual("dataset2", fileobj.get(1).name)
    self.assertEqual("data1", fileobj.get(1).get(0).name)
    self.assertEqual("q1", fileobj.get(1).get(0).get(0).name)

  def test_get_byIndex_2(self):
    fileobj = _fileobject.new()
    fileobj.create("/dataset1/data1")
    fileobj.create("/dataset1/data2")
    fileobj.create("/dataset2/data1")
    fileobj.create("/dataset2/data1/q1")

    self.assertEqual("dataset1", fileobj.get(0).name)
    try:
      fileobj.get(2)
      self.fail("Expected IndexError")
    except IndexError:
      pass

  def test_areNamesSet_1(self):
    fileobj = _fileobject.new()
    fileobj.create("/dataset1/data1")
    fileobj.create("/dataset1/data2")
    fileobj.create("/dataset2/data1")
    self.assertEqual(True, fileobj.areNamesSet())

  def test_areNamesSet_3(self):
    fileobj = _fileobject.new()
    fileobj.create("/dataset1/data1")
    fileobj.create("/dataset1/data2")
    k = fileobj.create("/dataset2/data1")
    k.name = ""
    self.assertEqual(False, fileobj.areNamesSet())

  def test_toString(self):
    fileobj = _fileobject.new()
    fileobj.create("/dataset1/data1")
    d2 = fileobj.create("/dataset1/data2")
    d2.attributes["name"] = "HELLO"
    self.assertEqual("""GROUP '/' {
  GROUP 'dataset1' {
    GROUP 'data1' {
    }
    GROUP 'data2' {
      name = HELLO
    }
  }
}
""", str(fileobj))

  def test_numberOfGroups(self):
    fileobj = _fileobject.new()
    fileobj.create("/dataset1/data1").attributes["a"] = "A"
    fileobj.create("/dataset1/data2").attributes["a"] = "B"
    fileobj.create("/dataset2/data1").attributes["a"] = "C"
    fileobj.create("/dataset2/data1/q1").attributes["a"] = "D"
    self.assertEqual(2, fileobj.numberOfGroups)
    self.assertEqual(2, fileobj["/dataset1"].numberOfGroups)
    self.assertEqual(0, fileobj["/dataset1/data1"].numberOfGroups)
    self.assertEqual(0, fileobj["/dataset1/data2"].numberOfGroups)
    self.assertEqual(1, fileobj["/dataset2"].numberOfGroups)
    self.assertEqual(1, fileobj["/dataset2/data1"].numberOfGroups)
    self.assertEqual(0, fileobj["/dataset2/data1/q1"].numberOfGroups)

  def test_exists(self):
    fileobj = _fileobject.new()
    fileobj.create("/dataset1/data1")
    fileobj.create("/dataset1/data2")
    fileobj.create("/dataset2/data1")
    fileobj.create("/dataset2/data1/q1")
    self.assertTrue("/dataset1" in fileobj)
    self.assertTrue("/dataset1/data1" in fileobj)
    self.assertTrue("/dataset1/data2" in fileobj)
    self.assertTrue("/dataset2" in fileobj)
    self.assertTrue("/dataset2/data1" in fileobj)
    self.assertTrue("/dataset2/data1/q1" in fileobj)

    self.assertFalse("/dataset3" in fileobj)
    self.assertFalse("/dataset1/data3" in fileobj)
    self.assertFalse("/dataset1/data4" in fileobj)
    self.assertFalse("/dataset4" in fileobj)
    self.assertFalse("/dataset4/data1" in fileobj)
    self.assertFalse("/dataset4/data1/q1" in fileobj)

  def test_groups(self):
    fileobj = _fileobject.new()
    fileobj.create("/dataset1/data1")
    fileobj.create("/dataset1/data2")
    fileobj.create("/dataset2/data1")
    fileobj.create("/dataset2/data1/q1")
    self.assertEqual(2, len(fileobj.groups))
    self.assertEqual("dataset1", fileobj.groups[0].name)
    self.assertEqual("dataset2", fileobj.groups[1].name)
    self.assertEqual(2, len(fileobj.groups[0].groups))
    self.assertEqual(0, len(fileobj.groups[0].groups[0].groups))
    self.assertEqual("data1", fileobj.groups[0].groups[0].name)
    self.assertEqual(0, len(fileobj.groups[0].groups[1].groups))
    self.assertEqual("data2", fileobj.groups[0].groups[1].name)
    self.assertEqual(1, len(fileobj.groups[1].groups))
    self.assertEqual("data1", fileobj.groups[1].groups[0].name)
    self.assertEqual(1, len(fileobj.groups[1].groups[0].groups))
    self.assertEqual("q1", fileobj.groups[1].groups[0].groups[0].name)

  def test_isDataset(self):
    fileobj = _fileobject.new()
    self.assertFalse(fileobj.isDataset())
    fileobj.data = numpy.zeros((2,2), numpy.uint8)
    self.assertTrue(fileobj.isDataset())

  def test_isDatasetLoaded(self):
    fileobj = _fileobject.new()
    self.assertFalse(fileobj.isDatasetLoaded())
    fileobj.data = numpy.zeros((2,2), numpy.uint8)
    self.assertTrue(fileobj.isDatasetLoaded())

  def test_xsize_ysize_datatype(self):
    fileobj = _fileobject.new()
    fileobj.data = numpy.zeros((2,3), numpy.uint8)
    self.assertEqual(3, fileobj.xsize)
    self.assertEqual(2, fileobj.ysize)
    self.assertEqual(_rave.RaveDataType_UCHAR, fileobj.datatype)

  def test_xsize_ysize_datatype_notdataset(self):
    fileobj = _fileobject.new()
    try:
      fileobj.xsize
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

    try:
      fileobj.ysize
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

    try:
      fileobj.datatype
      self.fail("Expected AttributeError")
    except AttributeError:
      pass

  def test_load_rhi_open(self):
    """ We use raveio to verify that file object loading is working but we keep tests in here.
    """
    fileobj=_raveio.open(self.FIXTURE_RHI, True).object
    self.assertEqual("ODIM_H5/V2_3", fileobj.attributes["Conventions"])
    self.assertEqual("085918",   fileobj["/what"].attributes["time"])
    self.assertEqual("20251215", fileobj["/what"].attributes["date"])
    self.assertEqual("ELEV",     fileobj["/what"].attributes["object"])
    self.assertEqual("H5rad 2.3",     fileobj["/what"].attributes["version"])
    self.assertEqual("WMO:02570,RAD:SE48,PLC:Åtvidaberg(Vilebo),NOD:seatv,ORG:82,CTY:643,CMT:Swedish radar",     fileobj["/what"].attributes["source"])

    self.assertAlmostEqual(58.1059,   fileobj["/where"].attributes["lat"],4)
    self.assertAlmostEqual(15.9365,   fileobj["/where"].attributes["lon"],4)
    self.assertAlmostEqual(222,   fileobj["/where"].attributes["height"],4)

    self.assertEqual(-32767, fileobj["/dataset1/data1/what"].attributes["undetect"])
    self.assertEqual(-32768, fileobj["/dataset1/data1/what"].attributes["nodata"])
    self.assertAlmostEqual(0.01,   fileobj["/dataset1/data1/what"].attributes["gain"], 4)
    self.assertAlmostEqual(0.0,    fileobj["/dataset1/data1/what"].attributes["offset"], 4)
    self.assertEqual("TH",   fileobj["/dataset1/data1/what"].attributes["quantity"], 4)

    self.assertFalse(fileobj.isDatasetLoaded())
    self.assertEqual(480, fileobj["/dataset1/data1/data"].xsize)
    self.assertEqual(58, fileobj["/dataset1/data1/data"].ysize)
    self.assertEqual(_rave.RaveDataType_SHORT, fileobj["/dataset1/data1/data"].datatype)
    self.assertEqual((58, 480), fileobj["/dataset1/data1/data"].data.getData().shape)
    self.assertTrue(fileobj["/dataset1/data1/data"].isDatasetLoaded())

  def test_load_rhi_openFileObject(self):
    fileobj=_raveio.openFileObject(self.FIXTURE_RHI, True).object
    self.assertEqual("ODIM_H5/V2_3", fileobj.attributes["Conventions"])
    self.assertEqual("085918",   fileobj["/what"].attributes["time"])
    self.assertEqual("20251215", fileobj["/what"].attributes["date"])
    self.assertEqual("ELEV",     fileobj["/what"].attributes["object"])
    self.assertEqual("H5rad 2.3",     fileobj["/what"].attributes["version"])
    self.assertEqual("WMO:02570,RAD:SE48,PLC:Åtvidaberg(Vilebo),NOD:seatv,ORG:82,CTY:643,CMT:Swedish radar",     fileobj["/what"].attributes["source"])

    self.assertAlmostEqual(58.1059,   fileobj["/where"].attributes["lat"],4)
    self.assertAlmostEqual(15.9365,   fileobj["/where"].attributes["lon"],4)
    self.assertAlmostEqual(222,   fileobj["/where"].attributes["height"],4)

    self.assertEqual(-32767, fileobj["/dataset1/data1/what"].attributes["undetect"])
    self.assertEqual(-32768, fileobj["/dataset1/data1/what"].attributes["nodata"])
    self.assertAlmostEqual(0.01,   fileobj["/dataset1/data1/what"].attributes["gain"], 4)
    self.assertAlmostEqual(0.0,    fileobj["/dataset1/data1/what"].attributes["offset"], 4)
    self.assertEqual("TH",   fileobj["/dataset1/data1/what"].attributes["quantity"], 4)

    self.assertFalse(fileobj.isDatasetLoaded())
    self.assertEqual(480, fileobj["/dataset1/data1/data"].xsize)
    self.assertEqual(58, fileobj["/dataset1/data1/data"].ysize)
    self.assertEqual(_rave.RaveDataType_SHORT, fileobj["/dataset1/data1/data"].datatype)
    self.assertEqual((58, 480), fileobj["/dataset1/data1/data"].data.getData().shape)
    self.assertTrue(fileobj["/dataset1/data1/data"].isDatasetLoaded())

  def test_load_scan_openFileObject(self):
    fileobj=_raveio.openFileObject(self.FIXTURE_SCAN, False).object
    self.assertEqual("ODIM_H5/V2_2", fileobj.attributes["Conventions"])
    self.assertEqual("160000",   fileobj["/what"].attributes["time"])
    self.assertEqual("20200414", fileobj["/what"].attributes["date"])
    self.assertEqual("SCAN",     fileobj["/what"].attributes["object"])
    self.assertEqual("H5rad 2.2",     fileobj["/what"].attributes["version"])
    self.assertEqual("WMO:02588,RAD:SE47,PLC:Hemse(Ase),NOD:sehem,ORG:82,CTY:643,CMT:Swedish radar",     fileobj["/what"].attributes["source"])

    self.assertAlmostEqual(57.3034,   fileobj["/where"].attributes["lat"],4)
    self.assertAlmostEqual(18.4003,   fileobj["/where"].attributes["lon"],4)
    self.assertAlmostEqual(85,   fileobj["/where"].attributes["height"],4)

    self.assertEqual(0.0, fileobj["/dataset1/data1/what"].attributes["undetect"])
    self.assertEqual(255, fileobj["/dataset1/data1/what"].attributes["nodata"])
    self.assertAlmostEqual(0.5,   fileobj["/dataset1/data1/what"].attributes["gain"], 4)
    self.assertAlmostEqual(-32.0,    fileobj["/dataset1/data1/what"].attributes["offset"], 4)
    self.assertEqual("DBZH",   fileobj["/dataset1/data1/what"].attributes["quantity"], 4)

    self.assertTrue(fileobj["/dataset1/data1/data"].isDatasetLoaded())
    self.assertEqual(480, fileobj["/dataset1/data1/data"].xsize)
    self.assertEqual(360, fileobj["/dataset1/data1/data"].ysize)
    self.assertEqual(_rave.RaveDataType_UCHAR, fileobj["/dataset1/data1/data"].datatype)
    self.assertEqual((360, 480), fileobj["/dataset1/data1/data"].data.getData().shape)

  def test_load_save_fileObject_scan(self):
    fileobj=_raveio.openFileObject(self.FIXTURE_SCAN, False).object
    rio = _raveio.new()
    rio.object = fileobj
    rio.save(self.TEMPORARY_FILE)

    scan = _raveio.open(self.TEMPORARY_FILE).object
    self.assertTrue(_polarscan.isPolarScan(scan))
    self.assertEqual("160000",   scan.time)
    self.assertEqual("20200414", scan.date)
    self.assertEqual("WMO:02588,RAD:SE47,PLC:Hemse(Ase),NOD:sehem,ORG:82,CTY:643,CMT:Swedish radar", scan.source)

    self.assertAlmostEqual(57.3034,   scan.latitude * 180.0/math.pi, 4)
    self.assertAlmostEqual(18.4003,   scan.longitude * 180.0/math.pi, 4)
    self.assertAlmostEqual(85,   scan.height, 4)
