'''
Copyright (C) 2014 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the compositing class

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2014-11-24
'''
import unittest
import _rave
import compositing
import _pycomposite
import rave_quality_plugin, rave_pgf_quality_registry
import mock

class scan_mock(mock.MagicMock):

  def __init__(self, *args, **kwargs):
    super(scan_mock, self).__init__(*args, **kwargs)
    self.source = ""
    self.attribute_map = {}
    self.date = ""
    self.time = ""

  def _get_child_mock(self, **kwargs):
    return mock.MagicMock(**kwargs)
    
  def set_source(self, source):
    self.source = source
    
  def set_attribute(self, key, value):
    self.attribute_map[key] = value
    
  def hasAttribute(self, attribute):
    return attribute in self.attribute_map
  
  def getAttribute(self, attribute):
    return self.attribute_map.get(attribute)
  
  
class volume_mock(mock.MagicMock):

  def __init__(self, *args, **kwargs):
    super(volume_mock, self).__init__(*args, **kwargs)
    self.source = ""
    self.date = ""
    self.time = ""
    self.scans = []

  def _get_child_mock(self, **kwargs):
    return mock.MagicMock(**kwargs)
    
  def set_source(self, source):
    self.source = source
    
  def set_attribute(self, key, value):
    self.attribute_map[key] = value
    
  def add_scan(self, scan):
    self.scans.append(scan)
    
  def getScan(self, index):
    return self.scans[index]
  
  def getNumberOfScans(self):
    return len(self.scans)
  
  def removeScan(self, index):
    self.scans.remove(self.scans[index])
        

class compositing_test(unittest.TestCase):
  
  
  def setUp(self):
    self.qc_check_1_mock = mock.Mock(spec=rave_quality_plugin.rave_quality_plugin)
    self.qc_check_2_mock = mock.Mock(spec=rave_quality_plugin.rave_quality_plugin)
    rave_pgf_quality_registry.add_plugin("qc.check.1", self.qc_check_1_mock)
    rave_pgf_quality_registry.add_plugin("qc.check.2", self.qc_check_2_mock)
    self.classUnderTest = compositing.compositing()
    
    self.sources = {"WMO:02262,RAD:SE43,PLC:Ornskoldsvik,CMT:seovi" : "seovi",  "WMO:02600,RAD:SE49,PLC:Vara,CMT:sevar" : "sevax"}

  def tearDown(self):
    self.classUnderTest = None
    rave_pgf_quality_registry.remove_plugin("qc.check.1")
    rave_pgf_quality_registry.remove_plugin("qc.check.2")
    
  def setup_default_scan_mock(self, index, malfunc=False):
    scan = scan_mock()
    scan.set_source(self.sources.keys()[index%len(self.sources)])
    scan.set_attribute("how/task", "how_task" + str(index))
    scan.set_attribute("how/malfunc", str(malfunc))
    return scan
  
  def setup_default_volume_mock(self, index):
    vol = volume_mock()
    vol.set_source(self.sources.keys()[index])
    return vol
  
  def add_default_scan_to_volume_mock(self, vol_mock, index, malfunc=False):
    scan = self.setup_default_scan_mock(index, malfunc)
    vol_mock.add_scan(scan)
    return scan

  @mock.patch('_polarvolume.isPolarVolume')
  @mock.patch('_polarscan.isPolarScan')
  def test_fetch_objects_scan(self, mock_ispolarscan, mock_ispolarvolume):
    file_obj1 = self.setup_default_scan_mock(0)
    file_obj2 = self.setup_default_scan_mock(1)
    
    file_map = {"file1" : file_obj1, "file2" : file_obj2}
    
    self.classUnderTest.filenames = file_map.keys()
    
    self.classUnderTest.ravebdb = mock.Mock()
    
    self.classUnderTest.ravebdb.get_rave_object.side_effect = lambda x : file_map[x]
    mock_ispolarscan.return_value = True
    mock_ispolarvolume.return_value = False
        
    objects, nodes, how_tasks, all_files_malfunc = self.classUnderTest.fetch_objects()
        
    get_rave_object_calls = [] 
    for file_name in file_map.keys():
      get_rave_object_calls.append(mock.call(file_name))
      
      self.assertTrue(file_name in objects)
      self.assertEqual(objects[file_name], file_map[file_name])
    
    self.classUnderTest.ravebdb.get_rave_object.assert_has_calls(get_rave_object_calls)
    
    mock_ispolarvolume.assert_not_called()
    
    self.assertFalse(all_files_malfunc)
    
    nodes_list = nodes.replace("'", "").split(",")
    self.assertEqual(len(nodes_list), 2)
    expected_nodes = [self.sources.get(file_obj1.source), self.sources.get(file_obj2.source)]
    self.assertEqual(set(nodes_list), set(expected_nodes))

    how_task_list = how_tasks.split(",")
    self.assertEqual(len(how_task_list), 2)
    expected_how_tasks = [file_obj1.getAttribute("how/task"), file_obj2.getAttribute("how/task")]
    self.assertEqual(set(how_task_list), set(expected_how_tasks))
      
  @mock.patch('_polarvolume.isPolarVolume')
  @mock.patch('_polarscan.isPolarScan')
  def test_fetch_objects_scan__one_malfunc(self, mock_ispolarscan, mock_ispolarvolume):
    file_obj1 = self.setup_default_scan_mock(0, malfunc=True)
    file_obj2 = self.setup_default_scan_mock(1, malfunc=False)
    
    file_map = {"file1" : file_obj1, "file2" : file_obj2}
    
    self.classUnderTest.filenames = file_map.keys()
    self.classUnderTest.ignore_malfunc = True
    
    self.classUnderTest.ravebdb = mock.Mock()
    
    self.classUnderTest.ravebdb.get_rave_object.side_effect = lambda x : file_map[x]
    mock_ispolarscan.return_value = True
    mock_ispolarvolume.return_value = False
        
    objects, nodes, how_tasks, all_files_malfunc = self.classUnderTest.fetch_objects()
        
    get_rave_object_calls = [] 
    for file_name in file_map.keys():
      get_rave_object_calls.append(mock.call(file_name))
      
    self.assertFalse(all_files_malfunc)
      
    self.assertEqual(len(objects), 1)
    self.assertEqual(objects["file2"], file_map["file2"])
    
    self.classUnderTest.ravebdb.get_rave_object.assert_has_calls(get_rave_object_calls)
    
    self.assertEqual(nodes.strip("'"), self.sources.get(file_obj2.source))

    self.assertEqual(how_tasks, file_obj2.getAttribute("how/task"))
      
  @mock.patch('_polarvolume.isPolarVolume')
  @mock.patch('_polarscan.isPolarScan')
  def test_fetch_objects_volume(self, mock_ispolarscan, mock_ispolarvolume):
    file_obj1 = self.setup_default_volume_mock(0)
    scan1 = self.add_default_scan_to_volume_mock(file_obj1, 0)

    file_obj2 = self.setup_default_volume_mock(1)
    scan2 = self.add_default_scan_to_volume_mock(file_obj2, 1)
    scan3 = self.add_default_scan_to_volume_mock(file_obj2, 2)
    
    file_map = {"file1" : file_obj1, "file2" : file_obj2}
    
    self.classUnderTest.filenames = file_map.keys()
    self.classUnderTest.ignore_malfunc = False
    
    self.classUnderTest.ravebdb = mock.Mock()
    
    self.classUnderTest.ravebdb.get_rave_object.side_effect = lambda x : file_map[x]
    mock_ispolarscan.return_value = False
    mock_ispolarvolume.return_value = True
        
    objects, nodes, how_tasks, all_files_malfunc = self.classUnderTest.fetch_objects()
        
    get_rave_object_calls = [] 
    for file_name in file_map.keys():
      get_rave_object_calls.append(mock.call(file_name))
      self.assertTrue(file_name in objects)
      self.assertEqual(objects[file_name], file_map[file_name])
    
    self.classUnderTest.ravebdb.get_rave_object.assert_has_calls(get_rave_object_calls)
    
    self.assertFalse(all_files_malfunc)

    nodes_list = nodes.replace("'", "").split(",")
    expected_nodes = [self.sources.get(file_obj1.source), 
                      self.sources.get(file_obj2.source)]
    self.assertEqual(set(nodes_list), set(expected_nodes))

    how_task_list = how_tasks.split(",")
    expected_how_tasks = [scan1.getAttribute("how/task"), 
                          scan2.getAttribute("how/task"),
                          scan3.getAttribute("how/task")]
    self.assertEqual(set(how_task_list), set(expected_how_tasks))

  @mock.patch('_polarvolume.isPolarVolume')
  @mock.patch('_polarscan.isPolarScan')
  def test_fetch_objects_volume__vol_with_one_scan_malfunc(self, mock_ispolarscan, mock_ispolarvolume):
    file_obj1 = self.setup_default_volume_mock(0)
    scan1 = self.add_default_scan_to_volume_mock(file_obj1, 0, malfunc=False)

    file_obj2 = self.setup_default_volume_mock(1)
    scan2 = self.add_default_scan_to_volume_mock(file_obj2, 1, malfunc=True)
    scan3 = self.add_default_scan_to_volume_mock(file_obj2, 2, malfunc=False)
    
    file_map = {"file1" : file_obj1, "file2" : file_obj2}
    
    self.classUnderTest.filenames = file_map.keys()
    self.classUnderTest.ignore_malfunc = True
    
    self.classUnderTest.ravebdb = mock.Mock()
    
    self.classUnderTest.ravebdb.get_rave_object.side_effect = lambda x : file_map[x]
    mock_ispolarscan.return_value = False
    mock_ispolarvolume.return_value = True
        
    objects, nodes, how_tasks, all_files_malfunc = self.classUnderTest.fetch_objects()
        
    get_rave_object_calls = [] 
    for file_name in file_map.keys():
      get_rave_object_calls.append(mock.call(file_name))
      
      self.assertTrue(file_name in objects)
      self.assertEqual(objects[file_name], file_map[file_name])
    
    self.classUnderTest.ravebdb.get_rave_object.assert_has_calls(get_rave_object_calls)

    self.assertFalse(all_files_malfunc)
    
    nodes_list = nodes.replace("'", "").split(",")
    expected_nodes = [self.sources.get(file_obj1.source), 
                      self.sources.get(file_obj2.source)]
    self.assertEqual(set(nodes_list), set(expected_nodes))

    how_task_list = how_tasks.split(",")
    expected_how_tasks = [scan1.getAttribute("how/task"),
                          scan3.getAttribute("how/task")]
    self.assertEqual(set(how_task_list), set(expected_how_tasks))
    
    self.assertFalse(scan2 in file_obj2.scans)
    
  @mock.patch('_polarvolume.isPolarVolume')
  @mock.patch('_polarscan.isPolarScan')
  def test_fetch_objects_volume__vol_with_all_scans_malfunc(self, mock_ispolarscan, mock_ispolarvolume):
    file_obj1 = self.setup_default_volume_mock(0)
    scan1 = self.add_default_scan_to_volume_mock(file_obj1, 0, malfunc=False)

    file_obj2 = self.setup_default_volume_mock(1)
    scan2 = self.add_default_scan_to_volume_mock(file_obj2, 1, malfunc=True)
    scan3 = self.add_default_scan_to_volume_mock(file_obj2, 2, malfunc=True)
    
    file_map = {"file1" : file_obj1, "file2" : file_obj2}
    
    self.classUnderTest.filenames = file_map.keys()
    self.classUnderTest.ignore_malfunc = True
    
    self.classUnderTest.ravebdb = mock.Mock()
    
    self.classUnderTest.ravebdb.get_rave_object.side_effect = lambda x : file_map[x]
    mock_ispolarscan.return_value = False
    mock_ispolarvolume.return_value = True
        
    objects, nodes, how_tasks, all_files_malfunc = self.classUnderTest.fetch_objects()
        
    get_rave_object_calls = [] 
    for file_name in file_map.keys():
      get_rave_object_calls.append(mock.call(file_name))
      
    self.assertEqual(len(objects), 1)
    self.assertEqual(objects["file1"], file_map["file1"])
    
    self.classUnderTest.ravebdb.get_rave_object.assert_has_calls(get_rave_object_calls)
    
    self.assertFalse(all_files_malfunc)
    
    nodes_list = nodes.replace("'", "").split(",")
    expected_nodes = [self.sources.get(file_obj1.source)]
    self.assertEqual(set(nodes_list), set(expected_nodes))

    how_task_list = how_tasks.split(",")
    expected_how_tasks = [scan1.getAttribute("how/task")]
    self.assertEqual(set(how_task_list), set(expected_how_tasks))
    
    self.assertFalse(scan2 in file_obj2.scans)
    self.assertFalse(scan3 in file_obj2.scans)
    
  @mock.patch('_polarvolume.isPolarVolume')
  @mock.patch('_polarscan.isPolarScan')
  def test_fetch_objects_volume__two_vols_with_all_scans_malfunc(self, mock_ispolarscan, mock_ispolarvolume):
    file_obj1 = self.setup_default_volume_mock(0)
    scan1 = self.add_default_scan_to_volume_mock(file_obj1, 0, malfunc=True)

    file_obj2 = self.setup_default_volume_mock(1)
    scan2 = self.add_default_scan_to_volume_mock(file_obj2, 1, malfunc=True)
    scan3 = self.add_default_scan_to_volume_mock(file_obj2, 2, malfunc=True)
    
    file_map = {"file1" : file_obj1, "file2" : file_obj2}
    
    self.classUnderTest.filenames = file_map.keys()
    self.classUnderTest.ignore_malfunc = True
    
    self.classUnderTest.ravebdb = mock.Mock()
    
    self.classUnderTest.ravebdb.get_rave_object.side_effect = lambda x : file_map[x]
    mock_ispolarscan.return_value = False
    mock_ispolarvolume.return_value = True
        
    objects, nodes, how_tasks, all_files_malfunc = self.classUnderTest.fetch_objects()
        
    get_rave_object_calls = [] 
    for file_name in file_map.keys():
      get_rave_object_calls.append(mock.call(file_name))
      
    self.assertEqual(len(objects), 0)
    
    self.classUnderTest.ravebdb.get_rave_object.assert_has_calls(get_rave_object_calls)
    
    self.assertTrue(all_files_malfunc)
    
    self.assertEqual(nodes, "")
    
    self.assertEqual(how_tasks, "")

    self.assertFalse(scan1 in file_obj1.scans)
    self.assertFalse(scan2 in file_obj2.scans)
    self.assertFalse(scan3 in file_obj2.scans)

  def test_quality_control_objects(self):
    o1 = object()
    o2 = object()
    detectors = ["qc.check.1","qc.check.2"]
    self.classUnderTest.detectors=detectors
    self.classUnderTest.reprocess_quality_field = True

    self.qc_check_1_mock.process.side_effect = lambda x,y: ({o1:o1,o2:o2}[x], [detectors[0]])
    self.qc_check_1_mock.algorithm.return_value = None
    self.qc_check_2_mock.process.side_effect = lambda x,y: ({o1:o1,o2:o2}[x], [detectors[1]])
    self.qc_check_2_mock.algorithm.return_value = None
    
    result, algorithm, qfields = self.classUnderTest.quality_control_objects({"s1.h5":o1,"s2.h5":o2})
    
    expected_qc_check_1_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True), mock.call.algorithm()]
    expected_qc_check_2_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True), mock.call.algorithm()]
    
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    self.assertTrue(expected_qc_check_2_calls == self.qc_check_2_mock.mock_calls)
    
    self.assertTrue(isinstance(result,dict))
    self.assertTrue(2 == len(result))
    self.assertTrue(result["s1.h5"] == o1)
    self.assertTrue(result["s2.h5"] == o2)
    self.assertTrue(algorithm == None)
    self.assertEquals(detectors, qfields, "Wrong qfields returned from quality_control_objects")
    

  def test_quality_control_objects_algorithm_on_first(self):
    o1 = object()
    o2 = object()
    a1 = object()
    detectors = ["qc.check.1","qc.check.2"]
    self.classUnderTest.detectors=detectors
    self.classUnderTest.reprocess_quality_field = True

    self.qc_check_1_mock.process.side_effect = lambda x,y: ({o1:o1,o2:o2}[x], [detectors[0]])
    self.qc_check_1_mock.algorithm.return_value = a1
    self.qc_check_2_mock.process.side_effect = lambda x,y: ({o1:o1,o2:o2}[x], [detectors[1]])
    self.qc_check_2_mock.algorithm.return_value = None
    
    result, algorithm, qfields = self.classUnderTest.quality_control_objects({"s1.h5":o1,"s2.h5":o2})
    
    expected_qc_check_1_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True), mock.call.algorithm()]
    expected_qc_check_2_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True), mock.call.algorithm()]
    
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    self.assertTrue(expected_qc_check_2_calls == self. qc_check_2_mock.mock_calls)
    
    self.assertTrue(isinstance(result,dict))
    self.assertTrue(2 == len(result))
    self.assertTrue(result["s1.h5"] == o1)
    self.assertTrue(result["s2.h5"] == o2)
    self.assertTrue(algorithm == a1)
    self.assertEquals(detectors, qfields, "Wrong qfields returned from quality_control_objects")


  def test_quality_control_objects_processor_returns_both_object_and_algorithm(self):
    o1 = object()
    o2 = object()
    a2 = object()
    detectors = ["qc.check.1","qc.check.2"]
    self.classUnderTest.detectors=detectors 
    self.classUnderTest.reprocess_quality_field = True
    
    self.qc_check_1_mock.process.side_effect = lambda x,y: ({o1:o1,o2:o2}[x], [detectors[0]])
    self.qc_check_1_mock.algorithm.return_value = None
    self.qc_check_2_mock.process.side_effect = lambda x,y: ({o1:o1,o2:(o2,a2)}[x], [detectors[1]])
    self.qc_check_2_mock.algorithm.return_value = None
    
    result, algorithm, qfields = self.classUnderTest.quality_control_objects({"s1.h5":o1,"s2.h5":o2})
    
    expected_qc_check_1_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True), mock.call.algorithm()]
    expected_qc_check_2_calls = [mock.call.process(o1,True), mock.call.algorithm(),mock.call.process(o2,True)]
    
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    self.assertTrue(expected_qc_check_2_calls == self. qc_check_2_mock.mock_calls)
    
    self.assertTrue(isinstance(result,dict))
    self.assertTrue(2 == len(result))
    self.assertTrue(result["s1.h5"] == o1)
    self.assertTrue(result["s2.h5"] == o2)
    self.assertTrue(algorithm == a2)
    self.assertEquals(detectors, qfields, "Wrong qfields returned from quality_control_objects")
    
  def test_quality_control_objects_plugin_returns_only_obj(self):
    o1 = object()
    o2 = object()
    detectors = ["qc.check.1","qc.check.2"]
    self.classUnderTest.detectors=detectors
    self.classUnderTest.reprocess_quality_field = True

    self.qc_check_1_mock.process.side_effect = lambda x,y: ({o1:o1,o2:o2}[x])
    self.qc_check_1_mock.algorithm.return_value = None
    self.qc_check_1_mock.getQualityFields.return_value = [detectors[0]]
    self.qc_check_2_mock.process.side_effect = lambda x,y: ({o1:o1,o2:o2}[x], [detectors[1]])
    self.qc_check_2_mock.algorithm.return_value = None
    
    result, algorithm, qfields = self.classUnderTest.quality_control_objects({"s1.h5":o1,"s2.h5":o2})
    
    expected_qc_check_1_calls = [mock.call.process(o1,True), mock.call.getQualityFields(), mock.call.algorithm(), mock.call.process(o2,True), mock.call.getQualityFields(), mock.call.algorithm()]
    expected_qc_check_2_calls = [mock.call.process(o1,True), mock.call.algorithm(), mock.call.process(o2,True), mock.call.algorithm()]
    
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    self.assertTrue(expected_qc_check_2_calls == self.qc_check_2_mock.mock_calls)
    
    self.assertTrue(isinstance(result,dict))
    self.assertTrue(2 == len(result))
    self.assertTrue(result["s1.h5"] == o1)
    self.assertTrue(result["s2.h5"] == o2)
    self.assertTrue(algorithm == None)
    self.assertEquals(detectors, qfields, "Wrong qfields returned from quality_control_objects")
    
    
  def test_set_product_from_string(self):
    prods = [("ppi", _rave.Rave_ProductType_PPI),
             ("cappi", _rave.Rave_ProductType_CAPPI),
             ("pcappi", _rave.Rave_ProductType_PCAPPI),
             ("pmax", _rave.Rave_ProductType_PMAX),
             ("max", _rave.Rave_ProductType_MAX)]

    for p in prods:
      self.classUnderTest.set_product_from_string(p[0])
      self.assertEquals(p[1], self.classUnderTest.product)

  def test_set_product_from_string_invalid(self):
    try:
      self.classUnderTest.set_product_from_string("nisse")
      self.fail("Expected ValueError")
    except ValueError:
      pass

  def test_set_method_from_string(self):
    methods = [("NEAREST_RADAR", _pycomposite.SelectionMethod_NEAREST),
               ("HEIGHT_ABOVE_SEALEVEL", _pycomposite.SelectionMethod_HEIGHT)]
    for m in methods:
      self.classUnderTest.set_method_from_string(m[0])
      self.assertEquals(m[1], self.classUnderTest.selection_method)
  
  def test_set_method_from_string_invalid(self):
    try:
      self.classUnderTest.set_method_from_string("nisse")
      self.fail("Expected ValueError")
    except ValueError:
      pass

  def test_strToNumber(self):
    self.assertEquals(1.5, self.classUnderTest._strToNumber("1.5"), 4)
    self.assertEquals(1, self.classUnderTest._strToNumber("1"))
    self.assertEquals(1.0, self.classUnderTest._strToNumber("1.0"), 4)

  def test_strToNumber_preserveValue(self):
    self.assertEquals(1.5, self.classUnderTest._strToNumber(1.5), 4)
    self.assertEquals(1, self.classUnderTest._strToNumber(1))
