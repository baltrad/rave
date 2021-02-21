'''
Copyright (C) 2010- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.

'''
## Tests the quality_chain plugin
## 
## @file
## @author Anders Henja, SMHI
## @date 2014-12-19
import unittest
import rave_quality_chain_plugin, rave_quality_chain_registry, rave_quality_plugin
import rave_pgf_quality_registry
import _raveio
import mock

class rave_quality_chain_plugin_test(unittest.TestCase):
  PVOL_SELEK_FIXTURE = "fixtures/pvol_selek_20090501T120000Z.h5"
  PVOL_SEKKR_FIXTURE = "fixtures/pvol_sekkr_20090501T120000Z.h5"
  def setUp(self):
    self.quality_chain_registry_mock = mock.Mock(spec=rave_quality_chain_registry.rave_quality_chain_registry)
    self.classUnderTest = rave_quality_chain_plugin.rave_quality_chain_plugin(self.quality_chain_registry_mock)
    self.qc_check_1_mock = mock.Mock(spec=rave_quality_plugin.rave_quality_plugin)
    self.qc_check_2_mock = mock.Mock(spec=rave_quality_plugin.rave_quality_plugin)
    rave_pgf_quality_registry.add_plugin("qc.check.1", self.qc_check_1_mock)
    rave_pgf_quality_registry.add_plugin("qc.check.2", self.qc_check_2_mock)
    
  def tearDown(self):
    self.quality_registry = None
    self.classUnderTest = None
    rave_pgf_quality_registry.remove_plugin("qc.check.1")
    rave_pgf_quality_registry.remove_plugin("qc.check.2")

  def test_getQualityFields(self):
    self.assertEqual("se.smhi.quality.chain.qc", self.classUnderTest.getQualityFields()[0])
    
  def test_process(self):
    pvol = _raveio.open(self.PVOL_SELEK_FIXTURE).object
    qfields = ["qc.check.1", "qc.check.2"]
    link_1 = rave_quality_chain_registry.link(qfields[0])
    link_2 = rave_quality_chain_registry.link(qfields[1])
    a_chain = rave_quality_chain_registry.chain("selek", "default", [link_1, link_2])
    self.quality_chain_registry_mock.get_chain.return_value = a_chain
    self.qc_check_1_mock.process.return_value = pvol, [qfields[0]]
    self.qc_check_1_mock.algorithm.return_value = None
    self.qc_check_2_mock.process.return_value = pvol, [qfields[1]]
    self.qc_check_2_mock.algorithm.return_value = None
    
    _, returned_qfields = self.classUnderTest.process(pvol)
    
    expected_chain_registry_calls = [mock.call.get_chain("selek")]
    expected_qc_check_1_calls = [mock.call.process(pvol,True,"analyze_and_apply",None), mock.call.algorithm()]
    expected_qc_check_2_calls = [mock.call.process(pvol,True,"analyze_and_apply",None), mock.call.algorithm()]
    
    self.assertTrue(expected_chain_registry_calls == self.quality_chain_registry_mock.mock_calls)
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    self.assertTrue(expected_qc_check_2_calls == self. qc_check_2_mock.mock_calls)
    self.assertEqual(returned_qfields, qfields, "Wrong qfields returned from process-function")

  def test_process_false(self):
    pvol = _raveio.open(self.PVOL_SELEK_FIXTURE).object
    qfields = ["qc.check.1", "qc.check.2"]
    link_1 = rave_quality_chain_registry.link(qfields[0])
    link_2 = rave_quality_chain_registry.link(qfields[1])
    a_chain = rave_quality_chain_registry.chain("selek", "default", [link_1, link_2])
    self.quality_chain_registry_mock.get_chain.return_value = a_chain
    self.qc_check_1_mock.process.return_value = pvol, [qfields[0]]
    self.qc_check_1_mock.algorithm.return_value = None
    self.qc_check_2_mock.process.return_value = pvol, [qfields[1]]
    self.qc_check_2_mock.algorithm.return_value = None
    
    _, returned_qfields = self.classUnderTest.process(pvol, False)
    
    expected_chain_registry_calls = [mock.call.get_chain("selek")]
    expected_qc_check_1_calls = [mock.call.process(pvol,False,"analyze_and_apply",None), mock.call.algorithm()]
    expected_qc_check_2_calls = [mock.call.process(pvol,False,"analyze_and_apply",None), mock.call.algorithm()]
    
    self.assertTrue(expected_chain_registry_calls == self.quality_chain_registry_mock.mock_calls)
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    self.assertTrue(expected_qc_check_2_calls == self. qc_check_2_mock.mock_calls)
    self.assertEqual(returned_qfields, qfields, "Wrong qfields returned from process-function")
  
  def test_process_missing_link_processor(self):
    pvol = _raveio.open(self.PVOL_SELEK_FIXTURE).object
    qfield1 = "qc.check.1"
    qfield3 = "qc.check.3"
    link_1 = rave_quality_chain_registry.link(qfield1)
    link_3 = rave_quality_chain_registry.link(qfield3)
    a_chain = rave_quality_chain_registry.chain("selek", "default", [link_1, link_3])
    self.quality_chain_registry_mock.get_chain.return_value = a_chain
    self.qc_check_1_mock.process.return_value = pvol, [qfield1]
    self.qc_check_1_mock.algorithm.return_value = None
    
    obj, returned_qfields = self.classUnderTest.process(pvol)
    
    expected_chain_registry_calls = [mock.call.get_chain("selek")]
    expected_qc_check_1_calls = [mock.call.process(pvol,True,"analyze_and_apply", None), mock.call.algorithm()]
    
    self.assertTrue(expected_chain_registry_calls == self.quality_chain_registry_mock.mock_calls)
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    
    self.assertTrue(pvol == obj)
    self.assertEqual(returned_qfields, [qfield1], "Wrong qfields returned from process-function")
    
  def test_process_no_such_chain(self):
    pvol = _raveio.open(self.PVOL_SEKKR_FIXTURE).object
    self.quality_chain_registry_mock.get_chain.side_effect = LookupError("appp")

    obj, returned_qfields = self.classUnderTest.process(pvol)

    expected_chain_registry_calls = [mock.call.get_chain("sekaa")]

    self.assertTrue(expected_chain_registry_calls == self.quality_chain_registry_mock.mock_calls)
    
    self.assertTrue(pvol == obj)
    self.assertTrue(returned_qfields == [])
    
  def test_process_bad_link_processor(self):
    pvol = _raveio.open(self.PVOL_SELEK_FIXTURE).object
    qfields = ["qc.check.1", "qc.check.2"]
    link_1 = rave_quality_chain_registry.link(qfields[0])
    link_2 = rave_quality_chain_registry.link(qfields[1])
    a_chain = rave_quality_chain_registry.chain("selek", "default", [link_1, link_2])
    self.quality_chain_registry_mock.get_chain.return_value = a_chain
    self.qc_check_1_mock.process.side_effect = AttributeError("appp")
    self.qc_check_2_mock.process.return_value = pvol, qfields[1]
    self.qc_check_2_mock.algorithm.return_value = None
    
    obj, _ = self.classUnderTest.process(pvol)

    expected_chain_registry_calls = [mock.call.get_chain("selek")]
    expected_qc_check_1_calls = [mock.call.process(pvol,True,"analyze_and_apply", None)]
    expected_qc_check_2_calls = [mock.call.process(pvol,True,"analyze_and_apply", None), mock.call.algorithm()]
    
    self.assertTrue(expected_chain_registry_calls == self.quality_chain_registry_mock.mock_calls)
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    self.assertTrue(expected_qc_check_2_calls == self. qc_check_2_mock.mock_calls)
    
    self.assertTrue(pvol == obj)

  def test_process_link_with_args(self):
    pvol = _raveio.open(self.PVOL_SELEK_FIXTURE).object
    link_args={"abc":10,"def":"something"}
    qfield1 = "qc.check.1"
    link_1 = rave_quality_chain_registry.link(qfield1, link_args)
    
    a_chain = rave_quality_chain_registry.chain("selek", "default", [link_1])
    self.quality_chain_registry_mock.get_chain.return_value = a_chain
    self.qc_check_1_mock.process.return_value = pvol, qfield1
    self.qc_check_1_mock.algorithm.return_value = None
    
    obj, _ = self.classUnderTest.process(pvol)

    expected_chain_registry_calls = [mock.call.get_chain("selek")]
    expected_qc_check_1_calls = [mock.call.process(pvol,True,"analyze_and_apply",link_args), mock.call.algorithm()]
    
    self.assertTrue(expected_chain_registry_calls == self.quality_chain_registry_mock.mock_calls)
    self.assertTrue(expected_qc_check_1_calls == self.qc_check_1_mock.mock_calls)
    
    self.assertTrue(pvol == obj)
    