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
## Plugin for performing qc-chains on various sources.
## 
## @file
## @author Anders Henja, SMHI
## @date 2014-12-06
import unittest
import os
import rave_quality_chain_registry

class rave_quality_chain_registry_test(unittest.TestCase):
  FIXTURE_XML = "fixtures/rave_quality_chain_registry_test.xml"
  REAL_XML = "../../config/rave_quality_chain_registry.xml"
  classUnderTest = None
  
  def setUp(self):
    self.classUnderTest = rave_quality_chain_registry.rave_quality_chain_registry(self.FIXTURE_XML)
    
  def tearDown(self):
    self.classUnderTest = None

  def test_get_chain_bySource(self):
    chain = self.classUnderTest.get_chain("selek")
    self.assertEqual("selek", chain.source())
    self.assertEqual("insect_detection", chain.category())
    links = chain.links()
    self.assertEqual(2, len(links))
    self.assertEqual("rave-overshooting", links[0].refname())
    self.assertEqual("radvol-att", links[1].refname())

  def test_get_chain_bySourceAndCategory(self):
    chain = self.classUnderTest.get_chain("sekkr","qpe")
    self.assertEqual("sekkr", chain.source())
    self.assertEqual("qpe", chain.category())
    links = chain.links()
    self.assertEqual(1, len(links))
    self.assertEqual("rave-overshooting", links[0].refname())

  def test_get_chain_haveArguments(self):
    chain = self.classUnderTest.get_chain("sekkr","insect_detection")
    arguments = chain.links()[0].arguments()
    self.assertEqual(1, len(arguments))
    self.assertEqual("some sort of value", arguments["something"])

  def test_get_chain_not_found(self):
    try:
      chain = self.classUnderTest.get_chain("sesss")
      self.fail("Expected LookupError")
    except LookupError:
      pass

  def test_get_chain_to_many_found(self):
    try:
      chain = self.classUnderTest.get_chain("sekkr")
      self.fail("Expected LookupError")
    except LookupError:
      pass

  def test_get_chain_not_found_invalid_category(self):
    try:
      chain = self.classUnderTest.get_chain("sekkr","nisse")
      self.fail("Expected LookupError")
    except LookupError:
      pass

  def test_find_chains_bySource(self):
    chains = self.classUnderTest.find_chains("sekkr")
    self.assertEqual(2, len(chains))
    self.assertEqual("sekkr", chains[0].source())
    self.assertEqual("insect_detection", chains[0].category())
    links = chains[0].links()
    self.assertEqual(2, len(links))
    self.assertEqual("rave-overshooting", links[0].refname())
    self.assertEqual(1, len(links[0].arguments()))
    self.assertEqual("some sort of value", links[0].arguments()["something"])
    self.assertEqual("radvol-spike", links[1].refname())
    
    self.assertEqual("sekkr", chains[1].source())
    self.assertEqual("qpe", chains[1].category())
    links = chains[1].links()
    self.assertEqual(1, len(links))
    self.assertEqual("rave-overshooting", links[0].refname())

  def test_find_chains_bySource_nothing_found(self):
    chains = self.classUnderTest.find_chains("sesss")
    self.assertEqual(0, len(chains))

  def test_find_chains_bySourceAndCategory(self):
    chains = self.classUnderTest.find_chains("sekkr", "qpe")
    self.assertEqual(1, len(chains))
    self.assertEqual("sekkr", chains[0].source())
    self.assertEqual("qpe", chains[0].category())
    links = chains[0].links()
    self.assertEqual(1, len(links))
    self.assertEqual("rave-overshooting", links[0].refname())

  def test_find_chains_bySourceAndCategory_nothing_found(self):
    chains = self.classUnderTest.find_chains("sesss", "qpe")
    self.assertEqual(0, len(chains))
    
  def test_real_qualityChainRegistryXml(self):
    realXmlClass = rave_quality_chain_registry.rave_quality_chain_registry(self.REAL_XML)
    
    chains = realXmlClass.find_chains("sekkr", "insect_detection")
    self.assertEqual(1, len(chains))
    self.assertEqual("sekkr", chains[0].source())
    self.assertEqual("insect_detection", chains[0].category())
    
    links = chains[0].links()
    self.assertEqual(2, len(links))
    link1 = links[0]
    self.assertEqual("rave-overshooting", link1.refname())
    arguments = link1.arguments()
    self.assertEqual("some sort of value", arguments["something"])
    link2 = links[1]
    self.assertEqual("radvol-spike", link2.refname())
    