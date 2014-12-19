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
  FIXTURE = "fixtures/rave_quality_chain_registry_test.xml"
  classUnderTest = None
  
  def setUp(self):
    self.classUnderTest = rave_quality_chain_registry.rave_quality_chain_registry(self.FIXTURE)
    
  def tearDown(self):
    self.classUnderTest = None

  def test_get_chain_bySource(self):
    chain = self.classUnderTest.get_chain("selek")
    self.assertEquals("selek", chain.source())
    self.assertEquals("insect_detection", chain.category())
    links = chain.links()
    self.assertEquals(2, len(links))
    self.assertEquals("rave-overshooting", links[0].refname())
    self.assertEquals("radvol-att", links[1].refname())

  def test_get_chain_bySourceAndCategory(self):
    chain = self.classUnderTest.get_chain("sekkr","qpe")
    self.assertEquals("sekkr", chain.source())
    self.assertEquals("qpe", chain.category())
    links = chain.links()
    self.assertEquals(1, len(links))
    self.assertEquals("rave-overshooting", links[0].refname())

  def test_get_chain_haveArguments(self):
    chain = self.classUnderTest.get_chain("sekkr","insect_detection")
    arguments = chain.links()[0].arguments()
    self.assertEquals(1, len(arguments))
    self.assertEquals("some sort of value", arguments["something"])

  def test_get_chain_not_found(self):
    try:
      chain = self.classUnderTest.get_chain("sesss")
      self.fail("Expected LookupError")
    except LookupError,e:
      pass

  def test_get_chain_to_many_found(self):
    try:
      chain = self.classUnderTest.get_chain("sekkr")
      self.fail("Expected LookupError")
    except LookupError,e:
      pass

  def test_get_chain_not_found_invalid_category(self):
    try:
      chain = self.classUnderTest.get_chain("sekkr","nisse")
      self.fail("Expected LookupError")
    except LookupError,e:
      pass

  def test_find_chains_bySource(self):
    chains = self.classUnderTest.find_chains("sekkr")
    self.assertEquals(2, len(chains))
    self.assertEquals("sekkr", chains[0].source())
    self.assertEquals("insect_detection", chains[0].category())
    links = chains[0].links()
    self.assertEquals(2, len(links))
    self.assertEquals("rave-overshooting", links[0].refname())
    self.assertEquals(1, len(links[0].arguments()))
    self.assertEquals("some sort of value", links[0].arguments()["something"])
    self.assertEquals("radvol-spike", links[1].refname())
    
    self.assertEquals("sekkr", chains[1].source())
    self.assertEquals("qpe", chains[1].category())
    links = chains[1].links()
    self.assertEquals(1, len(links))
    self.assertEquals("rave-overshooting", links[0].refname())

  def test_find_chains_bySource_nothing_found(self):
    chains = self.classUnderTest.find_chains("sesss")
    self.assertEquals(0, len(chains))

  def test_find_chains_bySourceAndCategory(self):
    chains = self.classUnderTest.find_chains("sekkr", "qpe")
    self.assertEquals(1, len(chains))
    self.assertEquals("sekkr", chains[0].source())
    self.assertEquals("qpe", chains[0].category())
    links = chains[0].links()
    self.assertEquals(1, len(links))
    self.assertEquals("rave-overshooting", links[0].refname())

  def test_find_chains_bySourceAndCategory_nothing_found(self):
    chains = self.classUnderTest.find_chains("sesss", "qpe")
    self.assertEquals(0, len(chains))
    