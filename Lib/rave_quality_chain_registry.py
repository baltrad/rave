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
## Registry for reading quality control chain configurations
## 
## @file
## @author Anders Henja, SMHI
## @date 2014-12-06

import xml.etree.ElementTree as ET
import os, copy

CONFIG_FILE = os.path.join(os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], 
                                        'config'), 'rave_quality_chain_registry.xml')

initialized = 0

class link(object):
  def __init__(self, refname, arguments=None):
    self._refname = refname
    self._arguments = arguments

  def refname(self):
    return self._refname
  
  def arguments(self):
    return self._arguments

class chain(object):
  def __init__(self, source, category, links=[]):
    self._source = source
    self._category = category
    self._links = links
  
  def source(self):
    return self._source
  
  def category(self):
    return self._category
  
  def links(self):
    return self._links

class rave_quality_chain_registry(object):
  def __init__(self, registryfile=CONFIG_FILE):
    self.chains = self.load(registryfile)
    
  def get_chain(self, source, category=None):
    result = self.find_chains(source, category)
    if len(result) != 1:
      raise LookupError("Number of found chains != 1")
    return result[0]

  def find_chains(self, source, category=None):
    result = []
    if source in self.chains:
      src_chains = self.chains[source]
      if category != None:
        for c in src_chains:
          if c.category() == category:
            result.append(c)
      else:
        result.extend(src_chains)
      
    return result
  
  def load(self, registryfile):
    chainelements = ET.parse(registryfile).getroot().findall("chain")
    chains={}
    for ce in chainelements:
      source = ce.attrib["source"]
      category = ce.attrib["category"]
      if source not in chains:
        chains[source] = []
      chains[source].append(self.create_chain(source,category,ce))
    return chains

  def create_chain(self, source, category, ce):
    linkelements=ce.findall("link")
    links = []
    for le in linkelements:
      refname = le.attrib["ref"]
      links.append(link(refname, self.create_link_arguments(le)))
    return chain(source, category, links)

  def create_link_arguments(self, le):
    result = {}
    linkargument = le.find("arguments")
    if linkargument is not None:
      linkarguments = linkargument.findall("argument")
      for la in linkarguments:
        result[la.attrib["name"]] = la.text
    return result
      
def get_global_registry():
  global initialized, QUALITY_CHAIN_REGISTRY
  if initialized: return QUALITY_CHAIN_REGISTRY
  QUALITY_CHAIN_REGISTRY=rave_quality_chain_registry()
  initialized = 1
  return QUALITY_CHAIN_REGISTRY

    