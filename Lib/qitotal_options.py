'''
Copyright (C) 2014- Swedish Meteorological and Hydrological Institute (SMHI)

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
##
# Keeps information on the qitotal options, like site weight and what fields that are included in the qi-total algorithm.
#
# This is no interactive registry, instead you will have to modify the qitotal_options.xml file manually
#

## 
# @file
# @author Anders Henja, SMHI
# @date 2014-03-13

import xml.etree.ElementTree as ET
import os, copy

CONFIG_FILE = os.path.join(os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], 
                                        'config'), 'qitotal_options.xml')

initialized = 0

QITOTAL_REGISTRY = {}

## Atempts to convert a string into a float
# @param sval the string value
# @return: a float value if possible
# @raise ValueError: if it wasn't possible to convert to a float 
def strToFloat(sval):
  result = 0.0
  try:
    result = float(sval)
  except ValueError:
    result = float(int(sval))
  return result

## Relation between a name (how/task quality field name) and a weight.
# 
class qifield_information(object):
  ## Constructor
  # @param name the quality field name
  # @param weight the weight
  def __init__(self, name, weight):
    self._name = name
    self._weight = weight
  
  ##
  # @return: the quality field name
  def name(self):
    return self._name
  
  ##
  # @return: the weight
  def weight(self):
    return self._weight
  
  ##
  # @return: the object representation of self
  def __repr__(self):
    return "qifield '%s' weight = %f"%(self._name, self._weight)

## Information on what quality fields that should be used for generating the qi-total field
#
class qitotal_site_information(object):
  ## Constructor
  # @param nod the site name (like sekkr)
  # @param qifields the quality indicator fields to be used for this qi-total field
  # @param weight the weight that should be used for this site when using the qi-total field in the composite generation
  def __init__(self, nod, qifields=[], weight=100.0):
    super(qitotal_site_information, self).__init__()
    self._nod = nod
    self._qifields = qifields
    self._weight = weight

  ##
  # @return: the site name
  def nod(self):
    return self._nod
  
  ##
  # @return: the qi fields to be used for this qi-total field
  def qifields(self):
    return copy.deepcopy(self._qifields)

  ##
  # @return: the weight to use on this sites qi-field
  def weight(self):
    return self._weight

  ##
  # @return: the string representation of self
  def __repr__(self):
    r = "qitotal_site_information (%s) weight = %f\n  fields:\n"%(self._nod, self._weight)
    for f in self._qifields:
      r = "%s\t%s\n"%(r, f.__repr__())
    return r

  #<qitotal nod="sehud" weight="1.0">
  #  <field name="se.smhi.test.1" weight="0.3"/>
  #  <field name="se.smhi.test.2" weight="0.7"/>
  #</qitotal>
def parse_qitotal_site_information(cfile):
  qitotal_sites = ET.parse(cfile).getroot().findall("qitotal")
  site_information = {}
  
  for site in qitotal_sites:
    nod = site.attrib["nod"]
    weight = 100.0
    fields = []
      
    try:
      w = site.attrib["weight"]
      weight = float(strToFloat(w))
    except Exception as e:
      print(e.__str__())
    
    qfields = site.findall("field")
    for f in qfields:
      fname = f.attrib["name"]
      fweight = 100.0
      try:
        w = f.attrib["weight"]
        fweight = float(strToFloat(w))
      except Exception as e:
        print(e.__str__())
      
      fields.append(qifield_information(fname, fweight))
        
    site_information[nod] = qitotal_site_information(nod, fields, weight)
      
  return site_information
  

def get_global_qitotal_site_information():
  global initialized, QITOTAL_REGISTRY
  if initialized: return QITOTAL_REGISTRY

  QITOTAL_REGISTRY = parse_qitotal_site_information(CONFIG_FILE)
  
  initialized = 1
  return QITOTAL_REGISTRY  

##
# Creates the qitotal site information
#
def get_qitotal_site_information(cfile=None):
  if cfile == None:
    return get_global_qitotal_site_information()
  
  return parse_qitotal_site_information(cfile)

if __name__=="__main__":
  print(get_qitotal_site_information().__repr__())
