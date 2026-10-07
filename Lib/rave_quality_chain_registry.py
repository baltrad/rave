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

# Standard python libs:
import os
import copy
import xml.etree.ElementTree as ET

from rave_defines import RAVE_QUALITY_CHAIN_REGISTRY_FILE

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
    def __init__(self, registryfile=RAVE_QUALITY_CHAIN_REGISTRY_FILE):
        self.chains = self.load(registryfile)

    def get(self, source, category=None):
        result = self.find_chains(source, category)
        if len(result) == 0:
            result = self.find_chains("default", category)
        if len(result) != 1:
            raise LookupError("Number of found chains != 1")
        return result[0]

    def get_chain(self, source, category=None):
        return self.get(source, category)

    def find_chains(self, source, category=None):
        result = []
        if source in self.chains:
            src_chains = self.chains[source]
            if category is not None:
                for c in src_chains:
                    if c.category() == category:
                        result.append(c)
            else:
                result.extend(src_chains)

        return result

    def load(self, registryfile):
        chainelements = list(ET.parse(registryfile).getroot())
        chains = {}
        for ce in chainelements:
            category = None
            if "category" in ce.attrib:
                category = ce.attrib["category"]
            if ce.tag not in chains:
                chains[ce.tag] = []
            chains[ce.tag].append(self.create_chain(ce.tag, category, ce))
        return chains

    def create_chain(self, source, category, ce):
        links = []
        if "links" in ce.attrib:
            linknames = [item.strip() for item in ce.attrib["links"].split(",")]
            for ln in linknames:
                links.append(link(ln))

        linkelements = ce.findall("link")
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
    if initialized:
        return QUALITY_CHAIN_REGISTRY
    QUALITY_CHAIN_REGISTRY = rave_quality_chain_registry()
    initialized = 1
    return QUALITY_CHAIN_REGISTRY
