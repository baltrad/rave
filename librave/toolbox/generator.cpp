/*
Copyright (C) 2025- Swedish Meteorological and Hydrological Institute (SMHI)

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

*/
//## C++ implementation to composite generator functionality.

//## @file
//## @author Anders Henja and Yngve Einarsson, SMHI
//## @date 2025-04-01
//import os, sys, math
//import _compositefactorymanager, _compositegenerator, _compositearguments, _raveproperties, _odimsources
//import _rave

#include "generator.h"
#include "rave_defines.h"

extern "C" {
#include "rave_object.h"
#include "rave_debug.h"
#include "compositefactorymanager.h"
#include "compositegenerator.h"
#include "compositearguments.h"
#include "rave_properties.h"
#include "odim_sources.h"
}

#include <sstream>
#include <cstring>

Generator::Generator() {
    _manager=NULL;
    _generator=NULL;
};
Generator::~Generator() {
    if (_manager != NULL) {
        RAVE_OBJECT_RELEASE(_manager);
    }
    if (_generator != NULL) {
        RAVE_OBJECT_RELEASE(_generator);
    }
};

int Generator::init(std::string generatorfilter){
    _manager = (CompositeFactoryManager_t *)RAVE_OBJECT_NEW(&CompositeFactoryManager_TYPE);
    if (_manager == NULL) {
        RAVE_CRITICAL0("Could not create CompositeFactoryManager!");
        return 0;
    }
    _generator = CompositeGenerator_create(_manager, generatorfilter.c_str());
    if (_generator == NULL) {
        RAVE_OBJECT_RELEASE(_manager);
        RAVE_CRITICAL0("Could not create CompositeGenerator!");
        return 0;
    }
    RaveProperties_t * the_props = load_properties();
    CompositeGenerator_setProperties(_generator, the_props);
    RAVE_OBJECT_RELEASE(the_props);
    return 1;

};

CompositeArguments_t * Generator::create_arguments(){
    return (CompositeArguments_t *)RAVE_OBJECT_NEW(&CompositeArguments_TYPE);
};

Cartesian_t * Generator::generate(CompositeArguments_t* arguments){
    return CompositeGenerator_generate(_generator, arguments);
};

void Generator::update_arguments_with_prodpar(CompositeArguments_t * arguments, std::string prodpar) {
    if (prodpar.length() != 0) {
        if ((!strcmp(CompositeArguments_getProduct(arguments),"CAPPI")) || (!strcmp(CompositeArguments_getProduct(arguments),"PCAPPI"))) {
            CompositeArguments_setHeight(arguments, _strToNumber(prodpar));
        }
        else if (!strcmp(CompositeArguments_getProduct(arguments),"PMAX")) {
            std::vector<std::string> pp;
            std::istringstream f(prodpar);
            std::string s;
            while (getline(f, s, ',')) {
                pp.push_back(s);
            }
            // FIXME: Do we need to strip withspaces?
            if (pp.size() == 2) {
                CompositeArguments_setHeight(arguments, _strToNumber(pp[0]));
                CompositeArguments_setRange(arguments, _strToNumber(pp[1]));
            } else if (pp.size() == 1) {
                CompositeArguments_setHeight(arguments, _strToNumber(pp[0]));
            }
        }
        else if (!strcmp(CompositeArguments_getProduct(arguments),"PPI")) {
            float v = _strToNumber(prodpar);
            CompositeArguments_setElevationAngle(arguments, v * M_PI / 180.0);
        }
    }
};

RaveProperties_t * Generator::load_properties(){
    std::string composite_generator_property_file_path = _RAVEROOT + COMPOSITE_GENERATOR_PROPERTY_FILE;
    RaveProperties_t * properties = RaveProperties_load(composite_generator_property_file_path.c_str());
    if (properties == NULL) {
        // Json file missing or not readable.
        properties = (RaveProperties_t *)RAVE_OBJECT_NEW(&RaveProperties_TYPE);
        if (properties == NULL) {
            RAVE_CRITICAL0("Could not create properties!");
            return NULL;
        }
        std::string acqva_cluttermap_dir_path = _RAVEROOT + ACQVA_CLUTTERMAP_DIR;
        RaveValue_t* value = RaveValue_createString(acqva_cluttermap_dir_path.c_str());
        RaveProperties_set(properties, "rave.acqva.cluttermap.dir", value);
        RAVE_OBJECT_RELEASE(value);
    }
    OdimSources_t* odims = RaveProperties_getOdimSources(properties);
    if (odims==NULL) {
        std::string odim_source_file_path = _RAVEROOT + ODIM_SOURCE_FILE;
        RaveProperties_setOdimSources(properties, OdimSources_load(odim_source_file_path.c_str()));
        RAVE_OBJECT_RELEASE(odims);
    }
    return properties;
};

