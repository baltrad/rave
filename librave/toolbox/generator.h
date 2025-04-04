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
#ifndef GENERATOR_H
#define GENERATOR_H

//## C++ interface to composite generator functionality.

//## @file
//## @author Anders Henja and Yngve Einarsson, SMHI
//## @date 2025-04-01
#include "rave_defines.h"

extern "C" {
#include "rave_object.h"
#include "composite.h"
#include "compositefactorymanager.h"
#include "compositegenerator.h"
#include "compositearguments.h"
#include "rave_properties.h"
#include "odim_sources.h"
//#include "rave.h"
}
//import os, sys, math
//import _compositefactorymanager, _compositegenerator, _compositearguments, _raveproperties, _odimsources
//import _rave

//from rave_defines import COMPOSITE_GENERATOR_FILTER_FILENAME, ACQVA_CLUTTERMAP_DIR, ODIM_SOURCE_FILE

class Generator {
public:
    Generator();
    ~Generator();
    void init(std::string generatorfilter);
    CompositeArguments_t * create_arguments();
    Cartesian_t * generate(CompositeArguments_t* arguments);
    void update_arguments_with_prodpar(CompositeArguments_t * arguments, std::string prodpar);
    RaveProperties_t * load_properties();


private:
    float _strToNumber(std::string sval){return std::stof(sval);};
    CompositeFactoryManager_t * _manager;
    CompositeGenerator_t * _generator;
};
#endif
