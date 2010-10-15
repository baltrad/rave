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
## Protocol converter that can be used to translate for example
## a beast generator command into a rave generator command.

## @file
## @author Anders Henja, SMHI
## @date 2010-10-14

import re

## Convenience function for accessing arguments from an Element
# @param elem Element containing arguments
# @param name string containing the type of arguments to access. Must be
# "strings", "ints", "floats", or "sequences".
def get_args(elem, name):
  entry = elem.get(name)
  if entry: return entry.split(',')
  else: return []
    
## Translates beast arguments into rave compatible arguments
# Basically translates --<key>=<value> into <key> <value>
# returns argumentlist
def convert_beast_arguments(algorithm_entry, arguments):
  ae_args = algorithm_entry.find('arguments')
  strings = get_args(ae_args, "strings")
  ints = get_args(ae_args, "ints")
  floats = get_args(ae_args, "floats")
  seqs = get_args(ae_args, "sequences")
  allarguments = strings + ints + floats + seqs
    
  # All beast commands is formulated like --<key>=<value>
  # so we just translate it.
  newargs=[]
  for arg in arguments:
    o = re.match("--([^=]+)=([^$]+)", arg)
    if o != None:
      key = o.group(1)
      value = o.group(2)
      if key in allarguments:
        newargs.append(key)
        if key in strings:
          newargs.append(value)
        elif key in ints:
          newargs.append(int(value))
        elif key in floats:
          newargs.append(float(value))
        else:
          newargs.append(value)
    else:
      o = re.match("--([^$]+)", arg)
      if o != None:
        key = o.group(1)
        if key in allarguments:
          newargs.append(key)
          newargs.append("true")
  
  return newargs

## Translates a command into a rave compatible command
# returns arguments
def convert_arguments(algorithm, algorithm_entry, arguments):
  if algorithm.startswith("eu.baltrad.beast"):
    return convert_beast_arguments(algorithm_entry, arguments)
  return arguments
