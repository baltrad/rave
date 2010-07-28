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
## Functionality for verifying whether arguments given to a product generator
# are those expected.
# Could be extended to perform sanity checks on the values of the arguments.

## @file
## @author Daniel Michelson, SMHI
## @date 2010-07-21

import types


## Convenience function for accessing arguments from an Element
# @param elem Element containing arguments
# @param name string containing the type of arguments to access. Must be
# "strings", "ints", "floats", or "sequences".
def get_args(elem, name):
    entry = elem.get(name)
    if entry: return entry.split(',')
    else: return []


## Verifies arguments passed to a "generate" algorithm.
# @param args list containing an even number of items in pairs, where the
# first member of each pair is the key and the second is its value.
# @param algorithm_entry Element of a \ref rave_pgf_registry.PGF_Registry object
# containing the algorithm name and its arguments.
# @return boolean True or False
def verify_generate_args(args, algorithm_entry):
    ae_args = algorithm_entry.find('arguments')
    strings = get_args(ae_args, "strings")
    ints = get_args(ae_args, "ints")
    floats = get_args(ae_args, "floats")
    seqs = get_args(ae_args, "sequences")
    All = strings + ints + floats + seqs

    # This loop will raise an IndexError if len(args) is odd.
    for i in range(0, len(args), 2):
        key, value = args[i], args[i+1]
        if key not in All:
            return False
        if key in strings and type(value) != types.StringType:
            return False
        if key in ints and type(value) != types.IntType:
            return False
        if key in floats and type(value) != types.FloatType:
            return False
        if key in seqs and type(value) not in (types.ListType, types.TupleType):
            return False
    return True


if __name__ == "__main__":
    print __doc__
