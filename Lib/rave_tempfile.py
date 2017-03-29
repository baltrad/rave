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
## Redefines the tempfile template for RAVE.

## @file
## @author Daniel Michelson, SMHI
## @date 2010-07-19

import os, tempfile
import rave, rave_defines

## Use shared memory space if available, Linux only. 
if os.path.isdir("/dev/shm") and os.access('/dev/shm', os.W_OK):
    tempfile.tempdir = "/dev/shm/rave"
else:
    tempfile.tempdir = os.path.split(rave.__file__)[0] + '/../tmp'
# Create the direcory if not already there.
if os.path.isdir(tempfile.tempdir) is False:
    os.makedirs(tempfile.tempdir)

## The redefined template.
RAVETEMP = tempfile.tempdir


## RAVE's tempfile constructor.
# @param suffix string, can be e.g. '.h5' for HDF5 files
# @param close string, set to "True" to close the file before continuing.
# The default value is "False".
# @return tuple containing an int containing an OS-level handle to an open
# file as would be returned by os.open(), that can be closed with os.close().,
# and a string containing the absolute pathname to the file.
# NOTE the file is created and opened. In order to prevent too many open files,
# you may have to close this file before continuing.
def mktemp(suffix='', close=False):
    PREFIX = "rave%d-" % os.getpid()
    t = tempfile.mkstemp(prefix=PREFIX, suffix=suffix)
    if eval(close):
        os.close(t[0])
    return t



if __name__ == "__main__":
    print(__doc__)
