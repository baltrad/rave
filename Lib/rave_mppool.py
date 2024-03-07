#!/usr/bin/env python
'''
Copyright (C) 2014- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
'''

## Class for creating non-daemonic process pools.

## @file
## @author Daniel Michelson, SMHI
## @date 2014-09-30


import multiprocessing
import multiprocessing.pool

from rave_defines import RAVE_MULTIPROCESSING_MAX_TASKS_PER_WORKER

## Inherit Process
#  @param Process object
class NonDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NonDaemonContext(type(multiprocessing.get_context())):
    Process = NonDaemonProcess

## We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
#  because the latter is only a wrapper function, not a proper class.
# @param Pool with our NonDaemonicProcess
class RavePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        self._maxtasksperchild = kwargs.pop('maxtasksperchild', RAVE_MULTIPROCESSING_MAX_TASKS_PER_WORKER)
        
        kwargs['context'] = NonDaemonContext()
        kwargs['maxtasksperchild'] = self._maxtasksperchild
        
        super(RavePool, self).__init__(*args, **kwargs)

if __name__ == "__main__":
    pass
