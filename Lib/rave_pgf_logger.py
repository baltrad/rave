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
## Management utilities for the logging system.

## @file
## @author Daniel Michelson, SMHI
## @date 2010-07-24

import os
import logging
import logging.handlers
from rave_defines import LOGFILE, LOGFILESIZE, LOGFILES


## Initializes the system logger.
# @param logger an instance returned by \ref logging.getLogger()
def init_logger(logger):
    logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(LOGFILE,
                                                   maxBytes = LOGFILESIZE,
                                                   backupCount = LOGFILES)
    # This formatter removes the fractions of a second in the time.
#     formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
#                                   '%Y-%m-%d %H:%M:%S %Z')
    # The default formatter contains fractions of a second in the time.
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-7s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Logging system initialized. Starting...")


if __name__ == "__main__":
    print __doc__
