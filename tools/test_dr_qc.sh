#!/bin/sh
############################################################
# Description: Script that performs the actual unit tests
#
# Author(s):   Anders Henja and Daniel Michelson
#
# Copyright:   Swedish Meteorological and Hydrological Institute, 2009   
#              The Crown (i.e. Her Majesty the Queen in Right of Canada), 2017
#
# History:     2009-06-15 Created by Anders Henja
#              2017-08-25 Modified by Daniel Michelson
############################################################
SCRFILE=`python -c "import os;print(os.path.abspath(\"$0\"))"`
SCRIPTPATH=`dirname "$SCRFILE"`

RES=255

if [ $# -gt 0 -a "$1" = "alltest" ]; then
  "$SCRIPTPATH/run_python_script.sh" "${SCRIPTPATH}/../test/pytest/DR_QCFullTestSuite.py" "${SCRIPTPATH}/../test/pytest"
  RES=$?
else
  "$SCRIPTPATH/run_python_script.sh" "${SCRIPTPATH}/../test/pytest/DR_QCTestSuite.py" "${SCRIPTPATH}/../test/pytest"
  RES=$?
fi

exit $RES
