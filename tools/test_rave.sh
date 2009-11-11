#!/bin/sh
############################################################
# Description: Script that performs the actual unit test of
#   rave.
#
# Author(s):   Anders Henja
#
# Copyright:   Swedish Meteorological and Hydrological Institute, 2009
#
# History:  2009-06-15 Created by Anders Henja
############################################################
SCRFILE=`python -c "import os;print os.path.abspath(\"$0\")"`
SCRIPTPATH=`dirname "$SCRFILE"`

"$SCRIPTPATH/run_python_script.sh" "${SCRIPTPATH}/../test/pytest/RaveTestSuite.py" "${SCRIPTPATH}/../test/pytest"
exit $?
