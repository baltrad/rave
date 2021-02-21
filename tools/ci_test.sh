#!/bin/sh
############################################################
# Description: Script that should be executed from a continous
# integration runner. It is nessecary to point out the proper
# paths to HLHDF oposite of the test_rave.sh script since
# this script should be run whenever HLHDF has been changed.
# It also assumes that the HDF5 libraries are available through
# the path.
#
# Author(s):   Anders Henja
#
# Copyright:   Swedish Meteorological and Hydrological Institute, 2009
#
# History:  2009-10-12 Created by Anders Henja
############################################################
SCRFILE=`python -c "import os;print(os.path.abspath(\"$0\"))"`
SCRIPTPATH=`dirname "$SCRFILE"`

"$SCRIPTPATH/run_python_script.sh" "${SCRIPTPATH}/../test/pytest/RaveXmlTestSuite.py" "${SCRIPTPATH}/../test/pytest"
exit $?
