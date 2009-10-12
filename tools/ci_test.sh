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
SCRFILE=`python -c "import os;print os.path.abspath(\"$0\")"`
SCRIPTPATH=`dirname "$SCRFILE"`

RESULT=0
BNAME=`python -c 'from distutils import util; import sys; print "lib.%s-%s" % (util.get_platform(), sys.version[0:3])'`

RBPATH="${SCRIPTPATH}/../build/${BNAME}"
XRUNNERPATH="${SCRIPTPATH}/../test/lib"
RAVE_LDPATH="${SCRIPTPATH}/../librave/transform:${SCRIPTPATH}/../librave/pyapi"

# Special hack for mac osx.
ISMACOS=no
case `uname -s` in
 Darwin*)
   ISMACOS=yes
   ;;
 darwin*)
   ISMACOS=yes
   ;;
esac

if [ "x$ISMACOS" = "xyes" ]; then
  if [ "$DYLD_LIBRARY_PATH" != "" ]; then
    export DYLD_LIBRARY_PATH="${RAVE_LDPATH}:${DYLD_LIBRARY_PATH}"
  else
    export DYLD_LIBRARY_PATH="${RAVE_LDPATH}"
  fi
else
  if [ "$LD_LIBRARY_PATH" != "" ]; then
    export LD_LIBRARY_PATH="${RAVE_LDPATH}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${RAVE_LDPATH}"
  fi
fi

export RAVEPATH="${RBPATH}:${XRUNNERPATH}"

if test "${PYTHONPATH}" != ""; then
  export PYTHONPATH="${RAVEPATH}:${PYTHONPATH}"
else
  export PYTHONPATH="${RAVEPATH}"
fi

cd "${SCRIPTPATH}/../test/pytest"
python RaveXmlTestSuite.py
VAL=$?
if [ $VAL != 0 ]; then
  RESULT=$VAL
fi

#RUN OTHER ESSENTIAL TESTS

# EXIT WITH A STATUS CODE, 0 == OK, ANY OTHER VALUE = FAIL
exit $RESULT
