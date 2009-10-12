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

DEF_MK_FILE="${SCRIPTPATH}/../librave/def.mk"

if [ ! -f "${DEF_MK_FILE}" ]; then
  echo "configure has not been run"
  exit 255
fi

RESULT=0

# RUN THE PYTHON TESTS
HLHDF_MKFFILE=`fgrep HLHDF_HLDEF_MK_FILE "${DEF_MK_FILE}" | sed -e"s/\(HLHDF_HLDEF_MK_FILE=[ \t]*\)//"`

# Get HDF5s ld path from hlhdfs mkf file
HDF5_LDPATH=`fgrep HDF5_LIBDIR "${HLHDF_MKFFILE}" | sed -e"s/\(HDF5_LIBDIR=[ \t]*-L\)//"`

# Get HLHDFs libpath from raves mkf file
HLHDF_LDPATH=`fgrep HLHDF_LIB_DIR "${DEF_MK_FILE}" | sed -e"s/\(HLHDF_LIB_DIR=[ \t]*\)//"`

BNAME=`python -c 'from distutils import util; import sys; print "lib.%s-%s" % (util.get_platform(), sys.version[0:3])'`

RBPATH="${SCRIPTPATH}/../build/${BNAME}"
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
    export DYLD_LIBRARY_PATH="${RAVE_LDPATH}:${HLHDF_LDPATH}:${HDF5_LDPATH}:${LD_LIBRARY_PATH}"
  else
    export DYLD_LIBRARY_PATH="${RAVE_LDPATH}:${HLHDF_LDPATH}:${HDF5_LDPATH}"
  fi
else
  if [ "$LD_LIBRARY_PATH" != "" ]; then
    export LD_LIBRARY_PATH="${RAVE_LDPATH}:${HLHDF_LDPATH}:${HDF5_LDPATH}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${RAVE_LDPATH}:${HLHDF_LDPATH}:${HDF5_LDPATH}"
  fi
fi

export RAVEPATH="${HLHDF_LDPATH}:${RBPATH}"

if test "${PYTHONPATH}" != ""; then
  export PYTHONPATH="${RAVEPATH}:${PYTHONPATH}"
else
  export PYTHONPATH="${RAVEPATH}"
fi

cd "${SCRIPTPATH}/../test/pytest"
python RaveTest.py
VAL=$?
if [ $VAL != 0 ]; then
  RESULT=$VAL
fi

#RUN OTHER ESSENTIAL TESTS

# EXIT WITH A STATUS CODE, 0 == OK, ANY OTHER VALUE = FAIL
exit $RESULT

