#!/bin/sh
############################################################
# Description: Script that executes a python script with proper
# settings in this build.
#
# Author(s):   Anders Henja
#
# Copyright:   Swedish Meteorological and Hydrological Institute, 2009
#
# History:  2009-10-22 Created by Anders Henja
############################################################
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

DEF_MK_FILE="${SCRIPTPATH}/../def.mk"

if [ ! -f "${DEF_MK_FILE}" ]; then
  echo "configure has not been run"
  exit 255
fi

RESULT=0

# Identify python version
PYTHON_BIN=`fgrep PYTHON_BIN "${DEF_MK_FILE}" | sed -e "s/\(PYTHON_BIN=[ \t]*\)//"`
if [ "$PYTHON_BIN" = "" ]; then
  PYTHON_BIN=python
fi


# RUN THE PYTHON TESTS
HLHDF_MKFFILE=`fgrep HLHDF_HLDEF_MK_FILE "${DEF_MK_FILE}" | sed -e"s/\(HLHDF_HLDEF_MK_FILE=[ \t]*\)//"`

# Get HDF5s ld path from hlhdfs mkf file
HDF5_LDPATH=`fgrep HDF5_LIBDIR "${HLHDF_MKFFILE}" | sed -e"s/\(HDF5_LIBDIR=[ \t]*-L\)//"`

# Get HLHDFs libpath from raves mkf file
HLHDF_LDPATH=`fgrep HLHDF_LIB_DIR "${DEF_MK_FILE}" | sed -e"s/\(HLHDF_LIB_DIR=[ \t]*\)//"`

PROJ_LDPATH=`fgrep PROJ_LIB_DIR "${DEF_MK_FILE}" | sed -e"s/\(PROJ_LIB_DIR=[ \t]*\)//" | sed -e"s/-L//"`

RBPATH="${SCRIPTPATH}/../Lib:${SCRIPTPATH}/../modules"
RAVE_LDPATH="${SCRIPTPATH}/../librave/tnc:${SCRIPTPATH}/../librave/toolbox:${SCRIPTPATH}/../librave/pyapi:${SCRIPTPATH}/../librave/scansun:${SCRIPTPATH}/../librave/radvol/lib"
XRUNNERPATH="${SCRIPTPATH}/../test/lib"

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
    export DYLD_LIBRARY_PATH="${RAVE_LDPATH}:${PROJ_LDPATH}:${HLHDF_LDPATH}:${HDF5_LDPATH}:${LD_LIBRARY_PATH}"
  else
    export DYLD_LIBRARY_PATH="${RAVE_LDPATH}:${PROJ_LDPATH}:${HLHDF_LDPATH}:${HDF5_LDPATH}"
  fi
else
  if [ "$LD_LIBRARY_PATH" != "" ]; then
    export LD_LIBRARY_PATH="${RAVE_LDPATH}:${PROJ_LDPATH}:${HLHDF_LDPATH}:${HDF5_LDPATH}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${RAVE_LDPATH}:${PROJ_LDPATH}:${HLHDF_LDPATH}:${HDF5_LDPATH}"
  fi
fi

export RAVEPATH="${HLHDF_LDPATH}:${RBPATH}:${XRUNNERPATH}"

if test "${PYTHONPATH}" != ""; then
  export PYTHONPATH="${RAVEPATH}:${PYTHONPATH}"
else
  export PYTHONPATH="${RAVEPATH}"
fi

# Syntax: run_python_script <pyscript> [<dir> - if script should be executed in a particular directory]

NARGS=$#
PYSCRIPT=
DIRNAME=
if [ $NARGS -eq 1 ]; then
  PYSCRIPT=`$PYTHON_BIN -c "import os;print(os.path.abspath(\"$1\"))"`
elif [ $NARGS -eq 2 ]; then
  PYSCRIPT=`$PYTHON_BIN -c "import os;print(os.path.abspath(\"$1\"))"`
  DIRNAME="$2"
elif [ $NARGS -eq 0 ]; then
  # Do nothing
  PYSCRIPT=
  DIRNAME=
else
  echo "Unknown command"
  exit 255
fi

if [ "$DIRNAME" != "" ]; then
  cd "$DIRNAME"
fi

if [ "$PYSCRIPT" != "" ]; then
  #valgrind -v $PYTHON_BIN "$PYSCRIPT" #--leak-check=full --show-leak-kinds=all 
  $PYTHON_BIN "$PYSCRIPT"
else
  $PYTHON_BIN
fi

VAL=$?
if [ $VAL != 0 ]; then
  RESULT=$VAL
fi

# EXIT WITH A STATUS CODE, 0 == OK, ANY OTHER VALUE = FAIL
exit $RESULT

