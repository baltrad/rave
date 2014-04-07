###########################################################################
# Copyright (C) 2012 Swedish Meteorological and Hydrological Institute, SMHI,
#
# This file is part of RAVE.
#
# RAVE is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# RAVE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------
# 
# Updates the rave_defines.py file with the proper configuration
# @file
# @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
# @date 2012-03-12
###########################################################################

if [ $# -ne 1 ]; then
  echo "Usage: $0 <rave_defines.py>"
  exit 127
fi

if [ ! -f "$1" ]; then
  echo "No such file '$1'"
  exit 127
fi  

if [ -n "$PGF_HOST" ]; then
  sed -i "s/^PGF_HOST.*/PGF_HOST = '$PGF_HOST'/g" "$1"
fi

if [ -n "$PGF_PORT" ]; then
  sed -i "s/^PGF_PORT.*/PGF_PORT = $PGF_PORT/g" "$1"
fi

if [ -n "$LOGPORT" ]; then
  sed -i "s/^LOGPORT.*/LOGPORT = $LOGPORT/g" "$1"
fi


if [ -n "$DEX_SPOE" ]; then
  sed -i "s/^DEX_SPOE.*/DEX_SPOE = \"http:\/\/$DEX_SPOE\/BaltradDex\"/g" "$1"
fi

if [ -n "$CENTER_ID" ]; then
  sed -i "s/^CENTER_ID.*/CENTER_ID = \"ORG:$CENTER_ID\"/g" "$1"
fi

if [ -n "$DEX_NODENAME" ]; then
  sed -i "s/^DEX_NODENAME.*/DEX_NODENAME = \"$DEX_NODENAME\"/g" "$1"
fi

if [ -n "$DEX_PRIVATEKEY" ]; then
  NEW_PKEY=`echo "$DEX_PRIVATEKEY" | sed -e "s;/;\\\\\/;g"`
  sed -i "s/^DEX_PRIVATEKEY.*/DEX_PRIVATEKEY = \"$NEW_PKEY\"/g" "$1"
fi

if [ -n "$BDB_CONFIG_FILE" ]; then
  NEW_CONFFILE=`echo "$BDB_CONFIG_FILE" | sed -e "s;/;\\\\\/;g"`
  sed -i "s/^BDB_CONFIG_FILE.*/BDB_CONFIG_FILE = \"$NEW_CONFFILE\"/g" "$1"
fi
