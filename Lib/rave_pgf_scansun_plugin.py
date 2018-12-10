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
## Plugin for scanning a polar volume for sun hits, using the RAVE product 
## generation framework.
## Register in the RAVE PGF with: % pgf_registry -a -H http://<host>:<port>/RAVE
## --name=eu.baltrad.beast.generatescansun -m rave_pgf_scansun_plugin -f generate
## -d 'Scans polar volumes for sun hits'

## @file
## @author Daniel Michelson, SMHI
## @date 2011-01-21
##
## @author Ulf E. Nordh, SMHI
## @date 2018-12-06. Updated so that an alternative output path (RAVESCANSUN_OUT) can be used instead of RAVEETC.
##                   The alternative path (RAVESCANSUN_OUT) must be defined in rave_defines.py
##                   The plugin will use RAVESCANSUN_OUT if it is defined, otherwise it will use RAVEETC.  

import os
from rave_defines import RAVEETC
import odim_source
import _scansun
import rave_pgf_logger

# Determining where to write the scansun output files depending on input from rave_defines.py
# If an alternative path, RAVESCANSUN_OUT, is defined in rave_defines.py it is used,
# otherwise the default RAVEETC is used
try:
  from rave_defines import RAVESCANSUN_OUT
  scansun_outputpath = RAVESCANSUN_OUT
except:
  scansun_outputpath = RAVEETC

ravebdb = None
try:
  import rave_bdb
  ravebdb = rave_bdb.rave_bdb()
except:
  pass

HEADER = "#Date    Time        Elevatn Azimuth   ElevSun   AzimSun    N  dBSunFlux   SunMean SunStdd   ZdrMean ZdrStdd  Refl  ZDR\n"
FORMAT = "%08i %010.3f  %7.3f %7.2f   %7.4f  %7.4f    %4i  %9.2f %9.2f  %6.3f %9.2f  %6.3f  %s   %s\n"

logger = rave_pgf_logger.create_logger()

## Convenience function. Gets the NOD identifier from /what/source string.
# Assumes that the NOD is there or can be looked up based on the WMO identifier.
# If WMO isn't there either, then a 'n/a' (not available) is returned.
# @param obj input SCAN or PVOL object
# @return the NOD identifier or 'n/a'
def NODfromSourceString(source):
  S = odim_source.ODIM_Source(source)
  if S.nod: return S.nod
  else:
    try:
      return odim_source.NOD[S.wmo]
    except KeyError:
      return None

## Creates a file name, preferably containing the NOD identifier for that radar. 
# If it can't be found, converts the whole /what/source string to a file string. 
# If its parent directory doesn't exist, it is created.
# @param source string containing the full value of /what/source
# @return string containing the complete path to a file
def Source2File(isource):
    source = NODfromSourceString(isource)
    if not source: source = isource.replace(';','_').replace(',','_').replace(':','-')
    path = os.path.join(scansun_outputpath, "scansun")
    if not os.path.isdir(path): os.makedirs(path)
    return os.path.join(path, source + '.scansun')

## Writes hits to file
# @param source string containing the full value of /what/source
# @param list containing sun hits
def writeHits(source, hits):
    ofstr = Source2File(source)
    fd = open(ofstr, 'a')

    if os.path.getsize(ofstr) == 0:
        fd.write(HEADER)
        
    for hit in hits:
        fd.write(FORMAT % hit)

    fd.close()

## Performs the sun scan.
# @param files list of files to scan. Keep in mind that each file can be
# from a different radar.
# @return nothing
def generate(files, arguments):
    for ifstr in files:
      fname = None
      removeme = None
      if os.path.exists(ifstr):
        fname = ifstr
      else:
        fname = ravebdb.get_file(ifstr)
        removeme = fname

      try:      
        source, hits = _scansun.scansun(fname)
      finally:
        if removeme != None and os.path.exists(removeme):
          os.unlink(removeme)

      if len(hits) > 0:
          writeHits(source, hits)

    return None

if __name__ == '__main__':
    pass
  
