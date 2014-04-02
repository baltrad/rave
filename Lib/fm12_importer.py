'''
Copyright (C) 2011 - Swedish Meteorological and Hydrological Institute (SMHI)

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
## Class used for importing fm12 synop files into the rave database

## @file
## @author Anders Henja, SMHI
## @date 2013-11-12
import sys, getopt, os
import logging
from rave_dom import observation

#logger = logging.getLogger("fm12_importer")

class fm12_importer(object):
  def __init__(self, dburi="postgresql://baltrad:baltrad@127.0.0.1/baltrad", monitored_dir=None, pidfile="/tmp/fm12_importer.pid"):
    self.dburi = dburi
    self.pidfile = pidfile
    self.monitored_dir = monitored_dir
    self.janitor = False
    self.docatchup = False
    self._init_logger()

  ## Initializes the default logger    
  def _init_logger(self):
    self._logger = logging.getLogger("fm12_importer")
    self._logger.setLevel(logging.ERROR)
    
  ## Sets an external logger if it is necessary
  def set_logger(self, logger):
    self._logger = logger

  ## Determines whether the daemon is running, based on the PID in the PIDFILE.
  # @return True if the daemon is running, otherwise False
  def isalive(self):
    if os.path.isfile(self.pidfile):
      fd = open(self.pidfile)
      c = fd.read()
      fd.close()
      try:
        s.getpgid(int(c))
        return True
      except:
        return False
    else:
      return False
    
  def killme(self):
    import signal
    try:
      self._logger.info("Terminating fm12_importer.")
      fd = open(self.pidfile)
      c = fd.read()
      fd.close()
      try:
        os.kill(int(c), signal.SIGHUP)
      except:
        os.kill(int(c), signal.SIGKILL)
    except:
      self._logger.warn("Failed to kill daemon. Check pid!")
      print "Could not kill daemon. Check pid."
    finally:
      os.remove(self.pidfile)
    
  def catchup(self):
    import glob
    
    if self.monitored_dir != None and self.docatchup:
      while 1:
        flist = glob.glob(os.path.join(self.monitored_dir, '*'))
        if len(flist) == 0:
          break
        else:
            for fstr in flist:
              self.import_file(fstr)
              if self.janitor:
                try:
                  os.unlink(fstr)
                except:
                  self._logger.exception("Failed to remove '%s'")
        if not self.janitor:
          break # If no janitor we will never leave this loop which is a bit hazardous
    
  def convert_fm12_to_obs(self, fm12obs, dbinstance):
    wmost = dbinstance.get_station(fm12obs.station)
    if wmost == None:
      self._logger.info("No station named %s"%fm12obs.station)
      return None
    else:
      self._logger.debug("Reading station information %s"%fm12obs.station)
    result = observation(wmost.stationnumber, wmost.country, fm12obs.type, fm12obs.date, fm12obs.time, wmost.longitude, wmost.latitude)
    result.visibility = fm12obs.visibility
    result.windtype = fm12obs.windtype
    result.cloudcover = fm12obs.cloudcover
    result.winddirection = fm12obs.winddirection
    result.windspeed = fm12obs.windspeed
    result.temperature = fm12obs.temperature
    result.dewpoint = fm12obs.dewpoint
    result.relativehumidity = fm12obs.relativehumidity
    result.pressure = fm12obs.pressure
    result.sea_lvl_pressure = fm12obs.sea_lvl_pressure
    result.pressure_change = fm12obs.pressure_change
    result.liquid_precipitation = fm12obs.liquid_precipitation
    result.accumulation_period = fm12obs.accumulation_period

    return result

  def convert_fm12list_to_obs(self, fm12list, dbinstance):
    result=[]
    for fm12obs in fm12list:
      nobs = self.convert_fm12_to_obs(fm12obs, dbinstance)
      if nobs != None:
        result.append(nobs)
      
    return result

  def _import_file_internal(self, fname):
    from rave_fm12 import fm12_parser
    import rave_dom_db
    
    try:
      db = rave_dom_db.create_db(self.dburi, True)
    
      parser = fm12_parser()
    
      result = parser.parse(fname)

      olist = self.convert_fm12list_to_obs(result, db)
    
      db.merge(olist)
    
      self._logger.info("Imported %d observations"%len(olist))
    except Exception, e:
      self._logger.exception("Failed to import %s"%fname)      

  def import_file(self, fname):
    if os.path.isfile(fname):
      self._logger.info("Trying to import file with size %d"%os.path.getsize(fname))
      if os.path.getsize(fname) != 0:
        self._import_file_internal(fname)
      try:
        if self.janitor:
          os.unlink(fname)
      except Exception, e:
        self._logger.exception("Janitor: Failed to remove file %s"%fname)
          

def get_dburi_from_conf(configfile, propname = "rave.db.uri"):
    properties = {}
    try:
      with open(configfile) as fp:
        properties = jprops.load_properties(fp)
    except Exception,e:
      print e.__str__()

    if properties.has_key(propname):
      return properties[propname]
    return None
    
