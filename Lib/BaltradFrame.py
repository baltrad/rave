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
## Equivalent functionality to using OdimH5.jar on the command line to inject
# individual files to a BALTRAD node.

## @file
## @author Daniel Michelson, SMHI
## @date 2010-07-08

import os, pycurl
import BaltradMessageXML
from rave_defines import DEX_SPOE


## Class for managing and sending BaltradFrame messages.
class BaltradFrameMessage:

  ## Constructor
  # @param url string containing the initialized URL
  def __init__(self, url):
    self._ctr = 0
    ## @param message initialized empty message
    self.message = []
    ## @param url string containing the initialized URL
    self.url = url

  ## Changes the URL to that given.
  # @param url string containing the URL to which to transmit the message
  def setURL(self, url=DEX_SPOE):
    self.url = url
    
  ## Adds an XML envelope to the message.
  # @param channel string denoting the identifier of the data channel
  # @param filename input file string, must be relative to the DEX's
  # IncomingData directory!
  # @param sender string, the sender's identifier. Totally bogus in this
  # version of the DEX
  def addEnvelope(self, channel, filename, sender='smhi'):
    envelope = BaltradMessageXML.MakeBaltradFrameXML(sender, channel, filename)
    self.message.append( ('<bf_xml/>',
                          (pycurl.FORM_CONTENTTYPE,
                           'multipart/form-data; charset=UTF-8',
                           pycurl.FORM_CONTENTS, envelope)))
  
  ## Adds a binary payload to the message. The file's physical location
  # must be relative to the DEX's IncomingData directory.
  # @param filename string containing the input data to include in the message.
  # This is an ODIM_H5 file, and the path must be relative to the DEX's
  # IncomingData directory.
  def addBinaryPayload(self, filename):
    self.message.append( ('<bf_file/>',
                          (pycurl.FORM_FILE, filename)))
    self._ctr = self._ctr + 1
  
  ## Creates a pycurl.Curl object, puts the message inside it, and injects
  # it into the DEX.
  def send(self):
    c = pycurl.Curl()
    c.setopt(c.POST, 1)
    c.setopt(c.URL, self.url)
    c.setopt(c.HTTPPOST, self.message)
    c.perform()
    c.close()


## Convenience function for injecting a file into the DEX of a baltrad-node
# @param filename input file string, must be relative to the DEX's IncomingData
# directory!
# @param channel string denoting the identifier of the data channel
# @param url string containing the URL to which to transmit the message
# @param sender string, the sender's identifier. Totally bogus in this version
# of the DEX
def inject(filename, channel='bogus_channel', url=DEX_SPOE, sender='smhi'):
  import _pyhl

  if os.path.isfile(filename) and os.path.getsize(filename) > 0 and _pyhl.is_file_hdf5(filename):
    path, fstr = os.path.split(filename)
    if len(path):
      here = os.getcwd()
      os.chdir(path)
    this = BaltradFrameMessage(url)
    this.addEnvelope(channel, fstr)
    this.addBinaryPayload(fstr)
    this.send()
    if len(path): os.chdir(here)  # Critical that we don't chdir before sending.
  else:
    raise IOError, "File %s is not a regular file, it is zero length, or it is not an HDF5 file." % filename



if __name__=="__main__":
  import sys
#  inject(sys.argv[1])
  inject(sys.argv[1], channel=sys.argv[2],
         url=sys.argv[3], sender=sys.argv[4])
