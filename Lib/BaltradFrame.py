import httplib
import mimetools
import mimetypes
import os
import sys
import time
import datetime
import urlparse
import base64

from keyczar import keyczar 

from rave_defines import DEX_NODENAME, DEX_PRIVATEKEY, DEX_SPOE

class BaltradFrame(object):
  _uri = None
  _private_key = None
  _nodename = None
  _signer = None
  
  def __init__(self, dex_uri, dex_privatekey=DEX_PRIVATEKEY, dex_nodename=DEX_NODENAME):
    self._dex_uri = dex_uri
    self._private_key = dex_privatekey
    self._nodename = dex_nodename
    self._signer = keyczar.Signer.Read(dex_privatekey)
  
  def _generate_headers(self, uri):
    datestr = datetime.datetime.now().strftime("%a, %e %B %Y %H:%M:%S")
    contentMD5 = base64.b64encode(uri)
    message = ("POST" + '\n' + uri + '\n' + "application/x-hdf5" + '\n' + contentMD5 + '\n' + datestr)
    signature = self._signer.Sign(message)
    headers = {"Node-Name": self._nodename, 
               "Content-Type": "application/x-hdf5",
               "Content-MD5": contentMD5, 
               "Date": datestr, 
               "Authorization": self._nodename + ':' + signature}
    return headers
  
  def _split_uri(self, uri):
    urlparts = urlparse.urlsplit(uri)
    host = urlparts[1]
    query = urlparts[2]    
    return (host, query)
  
  def _post(self, host, query, data, headers):
    conn = httplib.HTTPConnection(host)
    try:
      conn.request("POST", query, data, headers)
      response = conn.getresponse()
    finally:
      conn.close();
      
    return response.status, response.reason, response.read()
  
  def send_file(self, path):
    uri = "%s/post_file.htm"%self._dex_uri
    
    (host, query) = self._split_uri(uri)
    headers = self._generate_headers(uri)
 
    fp = open(path, 'r')
    
    try:
      return self._post(host, query, fp, headers)
    finally:
      fp.close()

  def send_message(self, message):
    uri = "%s/post_message.htm"%self._dex_uri
    (host, query) = self._split_uri(uri)
    headers = self._generate_headers(uri)

    return self._post(host, query, message, headers)

def inject_file(path, dex_uri, dex_privatekey=DEX_PRIVATEKEY, dex_nodename=DEX_NODENAME):
  return BaltradFrame(dex_uri, dex_privatekey, dex_nodename).send_file(path)

def send_message(msg, dex_uri, dex_privatekey=DEX_PRIVATEKEY, dex_nodename=DEX_NODENAME):
  return BaltradFrame(dex_uri, dex_privatekey, dex_nodename).send_message(msg)

if __name__ == "__main__":
  dex_url = DEX_SPOE
  if sys.argv[1] == "certificate" or sys.argv[1] == "file":
    filename = sys.argv[2]
    if len(sys.argv) > 3:
      dex_url = sys.argv[3]
  else:
    print "Syntax is BaltradFrame.py <command> <file> [<url>}"
    print "where command either is one of: 'file'"
    print "  if file, then a hdf 5 file should be provided as <file>"
    print ""
    print "url is optional, if not specified, then it will default to DEX_SPOE in rave_defines"
    print ""
    sys.exit(0)
  
  if sys.argv[1] == "file":
    print inject_file(filename, dex_url)
