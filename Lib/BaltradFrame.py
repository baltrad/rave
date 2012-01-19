import httplib
import mimetools
import mimetypes
import os
import sys
import time
import urlparse

from keyczar import keyczar 

from rave_defines import DEX_NODENAME, DEX_PRIVATEKEY, DEX_SPOE

class BaltradFrame(object):
  def __init__(self):
    self.fields = {}
    self.files = {}
    self.set_timestamp(int(time.time() * 1000))
    
  def set_request_type(self, type_):
    self.fields["BF_RequestType"] = type_
    
  def set_timestamp(self, timestamp):
    self.fields["BF_TimeStamp"] = str(timestamp)

  def set_local_uri(self, uri):
    self.fields["BF_LocalURI"] = uri
    
  def set_node_name(self, name):
    self.fields["BF_NodeName"] = name
    
  def set_certificate(self, path):
    with open(path) as f:
      self.files["BF_CertificateFileField"] = (path, f.read())
    
  def set_payload_file(self, path):
    with open(path) as f:
      self.files["BF_PayloadFileField"] = (path, f.read())

  def set_message(self, msg):
    self.fields["BF_MessageField"] = msg

  def sign(self, signer):
    signature = signer.Sign(self.fields["BF_TimeStamp"])
    self.fields["BF_SignatureField"] = signature
    
  def post(self, url):
    fields = []
    for key, value in self.fields.iteritems():
      fields.append((key, value))
    files = []

    for key, (filename, value) in self.files.iteritems():
      files.append((key, filename, value))

    urlparts = urlparse.urlsplit(url)
    host = urlparts[1]
    query = urlparts[2]
    return post_multipart(host, query, fields, files)


def post_multipart(host, selector, fields, files):
  content_type, body = encode_multipart_formdata(fields, files)
  conn = httplib.HTTPConnection(host)
  headers = {
    "Content-Type": content_type
  }
  try:
    conn.request('POST', selector, body, headers)
    response = conn.getresponse()
  finally:
    conn.close()
  return response.status, response.reason, response.read()

def encode_multipart_formdata(fields, files):
  """
  :param fields: a sequence of (name, value) elements for regular form
                 fields.
  :param files: a sequence of (name, filename, value) elements for data
                to be uploaded as files
  :return: *(content_type, body)* ready for httplib.HTTP instance
  """
  BOUNDARY = mimetools.choose_boundary()
  L = []
  for (key, value) in fields:
    L.append('--' + BOUNDARY)
    L.append('Content-Disposition: form-data; name="%s"' % key)
    L.append('')
    L.append(value)
  for (key, filename, value) in files:
    L.append('--' + BOUNDARY)
    L.append('Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filename))
    L.append('Content-Type: %s' % get_content_type(filename))
    L.append('')
    L.append(value)
    
  L.append('--' + BOUNDARY + '--')
  L.append('')
  content_type = 'multipart/form-data; boundary=%s' % BOUNDARY
  return content_type, "\r\n".join(L)

def get_content_type(filename):
  return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

def inject_file(path, dex_uri, dex_privatekey=DEX_PRIVATEKEY, dex_nodename=DEX_NODENAME):
  if dex_privatekey == None:
    raise RuntimeError("BaltradFrame only support encrypted communication")
  if not os.path.exists(dex_privatekey):
    raise RuntimeError("Private key does not exist: '%s'" % dex_privatekey)

  frame = BaltradFrame()
  frame.set_request_type("BF_PostDataDeliveryRequest")
  frame.set_node_name(dex_nodename)
  frame.set_local_uri("http://localhost")
  
  frame.set_payload_file(path)
  signer = keyczar.Signer.Read(dex_privatekey)
  frame.sign(signer)
  return frame.post(dex_uri)

##
# Sends a message to the dex.
# @param msg: the message to send
# @param dex_uri: the dex to send the message to
# @param dex_privatekey: the private key for signing
# @param dex_nodename: the name of this node
#
def send_message(msg, dex_uri, dex_privatekey=DEX_PRIVATEKEY, dex_nodename=DEX_NODENAME):
  if dex_privatekey == None:
    raise RuntimeError("BaltradFrame only support encrypted communication")
  if not os.path.exists(dex_privatekey):
    raise RuntimeError("Private key does not exist: '%s'" % dex_privatekey)

  frame = BaltradFrame()
  frame.set_request_type("BF_PostMessageDeliveryRequest")
  frame.set_node_name(dex_nodename)
  frame.set_local_uri("http://localhost")
  
  frame.set_message(msg)
  signer = keyczar.Signer.Read(dex_privatekey)
  frame.sign(signer)
  return frame.post(dex_uri)

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
