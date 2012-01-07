import httplib
import mimetools
import mimetypes
import sys
import time
import urlparse
from Crypto.Util import asn1, number
from Crypto.PublicKey import DSA
import Crypto.Hash.SHA as SHA
from Crypto.Util.asn1 import DerSequence

from rave_defines import DEX_NODENAME, DEX_PRIVATEKEY, DEX_SPOE

PRIVKEY=None
try:
  PRIVKEY=open(DEX_PRIVATEKEY).read()
except:
  print "Failed to read the private key: '%s'"%DEX_PRIVATEKEY
  print "Secure communication disabled"
  

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
    
  def sign(self, key):
    hash=SHA.new(self.fields["BF_TimeStamp"]).digest()
    sig = key.sign(hash, 2)
    der = DerSequence()
    der.append(sig[0])
    der.append(sig[1])
    data = der.encode()
    self.files["BF_SignatureFileField"] = ("signature.file", data)
    
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

def createDsaKey(key):
    seq = asn1.DerSequence()
    data = "\n".join(key.strip().split("\n")[1:-1]).decode("base64")
    seq.decode(data)
    p, q, g, y, x = seq[1:]
    return DSA.construct((y, g, p, q, x))

def get_content_type(filename):
    return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

def inject_certificate(dex_uri, certificate):
    frame = BaltradFrame()
    frame.set_request_type("BF_PostCertificate")
    frame.set_node_name(DEX_NODENAME)
    frame.set_local_uri("http://localhost")
    frame.set_certificate(certificate)
    return frame.post(dex_uri)

def inject_file(path, dex_uri):
    if PRIVKEY == None:
      raise Exception, "BaltradFrame only support encrypted communication"

    frame = BaltradFrame()
    frame.set_request_type("BF_PostDataDeliveryRequest")
    frame.set_node_name(DEX_NODENAME)
    frame.set_local_uri("http://localhost")

    key = createDsaKey(PRIVKEY)
    frame.sign(key)
    frame.set_payload_file(path)
    return frame.post(dex_uri)


#PRIVKEY2 = """-----BEGIN DSA PRIVATE KEY-----
#MIIBuwIBAAKBgQD9f1OBHXUSKVLfSpwu7OTn9hG3UjzvRADDHj+AtlEmaUVdQCJR
#+1k9jVj6v8X1ujD2y5tVbNeBO4AdNG/yZmC3a5lQpaSfn+gEexAiwk+7qdf+t8Yb
#+DtX58aophUPBPuD9tPFHsMCNVQTWhaRMvZ1864rYdcq7/IiAxmd0UgBxwIVAJdg
#UI8VIwvMspK5gqLrhAvwWBz1AoGBAPfhoIXWmz3ey7yrXDa4V7l5lK+7+jrqgvlX
#TAs9B4JnUVlXjrrUWU/mcQcQgYC0SRZxI+hMKBYTt88JMozIpuE8FnqLVHyNKOCj
#rh4rs6Z1kW6jfwv6ITVi8ftiegEkO8yk8b6oUZCJqIPf4VrlnwaSi2ZegHtVJWQB
#TDv+z0kqAoGAafFTFzcMqfCwbf0Q/BnmCsk6PRU3Efy/iz2D4s8WHRDM9lg2drBb
#yCIpPLJkmzi+qFFgSMMcKKYVMNbEs3BdyjsrnjW3zXc6zZWzDYY9nraoVVXPJ7Ik
#N3u2HwtB1m/afHED8mcBf2OpcfJ6GWyeB1kh56BB2b3jSW+hj6Gsw7ACFD+SquSp
#SGNQVh+cc+gW72maAOR3
#-----END DSA PRIVATE KEY-----"""

#PRIVKEY = """-----BEGIN DSA PRIVATE KEY-----
#MIIBvAIBAAKBgQD9f1OBHXUSKVLfSpwu7OTn9hG3UjzvRADDHj+AtlEmaUVdQCJR
#+1k9jVj6v8X1ujD2y5tVbNeBO4AdNG/yZmC3a5lQpaSfn+gEexAiwk+7qdf+t8Yb
#+DtX58aophUPBPuD9tPFHsMCNVQTWhaRMvZ1864rYdcq7/IiAxmd0UgBxwIVAJdg
#UI8VIwvMspK5gqLrhAvwWBz1AoGBAPfhoIXWmz3ey7yrXDa4V7l5lK+7+jrqgvlX
#TAs9B4JnUVlXjrrUWU/mcQcQgYC0SRZxI+hMKBYTt88JMozIpuE8FnqLVHyNKOCj
#rh4rs6Z1kW6jfwv6ITVi8ftiegEkO8yk8b6oUZCJqIPf4VrlnwaSi2ZegHtVJWQB
#TDv+z0kqAoGBANBoawdam6WE3jAebLhUQ7XP5aD9EimPTAjtt+PdTWx1zqH4hKFr
#ygmL/8nh/BsQOiLKo/XOUmmafuby8P5Uke30t/BphU07fVepan1oQicYdlNVemNX
#AIGdB3a58AJ8XAkx9ayDRDQ9w2VDMd2R75qF1gwjxPYPSL6XO39E1FQVAhQXiFyh
#AmWPEVj0t8HEp4nAePTX2w==
#-----END DSA PRIVATE KEY-----"""

if __name__ == "__main__":
  dex_url = DEX_SPOE
  if sys.argv[1] == "certificate" or sys.argv[1] == "file":
    filename = sys.argv[2]
    if len(sys.argv) > 3:
      dex_url = sys.argv[2]
  else:
    print "Syntax is BaltradFrame.py <command> <file> [<url>}"
    print "where command either is certificate or file."
    print "  if certificate, then a CERTIFICATE should be provided as <file>"
    print "  if file, then a hdf 5 file should be provided as <file>"
    print ""
    print "url is optional, if not specified, then it will default to DEX_SPOE in rave_defines"
    print ""
    sys.exit(0)
    #dex_url = "http://localhost:9001/BaltradDex/dispatch.htm"
    #node_name = "test_node"
    #    print inject_certificate(dex_url, node_name, sys.argv[1])
    if sys.argv[1] == "certificate":
      print inject_certificate(dex_url, filename)
    else:
      print inject_file(filename, dex_url)
