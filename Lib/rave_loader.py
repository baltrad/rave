'''
Utility class for loading a HDF5 file in the ODIM format and create
the _rave - module objects.

@author: Anders Henja
'''
import _rave
import _pyhl
import math

class rave_loader:
  def __init__(self):
    pass
  
  def load_file(self, filename, quantity):
    nodelist = _pyhl.read_nodelist(filename)
    names = nodelist.getNodeNames()

    index = 1
    finished = False

    vol = None
    if names.has_key("/Conventions"):
      conv = nodelist.fetchNode("/Conventions").data()
      if conv == "ODIM_H5/V2_0":
        vol = _rave.volume()

    if vol == None:
      return None
    
    vol.longitude = nodelist.fetchNode("/where/lon").data() * math.pi / 180.0
    vol.latitude = nodelist.fetchNode("/where/lat").data() * math.pi / 180.0
    vol.height = nodelist.fetchNode("/where/height").data()
      
    while not finished:
      scanname = "/dataset%d"%index
      if names.has_key(scanname):
        scan = self.load_scan_from_file(scanname, nodelist, names, quantity)
        if scan != None:
          vol.addScan(scan)
      else:
        finished = True
        
      index = index + 1

    #vol.sortByElevations(1) # Ascending sort
    
    return vol

  def load_scan_from_file(self, name, nodelist, names, quantity):
    index = 1
    finished = False
    
    while not finished:
      scanname = "%s/data%d"%(name, index)
      if names.has_key(scanname):
        quantityname = "%s/what/quantity"%scanname
        if names.has_key(quantityname):
          q = nodelist.fetchNode(quantityname).data()
          if q == quantity:
            scan = _rave.scan()
            scan.gain = nodelist.fetchNode("%s/what/gain"%scanname).data()
            scan.offset = nodelist.fetchNode("%s/what/offset"%scanname).data()
            scan.nodata = nodelist.fetchNode("%s/what/nodata"%scanname).data()
            scan.undetect = nodelist.fetchNode("%s/what/undetect"%scanname).data()
            scan.quantity = nodelist.fetchNode("%s/what/quantity"%scanname).data()
            
            scan.elangle = nodelist.fetchNode("%s/where/elangle"%name).data() * math.pi/180.0
            scan.a1gate = nodelist.fetchNode("%s/where/a1gate"%name).data()
            #scan.nbins = nodelist.fetchNode("%s/where/nbins"%name).data()
            #scan.nrays = nodelist.fetchNode("%s/where/nrays"%name).data()
            scan.rscale = nodelist.fetchNode("%s/where/rscale"%name).data()
            scan.rstart = nodelist.fetchNode("%s/where/rstart"%name).data()
            
            scan.setData(nodelist.fetchNode("%s/data"%scanname).data())
            
            return scan
      else:
        finished = True
    return None