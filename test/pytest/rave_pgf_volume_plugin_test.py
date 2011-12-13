'''
Created on Dec 13, 2011
def generateVolume(files, args):
  if len(files) <=0:
    raise AttributeError, "Volume must consist of at least 1 scan"

  firstscan=False  
  volume = _polarvolume.new()
  volume.date = args['date']
  volume.time = args['time']
  
  #'longitude', 'latitude', 'height', 'time', 'date', 'source'

  for fname in files:
    rio = _raveio.open(fname)
    if firstscan == False:
      firstscan = True
      volume.longitude = rio.object.longitude
      volume.latitude = rio.object.latitude
      volume.height = rio.object.height
    volume.addScan(rio.object)

  volume.source = rio.object.source  # Recycle the last input, it won't necessarily be correct ...
  odim_source.CheckSo
@author: anders
'''
import unittest
import os
import math
import string
import rave_pgf_volume_plugin

class rave_pgf_volume_plugin_test(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_generateVolume(self):
    args={}
    args["date"] = "20110101"
    args["time"] = "100000"
    
    files=["fixtures/scan_sehud_0.5_20110126T184500Z.h5",
           "fixtures/scan_sehud_1.0_20110126T184600Z.h5",
           "fixtures/scan_sehud_1.5_20110126T184600Z.h5"]
    
    
    result = rave_pgf_volume_plugin.generateVolume(files, args)
    self.assertEquals(3, result.getNumberOfScans())
    self.assertAlmostEquals(61.5771, result.latitude * 180.0 / math.pi, 4)
    self.assertAlmostEquals(16.7144, result.longitude * 180.0 / math.pi, 4)
    self.assertAlmostEquals(389.0, result.height, 4)
    self.assertTrue(string.find(result.source, "RAD:SE44") >= 0)
    self.assertAlmostEquals(0.86, result.beamwidth * 180.0 / math.pi, 4)
    self.assertAlmostEquals(0.86, result.getScan(0).beamwidth * 180.0 / math.pi, 4)
    self.assertAlmostEquals(0.86, result.getScan(1).beamwidth * 180.0 / math.pi, 4)
    self.assertAlmostEquals(0.86, result.getScan(2).beamwidth * 180.0 / math.pi, 4)
    
    