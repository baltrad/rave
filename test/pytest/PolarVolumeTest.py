'''
Created on Oct 14, 2009
@author: Anders Henja
'''
import unittest
import os
import _polarvolume
import string

class PolarVolumeTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass
    
  def testNewVolume(self):
    obj = _polarvolume.volume()
    
    result = string.find(`type(obj)`, "PolarVolumeCore")
    self.assertNotEqual(-1, result) 
    self.assertTrue("cappi" in dir(obj))

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()