'''
Created on 13 April 2011

@author: Daniel Mattsson
'''
import unittest
import _raveio
import _polarscan
import _rack

class RaveRack(unittest.TestCase):
    # KNMI PVOL from Den Helder with a nice sun hit
    PVOL_TESTFILE="fixtures/KNMI-PVOL-Den_Helder.h5"
    SCAN_TESTFILE="fixtures/scan_sevil_20100702T113200Z.h5"


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testRackMainWithPvol(self):
	obj = _raveio.open(self.PVOL_TESTFILE)
	self.assertEquals(_raveio.Rave_ObjectType_PVOL, obj.objectType)
	obj2 = _rack.rack(obj.object, "--aSpeckle 32,30 --aEmitter 3,6 --rShip 32,6 --rBiomet 16,4,500,50 --aPaste --aGapFill 3,3")
	self.assertTrue(obj2 != None)


    def testRackMainWithScan(self):
	obj = _raveio.open(self.SCAN_TESTFILE)
	self.assertEquals(_raveio.Rave_ObjectType_SCAN, obj.objectType)
	obj2 = _rack.rack(obj.object, "--aSpeckle 32,30 --quantity DBZH --aEmitter 3,6 --rShip 32,6 --rBiomet 16,4,500,50 --aPaste --aGapFill 3,3")
	self.assertTrue(obj2 != None)


    def testRackAndreSpeckle(self):
	obj = _raveio.open(self.PVOL_TESTFILE)
	self.assertEquals(_raveio.Rave_ObjectType_PVOL, obj.objectType)
	obj2 = _rack.rack(obj.object, "--aSpeckle 32,30 --aPaste")
	self.assertTrue(obj2 != None)


    def testRackAndreEmitter(self):
	obj = _raveio.open(self.PVOL_TESTFILE)
	self.assertEquals(_raveio.Rave_ObjectType_PVOL, obj.objectType)
	obj2 = _rack.rack(obj.object, "--aEmitter 3,6 --aPaste")
	self.assertTrue(obj2 != None)


    def testRackRopoShip(self):
	obj = _raveio.open(self.PVOL_TESTFILE)
	self.assertEquals(_raveio.Rave_ObjectType_PVOL, obj.objectType)
	obj2 = _rack.rack(obj.object, "--rShip 32,6 --aPaste")
	self.assertTrue(obj2 != None)


    def testRackRopoBiomet(self):
	obj = _raveio.open(self.PVOL_TESTFILE)
	self.assertEquals(_raveio.Rave_ObjectType_PVOL, obj.objectType)
	obj2 = _rack.rack(obj.object, "--rBiomet 16,4,500,50 --aPaste")
	self.assertTrue(obj2 != None)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
