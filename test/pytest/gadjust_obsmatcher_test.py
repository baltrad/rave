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
import unittest, os, datetime, math
import rave_dom_db
from rave_dom import observation
from gadjust import obsmatcher
import mock
import _raveio, _rave

##
# Tests that the obsmatcher works as expected. These tests require mock 1.0 in order to work
# float(mock.__version__[:3]) >= 1.0
#
class gadjust_obsmatcher_test(unittest.TestCase):
  def setUp(self):
    self.dbmock = mock.Mock(spec=rave_dom_db.rave_db) 
    self.classUnderTest = obsmatcher.obsmatcher(self.dbmock) 

  def tearDown(self):
    self.db = None
    self.classUnderTest = None
  
  def test_match_1(self):
    acrrmock = mock.Mock(date="20101010",time="121500")
    acrrmock.getExtremeLonLatBoundaries.return_value = ((0.1,0.2),(0.3,0.4))
    
    self.dbmock.get_observations_in_bbox.return_value = [observation("01234","S",observation.SYNOP, "20101010", "001500", 13.0, 60.0, liquid_precipitation=1.0),
                                                         observation("01235","S",observation.SYNOP, "20101010", "001500", 13.5, 60.0, liquid_precipitation=1.0),
                                                         observation("01236","S",observation.SYNOP, "20101010", "001500", 13.5, 60.5, liquid_precipitation=1.0)]
    acrrmock.getConvertedValueAtLonLat.return_value = (2,48.0)
    acrrmock.getQualityValueAtLonLat.return_value = 200.0
    acrrmock.getConvertedValueAtLonLat.return_value = (2,49.0)
    acrrmock.getQualityValueAtLonLat.return_value = 25.5
    acrrmock.getConvertedValueAtLonLat.return_value = (2,30.0)
    acrrmock.getQualityValueAtLonLat.return_value = 123456.0
    
    result = self.classUnderTest.match(acrrmock, 12)
    
    # Expects
    expected_acrrmock_calls=[mock.call.getExtremeLonLatBoundaries(), 
                             mock.call.getConvertedValueAtLonLat((13.0*math.pi/180.0, 60.0*math.pi/180.0)),
                             mock.call.getQualityValueAtLonLat((13.0*math.pi/180.0, 60.0*math.pi/180.0), 'se.smhi.composite.distance.radar'),
                             mock.call.getConvertedValueAtLonLat((13.5*math.pi/180.0, 60.0*math.pi/180.0)),
                             mock.call.getQualityValueAtLonLat((13.5*math.pi/180.0, 60.0*math.pi/180.0), 'se.smhi.composite.distance.radar'),
                             mock.call.getConvertedValueAtLonLat((13.5*math.pi/180.0, 60.5*math.pi/180.0)),
                             mock.call.getQualityValueAtLonLat((13.5*math.pi/180.0, 60.5*math.pi/180.0), 'se.smhi.composite.distance.radar')]
    self.assertTrue(expected_acrrmock_calls == acrrmock.mock_calls)
    self.assertEquals(3, len(result))

  def test_match_2(self):
    acrrmock = mock.Mock(date="20101010",time="121500")
    acrrmock.getExtremeLonLatBoundaries.return_value = ((0.1,0.2),(0.3,0.4))
    
    self.dbmock.get_observations_in_bbox.return_value = [observation("01234","S",observation.SYNOP, "20101010", "001500", 13.0, 60.0, liquid_precipitation=1.0)]
    acrrmock.getConvertedValueAtLonLat.return_value = (2,48.0)
    acrrmock.getQualityValueAtLonLat.return_value = 200.0
    
    result = self.classUnderTest.match(acrrmock, 12)
    
    # Expects
    expected_acrrmock_calls=[mock.call.getExtremeLonLatBoundaries(), 
                             mock.call.getConvertedValueAtLonLat((13.0*math.pi/180.0, 60.0*math.pi/180.0)),
                             mock.call.getQualityValueAtLonLat((13.0*math.pi/180.0, 60.0*math.pi/180.0), 'se.smhi.composite.distance.radar')]
    self.assertTrue(expected_acrrmock_calls == acrrmock.mock_calls)
    self.assertEquals(1, len(result))
    self.assertEquals(result[0].radarvaluetype, _rave.RaveValueType_DATA)
    self.assertAlmostEquals(result[0].radarvalue, 48.0, 4)
    self.assertAlmostEquals(200.0, result[0].radardistance, 4)
    self.assertAlmostEquals(13.0, result[0].longitude, 4)
    self.assertAlmostEquals(60.0, result[0].latitude, 4)
    self.assertEquals("20101010", result[0].date)
    self.assertEquals("001500", result[0].time)

  def test_match_3(self):
    acrrmock = mock.Mock(date="20101010",time="121500")
    acrrmock.getExtremeLonLatBoundaries.return_value = ((0.1,0.2),(0.3,0.4))
    
    self.dbmock.get_observations_in_bbox.return_value = [observation("01234","S",observation.SYNOP, "20101010", "001500", 13.0, 60.0)]
    acrrmock.getConvertedValueAtLonLat.return_value = (1,48.0)
    acrrmock.getQualityValueAtLonLat.return_value = 200.0
    
    result = self.classUnderTest.match(acrrmock, 12)
    
    # Expects
    expected_acrrmock_calls=[mock.call.getExtremeLonLatBoundaries(), 
                             mock.call.getConvertedValueAtLonLat((13.0*math.pi/180.0, 60.0*math.pi/180.0))]
    self.assertTrue(expected_acrrmock_calls == acrrmock.mock_calls)
    self.assertEquals(0, len(result))
    

  def Xtest_match_2(self):
    acrrmock = mock.Mock(date="20101010",time="121500")
    acrrmock.getExtremeLonLatBoundaries.return_value = ((0.1,0.2),(0.3,0.4))
    
    self.dbmock.get_observations_in_bbox.return_value = [1,2,3]
    
    self.classUnderTest.match(acrrmock, 13)
    
    # Expects
    self.dbmock.get_observations_in_bbox.assert_called_with(0.1,0.2,0.3,0.4, datetime.datetime(2010,10,9,23,15,00))
    
