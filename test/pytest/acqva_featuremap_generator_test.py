'''
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/

Tests the acqva_featuremap_generator class

@file
@author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
@date 2025-11-14
'''
import unittest
import _raveio, _polarvolume, _polarscan, _polarscanparam
from acqva_featuremap_generator import acqva_featuremap_generator
import string, json, datetime, math, os
import numpy
from numpy import array, reshape, uint8

class acqva_featuremap_generator_test(unittest.TestCase):
  TEMPORARY_FILE="acqva_featuremap_generator_test_iotest.h5"

  def setUp(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  def tearDown(self):
    if os.path.isfile(self.TEMPORARY_FILE):
      os.unlink(self.TEMPORARY_FILE)

  
  def validate_field(self, field, nbins, nrays, elangle, rscale, rstart, beamwidth, fillvalue):
    self.assertEqual(nbins, field.nbins)
    self.assertEqual(nrays, field.nrays)
    self.assertAlmostEqual(elangle, field.elangle, 4)
    self.assertAlmostEqual(rscale, field.rscale, 4)
    self.assertAlmostEqual(rstart, field.rstart, 4)
    self.assertAlmostEqual(beamwidth, field.beamwidth, 4)
    self.assertTrue(numpy.all(numpy.ones((nrays, nbins), numpy.uint8)==field.getData()))

  def test_create_featuremap_from_config_1(self):
    cfg = json.loads("""
      {
        "nod":"seang",
        "longitude":14.0,
        "latitude":60.0,
        "height":123,
        "scans":[
            {"nbins":240, "nrays":360, "elangle":0.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":240, "nrays":360, "elangle":0.5, "rscale":1000, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":0.5, "rscale":500, "rstart":3, "beamwidth":2.0},
            {"nbins":480, "nrays":360, "elangle":1.0, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":1.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":2.0, "rscale":500, "rstart":0, "beamwidth":1.0}
        ]
      }
      """)
    featuremap = acqva_featuremap_generator.create_featuremap_from_config(cfg, "seang", "20250101", "20250201")
    self.assertEqual("seang", featuremap.nod)
    self.assertAlmostEqual(14.0*math.pi / 180.0, featuremap.longitude, 4)
    self.assertAlmostEqual(60.0*math.pi / 180.0, featuremap.latitude, 4)
    self.assertAlmostEqual(123.0, featuremap.height, 4)
    self.assertEqual("20250101", featuremap.startdate)
    self.assertEqual("20250201", featuremap.enddate)

    self.assertEqual(4, featuremap.getNumberOfElevations())
    
    self.validate_field(featuremap.getElevation(0).get(0), 240, 360, 0.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(0).get(1), 240, 360, 0.5*math.pi/180.0, 1000.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(0).get(2), 480, 360, 0.5*math.pi/180.0, 500.0, 3.0, 2.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(1).get(0), 480, 360, 1.0*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(2).get(0), 480, 360, 1.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(3).get(0), 480, 360, 2.0*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)


  def test_create_featuremap_from_config_2(self):
    cfg = json.loads("""
      {
        "nod":"seang",
        "startdate":"20250201",
        "enddate":"20250301",
        "longitude":14.0,
        "latitude":60.0,
        "height":123,
        "scans":[
            {"nbins":240, "nrays":360, "elangle":0.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":240, "nrays":360, "elangle":0.5, "rscale":1000, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":0.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":1.0, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":1.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":2.0, "rscale":500, "rstart":0, "beamwidth":1.0}
        ]
      }
      """)
    featuremap = acqva_featuremap_generator.create_featuremap_from_config(cfg, "seang", "20250101", "20250201")
    self.assertEqual("seang", featuremap.nod)
    self.assertAlmostEqual(14.0*math.pi / 180.0, featuremap.longitude, 4)
    self.assertAlmostEqual(60.0*math.pi / 180.0, featuremap.latitude, 4)
    self.assertAlmostEqual(123.0, featuremap.height, 4)
    self.assertEqual("20250101", featuremap.startdate)
    self.assertEqual("20250201", featuremap.enddate)

    self.assertEqual(4, featuremap.getNumberOfElevations())
    
    self.validate_field(featuremap.getElevation(0).get(0), 240, 360, 0.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(0).get(1), 240, 360, 0.5*math.pi/180.0, 1000.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(0).get(2), 480, 360, 0.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(1).get(0), 480, 360, 1.0*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(2).get(0), 480, 360, 1.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(3).get(0), 480, 360, 2.0*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)

  def test_create_featuremap_from_config_3(self):
    cfg = json.loads("""
      {
        "nod":"seang",
        "startdate":"20250201",
        "enddate":"20250301",
        "longitude":14.0,
        "latitude":60.0,
        "height":123,
        "scans":[
            {"nbins":240, "nrays":360, "elangle":0.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":240, "nrays":360, "elangle":0.5, "rscale":1000, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":0.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":1.0, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":1.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":2.0, "rscale":500, "rstart":0, "beamwidth":1.0}
        ]
      }
      """)
    featuremap = acqva_featuremap_generator.create_featuremap_from_config(cfg, "seang")
    self.assertEqual("seang", featuremap.nod)
    self.assertAlmostEqual(14.0*math.pi / 180.0, featuremap.longitude, 4)
    self.assertAlmostEqual(60.0*math.pi / 180.0, featuremap.latitude, 4)
    self.assertAlmostEqual(123.0, featuremap.height, 4)
    self.assertEqual("20250201", featuremap.startdate)
    self.assertEqual("20250301", featuremap.enddate)

    self.assertEqual(4, featuremap.getNumberOfElevations())
    
    self.validate_field(featuremap.getElevation(0).get(0), 240, 360, 0.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(0).get(1), 240, 360, 0.5*math.pi/180.0, 1000.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(0).get(2), 480, 360, 0.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(1).get(0), 480, 360, 1.0*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(2).get(0), 480, 360, 1.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(3).get(0), 480, 360, 2.0*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)

  def test_create_featuremap_from_config_4(self):
    cfg = json.loads("""
      {
        "nod":"seang",
        "startdate":"20250201",
        "enddate":"20250301",
        "longitude":14.0,
        "latitude":60.0,
        "height":123,
        "scans":[
            {"nbins":240, "nrays":360, "elangle":0.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":240, "nrays":360, "elangle":0.5, "rscale":1000, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":0.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":1.0, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":1.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":2.0, "rscale":500, "rstart":0, "beamwidth":1.0}
        ]
      }
      """)
    featuremap = acqva_featuremap_generator.create_featuremap_from_config(cfg)
    self.assertEqual("seang", featuremap.nod)
    self.assertAlmostEqual(14.0*math.pi / 180.0, featuremap.longitude, 4)
    self.assertAlmostEqual(60.0*math.pi / 180.0, featuremap.latitude, 4)
    self.assertAlmostEqual(123.0, featuremap.height, 4)
    self.assertEqual("20250201", featuremap.startdate)
    self.assertEqual("20250301", featuremap.enddate)

    self.assertEqual(4, featuremap.getNumberOfElevations())
    
    self.validate_field(featuremap.getElevation(0).get(0), 240, 360, 0.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(0).get(1), 240, 360, 0.5*math.pi/180.0, 1000.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(0).get(2), 480, 360, 0.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(1).get(0), 480, 360, 1.0*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(2).get(0), 480, 360, 1.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)
    self.validate_field(featuremap.getElevation(3).get(0), 480, 360, 2.0*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0, 1)

  def test_create_featuremap_from_config_conflicting_dims(self):
    cfg = json.loads("""
      {
        "nod":"seang",
        "startdate":"20250201",
        "enddate":"20250301",
        "longitude":14.0,
        "latitude":60.0,
        "height":123,
        "scans":[
            {"nbins":240, "nrays":360, "elangle":0.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":240, "nrays":360, "elangle":1.0, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":240, "nrays":360, "elangle":0.5, "rscale":500, "rstart":0, "beamwidth":1.0}
        ]
      }
      """)
    try:
        featuremap = acqva_featuremap_generator.create_featuremap_from_config(cfg)
        self.fail("Expected RuntimeError")
    except RuntimeError:
        pass

  def test_create_1(self):
    featuremapcfg = json.loads("""
      {
        "acqva_static_volume": {
          "seang": {
            "blockings":[
              { "scans":"0 -> 2",
                "bins":"0 -> 10",
                "rays":"300 -> 50",
                "blocked":true
              },
              { "scans":"0 -> 3",
                "bins":"0 -> 20",
                "rays":"10 -> 50",
                "blocked":true
              }
            ]
          }
        }
      }
    """)

    volumecfg = json.loads("""
      {
        "nod":"seang",
        "startdate":"20250201",
        "enddate":"20250301",
        "longitude":14.0,
        "latitude":60.0,
        "height":123,
        "scans":[
            {"nbins":480, "nrays":360, "elangle":0.5, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":1.0, "rscale":500, "rstart":0, "beamwidth":1.0},
            {"nbins":480, "nrays":360, "elangle":1.5, "rscale":500, "rstart":0, "beamwidth":1.0}
        ]
      }
      """)

    featuremap = acqva_featuremap_generator(featuremapcfg).create(volumecfg)
    featuremap.save(self.TEMPORARY_FILE)

    field1 = featuremap.getElevation(0).get(0)
    self.assertAlmostEqual(500.0, field1.rscale, 4)
    data = field1.getData()
    self.assertTrue(numpy.any(data[0:50,0:10] == numpy.zeros((50,10), numpy.uint8)))
    self.assertTrue(numpy.any(data[0:50,10:480] == numpy.ones((50,470), numpy.uint8)))
    self.assertTrue(numpy.any(data[10:50,0:20] == numpy.zeros((40,20), numpy.uint8)))

  def create_scan(self, quantity, nbins, nrays, elangle, rscale, rstart, beamwidth):
    scan = _polarscan.new()
    scan.elangle = elangle
    scan.rscale = rscale
    scan.rstart = rstart
    scan.beamwidth = beamwidth
    param = _polarscanparam.new()
    param.setData(numpy.zeros((nrays, nbins), numpy.uint16))
    param.quantity = quantity
    scan.addParameter(param)
    return scan

  def test_create_volume_config(self):
    vol1 = _polarvolume.new()
    vol1.longitude = 14.0 * math.pi / 180.0
    vol1.latitude = 60.0 * math.pi / 180.0
    vol1.height = 212.0
    vol1.source = "NOD:seang"
    vol1.addScan(self.create_scan("DBZH", 480, 360, 0.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0))
    vol1.addScan(self.create_scan("DBZH", 480, 360, 1.0*math.pi/180.0, 500.0, 1.0, 2.0*math.pi/180.0))
    vol1.addScan(self.create_scan("DBZH", 480, 360, 1.5*math.pi/180.0, 500.0, 2.0, 3.0*math.pi/180.0))

    cfg1 = acqva_featuremap_generator.create_volum_config_from_volume(vol1)
    self.assertEqual(cfg1, {'longitude': 14.0, 'latitude': 60.0, 'height': 212.0, 'nod': 'seang', 
                            'scans': [
                              {'nbins': 480, 'nrays': 360, 'elangle': 0.5, 'rscale': 500.0, 'rstart': 0.0, 'beamwidth': 1.0}, 
                              {'nbins': 480, 'nrays': 360, 'elangle': 1.0, 'rscale': 500.0, 'rstart': 1.0, 'beamwidth': 2.0}, 
                              {'nbins': 480, 'nrays': 360, 'elangle': 1.5, 'rscale': 500.0, 'rstart': 2.0, 'beamwidth': 3.0}
                            ]
                           })

  def test_merge_volume_config(self):
    vol1 = _polarvolume.new()
    vol1.longitude = 14.0 * math.pi / 180.0
    vol1.latitude = 60.0 * math.pi / 180.0
    vol1.height = 212.0
    vol1.source = "NOD:seang"
    vol1.addScan(self.create_scan("DBZH", 480, 360, 0.5*math.pi/180.0, 500.0, 0.0, 1.0*math.pi/180.0))
    vol1.addScan(self.create_scan("DBZH", 480, 360, 1.0*math.pi/180.0, 500.0, 1.0, 2.0*math.pi/180.0))
    vol1.addScan(self.create_scan("DBZH", 480, 360, 1.5*math.pi/180.0, 500.0, 2.0, 3.0*math.pi/180.0))

    vol2 = _polarvolume.new()
    vol2.longitude = 14.0 * math.pi / 180.0
    vol2.latitude = 60.0 * math.pi / 180.0
    vol2.height = 212.0
    vol2.source = "NOD:seang"
    vol2.addScan(self.create_scan("DBZH", 480, 360, 1.0*math.pi/180.0, 500.0, 1.0, 2.0*math.pi/180.0))
    vol2.addScan(self.create_scan("DBZH", 480, 360, 2.0*math.pi/180.0, 500.0, 1.0, 2.0*math.pi/180.0))
    vol2.addScan(self.create_scan("DBZH", 480, 360, 4.0*math.pi/180.0, 500.0, 2.0, 3.0*math.pi/180.0))

    cfg1 = acqva_featuremap_generator.create_volum_config_from_volume(vol1)
    cfg2 = acqva_featuremap_generator.create_volum_config_from_volume(vol2)
    self.assertEqual(cfg1, {'longitude': 14.0, 'latitude': 60.0, 'height': 212.0, 'nod': 'seang', 
                            'scans': [
                              {'nbins': 480, 'nrays': 360, 'elangle': 0.5, 'rscale': 500.0, 'rstart': 0.0, 'beamwidth': 1.0}, 
                              {'nbins': 480, 'nrays': 360, 'elangle': 1.0, 'rscale': 500.0, 'rstart': 1.0, 'beamwidth': 2.0}, 
                              {'nbins': 480, 'nrays': 360, 'elangle': 1.5, 'rscale': 500.0, 'rstart': 2.0, 'beamwidth': 3.0}
                            ]
                           })

    self.assertEqual(cfg2, {'longitude': 14.0, 'latitude': 60.0, 'height': 212.0, 'nod': 'seang', 
                            'scans': [
                              {'nbins': 480, 'nrays': 360, 'elangle': 1.0, 'rscale': 500.0, 'rstart': 1.0, 'beamwidth': 2.0}, 
                              {'nbins': 480, 'nrays': 360, 'elangle': 2.0, 'rscale': 500.0, 'rstart': 1.0, 'beamwidth': 2.0}, 
                              {'nbins': 480, 'nrays': 360, 'elangle': 4.0, 'rscale': 500.0, 'rstart': 2.0, 'beamwidth': 3.0}
                            ]
                           })

    merged = acqva_featuremap_generator.merge_volume_config(cfg1, cfg2)

    self.assertEqual(merged, {'longitude': 14.0, 'latitude': 60.0, 'height': 212.0, 'nod': 'seang', 
                            'scans': [
                              {'nbins': 480, 'nrays': 360, 'elangle': 0.5, 'rscale': 500.0, 'rstart': 0.0, 'beamwidth': 1.0}, 
                              {'nbins': 480, 'nrays': 360, 'elangle': 1.0, 'rscale': 500.0, 'rstart': 1.0, 'beamwidth': 2.0}, 
                              {'nbins': 480, 'nrays': 360, 'elangle': 1.5, 'rscale': 500.0, 'rstart': 2.0, 'beamwidth': 3.0},
                              {'nbins': 480, 'nrays': 360, 'elangle': 2.0, 'rscale': 500.0, 'rstart': 1.0, 'beamwidth': 2.0},
                              {'nbins': 480, 'nrays': 360, 'elangle': 4.0, 'rscale': 500.0, 'rstart': 2.0, 'beamwidth': 3.0}
                            ]
                           })
