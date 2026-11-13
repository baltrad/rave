'''
Copyright (C) 2026 Swedish Meteorological and Hydrological Institute, SMHI,

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

Tests the HOOF dealiasing functionality

@file
@author Peter Smerkol, Vito Švagelj, ARSO
@date 2026-11-13
'''
import unittest
import os
import tempfile
import math
import numpy
import _raveio
import _hoofdealias
import _polarvolume, _polarscan, _polarscanparam

class PyHoofDealiasTest(unittest.TestCase):
    FIXTURE = "fixtures/sehem_pvol_pn215_20171204T071500Z_0x81540b.h5"
    NO_NI_FIXTURE = "fixtures/var-20111127T194500Z.h5"
    TASK = "si.arso.detector.hoofdealias"

    def setUp(self):
        pass

    def tearDown(self):
        pass

    ##
    # Creates a polar volume with a uniform wind (u, v), with the radial velocities aliased into
    # [-NI, NI). Returns the volume and the true radial velocities (NaN where nodata) for each scan.
    def createSyntheticVolume(self, u=15.0, v=-20.0, ni=10.0, elangles=(0.5, 1.5, 3.0), quantity="VRADH", normalized=False):
        nrays, nbins, rscale = 360, 200, 500.0
        pvol = _polarvolume.new()
        pvol.longitude = 14.5 * math.pi / 180.0
        pvol.latitude = 46.0 * math.pi / 180.0
        pvol.height = 1000.0
        scale = ni if normalized else 1.0
        gain = 2.0 * ni / scale / 253.0
        offset = -ni / scale - gain
        truths = []
        for el in elangles:
            scan = _polarscan.new()
            scan.elangle = el * math.pi / 180.0
            scan.rscale = rscale
            scan.rstart = 0.0
            scan.addAttribute("how/NI", ni)
            az = (numpy.arange(nrays) * 2.0 * math.pi / nrays)[:, None] * numpy.ones((1, nbins))
            vr = math.cos(el * math.pi / 180.0) * (u * numpy.sin(az) + v * numpy.cos(az))
            va = numpy.mod(vr + ni, 2.0 * ni) - ni
            raw = numpy.clip(numpy.floor((va / scale - offset) / gain + 0.5), 1, 254).astype(numpy.uint8)
            raw[:, :4] = 255
            param = _polarscanparam.new()
            param.quantity = quantity
            param.gain = gain
            param.offset = offset
            param.nodata = 255.0
            param.undetect = 0.0
            param.setData(raw)
            scan.addParameter(param)
            pvol.addScan(scan)
            truths.append(numpy.where(raw == 255, numpy.nan, vr))
        return pvol, truths

    ##
    # Returns the decoded values of a parameter, NaN where nodata or undetect
    def decode(self, param):
        data = param.getData().astype(float)
        return numpy.where((data == param.nodata) | (data == param.undetect), numpy.nan, data * param.gain + param.offset)

    ##
    # Returns the quality field of a parameter with the HOOF dealiasing task, or None
    def getQualityField(self, param):
        for i in range(param.getNumberOfQualityFields()):
            field = param.getQualityField(i)
            if field.getAttribute("how/task") == self.TASK:
                return field
        return None

    ##
    # Checks that the dealiased volume reproduces the true radial velocities. The highest bins
    # of the volume lie in the top height layer, which has too few points for a wind model.
    def assertSyntheticDealiased(self, pvol, truths):
        nvalid = nrecovered = 0
        for i, truth in enumerate(truths):
            param = pvol.getScan(i).getParameter("VRADH")
            values = self.decode(param)
            valid = numpy.isfinite(truth)
            recovered = numpy.isfinite(values)
            self.assertFalse(numpy.any(recovered & ~valid))
            self.assertLessEqual(numpy.max(numpy.abs(values[recovered] - truth[recovered])), param.gain)
            nvalid += valid.sum()
            nrecovered += recovered.sum()
        self.assertGreater(nrecovered, 0.995 * nvalid)

    def test_notDealiased(self):
        scan = _polarscan.new()
        vrad = _polarscanparam.new()
        vrad.quantity = "VRADH"
        scan.addParameter(vrad)
        self.assertFalse(_hoofdealias.dealiased(scan))

    def test_dealiased(self):
        scan = _polarscan.new()
        vrad = _polarscanparam.new()
        vrad.quantity = "VRADH"
        vrad.addAttribute("how/dealiased", "True")
        scan.addParameter(vrad)
        self.assertTrue(_hoofdealias.dealiased(scan))
        self.assertFalse(_hoofdealias.dealiased(scan, "VRADV"))

    def test_dealias_synthetic(self):
        pvol, truths = self.createSyntheticVolume()
        self.assertTrue(_hoofdealias.dealias(pvol))
        self.assertSyntheticDealiased(pvol, truths)

    def test_dealias_synthetic_normalized(self):
        pvol, truths = self.createSyntheticVolume(normalized=True)
        self.assertTrue(_hoofdealias.dealias(pvol))
        self.assertSyntheticDealiased(pvol, truths)

    def test_dealias_synthetic_quantity(self):
        pvol, truths = self.createSyntheticVolume(quantity="VRAD")
        self.assertFalse(_hoofdealias.dealias(pvol))
        self.assertTrue(_hoofdealias.dealias(pvol, "VRAD"))
        self.assertTrue(_hoofdealias.dealiased(pvol.getScan(0), "VRAD"))

    def test_dealias_synthetic_encoding(self):
        pvol, truths = self.createSyntheticVolume()
        _hoofdealias.dealias(pvol)
        for i in range(pvol.getNumberOfScans()):
            param = pvol.getScan(i).getParameter("VRADH")
            data = param.getData()
            self.assertEqual(255.0, param.nodata)
            self.assertEqual(0.0, param.undetect)
            self.assertEqual("True", param.getAttribute("how/dealiased"))
            self.assertEqual(self.TASK, param.getAttribute("how/task"))
            field = self.getQualityField(param)
            self.assertIsNotNone(field)
            quality = field.getData()
            self.assertEqual(set([0, 255]), set(numpy.unique(quality)))
            self.assertTrue(numpy.array_equal(quality == 255, (data != 255) & (data != 0)))
            self.assertTrue(numpy.all((data[quality == 255] >= 1) & (data[quality == 255] <= 254)))

    def test_dealias_synthetic_maxwind(self):
        # with the maximum wind below the wind speed there is no wind model, so nothing can be dealiased
        pvol, truths = self.createSyntheticVolume()
        self.assertTrue(_hoofdealias.dealias(pvol, maxwind=20.0))
        for i in range(pvol.getNumberOfScans()):
            self.assertTrue(numpy.all(numpy.isnan(self.decode(pvol.getScan(i).getParameter("VRADH")))))

    def test_dealias_twice(self):
        pvol, truths = self.createSyntheticVolume()
        self.assertTrue(_hoofdealias.dealias(pvol))
        self.assertFalse(_hoofdealias.dealias(pvol))

    def test_dealias_fixture(self):
        pvol = _raveio.open(self.FIXTURE).object
        before = [self.decode(pvol.getScan(i).getParameter("VRADH")) for i in range(pvol.getNumberOfScans())]
        self.assertTrue(_hoofdealias.dealias(pvol))
        ndealiased = 0
        for i in range(pvol.getNumberOfScans()):
            scan = pvol.getScan(i)
            param = scan.getParameter("VRADH")
            after = self.decode(param)
            self.assertTrue(_hoofdealias.dealiased(scan))
            # only measured bins can be dealiased
            self.assertFalse(numpy.any(numpy.isfinite(after) & numpy.isnan(before[i])))
            # dealiased values differ from the measurements by multiples of 2 NI
            both = numpy.isfinite(after)
            ni = scan.getAttribute("how/NI")
            n = (after[both] - before[i][both]) / (2.0 * ni)
            self.assertLessEqual(numpy.max(numpy.abs(n - numpy.round(n))) * 2.0 * ni, param.gain)
            ndealiased += both.sum()
        self.assertGreater(ndealiased, 0)

    def test_dealias_rstart_odim_versions(self):
        # rstart is in km before ODIM 2.4 and in metres from 2.4 on, the result must not depend on the
        # ODIM version of the file
        results = []
        for version in (None, _raveio.RaveIO_ODIM_Version_2_2, _raveio.RaveIO_ODIM_Version_2_4):
            pvol = _raveio.open(self.FIXTURE).object
            for i in range(pvol.getNumberOfScans()):
                pvol.getScan(i).rstart = 0.6
            if version is not None:
                fd, path = tempfile.mkstemp(suffix=".h5")
                os.close(fd)
                try:
                    writer = _raveio.new()
                    writer.object = pvol
                    writer.version = version
                    writer.save(path)
                    pvol = _raveio.open(path).object
                finally:
                    os.unlink(path)
            self.assertAlmostEqual(0.6, pvol.getScan(0).rstart, 6)
            self.assertTrue(_hoofdealias.dealias(pvol))
            results.append([self.decode(pvol.getScan(i).getParameter("VRADH")) for i in range(pvol.getNumberOfScans())])
        for result in results[1:]:
            for a, b in zip(results[0], result):
                self.assertTrue(numpy.array_equal(a, b, equal_nan=True))

    def test_dealias_no_NI(self):
        pvol = _raveio.open(self.NO_NI_FIXTURE).object
        before = pvol.getScan(0).getParameter("VRADH").getData().copy()
        self.assertFalse(_hoofdealias.dealias(pvol))
        self.assertTrue(numpy.array_equal(before, pvol.getScan(0).getParameter("VRADH").getData()))
        self.assertFalse(_hoofdealias.dealiased(pvol.getScan(0)))

    def test_dealias_scan(self):
        pvol, truths = self.createSyntheticVolume()
        try:
            _hoofdealias.dealias(pvol.getScan(0))
            self.fail("Expected AttributeError")
        except AttributeError:
            pass

    def test_dealias_bad_parameters(self):
        pvol, truths = self.createSyntheticVolume()
        for kwargs in ({"zsector": 0.0}, {"maxwind": -1.0}):
            try:
                _hoofdealias.dealias(pvol, **kwargs)
                self.fail("Expected ValueError")
            except ValueError:
                pass

if __name__ == "__main__":
    unittest.main()
