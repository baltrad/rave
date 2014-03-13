/* --------------------------------------------------------------------
Copyright (C) 2012 Institute of Meteorology and Water Management -
National Research Institute, IMGW-PIB

This file is part of Radvol-QC package.

Radvol-QC is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Radvol-QC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Radvol-QC.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/
/**
 * Tests for radvolatt.
 * @file testRadvolAtt.c
 * @author Katarzyna Osrodka (Institute of Meteorology and Water Management, IMGW-PIB)
 * @date 2012-10-15
 */

#include <stdio.h>
#include <stdlib.h>
#include "CUnit/Basic.h"
#include "radvolatt.h"
#include "rave_io.h"
#include "rave_alloc.h"
#include "rave_attribute.h"
#include "rave_field.h"
#include "polarvolume.h"
#include "polarscan.h"
#include "polarscanparam.h"

/*
 * CUnit Test Suite
 */

char* XML_FILE;
char* H5_FILE;
char* H5_FILE_COR;
static FILE* temp_file = NULL;

int init_suite_testRadvolAtt(void) {
  XML_FILE = "fixtures/radvol_params.xml";
  if (NULL == (temp_file = fopen(XML_FILE, "r"))) {
    return -1;
  } else {
    fclose(temp_file);
  }
  H5_FILE = "fixtures/fake_att.h5";
  if (NULL == (temp_file = fopen(H5_FILE, "r"))) {
    return -1;
  } else {
    fclose(temp_file);
  }
  H5_FILE_COR = "fixtures/fake_att_cor.h5";
  if (NULL == (temp_file = fopen(H5_FILE_COR, "r"))) {
    return -1;
  } else {
    fclose(temp_file);
  }
  return 0;
}

int clean_suite_testRadvolAtt(void) {
  return 0;
}

void testRadvolAtt_attCorrection(void) {
  PolarVolume_t* pvol = NULL;
  PolarScan_t* scan = NULL;
  RaveIO_t* raveio = NULL;
  PolarScanParam_t* parameter = NULL;
  RaveAttribute_t* attribute = NULL;
  char* value = NULL;

  raveio = RaveIO_open(H5_FILE);
  CU_ASSERT_PTR_NOT_NULL_FATAL(raveio);
  pvol = (PolarVolume_t*) RaveIO_getObject(raveio);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol);
  RadvolAtt_attCorrection_pvol(pvol, XML_FILE);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol);
  CU_ASSERT_EQUAL_FATAL(PolarVolume_getNumberOfScans(pvol), 1);
  scan = PolarVolume_getScan(pvol, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan);
  parameter = PolarScan_getParameter(scan, "DBZH");
  CU_ASSERT_PTR_NOT_NULL_FATAL(parameter);
  attribute = PolarScanParam_getAttribute(parameter, "how/task");
  CU_ASSERT_PTR_NOT_NULL_FATAL(attribute);
  RaveAttribute_getString(attribute, &value);
  CU_ASSERT_STRING_EQUAL(value, "pl.imgw.radvolqc.att");
  attribute = PolarScanParam_getAttribute(parameter, "how/task_args");
  CU_ASSERT_PTR_NOT_NULL_FATAL(attribute);
  RaveAttribute_getString(attribute, &value);
  CU_ASSERT_STRING_EQUAL(value, "ATT: ATT_QI1=1.0, ATT_QI0=5.0, ATT_QIUn=0.9, ATT_a= 0.0044, ATT_b= 1.17, ATT_ZRa=200.0, ATT_ZRb= 1.6, ATT_Refl= 4.0, ATT_Last= 1.0, ATT_Sum= 5.0");

  RAVE_OBJECT_RELEASE(pvol);
  RAVE_OBJECT_RELEASE(scan);
  RAVE_OBJECT_RELEASE(parameter);
  RAVE_OBJECT_RELEASE(attribute);
  RAVE_OBJECT_RELEASE(raveio);
}

void testRadvolAtt_attCorrection_topLevel_correction(void) {
  RaveIO_t* raveio_in = NULL;
  PolarVolume_t* pvol_in = NULL;
  PolarScan_t* scan_in = NULL;
  RaveIO_t* raveio_cor = NULL;
  PolarVolume_t* pvol_cor = NULL;
  PolarScan_t* scan_cor = NULL;
  int nbin;
  int nray;
  int bi, ri;
  double value_in, value_cor;

  raveio_in = RaveIO_open(H5_FILE);
  CU_ASSERT_PTR_NOT_NULL_FATAL(raveio_in);
  pvol_in = (PolarVolume_t*) RaveIO_getObject(raveio_in);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol_in);
  RadvolAtt_attCorrection_pvol(pvol_in, XML_FILE);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol_in);
  CU_ASSERT_EQUAL_FATAL(PolarVolume_getNumberOfScans(pvol_in), 1);
  scan_in = PolarVolume_getScan(pvol_in, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan_in);

  raveio_cor = RaveIO_open(H5_FILE_COR);
  CU_ASSERT_PTR_NOT_NULL_FATAL(raveio_cor);
  pvol_cor = (PolarVolume_t*) RaveIO_getObject(raveio_cor);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol_cor);
  CU_ASSERT_EQUAL_FATAL(PolarVolume_getNumberOfScans(pvol_cor), 1);
  scan_cor = PolarVolume_getScan(pvol_cor, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan_cor);

  CU_ASSERT_TRUE_FATAL(PolarScan_hasParameter(scan_in, "DBZH"));
  CU_ASSERT_TRUE_FATAL(PolarScan_hasParameter(scan_cor, "DBZH"));

  nbin = PolarScan_getNbins(scan_in);
  nray = PolarScan_getNrays(scan_in);
  CU_ASSERT_EQUAL_FATAL(PolarScan_getNbins(scan_cor), nbin);
  CU_ASSERT_EQUAL_FATAL(PolarScan_getNrays(scan_cor), nray);

  for (ri = 0; ri < nray; ri++) {
    for (bi = 0; bi < nbin; bi++) {
      PolarScan_getParameterValue(scan_in, "DBZH", bi, ri, &value_in);
      PolarScan_getParameterValue(scan_cor, "DBZH", bi, ri, &value_cor);
      if (value_in != value_cor) {
        printf("\nbin=%d ray=%d value_in=%d value_cor=%d\n", bi, ri, (int) value_in, (int) value_cor);
        CU_ASSERT_FALSE_FATAL(fabs(value_in - value_cor) > 1);
      }
    }
  }

  RAVE_OBJECT_RELEASE(pvol_in);
  RAVE_OBJECT_RELEASE(scan_in);
  RAVE_OBJECT_RELEASE(raveio_in);
  RAVE_OBJECT_RELEASE(pvol_cor);
  RAVE_OBJECT_RELEASE(scan_cor);
  RAVE_OBJECT_RELEASE(raveio_cor);
}

void testRadvolAtt_attCorrection_topLevel_quality(void) {
  RaveIO_t* raveio_in = NULL;
  PolarVolume_t* pvol_in = NULL;
  PolarScan_t* scan_in = NULL;
  RaveField_t* field_in = NULL;
  RaveIO_t* raveio_cor = NULL;
  PolarVolume_t* pvol_cor = NULL;
  PolarScan_t* scan_cor = NULL;
  RaveField_t* field_cor = NULL;
  int nbin;
  int nray;
  int bi, ri;
  double value_in, value_cor;

  raveio_in = RaveIO_open(H5_FILE);
  CU_ASSERT_PTR_NOT_NULL_FATAL(raveio_in);
  pvol_in = (PolarVolume_t*) RaveIO_getObject(raveio_in);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol_in);
  RadvolAtt_attCorrection_pvol(pvol_in, XML_FILE);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol_in);
  CU_ASSERT_EQUAL_FATAL(PolarVolume_getNumberOfScans(pvol_in), 1);
  scan_in = PolarVolume_getScan(pvol_in, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan_in);

  raveio_cor = RaveIO_open(H5_FILE_COR);
  CU_ASSERT_PTR_NOT_NULL_FATAL(raveio_cor);
  pvol_cor = (PolarVolume_t*) RaveIO_getObject(raveio_cor);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol_cor);
  CU_ASSERT_EQUAL_FATAL(PolarVolume_getNumberOfScans(pvol_cor), 1);
  scan_cor = PolarVolume_getScan(pvol_cor, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan_cor);

  CU_ASSERT_EQUAL_FATAL(PolarScan_getNumberOfQualityFields(scan_in), 1);
  field_in = PolarScan_getQualityField(scan_in, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(field_in);
  CU_ASSERT_EQUAL_FATAL(PolarScan_getNumberOfQualityFields(scan_cor), 1);
  field_cor = PolarScan_getQualityField(scan_cor, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(field_cor);

  nbin = PolarScan_getNbins(scan_in);
  nray = PolarScan_getNrays(scan_in);
  CU_ASSERT_EQUAL_FATAL(PolarScan_getNbins(scan_cor), nbin);
  CU_ASSERT_EQUAL_FATAL(PolarScan_getNrays(scan_cor), nray);

  for (ri = 0; ri < nray; ri++) {
    for (bi = 0; bi < nbin; bi++) {
      RaveField_getValue(field_in, bi, ri, &value_in);
      RaveField_getValue(field_cor, bi, ri, &value_cor);
      if (value_in != value_cor) {
        printf("\nbin=%d ray=%d value_in=%d value_cor=%d\n", bi, ri, (int) value_in, (int) value_cor);
        CU_ASSERT_FALSE_FATAL(fabs(value_in - value_cor) > 1);
      }
    }
  }

  RAVE_OBJECT_RELEASE(pvol_in);
  RAVE_OBJECT_RELEASE(scan_in);
  RAVE_OBJECT_RELEASE(field_in);
  RAVE_OBJECT_RELEASE(raveio_in);
  RAVE_OBJECT_RELEASE(pvol_cor);
  RAVE_OBJECT_RELEASE(scan_cor);
  RAVE_OBJECT_RELEASE(field_cor);
  RAVE_OBJECT_RELEASE(raveio_cor);
}

int testRadvolAtt_main(void) {
  CU_pSuite pSuiteAtt = NULL;

  /* Add a suite to the registry */
  pSuiteAtt = CU_add_suite("testRadvolAtt", init_suite_testRadvolAtt, clean_suite_testRadvolAtt);
  if (NULL == pSuiteAtt) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  /* Add the tests to the suite */
  if ((NULL == CU_add_test(pSuiteAtt, "testRadvolAtt_attCorrection", testRadvolAtt_attCorrection)) ||
          (NULL == CU_add_test(pSuiteAtt, "testRadvolAtt_attCorrection_topLevel_correction", testRadvolAtt_attCorrection_topLevel_correction)) ||
          (NULL == CU_add_test(pSuiteAtt, "testRadvolAtt_attCorrection_topLevel_quality", testRadvolAtt_attCorrection_topLevel_quality))) {
    CU_cleanup_registry();
    return CU_get_error();
  }
  return 0;
}

