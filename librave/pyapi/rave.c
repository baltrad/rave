/**
 * rave object wrapper functionality, accesses the python rave object through
 * the c-api.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-08-07
 */
#include "rave.h"
#include "h5rad.h"
#include "polar.h"
#include "getpy.h"
#include <limits.h>
#include "rave_alloc.h"

/**
 * Sets a python error message and then performs goto fail.
 */
#define RaiseException(type, message) {PyErr_SetString(type, message); goto fail; }

void initialize_RaveObject(RaveObject* v)
{
  v->data = NULL;
  v->topo = NULL;
  v->bitmap = NULL;
  v->info = NULL;
}

char array_type_2d(PyArrayObject* arr)
{
  return arr->descr->type;
}

int array_type_num_2d(PyArrayObject* arr)
{
  return arr->descr->type_num;
}

int array_stride_xsize_2d(PyArrayObject* arr)
{
  return arr->strides[0] / arr->strides[1];
}

unsigned char* array_data_2d(PyArrayObject* arr)
{
  return (unsigned char*) arr->data;
}

RaveDataType translate_pytype_to_ravetype(char type)
{
  RaveDataType result = RaveDataType_UNDEFINED;
  switch(type) {
  case 'c':
  case '1':
    result = RaveDataType_CHAR;
    break;
  case 'b':
  case 'B':
    result = RaveDataType_UCHAR;
    break;
  case 's':
    result = RaveDataType_SHORT;
    break;
  case 'i':
    result = RaveDataType_INT;
    break;
  case 'l':
    result = RaveDataType_LONG;
    break;
  case 'f':
    result = RaveDataType_FLOAT;
    break;
  case 'd':
    result = RaveDataType_DOUBLE;
    break;
  default:
    result = RaveDataType_UNDEFINED;
    break;
  }
  return result;
}

RaveDataType translate_pyarraytype_to_ravetype(int type)
{
  RaveDataType result = RaveDataType_UNDEFINED;
  switch(type) {
  case PyArray_CHAR:
  case PyArray_BYTE:
    result = RaveDataType_CHAR;
    break;
  case PyArray_UBYTE:
    result = RaveDataType_UCHAR;
    break;
  case PyArray_SHORT:
  case PyArray_USHORT:
    result = RaveDataType_SHORT;
    break;
  case PyArray_INT:
  case PyArray_UINT:
    result = RaveDataType_INT;
    break;
  case PyArray_LONG:
  case PyArray_ULONG:
    result = RaveDataType_LONG;
    break;
  case PyArray_FLOAT:
    result = RaveDataType_FLOAT;
    break;
  case PyArray_DOUBLE:
    result = RaveDataType_DOUBLE;
    break;
  default:
    result = RaveDataType_UNDEFINED;
    break;
  }
  return result;
}

int translate_ravetype_to_pyarraytype(RaveDataType type)
{
  int result = PyArray_NOTYPE;
  switch(type) {
  case RaveDataType_CHAR:
    result = PyArray_BYTE;
    break;
  case RaveDataType_UCHAR:
    result = PyArray_UBYTE;
    break;
  case RaveDataType_SHORT:
    result = PyArray_SHORT;
    break;
  case RaveDataType_INT:
    result = PyArray_INT;
    break;
  case RaveDataType_LONG:
    result = PyArray_LONG;
    break;
  case RaveDataType_FLOAT:
    result = PyArray_FLOAT;
    break;
  case RaveDataType_DOUBLE:
    result = PyArray_DOUBLE;
    break;
  default:
    result = PyArray_NOTYPE;
    break;
  }
  return result;
}


int pyarraytype_from_type(char type)
{
  switch (type) {
  case 'c': /* PyArray_CHAR */
    return PyArray_CHAR;
    break;
  case '1': /* PyArray_SBYTE */
    return PyArray_BYTE;
    break;
  case 'b': /* PyArray_UBYTE */
    return PyArray_UBYTE;
    break;
  case 'B': /* PyArray_UBYTE */
    return PyArray_UBYTE;
    break;
  case 's': /* PyArray_SHORT */
    return PyArray_SHORT;
    break;
  case 'i': /* PyArray_INT */
    return PyArray_INT;
    break;
  case 'l': /* PyArray_LONG */
    return PyArray_LONG;
    break;
  case 'f': /* PyArray_FLOAT */
    return PyArray_FLOAT;
    break;
  case 'd': /* PyArray_DOUBLE */
    return PyArray_DOUBLE;
    break;
  case 'F': /* PyArray_CFLOAT */
  case 'D': /* PyArray_CDOUBLE */
  case 'O': /* PyArray_OBJECT */
  default:
    /* Ignored for now => 0.0 */
    printf("pyarraytype_from_type: Unsupported type: '%c'\n", type);
  }
  return -1;
}

double calculate_cressman(double r, double R)
{
  return (R * R - r * r) / (R * R + r * r);
}

double calculate_inverse(double r, double R)
{
  return 1.0 - r / R;
}

PJ* get_rave_projection(PyObject* obj)
{
  PyObject *tmp = NULL;
  PyObject *projdef = NULL;
  PJ* retpj = NULL;

  tmp = GetPyStringFromINFO(obj, "/where/projdef");
  projdef = PyObject_CallMethod(tmp, "split", "s", " ");
  retpj = initProjection(projdef);

  Py_XDECREF(tmp);
  Py_XDECREF(projdef);

  return retpj;
}

static int _missing_attribute(char* string)
{
  PyErr_SetString(PyExc_AttributeError, string);
  return 0;
}

int fill_rave_image_info(PyObject *inobj, RaveImageStruct *p, int set)
{
  PyObject* po;
  PyObject* py_nod;
  PyObject* py_und;
  char* nodata;
  char* undetect;

  po = GetSequenceFromINFO(inobj, "/how/extent");
  if (!GetDoubleFromSequence(po, 0, &p->lowleft.u))
    return _missing_attribute("No extent in info");
  if (!GetDoubleFromSequence(po, 1, &p->lowleft.v))
    return _missing_attribute("No extent in info");
  if (!GetDoubleFromSequence(po, 2, &p->uppright.u))
    return _missing_attribute("No extent in info");
  if (!GetDoubleFromSequence(po, 3, &p->uppright.v))
    return _missing_attribute("No extent in info");
  Py_XDECREF(po);

  if (!GetIntFromINFO(inobj, "/where/xsize", &p->xsize))
    return _missing_attribute("No xsize in info");
  if (!GetIntFromINFO(inobj, "/where/ysize", &p->ysize))
    return _missing_attribute("No ysize in info");
  if (!GetDoubleFromINFO(inobj, "/where/xscale", &p->xscale))
    return _missing_attribute("No xscale in info");
  if (!GetDoubleFromINFO(inobj, "/where/yscale", &p->yscale))
    return _missing_attribute("No yscale in info");

  py_nod = PyString_FromFormat("/image%d/what/nodata", set);
  py_und = PyString_FromFormat("/image%d/what/undetect", set);
  nodata = PyString_AsString(py_nod);
  undetect = PyString_AsString(py_und);

  if (!GetDoubleFromINFO(inobj, nodata, &p->nodata))
    return _missing_attribute("No nodata in info");
  if (!GetDoubleFromINFO(inobj, undetect, &p->noecho))
    return _missing_attribute("No undetect in info");

  Py_XDECREF(py_nod);
  Py_XDECREF(py_und);
  PyErr_Clear();
  return 1;
}

int fill_rave_area_extent(PyObject* dict, RaveImageStruct* p)
{
  PyObject* po;
  po = PyMapping_GetItemString(dict, "area_extent");

  if (po) {
    int wasok = 0;
    wasok |= !getIdxDoubleFromTuple(0, &p->lowleft.u, po);
    wasok |= !getIdxDoubleFromTuple(1, &p->lowleft.v, po);
    wasok |= !getIdxDoubleFromTuple(2, &p->uppright.u, po);
    wasok |= !getIdxDoubleFromTuple(3, &p->uppright.v, po);
    Py_DECREF(po);
    if (wasok) {
      return _missing_attribute("Area extent corrupt");
    }
  } else {
    return _missing_attribute("No area extent in info");
  }
  return 1;
}

int fill_rave_object(PyObject *robj, RaveObject *obj, int set,
  const char *dsetname)
{
  PyObject *po;
  PyObject *py_dat;
  char* dataset;

  py_dat = PyString_FromFormat("%s%d", dsetname, set);
  dataset = PyString_AsString(py_dat);

  po = PyObject_GetAttrString(robj, "data");
  if (!po || !PyDict_CheckExact(po)) {
    obj->data = NULL;
    Py_XDECREF(po);
    Py_XDECREF(py_dat);
    printf("Failed to fill data\n");
    return 0;
  } else {
    obj->data = (PyArrayObject*) PyDict_GetItemString(po, dataset);
  }
  Py_XDECREF(po);

  po = PyObject_GetAttrString(robj, "info");
  if (!po) {
    obj->info = NULL;
    Py_XDECREF(po);
    printf("Failed to fill info\n");
    return 0;
  } else {
    obj->info = po;
  }
  obj->topo = NULL; /* Deprecated objects must be initialized */
  obj->bitmap = NULL; /* in order to be safely ignored. */

  Py_XDECREF(po);
  Py_XDECREF(py_dat);
  PyErr_Clear();
  return 1;
}

void free_rave_object(RaveObject* obj)
{
  if (obj->info) {
    Py_XDECREF(obj->info);
  }
  if (obj->data) {
    Py_XDECREF(obj->data);
  }
  if (obj->topo) {
    Py_XDECREF(obj->topo);
  }
  if (obj->bitmap) {
    Py_XDECREF(obj->bitmap);
  }
}

void free_pypolar_volume(PyRavePolarVolume* vol)
{
  if (vol != NULL) {
    if (vol->fields != NULL) {
      int i = 0;
      for (i = 0; i < vol->fieldsn;i++) {
        Py_XDECREF(vol->fields[i]);
      }
      RAVE_FREE(vol->fields);
    }
  }
}

int fill_pypolar_volume(PyRavePolarVolume* vol, PyObject* pyobj)
{
  int result = 0;
  PyObject *info = NULL, *data = NULL, *po = NULL, *p_area = NULL;
  int i, n;

  if (!(info = PyObject_GetAttrString(pyobj, "info"))) {
    return 0;
  }
  if (!(data = PyObject_GetAttrString(pyobj, "data"))) {
    Py_XDECREF(info);
    return 0;
  }

  vol->fields = NULL;
  vol->volume.fields = NULL;
  vol->volume.got_cressmanxy = 0;
  vol->volume.got_cressmanz = 0;
  vol->fieldsn = 0;
  vol->volume.fieldsn = 0;

  if (!PyList_Check(data)) {
    PyErr_SetString(PyExc_TypeError,
                    "Volume does not contain a list of elevations");
    goto fail;
  }

  n = PyObject_Length(data);
  po = PyMapping_GetItemString(info, "elev");
  if (!po) {
    PyErr_SetString(PyExc_AttributeError,
                    "Volume does not contain list of elevs");
    goto fail;
  }

  if (PyObject_Length(po) != n) {
    PyErr_SetString(PyExc_ValueError, "Not same number of elevs as fields");
    goto fail;
  }

  vol->fields = RAVE_MALLOC(sizeof(PyArrayObject*) * n);
  vol->volume.fields = RAVE_MALLOC(sizeof(RavePolarField) * n);
  if (vol->fields == NULL || vol->volume.fields == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory");
    goto fail;
  }
  vol->fieldsn = n;
  vol->volume.fieldsn = n;

  for (i = 0; i < n; i++) {
    vol->fields[i] = NULL;
    memset(&vol->volume.fields[i], 0, sizeof(RavePolarField));
  }

  for (i = 0; i < n; i++) {
    PyObject* po2 = NULL; // Will not own reference so do not release
    if (!(vol->fields[i] = (PyArrayObject*) PySequence_GetItem(data, i))) {
      PyErr_SetString(PyExc_ValueError, "Could not get data from field sequence");
      goto fail;
    }
    vol->volume.fields[i].src = array_data_2d(vol->fields[i]);
    vol->volume.fields[i].type = array_type_2d(vol->fields[i]);
    vol->volume.fields[i].stride = array_stride_xsize_2d(vol->fields[i]);
    if (!(po2 = PyList_GetItem(po, i))) {
      PyErr_SetString(PyExc_ValueError, "Could not get elevation for field");
      goto fail;
    }
    vol->volume.fields[i].elev = PyFloat_AsDouble(po2) * DEG_TO_RAD;
  }

  if (!getDoubleFromDictionary("nodata", &vol->volume.nodata, info)) {
    RaiseException(PyExc_AttributeError, "No nodata in info");
  }

  if (!getDoubleFromDictionary("noecho", &vol->volume.noecho, info)) {
    RaiseException(PyExc_AttributeError, "No noecho in info");
  }

  if (!getDoubleFromDictionary("beamwidth", &vol->volume.beamwidth, info)) {
    RaiseException(PyExc_AttributeError, "No beamwidth in info");
  }

  if (!getDoubleFromDictionary("geo_height", &vol->volume.geo_height, info)) {
    RaiseException(PyExc_AttributeError, "No geo_height in info");
  }

  if (!getIntFromDictionary("range_uppb", &vol->volume.range_uppb, info)) {
    RaiseException(PyExc_AttributeError, "No range_uppb in info");
  }

  if (!getIntFromDictionary("azim_uppb", &vol->volume.azim_uppb, info)) {
    RaiseException(PyExc_AttributeError, "No azim_uppb in info");
  }

  if (vol->volume.azim_uppb == 0.0) {
    RaiseException(PyExc_AttributeError, "azim_uppb == 0.0 => Null division");
  }

  vol->volume.azimuth_offset = (360.0 / vol->volume.azim_uppb) * DEG_TO_RAD;
  vol->volume.beamwidth *= DEG_TO_RAD;

  if (getDoubleFromDictionary("cressman_xy", &vol->volume.cressmanxy, info))
    vol->volume.got_cressmanxy = 1;

  if (getDoubleFromDictionary("cressman_z", &vol->volume.cressmanz, info))
    vol->volume.got_cressmanz = 1;

  Py_XDECREF(po);
  if ((po = PyMapping_GetItemString(info, "geo_coord")) == NULL) {
    goto fail;
  }

  if (!getIdxDoubleFromTuple(0, &vol->volume.lon, po)) {
    RaiseException(PyExc_AttributeError, "Not proper lon info in geo_coord");
  }
  if (!getIdxDoubleFromTuple(1, &vol->volume.lat, po)) {
    RaiseException(PyExc_AttributeError, "Not proper lat info in geo_coord");
  }

  vol->volume.lon *= DEG_TO_RAD;
  vol->volume.lat *= DEG_TO_RAD;

  if ((p_area = PyObject_GetAttrString(pyobj, "p_area")) == NULL) {
    RaiseException(PyExc_TypeError, "source.p_area does not exist");
  }

  Py_XDECREF(po);
  if ((po = PyObject_GetAttrString(p_area, "range_size")) == NULL) {
    vol->volume.scale = 2000.0;
    PyErr_Clear();
  } else {
    vol->volume.scale = PyFloat_AsDouble(po);
  }

  result = 1;
fail:
  Py_XDECREF(po);
  Py_XDECREF(p_area);
  Py_XDECREF(info);
  Py_XDECREF(data);
  return result;
}
