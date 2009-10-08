/**
 * Compositing functionality in RAVE.
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2006-
 */
#include <Python.h>
#include <arrayobject.h>
#include "rave.h"

static PyObject *ErrorObject;

/**
 * Sets a python error exception.
 */
#define Raise(type,msg) {PyErr_SetString(type,msg);}

typedef struct {
  UV lowleft;
  UV center;

  int llx;
  int lly;
  int cx;
  int cy;
  int ux;
  int uy;

  unsigned char* src;
  PyArrayObject* data;

  unsigned char* srctopo;
  PyArrayObject* topodata;

  int maxx;
  int maxy;
  int inxsize;
  int topoinxsize;
} RAD;

typedef struct {
  int noi;
  int* ridx;
  int* xpos;
  int* ypos;
  double* vidx;
} HiLi; /*Hit list*/

typedef struct {
  UV llC;
  UV ulC;
  /*  double upper;*/

  RAD* rad;
  PJ* proj;
  int noofrad;
  int outxsize;
  int maxx;
  int maxy;
  double ux;
  double uy;
  int type;
  int topotype;
  /*int nodata;*/
  double nodata;
  double xscale;
  double yscale;
  unsigned char* desta;
  unsigned char* bitmapa;

  int bitmode;

  /*For out topography, will actually contain index colors for later use*/
  unsigned char* outtopo;
  int outtopotype;
  int outtopoxsize;

} CompWrapper;

static double getarritem(RAD* rad, int x, int y, CompWrapper* wrap)
{
  return get_array_item_2d(rad->src, x, y, wrap->type, rad->inxsize);
}

static double gettopoarritem(RAD* rad, int x, int y, CompWrapper* wrap)
{
  return get_array_item_2d(rad->srctopo, x, y, wrap->topotype, rad->topoinxsize);
}

static void setarritem(int x, int y, double v, CompWrapper* wrap)
{
  set_array_item_2d(wrap->desta, x, y, v, wrap->type, wrap->outxsize);
}

static void settopoarritem(int x, int y, double v, CompWrapper* wrap)
{
  set_array_item_2d(wrap->outtopo, x, y, v, wrap->outtopotype,
                    wrap->outtopoxsize);
}

static void setBitmapArritem(int x, int y, double v, CompWrapper* wrap)
{
  if (wrap->bitmode == 0 || wrap->bitmode == 3)
    return;
  set_array_item_2d(wrap->bitmapa, x, y, v, 'b', wrap->outxsize);
}

/*
 The following three functions have been commented since they're not being
 used right now. They may be activated later, but for now we just want
 to shut up the compiler...
 */
/* static double getBitmapArritem(int x, int y, CompWrapper* wrap) */
/* { */
/*    return get_array_item_2d(wrap->bitmapa, x, y, 'b', wrap->outxsize); */
/* } */

/* static double getddarritem(int x, int y, CompWrapper* wrap) */
/* { */
/*    return get_array_item_2d(wrap->desta, x, y, wrap->type, wrap->outxsize); */
/* } */

/* static int checkOverlapping(int myindex,CompWrapper* wrap) */
/* { */
/*    int x,y,i; */
/*    int overlapn=0; */
/*    int offx,offy; */
/*    int rx,ry; */
/*    //   int debug=0; */
/*    double refr; */
/*    double myrefr; */
/*    double dist; */
/*    int wasoverlapping=0; */

/*    int radarok=1; */

/*    int this_overDbz0=0; */
/*    int both_overDbz0=0; */
/*    int both_sameDbz=0; */

/*    int this_xm=(int)(wrap->rad[myindex].maxx/2); */
/*    int this_ym=(int)(wrap->rad[myindex].maxy/2); */
/*    int that_xm=0; */
/*    int that_ym=0; */

/*    for(y=0;y<wrap->rad[myindex].maxy;y++) { */
/*       offy=y+wrap->rad[myindex].uy; */
/*       for(x=0;x<wrap->rad[myindex].maxx;x++) { */
/* 	 offx=x+wrap->rad[myindex].ux; */
/* 	 for(i=0;i<wrap->noofrad;i++) { */
/* 	    if(i==myindex) */
/* 	       continue; */

/* 	    rx=offx-wrap->rad[i].ux; */
/* 	    ry=offy-wrap->rad[i].uy; */
/* 	    if(rx>=0 && rx<wrap->rad[i].maxx && */
/* 	       ry>=0 && ry<wrap->rad[i].maxy) { */
//	       wasoverlapping=1; /*Indicate that this radar has got something which
//				  *is overlapped*/
/* 	       dist=sqrt((double)((this_xm-x)*(this_xm-x)+(this_ym-y)*(this_ym-y))); */
/* 	       if(dist>90) */
/* 		  continue; */
/* 	       myrefr=getarritem(&wrap->rad[myindex],x,y,wrap); */
/* 	       if(myrefr==wrap->nodata) */
/* 		  continue; */

/* 	       that_xm=(int)(wrap->rad[i].maxx/2); */
/* 	       that_ym=(int)(wrap->rad[i].maxy/2); */

/* 	       dist=sqrt((double)((that_xm-rx)*(that_xm-rx)+(that_ym-ry)*(that_ym-ry))); */

/* 	       if(dist>90) */
/* 		  continue; */

/* 	       refr=getarritem(&wrap->rad[i],rx,ry,wrap); */
/* 	       if(refr!=wrap->nodata) { */
/* 		  overlapn++; */
/* 		  this_overDbz0+=(myrefr>75); */
/* 		  both_overDbz0+=(myrefr>75&&refr>75); */
/* 		  both_sameDbz+=((myrefr>75&&refr>75) || (myrefr<75&&refr<75)); */
/* 	       } */
/* 	    } */
/* 	 } */
/*       } */
/*    } */

/*    if(overlapn>100) { */
/*       double double_thisOverDbz0, double_bothOverDbz0, double_sameDbz; */
/*       double_thisOverDbz0=(double)this_overDbz0/(double)overlapn; */
/*       double_sameDbz=(double)both_sameDbz/(double)overlapn; */
/*       double_bothOverDbz0=(double)(both_overDbz0+1)/(double)(this_overDbz0+1); */

/*       if(double_thisOverDbz0>0.03) { */
/* 	 if(double_bothOverDbz0<0.5) */
/* 	    radarok=0; */
/* 	 else */
/* 	    radarok=1; */
/*       } else if(double_sameDbz<0.9) { */
/* 	 radarok=0; */
/*       } else { */
/* 	 radarok=1; */
/*       } */
/*    } else { */
/*       if(wasoverlapping) */
/* 	 radarok=0; */
/*       else */
/* 	 radarok=1; */
/*    } */

/*    return radarok; */
/* } */

static void NearestLoop(CompWrapper* wrap)
{
  int x, y;
  int i, radidx = -1, old_radidx = -1;
  int offx, offy;
  double dist;
  double mindist = 1e10;
  double value;
  //   double value,old_value=wrap->nodata;
  int* lastrow = malloc(sizeof(int) * wrap->maxx);

  for (i = 0; i < wrap->maxx; i++) {
    lastrow[i] = -1;
  }

  for (y = 0; y < wrap->maxy; y++) {
    for (x = 0, old_radidx = -1, radidx = -1; x < wrap->maxx; x++) {
      for (i = 0, radidx = -1, mindist = 1e10; i < wrap->noofrad; i++) {
        double dx, dy;
        dx = wrap->rad[i].cx - x;
        dy = wrap->rad[i].cy - y;

        dist = sqrt(dx * dx + dy * dy);

        offx = x - wrap->rad[i].ux;
        offy = y - wrap->rad[i].uy;
        value = wrap->nodata;
        if (offx >= 0 && offx < wrap->rad[i].maxx && offy >= 0 && offy
            < wrap->rad[i].maxy) {
          value = getarritem(&wrap->rad[i], offx, offy, wrap);
        }

        if (dist < mindist) {
          if (value != wrap->nodata) {
            mindist = dist;
            radidx = i;
          }
        }
      }
      if (radidx == -1) {
        setBitmapArritem(x, y, 0, wrap);
        lastrow[x] = -1;
        old_radidx = -1;
        continue;
      }

      offx = x - wrap->rad[radidx].ux;
      offy = y - wrap->rad[radidx].uy;

      value = wrap->nodata;

      if (offx >= 0 && offx < wrap->rad[radidx].maxx && offy >= 0 && offy
          < wrap->rad[radidx].maxy) {
        value = getarritem(&wrap->rad[radidx], offx, offy, wrap);

        if (value != wrap->nodata) {
          setarritem(x, y, value, wrap);
          if (wrap->outtopo)
            settopoarritem(x, y, (double) radidx, wrap);
        }
      }

      if (value != wrap->nodata && x > 0 && radidx != -1 && old_radidx != -1
          && lastrow[x] != -1 && (radidx != old_radidx || radidx != lastrow[x])) {
        setBitmapArritem(x, y, 1, wrap);
      } else {
        setBitmapArritem(x, y, 0, wrap);
      }

      lastrow[x] = old_radidx = radidx;
    }
  }

  free(lastrow);
}

static void LowestLoop(CompWrapper* wrap)
{
  int x, y;
  int i, radidx = -1, old_radidx = -1;
  int offx, offy;
  double dist;
  double mindist = 1e10;
  double value;

  int* lastrow = malloc(sizeof(int) * wrap->maxx);
  for (i = 0; i < wrap->maxx; i++) {
    lastrow[i] = -1;
  }

  for (y = 0; y < wrap->maxy; y++) {
    for (x = 0, old_radidx = -1; x < wrap->maxx; x++) {
      for (i = 0, mindist = 1e10, radidx = -1; i < wrap->noofrad; i++) {
        offx = x - wrap->rad[i].ux;
        offy = y - wrap->rad[i].uy;

        if (offx >= 0 && offx < wrap->rad[i].maxx && offy >= 0 && offy
            < wrap->rad[i].maxy) {

          dist = gettopoarritem(&wrap->rad[i], offx, offy, wrap);
          if (getarritem(&wrap->rad[i], offx, offy, wrap) == wrap->nodata) {
            continue;
          }

          if (dist < mindist) {
            mindist = dist;
            radidx = i;
          }
        }
      }

      value = wrap->nodata;

      if (radidx != -1) {
        offx = x - wrap->rad[radidx].ux;
        offy = y - wrap->rad[radidx].uy;
        value = getarritem(&wrap->rad[radidx], offx, offy, wrap);
        if (value != wrap->nodata) {
          setarritem(x, y, value, wrap);
          if (wrap->outtopo) {
            settopoarritem(x, y, (double) radidx, wrap);
          }
        }
      }

      if (value != wrap->nodata && x > 0 && (old_radidx != -1 && lastrow[x]
          != -1) && /*TEST*/
      (radidx != old_radidx || radidx != lastrow[x])) {
        setBitmapArritem(x, y, 1L, wrap);
      } else {
        setBitmapArritem(x, y, 0, wrap);
      }

      lastrow[x] = old_radidx = radidx;
    }
  }

  free(lastrow);
}

static void createSurroundingXBitmap(CompWrapper* wrap)
{
  int x, y;
  int i;
  int offx, offy;

  HiLi* hitlist = malloc(sizeof(HiLi) * wrap->maxx);

  for (i = 0; i < wrap->maxx; i++) {
    hitlist[i].ridx = malloc(sizeof(int) * wrap->noofrad);
    hitlist[i].vidx = malloc(sizeof(double) * wrap->noofrad);
    hitlist[i].xpos = malloc(sizeof(int) * wrap->noofrad);
    hitlist[i].ypos = malloc(sizeof(int) * wrap->noofrad);
  }

  for (y = 0; y < wrap->maxy; y++) {
    for (x = 0; x < wrap->maxx; x++) {
      hitlist[x].noi = 0;
      for (i = 0; i < wrap->noofrad; i++) {
        offx = x - wrap->rad[i].ux;
        offy = y - wrap->rad[i].uy;

        if (offx >= 0 && offx < wrap->rad[i].maxx && offy >= 0 && offy
            < wrap->rad[i].maxy) {
          if (getarritem(&wrap->rad[i], offx, offy, wrap) != wrap->nodata) {
            hitlist[x].ridx[hitlist[x].noi] = i;
            hitlist[x].vidx[hitlist[x].noi] = getarritem(&wrap->rad[i], offx,
                                                         offy, wrap);
            hitlist[x].xpos[hitlist[x].noi] = offx;
            hitlist[x].ypos[hitlist[x].noi] = offy;
            hitlist[x].noi++;
          }
        }
      }
    }

    for (i = 0; i < wrap->noofrad; i++) {
      offy = y - wrap->rad[i].uy;

      if (offy >= 0 && offy < wrap->rad[i].maxy) {
        for (x = 0; x < wrap->rad[i].maxx; x++) {
          offx = wrap->rad[i].ux + x;
          if (getarritem(&wrap->rad[i], x, offy, wrap) != wrap->nodata) {
            if (offx >= 0 && offx < wrap->maxx) {
              if (hitlist[offx].noi == 1 && hitlist[offx].ridx[0] == i)
                setBitmapArritem(offx, y, 1L, wrap);
            }
            break;
          }
        }
        for (x = wrap->rad[i].maxx - 1; x >= 0; x--) {
          offx = wrap->rad[i].ux + x;
          if (getarritem(&wrap->rad[i], x, offy, wrap) != wrap->nodata) {
            if (offx >= 0 && offx < wrap->maxx) {
              if (hitlist[offx].noi == 1 && hitlist[offx].ridx[0] == i)
                setBitmapArritem(offx, y, 1L, wrap);
            }
            break;
          }
        }
      }
    }
  }

  for (i = 0; i < wrap->maxx; i++) {
    free(hitlist[i].ridx);
    free(hitlist[i].vidx);
    free(hitlist[i].xpos);
    free(hitlist[i].ypos);
  }

  free(hitlist);
}

static void createSurroundingYBitmap(CompWrapper* wrap)
{
  int x, y;
  int i;
  int offx, offy;

  HiLi* hitlist = malloc(sizeof(HiLi) * wrap->maxy);

  for (i = 0; i < wrap->maxy; i++) {
    hitlist[i].ridx = malloc(sizeof(int) * wrap->noofrad);
    hitlist[i].vidx = malloc(sizeof(double) * wrap->noofrad);
    hitlist[i].xpos = malloc(sizeof(int) * wrap->noofrad);
    hitlist[i].ypos = malloc(sizeof(int) * wrap->noofrad);
  }

  for (x = 0; x < wrap->maxx; x++) {
    for (y = 0; y < wrap->maxy; y++) {
      hitlist[y].noi = 0;
      for (i = 0; i < wrap->noofrad; i++) {
        offx = x - wrap->rad[i].ux;
        offy = y - wrap->rad[i].uy;

        if (offx >= 0 && offx < wrap->rad[i].maxx && offy >= 0 && offy
            < wrap->rad[i].maxy) {
          if (getarritem(&wrap->rad[i], offx, offy, wrap) != wrap->nodata) {
            hitlist[y].ridx[hitlist[y].noi] = i;
            hitlist[y].vidx[hitlist[y].noi] = getarritem(&wrap->rad[i], offx,
                                                         offy, wrap);
            hitlist[y].xpos[hitlist[y].noi] = offx;
            hitlist[y].ypos[hitlist[y].noi] = offy;
            hitlist[y].noi++;
          }
        }
      }
    }

    for (i = 0; i < wrap->noofrad; i++) {
      double tmp;
      offx = x - wrap->rad[i].ux;

      if (offx >= 0 && offx < wrap->rad[i].maxx) {
        for (y = 0; y < wrap->rad[i].maxy; y++) {
          if (getarritem(&wrap->rad[i], offx, y, wrap) != wrap->nodata) {
            offy = wrap->rad[i].uy + y;

            if (offy >= 0 && offy < wrap->maxy) {
              if (hitlist[offy].noi == 1 && hitlist[offy].ridx[0] == i) {
                setBitmapArritem(x, offy, 1L, wrap);
              }
            }
            break;
          }
        }

        for (y = wrap->rad[i].maxy - 1; y >= 0; y--) {
          tmp = getarritem(&wrap->rad[i], offx, y, wrap);

          if (tmp != wrap->nodata) {
            offy = wrap->rad[i].uy + y;

            if (offy >= 0 && offy < wrap->maxy) {
              if (hitlist[offy].noi == 1 && hitlist[offy].ridx[0] == i) {
                setBitmapArritem(x, offy, 1L, wrap);
              }
            }
            break;
          }
        }
      }
    }
  }

  for (i = 0; i < wrap->maxy; i++) {
    free(hitlist[i].ridx);
    free(hitlist[i].vidx);
    free(hitlist[i].xpos);
    free(hitlist[i].ypos);
  }

  free(hitlist);
}

static PyObject* _nearest_composite(PyObject* self, PyObject* args)
{
  PyObject *in, *out;
  PyObject *worko; /* temp working object that can be recycled */
  PyObject *tmp;

  CompWrapper wrap;
  RaveObject outrave;
  RaveImageStruct rimage;
  int dex, dey, i;

  wrap.outtopo = NULL;

  if (!PyArg_ParseTuple(args, "OOi", &in, &out, &wrap.bitmode)) {
    return NULL;
  }

  if (!PySequence_Check(in)) {
    Raise(PyExc_AttributeError,"Input data is not a sequence");
  }

  /* Have to deal with output bitmap */
  if (!fill_rave_object(out, &outrave, 1, "image")) {
    if (!outrave.info || !outrave.data) {
      Raise(PyExc_AttributeError,"No info or data in output object");
    }
    if ((wrap.bitmode >= 3 && wrap.bitmode <= 5) && !outrave.topo) {
      Raise(PyExc_AttributeError,
          "No topo in output object but bitmode is 3, 4 or 5");
    }
  }
  /* Manage output composite bitmap. Assume it's in /image2 */
  worko = PyObject_GetAttrString(out, "data");
  tmp = PyDict_GetItemString(worko, "image2"); /* No need to DECREF later */
  Py_CLEAR(worko);
  if (!tmp) {
    Raise(PyExc_AttributeError, "No /image2 bitmap for output composite");
    return NULL;
  } else {
    outrave.bitmap = (PyArrayObject*) tmp;
  }

  if (!outrave.bitmap) {
    Raise(PyExc_AttributeError, "No /image2 bitmap for output composite");
    return NULL;
  }
  if (!fill_rave_image_info(out, &rimage, 1)) {
    return NULL;
  }

  if (outrave.topo) { /* What's this doing here? */
    wrap.outtopo = array_data_2d(outrave.topo);
    wrap.outtopotype = array_type_2d(outrave.topo);
    wrap.outtopoxsize = array_stride_xsize_2d(outrave.topo);
  }

  wrap.proj = get_rave_projection(out);
  if (!wrap.proj) {
    Raise(PyExc_AttributeError,"Could not create composite projection");
    return NULL;
  }

  /* Remember that 'in' is a sequence object */
  wrap.noofrad = PyObject_Length(in);
  wrap.rad = malloc(sizeof(RAD) * wrap.noofrad);

  wrap.xscale = rimage.xscale;
  wrap.yscale = rimage.yscale;
  wrap.nodata = rimage.nodata;
  wrap.ux = rimage.lowleft.u;
  wrap.uy = rimage.uppright.v;

  wrap.desta = array_data_2d(outrave.data);
  wrap.type = array_type_2d(outrave.data);
  wrap.outxsize = array_stride_xsize_2d(outrave.data);
  wrap.bitmapa = array_data_2d(outrave.bitmap);

  dey = outrave.data->dimensions[0];
  dex = outrave.data->dimensions[1];
  wrap.maxx = dex;
  wrap.maxy = dey;

  for (i = 0; i < wrap.noofrad; i++) {
    PyObject* po;
    RaveImageStruct istruct;
    UV here, here_c;

    po = PySequence_GetItem(in, i);

    /* Assume data to be composited is in /image1 */
    worko = PyObject_GetAttrString(po, "data");
    wrap.rad[i].data = (PyArrayObject*) PyDict_GetItemString(worko, "image1");
    Py_CLEAR(worko);
    wrap.rad[i].src = array_data_2d(wrap.rad[i].data);
    wrap.rad[i].inxsize = array_stride_xsize_2d(wrap.rad[i].data);

    /* No need to deal with topo */
    if (!fill_rave_image_info(po, &istruct, 1)) {
      Py_DECREF(po);
      return NULL;
    }
    wrap.rad[i].ux = (int) ((istruct.lowleft.u - wrap.ux) / wrap.xscale);
    wrap.rad[i].uy = (int) ((wrap.uy - istruct.uppright.v) / wrap.yscale);
    wrap.rad[i].maxy = wrap.rad[i].data->dimensions[0];
    wrap.rad[i].maxx = wrap.rad[i].data->dimensions[1];
    if (!GetDoubleFromINFO(po, "/how/lon_0", &here.u)) {
      Raise(PyExc_AttributeError,"No /how/lon_0 in an input image.");
      return NULL;
    }
    if (!GetDoubleFromINFO(po, "/how/lat_0", &here.v)) {
      Raise(PyExc_AttributeError,"No /how/lat_0 in an input image.");
      return NULL;
    }
    Py_DECREF(po);

    here.u *= DEG_TO_RAD;
    here.v *= DEG_TO_RAD;
    here_c = pj_fwd(here, wrap.proj);
    wrap.rad[i].cx = (int) ((here_c.u - wrap.ux) / wrap.xscale);
    wrap.rad[i].cy = (int) ((wrap.uy - here_c.v) / wrap.yscale);
  }

  NearestLoop(&wrap);

  if (wrap.bitmode == 2 || wrap.bitmode == 5) {
    createSurroundingXBitmap(&wrap);
    createSurroundingYBitmap(&wrap);
  }
  pj_free(wrap.proj);
  free(wrap.rad);

  PyErr_Clear();

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject* _lowest_composite(PyObject* self, PyObject* args)
{
  PyObject *in, *out;
  PyObject *worko;
  PyObject *tmp;

  //   RAD oneRAD;
  CompWrapper wrap;
  RaveObject outrave;
  RaveImageStruct rimage;
  int dex, dey, i;

  wrap.outtopo = NULL;

  if (!PyArg_ParseTuple(args, "OOi", &in, &out, &wrap.bitmode)) {
    return NULL;
  }

  if (!PySequence_Check(in)) {
    Raise(PyExc_AttributeError,"Input data is not sequence");
  }

  if (!fill_rave_object(out, &outrave, 1, "image")) {
    if (!outrave.info || !outrave.data) {
      Raise(PyExc_AttributeError,"No info or data in output object");
    }
    if ((wrap.bitmode == 3 || wrap.bitmode == 4) && !outrave.topo) {
      Raise(PyExc_AttributeError,"No topo in out object but bitmode is 3, 4, or 5");
    }
  }

  /* Manage output composite bitmap. Assume it's in /image2 */
  worko = PyObject_GetAttrString(out, "data");
  tmp = PyDict_GetItemString(worko, "image2"); /* No need to DECREF later */
  Py_CLEAR(worko);
  if (!tmp) {
    Raise(PyExc_AttributeError, "No /image2 bitmap for output composite");
    return NULL;
  } else {
    outrave.bitmap = (PyArrayObject*) tmp;
  }

  if (!outrave.bitmap) {
    Raise(PyExc_AttributeError, "No /image2 bitmap for output composite");
    return NULL;
  }

  if (outrave.topo) { /* What's this doing here? */
    wrap.outtopo = array_data_2d(outrave.topo);
    wrap.outtopotype = array_type_2d(outrave.topo);
    wrap.outtopoxsize = array_stride_xsize_2d(outrave.topo);
  }

  wrap.proj = get_rave_projection(out);
  if (!wrap.proj) {
    Raise(PyExc_AttributeError,"Could not create composite projection");
  }

  wrap.noofrad = PyObject_Length(in);
  wrap.rad = malloc(sizeof(RAD) * wrap.noofrad);

  if (!fill_rave_image_info(out, &rimage, 1)) {
    return NULL;
  }

  wrap.xscale = rimage.xscale;
  wrap.yscale = rimage.yscale;
  wrap.nodata = rimage.nodata;
  wrap.ux = rimage.lowleft.u;
  wrap.uy = rimage.uppright.v;

  wrap.desta = array_data_2d(outrave.data);
  wrap.type = array_type_2d(outrave.data);
  wrap.outxsize = array_stride_xsize_2d(outrave.data);
  wrap.bitmapa = array_data_2d(outrave.bitmap);

  dey = outrave.data->dimensions[0];
  dex = outrave.data->dimensions[1];
  wrap.maxx = dex;
  wrap.maxy = dey;

  for (i = 0; i < wrap.noofrad; i++) {
    PyObject* po;
    RaveImageStruct istruct;
    UV here, here_c;

    po = PySequence_GetItem(in, i);

    /* Assume data to be composited is in /image1 */
    worko = PyObject_GetAttrString(po, "data");
    wrap.rad[i].data = (PyArrayObject*) PyDict_GetItemString(worko, "image1");
    wrap.rad[i].src = array_data_2d(wrap.rad[i].data);
    wrap.rad[i].inxsize = array_stride_xsize_2d(wrap.rad[i].data);

    /* Assume associated radar_height LUT is in /image2 */
    /* Do I need a Py_INCREF here? */
    wrap.rad[i].topodata = (PyArrayObject*) PyDict_GetItemString(worko,
                                                                 "image2");
    Py_CLEAR(worko);
    wrap.rad[i].srctopo = array_data_2d(wrap.rad[i].topodata);
    wrap.topotype = array_type_2d(wrap.rad[i].topodata);
    wrap.rad[i].topoinxsize = array_stride_xsize_2d(wrap.rad[i].topodata);

    if (!fill_rave_image_info(po, &istruct, 1)) {
      Py_DECREF(po);
      return NULL;
    }
    wrap.rad[i].ux = (int) ((istruct.lowleft.u - wrap.ux) / wrap.xscale);
    wrap.rad[i].uy = (int) ((wrap.uy - istruct.uppright.v) / wrap.yscale);
    wrap.rad[i].maxy = wrap.rad[i].data->dimensions[0];
    wrap.rad[i].maxx = wrap.rad[i].data->dimensions[1];
    if (!GetDoubleFromINFO(po, "/how/lon_0", &here.u)) {
      Raise(PyExc_AttributeError,"No /how/lon_0 in an input image.");
      return NULL;
    }
    if (!GetDoubleFromINFO(po, "/how/lat_0", &here.v)) {
      Raise(PyExc_AttributeError,"No /how/lat_0 in an input image.");
      return NULL;
    }
    Py_DECREF(po);

    here.u *= DEG_TO_RAD;
    here.v *= DEG_TO_RAD;
    here_c = pj_fwd(here, wrap.proj);
    wrap.rad[i].cx = (int) ((here_c.u - wrap.ux) / wrap.xscale);
    wrap.rad[i].cy = (int) ((wrap.uy - here_c.v) / wrap.yscale);
  }

  LowestLoop(&wrap);

  if (wrap.bitmode == 2 || wrap.bitmode == 5) {
    createSurroundingXBitmap(&wrap);
    createSurroundingYBitmap(&wrap);
  }
  pj_free(wrap.proj);
  free(wrap.rad);

  PyErr_Clear();

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject* _test_method(PyObject* self, PyObject* args)
{
  printf("COMPOSITE: Check to see if it works\n");
  Py_INCREF(Py_None);
  return Py_None;
}

static struct PyMethodDef _composite_functions[] =
{
{ "nearest", (PyCFunction) _nearest_composite, METH_VARARGS },
{ "lowest", (PyCFunction) _lowest_composite, METH_VARARGS },
/*   {"check_overlap",(PyCFunction)_check_overlap,METH_VARARGS}, */
/*   {"check_mode",(PyCFunction)_check_mode,METH_VARARGS}, */
{ "test", (PyCFunction) _test_method, METH_VARARGS },
{ NULL, NULL } };

/**
 * Initializes the python mode _composite.
 */
PyMODINIT_FUNC init_composite(void)
{
  PyObject* m;
  m = Py_InitModule("_composite", _composite_functions);
  ErrorObject = PyString_FromString("_composite.error");
  if (ErrorObject == NULL || PyDict_SetItemString(PyModule_GetDict(m), "error",
                                                  ErrorObject) != 0)
    Py_FatalError("Can't define _composite.error");

  import_array(); /*To make sure I get access to the Numeric PyArray functions*/
}

