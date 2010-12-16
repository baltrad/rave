/* --------------------------------------------------------------------
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
/**
 * Provides support for reading and writing areas to and from
 * an xml-file.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-12-08
 */
#include "rave_simplexml.h"
#include "raveobject_list.h"
#include "raveobject_hashtable.h"
#include "rave_utilities.h"
#include "rave_alloc.h"
#include "rave_debug.h"
#include "expat.h"
#include <string.h>

/**
 * Represents a node
 */
struct _SimpleXmlNode_t {
  RAVE_OBJECT_HEAD /** Always on top */
  char* name; /**< the name of this node */
  char* text; /**< the trimmed character data */
  char* textBuffer; /**< text buffer, for managing untrimmed data */
  RaveObjectList_t* children; /**< the list of children */
  RaveObjectHashTable_t* attributes; /**< attributes for this node */
  SimpleXmlNode_t* parent; /**< this nodes parent */
};

typedef struct SimpleXmlNodeUserData_t {
  SimpleXmlNode_t* current;
  XML_Parser parser;
} SimpleXmlNodeUserData_t;

/*@{ Private functions */
/**
 * Constructor.
 */
static int SimpleXmlNode_constructor(RaveCoreObject* obj)
{
  SimpleXmlNode_t* this = (SimpleXmlNode_t*)obj;
  this->name = NULL;
  this->text = NULL;
  this->textBuffer = NULL;
  this->children = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  this->attributes = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  this->parent = NULL;

  if (this->children == NULL || this->attributes == NULL) {
    goto error;
  }

  return 1;
error:
  RAVE_OBJECT_RELEASE(this->children);
  RAVE_OBJECT_RELEASE(this->attributes);
  RAVE_OBJECT_RELEASE(this->parent);
  return 0;
}

/**
 * Copy constructor
 */
static int SimpleXmlNode_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  SimpleXmlNode_t* this = (SimpleXmlNode_t*)obj;
  SimpleXmlNode_t* src = (SimpleXmlNode_t*)srcobj;
  int nrchilds = 0, i = 0;

  this->name = RAVE_STRDUP(src->name);
  this->text = RAVE_STRDUP(src->text);
  this->textBuffer = RAVE_STRDUP(src->textBuffer);
  this->children = RAVE_OBJECT_CLONE(src->children);
  this->attributes = RAVE_OBJECT_CLONE(src->attributes);
  this->parent = NULL; /* You don't want to clone parent since children will be cloned -> parent cloned -> infinite */

  if (this->name == NULL || this->text == NULL ||
      this->children == NULL || this->attributes == NULL ||
      this->textBuffer == NULL) {
    goto error;
  }
  nrchilds = RaveObjectList_size(this->children);
  for (i = 0; i < nrchilds; i++) {
    SimpleXmlNode_t* node = (SimpleXmlNode_t*)RaveObjectList_get(this->children, i);
    SimpleXmlNode_setParent(node, this);
    RAVE_OBJECT_RELEASE(node);
  }
  return 1;
error:
  RAVE_FREE(this->name);
  RAVE_FREE(this->text);
  RAVE_FREE(this->textBuffer);
  RAVE_OBJECT_RELEASE(this->children);
  RAVE_OBJECT_RELEASE(this->attributes);
  return 0;
}

/**
 * Destroys the registry
 * @param[in] obj - the the AreaRegistry_t instance
 */
static void SimpleXmlNode_destructor(RaveCoreObject* obj)
{
  SimpleXmlNode_t* this = (SimpleXmlNode_t*)obj;
  RAVE_FREE(this->name);
  RAVE_FREE(this->text);
  RAVE_FREE(this->textBuffer);
  RAVE_OBJECT_RELEASE(this->children);
  RAVE_OBJECT_RELEASE(this->attributes);
  this->parent = NULL;
}


/**
 * Adds an attribute to the node.
 * @param[in] self - self
 * @param[in] attribute - Must have name set and value should preferrably be a string
 * @return 1 on success otherwise 0
 */
static int SimpleXmlNodeInternal_addAttribute(SimpleXmlNode_t* self, RaveAttribute_t* attribute)
{
  int result = 1;
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");
  if (RaveAttribute_getName(attribute) != NULL) {
    result = RaveObjectHashTable_put(self->attributes, RaveAttribute_getName(attribute), (RaveCoreObject*)attribute);
  } else {
    RAVE_ERROR0("No name supplied in the attribute");
    result = 0;
  }
  return result;
}

/**
 * Extracts and sets all attributes.
 * @param[in] node - the node that should get the attributes set
 * @param[in] userdata - pass along
 * @param[in] attr - the attributes
 * @return 1 on success otherwise 0
 */
static int SimpleXmlNodeInternal_setAttributes(SimpleXmlNode_t* node, SimpleXmlNodeUserData_t* userdata, const char** attr)
{
  int index = 0;
  int nrattrs = XML_GetSpecifiedAttributeCount(userdata->parser);
  int status = 1;

  for (index = 0; status == 1 && index < nrattrs; index += 2) {
    RaveAttribute_t* a = RAVE_OBJECT_NEW(&RaveAttribute_TYPE);
    if (a != NULL) {
      if (!RaveAttribute_setName(a, attr[index]) || !RaveAttribute_setString(a, attr[index+1])) {
        int lineno = (int)XML_GetCurrentLineNumber(userdata->parser);
        RAVE_ERROR2("Failed to create attribute for %s at line %d\n", attr[index], lineno);
        status = 0;
      } else {
        if (!SimpleXmlNodeInternal_addAttribute(node, a)) {
          int lineno = (int)XML_GetCurrentLineNumber(userdata->parser);
          RAVE_ERROR2("Failed to add attribute %s at line %d\n", attr[index], lineno);
          status = 0;
        }
      }
    }
    RAVE_OBJECT_RELEASE(a);
  }
  return status;
}

/**
 * Expats start tag handler function
 * @param[in] data - user data
 * @param[in] el - the name of the element
 * @param[in] attr - the attributes belonging to this element
 */
static void SimpleXmlNodeInternal_expatStartHandler(void *data, const char *el, const char **attr)
{
  SimpleXmlNodeUserData_t* userdata = (SimpleXmlNodeUserData_t*)data;
  int status = 1;
  SimpleXmlNode_t* node = RAVE_OBJECT_NEW(&SimpleXmlNode_TYPE);

  if (node == NULL || !SimpleXmlNode_setName(node, el)) {
    status = 0;
  } else {
    status = SimpleXmlNodeInternal_setAttributes(node, userdata, attr);
  }
  if (status == 1) {
    SimpleXmlNode_setParent(node, userdata->current);
    SimpleXmlNode_addChild(userdata->current, node);
    userdata->current = node;
  } else {
    XML_StopParser(userdata->parser, 0);
    RAVE_OBJECT_RELEASE(node);
  }
}

/**
 * Expats end tag handler
 * @param[in] data - user data
 * @param[in] el - the tag name
 */
static void SimpleXmlNodeInternal_expatEndHandler(void *data, const char *el)
{
  SimpleXmlNodeUserData_t* userdata = (SimpleXmlNodeUserData_t*)data;
  if (userdata->current != NULL) {
    const char* name = SimpleXmlNode_getName(userdata->current);
    if (name != NULL && strcasecmp(name, el)==0) {
      // good, we haven't got a bad scheme
      SimpleXmlNode_t* parent = SimpleXmlNode_getParent(userdata->current);
      RAVE_OBJECT_RELEASE(userdata->current);
      userdata->current = parent;
    } else {
      int lineno = (int)XML_GetCurrentLineNumber(userdata->parser);
      RAVE_ERROR2("Unexpected end tag at %d (%s)", lineno, el);
      XML_StopParser(userdata->parser, 0);
    }
  }
}

/**
 * Expats character data handler
 * @param[in] data - user data
 * @param[in] s - the text
 * @param[in] len - the length of the text
 */
static void SimpleXmlNodeInternal_characterDataHandler(void* data, const XML_Char* s, int len)
{
  SimpleXmlNodeUserData_t* userdata = (SimpleXmlNodeUserData_t*)data;
  if (userdata->current != NULL) {
    if (!SimpleXmlNode_addText(userdata->current, s, len)) {
      int lineno = (int)XML_GetCurrentLineNumber(userdata->parser);
      RAVE_ERROR1("Could not append text data when working with line %d", lineno);
      XML_StopParser(userdata->parser, 0);
    }
  }
}



/*@} End of Private functions */

/*@{ Interface functions */
SimpleXmlNode_t* SimpleXmlNode_parseFile(const char* filename)
{
  XML_Parser parser = NULL;
  FILE* fp = NULL;
  SimpleXmlNode_t* result = NULL;
  char buff[1024];
  int nread = 0;
  int finished = 0;
  SimpleXmlNodeUserData_t userdata;
  int nrchildren = 0;
  SimpleXmlNode_t* root = NULL;

  userdata.current = NULL;
  userdata.parser = NULL;

  RAVE_ASSERT((filename != NULL), "filename == NULL");

  if ((fp = fopen(filename, "r")) == NULL) {
    RAVE_ERROR1("Failed to open %s", filename);
    goto done;
  }
  if ((parser = XML_ParserCreate(NULL)) == NULL) {
    RAVE_ERROR0("Failed to create xml parser");
    goto done;
  }

  root = RAVE_OBJECT_NEW(&SimpleXmlNode_TYPE);
  if (root == NULL || !SimpleXmlNode_setName(root, "ROOT")) {
    RAVE_ERROR0("Failed to create simple xml node instance");
    goto done;
  }

  userdata.current = RAVE_OBJECT_COPY(root);
  userdata.parser = parser;

  XML_SetElementHandler(parser, SimpleXmlNodeInternal_expatStartHandler, SimpleXmlNodeInternal_expatEndHandler);
  XML_SetCharacterDataHandler(parser, SimpleXmlNodeInternal_characterDataHandler);
  XML_SetUserData(parser, &userdata);

  while (!finished && (nread = fread(buff, sizeof(char), 1024, fp)) > 0) {
    finished = (nread < 1024);
    if (XML_Parse(parser, buff, nread, finished) == XML_STATUS_ERROR) {
      long lineno = (long)XML_GetCurrentLineNumber(parser);
      const XML_LChar* msg = XML_ErrorString(XML_GetErrorCode(parser));
      RAVE_ERROR2("XML parser error at line %ld: %s", lineno, msg);
      goto done;
    }
  }

  nrchildren = RaveObjectList_size(userdata.current->children);
  if (nrchildren > 1) {
    result = RAVE_OBJECT_COPY(root);
  } else if (nrchildren == 1) {
    result = (SimpleXmlNode_t*)RaveObjectList_get(root->children, 0);
  } else {
    RAVE_INFO0("Empty object list");
  }
done:
  RAVE_OBJECT_RELEASE(userdata.current);
  RAVE_OBJECT_RELEASE(root);
  XML_ParserFree(parser);
  if (fp != NULL) fclose(fp);
  return result;
}

void SimpleXmlNode_setParent(SimpleXmlNode_t* self, SimpleXmlNode_t* parent)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->parent = parent;
}

SimpleXmlNode_t* SimpleXmlNode_getParent(SimpleXmlNode_t* self)
{
  return self->parent;
}

int SimpleXmlNode_setName(SimpleXmlNode_t* self, const char* name)
{
  int result = 1;
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_FREE(self->name);

  if (name != NULL) {
    self->name = RAVE_STRDUP(name);
    if (self->name == NULL) {
      result = 0;
    }
  }

  return result;
}

const char* SimpleXmlNode_getName(SimpleXmlNode_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->name;
}

int SimpleXmlNode_setText(SimpleXmlNode_t* self, const char* text, int len)
{
  int result = 1;
  RAVE_ASSERT((self != NULL), "self == NULL");

  RAVE_FREE(self->text);

  self->text = RaveUtilities_trimText(text, len);
  return result;
}

int SimpleXmlNode_addText(SimpleXmlNode_t* self, const char* text, int len)
{
  int result = 0;
  char* tmpptr = NULL;
  int clen = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (text != NULL) {
    if (self->textBuffer != NULL) {
      clen = strlen(self->textBuffer);
    } else if (self->text != NULL) {
      clen = strlen(self->text);
    }

    tmpptr = RAVE_MALLOC(sizeof(char)*(clen + len + 1));
    if (tmpptr == NULL) {
      goto done;
    }
    memset(tmpptr, 0, sizeof(char)*(clen + len + 1));

    if (self->textBuffer != NULL) {
      strcpy(tmpptr, self->textBuffer);
    } else if (self->text != NULL) {
      strcpy(tmpptr, self->text);
    }
    strncat(tmpptr, text, len);
    RAVE_FREE(self->textBuffer);
    self->textBuffer = tmpptr;
    SimpleXmlNode_setText(self, tmpptr, strlen(tmpptr));
  }
  result = 1;
done:
  return result;
}

const char* SimpleXmlNode_getText(SimpleXmlNode_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->text;
}


int SimpleXmlNode_addChild(SimpleXmlNode_t* self, SimpleXmlNode_t* child)
{
  int result = 1;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (!RaveObjectList_add(self->children, (RaveCoreObject*)child)) {
    RAVE_ERROR0("Failed to add child to children");
    result = 0;
  }
  return result;
}

int SimpleXmlNode_getNumberOfChildren(SimpleXmlNode_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectList_size(self->children);
}

SimpleXmlNode_t* SimpleXmlNode_getChild(SimpleXmlNode_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (SimpleXmlNode_t*)RaveObjectList_get(self->children, index);
}

SimpleXmlNode_t* SimpleXmlNode_getChildByName(SimpleXmlNode_t* self, const char* name)
{
  int nchilds = 0;
  int i = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  nchilds = RaveObjectList_size(self->children);

  for (i = 0; i < nchilds; i++) {
    SimpleXmlNode_t* child = (SimpleXmlNode_t*)RaveObjectList_get(self->children, i);
    if (SimpleXmlNode_getName(child) != NULL && strcasecmp(name, SimpleXmlNode_getName(child)) == 0) {
      return child;
    }
    RAVE_OBJECT_RELEASE(child);
  }

  return NULL;
}

int SimpleXmlNode_addAttribute(SimpleXmlNode_t* self, const char* key, const char* value)
{
  int result = 0;
  RaveAttribute_t* attribute = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");

  if (key == NULL) {
    goto done;
  }

  attribute = RaveAttributeHelp_createString(key, value);
  if (attribute == NULL) {
    goto done;
  }

  if (!RaveObjectHashTable_put(self->attributes, key, (RaveCoreObject*)attribute)) {
    goto done;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(attribute);
  return result;
}

const char* SimpleXmlNode_getAttribute(SimpleXmlNode_t* self, const char* key)
{
  RaveAttribute_t* attribute = NULL;
  char* value = NULL;
  const char* result = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  attribute = (RaveAttribute_t*)RaveObjectHashTable_get(self->attributes, key);
  if (attribute != NULL && RaveAttribute_getString(attribute, &value)) {
    result = (const char*)value;
  }
  RAVE_OBJECT_RELEASE(attribute);
  return result;
}

static int SimpleXmlNodeInternal_printNode(SimpleXmlNode_t* self, int scope, FILE* fp)
{
  int i;
  int nchildren = 0;
  int result = 0;
  RaveAttribute_t* attr = NULL;
  RaveList_t* attrkeys = NULL;
  for (i = 0; i < scope; i++) {
    if (fprintf(fp, "  ") < 0) {
      goto done;
    }
  }
  if (fprintf(fp, "<%s", SimpleXmlNode_getName(self))<0) {
    goto done;
  }
  attrkeys = RaveObjectHashTable_keys(self->attributes);
  if (attrkeys != NULL) {
    int nattrs = 0;
    nattrs = RaveList_size(attrkeys);
    for (i = 0; i < nattrs; i++) {
      const char* key = RaveList_get(attrkeys, i);
      char* value = NULL;
      attr = (RaveAttribute_t*)RaveObjectHashTable_get(self->attributes, key);
      if (attr != NULL) {
        if (key != NULL && RaveAttribute_getString(attr, &value)) {
          if (fprintf(fp, " %s=\"%s\"", key, (value != NULL)?value:"") < 0) {
            goto done;
          }
        }
      }
      RAVE_OBJECT_RELEASE(attr);
    }
  }
  if (fprintf(fp, ">\n") < 0) {
    goto done;
  }
  if (self->text != NULL && strcmp("", self->text)!=0) {
    for (i = 0; i < (scope+1); i++) {
      if (fprintf(fp, "  ") < 0) {
        goto done;
      }
    }
    if (fprintf(fp, "%s\n", self->text) < 0) {
      goto done;
    }
  }

  nchildren = RaveObjectList_size(self->children);
  for (i = 0; i < nchildren; i++) {
    SimpleXmlNode_t* node = (SimpleXmlNode_t*)RaveObjectList_get(self->children, i);
    if (node != NULL) {
      if (!SimpleXmlNodeInternal_printNode(node, scope + 1, fp)) {
        goto done;
      }
    }
    RAVE_OBJECT_RELEASE(node);
  }
  for (i = 0; i < scope; i++) {
    if (fprintf(fp, "  ") < 0) {
      goto done;
    }
  }
  if (fprintf(fp, "</%s>\n", SimpleXmlNode_getName(self)) < 0) {
    goto done;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(attr);
  RaveList_freeAndDestroy(&attrkeys);
  return result;
}

int SimpleXmlNode_write(SimpleXmlNode_t* self, FILE* fp)
{
  SimpleXmlNode_t* node = self;

  RAVE_ASSERT((self != NULL), "self == NULL");

  return SimpleXmlNodeInternal_printNode(node, 0, fp);
}

/*@} End of Interface functions */

RaveCoreObjectType SimpleXmlNode_TYPE = {
    "SimpleXmlNode",
    sizeof(SimpleXmlNode_t),
    SimpleXmlNode_constructor,
    SimpleXmlNode_destructor,
    SimpleXmlNode_copyconstructor
};
