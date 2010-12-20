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
 * Simple XML object frontend to expat.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-12-08
 */
#ifndef RAVE_SIMPLEXML_H
#define RAVE_SIMPLEXML_H
#include "rave_attribute.h"
#include "rave_object.h"
#include <stdio.h>

typedef struct _SimpleXmlNode_t SimpleXmlNode_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType SimpleXmlNode_TYPE;

/**
 * The parser function for parsing the xml-file into a node tree.
 * @param[in] filename - the file to parse
 * @return a node on success otherwise failure
 */
SimpleXmlNode_t* SimpleXmlNode_parseFile(const char* filename);

/**
 * Sets the parent for this node.
 * NOTE! The parent is just a raw pointer assignment so be careful when
 * using it.
 * @param[in] self - self
 * @param[in] parent - the parent
 */
void SimpleXmlNode_setParent(SimpleXmlNode_t* self, SimpleXmlNode_t* parent);

/**
 * Returns the parent for this node.
 * @param[in] self - self
 * @returns the parent
 */
SimpleXmlNode_t* SimpleXmlNode_getParent(SimpleXmlNode_t* self);

/**
 * Sets the tag name of this node.
 * @param[in] self - self
 * @param[in] name - the name of the node
 * @return 1 on success otherwise 0
 */
int SimpleXmlNode_setName(SimpleXmlNode_t* self, const char* name);

/**
 * Returns the tag name of this node
 * @param[in] self - self
 * @returns the tag name
 */
const char* SimpleXmlNode_getName(SimpleXmlNode_t* self);

/**
 * Sets the text. The added text will be stripped of leading and trailing
 * whitespaces (space, tab, newline and carriage return).
 * @param[in] self - self
 * @param[in] text - the text
 * @param[in] len - the length of text
 * @return 1 on success otherwise 0
 */
int SimpleXmlNode_setText(SimpleXmlNode_t* self, const char* text, int len);

/**
 * Adds text.
 * @param[in] self - self
 * @param[in] text - the text
 * @param[in] len - the length of the text
 * @return 1 on success otherwise 0
 */
int SimpleXmlNode_addText(SimpleXmlNode_t* self, const char* text, int len);

/**
 * Returns the text.
 * @param[in] self - self
 * @returns the text (if any)
 */
const char* SimpleXmlNode_getText(SimpleXmlNode_t* self);

/**
 * Adds a child to this node.
 * @param[in] self - self
 * @param[in] child - the child
 * @returns 1 on success otherwise 0
 */
int SimpleXmlNode_addChild(SimpleXmlNode_t* self, SimpleXmlNode_t* child);

/**
 * Removes the given child from the children list
 * @param[in] self - self
 * @param[in] child - the node to remove
 */
void SimpleXmlNode_remove(SimpleXmlNode_t* self, SimpleXmlNode_t* child);

/**
 * Returns the number of children
 * @param[in] self - self
 * @returns the number of children
 */
int SimpleXmlNode_getNumberOfChildren(SimpleXmlNode_t* self);

/**
 * Returns the child at specified index
 * @param[in] self - self
 * @param[in] index - the index
 * @returns the child node
 */
SimpleXmlNode_t* SimpleXmlNode_getChild(SimpleXmlNode_t* self, int index);

/**
 * Returns the child with the given name
 * @param[in] self - self
 * @param[in] name - the name of the node
 * @returns the child node
 */
SimpleXmlNode_t* SimpleXmlNode_getChildByName(SimpleXmlNode_t* self, const char* name);

/**
 * Adds an attribute to a node
 * @param[in] self - self
 * @param[in] key - the name of the attribute
 * @param[in] value - the value of the attribute
 * @returns 1 on success otherwise 0
 */
int SimpleXmlNode_addAttribute(SimpleXmlNode_t* self, const char* key, const char* value);

/**
 * Returns the attribute value for the specified attribute
 * @param[in] self - self
 * @param[in] key - the name of the attribute
 * @returns the value
 */
const char* SimpleXmlNode_getAttribute(SimpleXmlNode_t* self, const char* key);

/**
 * Writes the node to the file pointer.
 * @param[in] self - self
 * @param[in] fp - the file pointer to write to
 * @return 1 on success otherwise 0
 */
int SimpleXmlNode_write(SimpleXmlNode_t* self, FILE* fp);

/**
 * Creates a xml node. If node is given, the created node will be
 * created as a child to that node otherwise it will be created
 * as a root node.
 * @param[in] parent - the parent node (MAY BE NULL)
 * @param[in] name - the name of the node (MAY BE NULL)
 * @return the node on success or NULL on failure
 */
SimpleXmlNode_t* SimpleXmlNode_create(SimpleXmlNode_t* parent, const char* name);

#endif /* RAVE_SIMPLEXML_H */
