/*
 * Copyright (C) 2010 Ajax.org BV
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this library; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */
#ifndef J_C_XML_H
#define J_C_XML_H

namespace o3 {

	xmlFreeFunc xmlfree;
	xmlMallocFunc xmlmalloc;
	xmlReallocFunc xmlrealloc;
	xmlStrdupFunc xmlstrdup;

	void o3xmlFreeFunc(void *mem) {
		memFree(mem); 
		//xmlfree(mem);		
	}
	void* o3xmlMallocFunc(size_t size) {
		return memAlloc(size);
		//xmlmalloc(size);
	}
	void* o3xmlReallocFunc(void *mem, size_t size) {
		memFree(mem);
		return memAlloc(size);
		//return xmlrealloc(mem, size);
	
	}
	char* o3xmlStrdupFunc(const char *str) {
		char *d = (char *)(memAlloc(strlen (str) + 1));
		if (d != NULL)
			strcpy (d,str);
		return d;

		//return xmlstrdup(str);
	}


    struct cXml1: cScr, iXml {
        
            //root->regScr(scr_info(cXmlAttr1));
            //root->regScr(scr_info(cXmlCDATASection1));
            //root->regScr(scr_info(cXmlCharacterData1));
            //root->regScr(scr_info(cXmlComment1));
            //root->regScr(scr_info(cXmlDocument1));
            //root->regScr(scr_info(cXmlElement1));
            //root->regScr(scr_info(cXmlNamedNodeMap1));
            //root->regScr(scr_info(cXmlNode1));
            //root->regScr(scr_info(cXmlNodeArray1));
            //root->regScr(scr_info(cXmlNodeList1));
            //root->regScr(scr_info(cXmlText1));

        o3_begin_class(cScr)
            o3_add_iface(iXml)
        o3_end_class();

        o3_glue_gen()

        NodeMap m_node_map;

        static o3_ext("cO3") o3_get siXml xml(iCtx* ctx)
        {
            Var val = ctx->value("xml");
            siXml xml = val.toScr();
            if (xml)
                return xml;
            

			//xmlMemGet (&xmlfree, &xmlmalloc,
			//	&xmlrealloc, &xmlstrdup);
			
			//xmlMemSetup (o3xmlFreeFunc, o3xmlMallocFunc,
			//	o3xmlReallocFunc, o3xmlStrdupFunc);
            
			val = siScr(o3_new(cXml1)());
            ctx->setValue("xml", val);
            
            return val.toScr();
        }
        
        //o3_fun
        //siScr xmlFromString(const char* data, const char* type = "text/xml") {
        //    return parseFromString(data, type);
        //}        

        o3_fun siXmlNode parseFromString(iCtx* ctx, const char* str, const char* contentType = "text/xml") {
            o3_assert(str);
            o3_assert(contentType);

            if (!strEquals(contentType, "text/xml"))
                return siXmlNode();

            return wrapNode(ctx, (xmlNodePtr) xmlParseDoc(BAD_CAST str), 0);            
        }
    };
    
    siXmlNode wrapNode(iCtx* ctx, xmlNodePtr node, iXmlNode* owner_node) {
        if (!node)
            return siXmlNode();

        siXml ixml = ctx->value("xml").toScr();
        cXml1* xml = (cXml1*) ixml.ptr();
        NodeMap& node_map = xml->m_node_map;
        NodeMap::Iter found = node_map.find(node);
        if (found != node_map.end()) {
            siXmlNode ret = (*found).val;
            if (ret)
                return ret;
        }
        
        switch (node->type) {
        case XML_ELEMENT_NODE:
            return o3_new(cXmlElement1)(node, owner_node, &node_map);
		case XML_TEXT_NODE:
            return o3_new(cXmlText1)(node, owner_node, &node_map);
		case XML_COMMENT_NODE:
            return o3_new(cXmlComment1)(node, owner_node, &node_map);
        case XML_CDATA_SECTION_NODE:
            return o3_new(cXmlCDATASection1)(node, owner_node, &node_map);
        case XML_ATTRIBUTE_NODE:
            return o3_new(cXmlAttr1)((xmlAttrPtr) node, owner_node, &node_map);
        case XML_DOCUMENT_NODE:
            return o3_new(cXmlDocument1)((xmlDocPtr) node, &node_map);
        default:
            return o3_new(cXmlNode1)(node, owner_node, &node_map);
        }
    }

	void swapNode(iCtx* ctx, iXmlNode* old_node, xmlNodePtr new_node, iXmlNode* new_owner_node) {
		if (!old_node || !new_node)
			return;

		siXml ixml = ctx->value("xml").toScr();
		cXml1* xml = (cXml1*) ixml.ptr();
		NodeMap& node_map = xml->m_node_map;
		xmlNodePtr old = ((cXmlNode1*)old_node)->m_node;
		NodeMap::Iter found = node_map.find(old);
		if (found != node_map.end()) {
			siXmlNode ret = (*found).val;
			cXmlNode1* cret = (cXmlNode1*) ret.ptr();
			cret->m_node = new_node;
			cret->m_owner_node = new_owner_node;
			node_map.remove(old);
			node_map[new_node] = cret;
		}
	}
}

#endif // J_C_XML_H
