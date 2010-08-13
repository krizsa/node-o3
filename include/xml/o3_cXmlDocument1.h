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
#ifndef J_C_XML_DOCUMENT_H
#define J_C_XML_DOCUMENT_H

//#include <malloc.h>

namespace o3 {
    struct cXmlDocument1 : cXmlNode1, iXmlDocument {   
        cXmlDocument1(xmlDocPtr doc, NodeMap* node_map) 
            : cXmlNode1((xmlNodePtr) doc, 0, node_map), m_doc(doc) { 
				//printf("^xmlDoc\n");	
        }

        virtual ~cXmlDocument1()
        {
			//printf("dest doc %p %p", m_doc, this);
            xmlFreeDoc(m_doc);
			//xmlMemoryDump();

			//malloc_trim(0);
            
			m_node_map->remove(m_node);
			m_node = 0;
        }

        o3_begin_class(cXmlNode1)
            o3_add_iface(iXmlDocument)
        o3_end_class();
    
        o3_glue_gen()

        //o3_invoke(dump) {
        //    chr* res;
        //    int length;
        //    xmlDocDumpMemoryEnc(m_doc, (xmlChar**) &res, &length, "ISO-8859-1");
        //    o3_rval->set(Str(res));
        //    xmlFree(res);
        //    return true;
        //}
            
        virtual o3_get siXmlElement documentElement(iCtx* ctx) {
            o3_trace3 trace;
            return wrapNode(ctx, xmlDocGetRootElement(m_doc), this);
        }
        
        virtual o3_fun siXmlElement createElement(iCtx* ctx, const char* tagName) {
            o3_trace3 trace;
            o3_assert(tagName);
            xmlNodePtr node = xmlNewDocNode(m_doc, 0, BAD_CAST tagName, 0);

            return wrapNode(ctx,  node, 0);
        }
        
		virtual o3_fun siXmlText createTextNode(iCtx* ctx, const char* data) {
            o3_trace3 trace;
            o3_assert(data);
            xmlNodePtr node = xmlNewDocText(m_doc, BAD_CAST data);

            return wrapNode(ctx, node, 0);
        }

		virtual o3_fun siXmlComment createComment(iCtx* ctx, const char* data) {
            o3_trace3 trace;
            o3_assert(data);
			xmlNodePtr node = xmlNewDocComment(m_doc, BAD_CAST data);

            return wrapNode(ctx, node, 0);
		}
        
		virtual o3_fun siXmlCDATASection createCDATASection(iCtx* ctx, const char* data) {
            o3_trace3 trace;
            o3_assert(data);
            size_t length = strLen(data);
			xmlNodePtr node = xmlNewCDataBlock(m_doc, BAD_CAST data, length);

            return wrapNode(ctx, node, 0);
		}
        
        virtual o3_fun siXmlAttr createAttribute(iCtx* ctx, const char* name) {
            o3_trace3 trace;
            o3_assert(name);
            xmlAttrPtr attr = xmlNewDocProp(m_doc, BAD_CAST name, 0);

            return wrapNode(ctx, (xmlNodePtr) attr, 0);
        }

		virtual o3_fun siXmlNodeList getElementsByTagName(iCtx* ctx, const char* tagName) {
            o3_trace3 trace;
            o3_assert(tagName);
			return documentElement(ctx)->getElementsByTagName(tagName);
		}

        virtual xmlDocPtr docPtr() {
            o3_trace3 trace;
            return m_doc;
        }

//		xmldoc.setProperty("SelectionNamespaces",
//			"xmlns:wrox='http://www.wrox.com/' xmlns='http://www.amazon.com/'");

		o3_fun bool setProperty(const Str& prop, const Str& val) {
			if (!strEquals(prop.ptr(), "SelectionNamespaces"))
				return false;

			const char* p=val.ptr();
			while ((p = parseNextNS(p)) != 0);
			return true;
		}

		const tMap<Str,Str>& nameSpaces() {
			return m_ns;
		}

		const char* parseNextNS(const char* p) {
			if (!p || !*p)
				return 0;

			while(*p && *p==' ') 
				p++;

			if (!strEquals(p,"xmlns",5))
				return 0;
			p+=5;
			Str tag;
			const char* e;
			switch(*p) {
				case ':' :
					for(e=p;*e&&*e!='=';e++);

					if (!*e)
						return 0;
					tag = Str(p+1,e-p-1);
					p = e+1;
					break;
				case '=' :
					tag = "defaultNamespace";
					p++;
					break;
				default: 
					return 0;
			}
			switch(*p) {
				case '\'':
					for(e=p+1;*e&&*e!='\'';e++);
										
					break;
				case '\"':
					for(e=p+1;*e&&*e!='\"';e++);

					break;
				default:
					return 0;
			}
			if (!*e)
				return 0;
			
			m_ns[tag] = Str(p+1, e-p-1);
			return e+1;
		}

        xmlDocPtr m_doc;
		tMap<Str,Str> m_ns;
    };
}

#endif // J_C_XML_DOCUMENT_H
