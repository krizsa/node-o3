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
#ifndef J_C_XML_ELEMENT_H
#define J_C_XML_ELEMENT_H


namespace o3 {
    struct cXmlElement1 : cXmlNode1, iXmlElement {
        cXmlElement1(xmlNodePtr node, iXmlNode* owner_node, NodeMap* node_map) 
            : cXmlNode1(node, owner_node, node_map) {
            o3_trace3 trace;
        }

		virtual ~cXmlElement1()
		{

		}

        o3_begin_class(cXmlNode1)
            o3_add_iface(iXmlElement)
        o3_end_class();

        o3_glue_gen()
        
        o3_fun siXmlNode selectSingleNode(iCtx* ctx, const char* selection, iScr* ctxt = 0) {
            xmlXPathObjectPtr res = selectNodesInternal(
                selection, siXmlDocument(ctxt), ctx);
            if (!res || !res->nodesetval || !res->nodesetval->nodeTab) 
                return siXmlNode();
            siXmlNode ret = wrapNode(ctx, res->nodesetval->nodeTab[0], m_owner_node ? m_owner_node : this);
            xmlXPathFreeObject(res);
            return ret;
        }

        o3_fun siXmlNodeArray selectNodes(iCtx* ctx, const char* selection, iScr* ctxt = 0) {
            o3_trace3 trace;
            xmlXPathObjectPtr res = selectNodesInternal(
                selection, siXmlDocument(ctxt), ctx);
            if (!res)
                return siXmlNode();
            cXmlNodeArray1* nodes = o3_new(cXmlNodeArray1)(m_owner_node ? m_owner_node : this);
            nodes->m_nodes = res->nodesetval;
            nodes->m_xpath_obj = res;
            return siXmlNodeArray(nodes);            
        }
        
        virtual o3_get Str tagName() {
            o3_trace3 trace;
            return Str((const char*) m_node->name);
        }
        
        virtual o3_fun Str getAttribute(iCtx* ctx, const char* name) {
			o3_trace3 trace;
			o3_assert(name);
			siXmlAttr attr = getAttributeNode(ctx, name);
			if (attr)
				return attr->value();

			if (strCompare("xmlns", name, 5))
				return Str();
			
			const char* pre = *(name+5) == ':' ? name+6 : 0;

			xmlNsPtr ns = m_node->nsDef;
			while (ns) {
				if (!ns->prefix && !pre)
					return Str((const char*) ns->href);
				
				if (ns->prefix && pre && !strCompare((const char*) ns->prefix, pre)) 
					return Str((const char*) ns->href);
				
				ns = ns->next;
			}

			return Str();
        }
        
        virtual o3_fun void setAttribute(iCtx* ctx, const char* name, const char* value) {
            o3_trace3 trace;
            o3_assert(name);
            o3_assert(value);
            siXmlAttr new_attr;
			if (ownerDocument(ctx))
				new_attr = ownerDocument(ctx)->createAttribute(ctx, name);
            else{
				xmlAttrPtr attr = xmlNewDocProp(0, BAD_CAST name, 0);
				new_attr = wrapNode(ctx, (xmlNodePtr) attr, 0);
			}
						
			new_attr->setValue(value);
            setAttributeNode(ctx, new_attr);
        }
        
        virtual o3_fun void removeAttribute(iCtx* ctx, const char* name) {
            o3_trace3 trace;
            o3_assert(name);
            removeAttributeNode(getAttributeNode(ctx, name));
        }
        
        o3_fun siXmlAttr getAttributeNode(iCtx* ctx, const char* name) {
            o3_trace3 trace;
            o3_assert(name);
            xmlAttrPtr attr = xmlHasProp(m_node, BAD_CAST name);
            
            if (attr == NULL)
                return siXmlAttr();
            return wrapNode(ctx, (xmlNodePtr) attr, m_owner_node ? m_owner_node : this);
        }
        
        o3_fun siXmlAttr setAttributeNode(iCtx* ctx, iXmlAttr* new_attr) {
            o3_trace3 trace;            
            return new_attr ? appendChild(ctx, siXmlNode(new_attr)) : siXmlNode();
        }
        
        o3_fun siXmlAttr removeAttributeNode(iXmlAttr* old_attr) {
            o3_trace3 trace;                
            return old_attr ? removeChild(siXmlNode(old_attr)) : siXmlNode();
        }
        
		o3_fun siXmlNodeList getElementsByTagName(const char* name) {
            o3_trace3 trace;
            o3_assert(name);
            return siXmlNodeList(o3_new(cXmlNodeList1)(this, name));
		}
        
        xmlXPathObjectPtr selectNodesInternal(const char* expr, iXmlDocument* ctxt, iCtx* ctx) {
            o3_trace3 trace;
            o3_assert(expr);

            xmlXPathObjectPtr xpathObj;
            xmlXPathContextPtr xpathCtx;             
            // Create xpath evaluation context
            if (ctxt){
                //search from the doc
                xpathCtx = xmlXPathNewContext(ctxt->docPtr());
                o3_assert(xpathCtx);
            } else {
                //search from current node
                o3_assert(m_node);
                xpathCtx = xmlXPathNewContext( m_node->doc );
                o3_assert(xpathCtx);
                xpathCtx->node = m_node;                
            }
            //registering namespaces
            siXmlDocument doc = ownerDocument(ctx);
			if (doc){				
				const tMap<Str,Str> ns = doc->nameSpaces();
				tMap<Str,Str>::ConstIter it;
				for (it=ns.begin(); it != ns.end(); ++it) {
					if (xmlXPathRegisterNs(xpathCtx, 
						BAD_CAST (*it).key.ptr(),
						BAD_CAST (*it).val.ptr()))
					{
						o3_assert(false);
					}
				}
            }
            //do the xpath eval
            xpathObj = xmlXPathEval(BAD_CAST expr, xpathCtx);
            if(!xpathObj) {
                xmlXPathFreeContext(xpathCtx); 
                return 0;
            }
            
            xmlXPathFreeContext(xpathCtx); 
            return xpathObj;            
        }
        
        o3_fun void normalize() {
            o3_trace3 trace;
        }
        
        virtual xmlAttrPtr firstAttr() {
            o3_trace3 trace;
            return m_node->properties;
        }
    };
}

#endif // J_C_XML_ELEMENT_H
