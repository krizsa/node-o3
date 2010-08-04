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
#ifndef J_C_XML_NODE_ARRAY_H
#define J_C_XML_NODE_ARRAY_H

namespace o3 {
    struct cXmlNodeArray1 : cScr, iXmlNodeArray {
        cXmlNodeArray1(iXmlNode* owner_node) :
            m_xpath_obj(0), m_nodes(0), m_owner_node(owner_node) {
               o3_trace3 trace;
        }

        virtual ~cXmlNodeArray1() {
            o3_trace3 trace;
            if (m_xpath_obj)
                xmlXPathFreeObject(m_xpath_obj);
        }


        o3_begin_class(cScr)
            o3_add_iface(iXmlNodeArray)
        o3_end_class();

        o3_glue_gen()

        o3_fun bool __query__(int idx) {
            o3_trace3 trace;
            return (idx < length() && idx > 0);
        } 
        
        o3_fun siXmlNode __getter__(iCtx* ctx, int idx, siEx* ex = 0) {
            ex;
            o3_trace3 trace;
            if (idx >= length() || idx<0) {
                    //o3_set_ex(ex_invalid_value);
                return siXmlNode();
            }
            return item(ctx, idx);
        }

        virtual siXmlNode item(iCtx* ctx, int index) {
            o3_trace3 trace;
            o3_assert(m_nodes->nodeTab[index]);
            return wrapNode(ctx, m_nodes->nodeTab[index], m_owner_node ? m_owner_node : this);
        }
        
        virtual o3_get int length() {
            o3_trace3 trace;
            return  (int) (m_nodes ? m_nodes->nodeNr : 0);                
        }

        xmlXPathObjectPtr m_xpath_obj;
        xmlNodeSetPtr m_nodes;    
        siXmlNode m_owner_node;
    };
}

#endif // J_C_XML_NODE_ARRAY_H
