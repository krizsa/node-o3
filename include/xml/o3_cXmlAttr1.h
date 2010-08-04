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
#ifndef J_C_XML_ATTR_H
#define J_C_XML_ATTR_H

namespace o3 {
    struct cXmlAttr1 : cXmlNode1, iXmlAttr {
        cXmlAttr1(xmlAttrPtr attr, iXmlNode* owner_node, NodeMap* node_map) 
            : cXmlNode1((xmlNodePtr)attr, owner_node, node_map), m_attr(attr) {
            o3_trace3 trace;            
        }

		virtual ~cXmlAttr1()
		{

		}

        o3_begin_class(cXmlNode1)
            o3_add_iface(iXmlAttr)
        o3_end_class();
       
        o3_glue_gen()

        o3_get Str name() {
            o3_trace3 trace;
            return Str((const char*) m_attr->name);
        }

        o3_get bool specified() {
            o3_trace3 trace;
            return true;
        }

        virtual o3_get Str value() {
            o3_trace3 trace;
            return nodeValue();
        }

        virtual o3_set void setValue(const char* value) {
            o3_trace3 trace;
            setNodeValue(value);
        }

        xmlAttrPtr m_attr;
    };
}

#endif // J_C_XML_ATTR_H
