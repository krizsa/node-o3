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
namespace o3 {

    
    o3_iid(iXmlNodeList, 0x77b83a81, 0x15c2, 0x4304, 0x9b, 0xcf, 0xb5, 0xe7, 0xde, 0xcb, 0xaa, 0x34);    
	o3_iid(iXmlNodeArray, 0xfdede7dd, 0xcd4b, 0x4a2f, 0x94, 0xd3, 0xdd, 0xca, 0xa7, 0x84, 0x43, 0x84); 
    o3_iid(iXmlNamedNodeMap, 0x455d1b4c, 0xcdce, 0x4968, 0x9b, 0x46, 0xa, 0x5d, 0xd3, 0x85, 0xa2, 0xdf);
	o3_iid(iXmlNode, 0x2aac3ccb, 0x6394, 0x4dec, 0x91, 0x33, 0x49, 0x1c, 0xe, 0x8, 0x3f, 0xe3);
    o3_iid(iXmlElement, 0x9d5c3325, 0x4567, 0x4d3f, 0x89, 0x3d, 0x30, 0x2c, 0xe0, 0xda, 0x6a, 0x6b);
	o3_iid(iXmlText, 0x9df0bee1, 0xd05b, 0x4587, 0x98, 0x85, 0x6, 0xcd, 0x70, 0xf8, 0x39, 0xb6);
    o3_iid(iXmlCDATASection,0x46e65a93, 0x36c4, 0x4be2, 0x9a, 0x2, 0x40, 0x60, 0xab, 0x1c, 0x38, 0xc3);
	o3_iid(iXmlComment, 0x2d13412e, 0xf3e0, 0x4f85, 0x9b, 0x42, 0x6c, 0xc9, 0x18, 0x6c, 0x3a, 0x95);
	o3_iid(iXmlDocument, 0xb8508e87, 0xc49d, 0x4951, 0xba, 0x49, 0xb2, 0x6f, 0x26, 0xc2, 0xc0, 0x94);
    o3_iid(iXml, 0x9e5245a9, 0x582b, 0x4e1a, 0x8c, 0x97, 0xb8, 0xc9, 0xbf, 0x1c, 0xd4, 0xcb);
    o3_iid(iXmlCharacterData, 0x143a421d, 0xe895, 0x409b, 0xac, 0x55, 0xdb, 0x5e, 0xab, 0x14, 0x15, 0x89);
    o3_iid(iXmlAttr, 0x4fc036d5, 0xf100, 0x4b2d, 0xa6, 0x88, 0xd0, 0x30, 0x2a, 0x5a, 0x84, 0x84);

    typedef tMap<xmlNodePtr, iXmlNode*> NodeMap;

    struct iXml : iUnk {

    };

    struct iXmlAttr : iUnk {
            virtual Str name() = 0;
            virtual bool specified() = 0;
            virtual Str value() = 0;
            virtual void setValue(const char* value) = 0;
    };

    struct iXmlCDATASection : iUnk {

	};

	struct iXmlCharacterData : iUnk {
		virtual Str data() = 0;
		virtual void setData(const char* data) = 0;
		virtual int length() = 0;

        virtual Str substringData(int offset, int count, siEx* ex = 0) = 0;
		virtual void appendData(const char* arg) = 0;
		virtual void insertData(int offset, const char* arg, siEx* ex = 0) = 0;
		virtual void deleteData(int offset, int count, siEx* ex = 0) = 0;
		virtual void replaceData(int offset, int count, const char* arg, siEx* ex = 0) = 0;
	};

    struct iXmlComment : iUnk {

	};

    struct iXmlDocument : iUnk {
        virtual siXmlElement documentElement(iCtx* ctx) = 0;
        virtual siXmlElement createElement(iCtx* ctx, const char* tagName) = 0;
		// TODO: createDocumentFragment()
        virtual siXmlText createTextNode(iCtx* ctx, const char* data) = 0;
		virtual siXmlComment createComment(iCtx* ctx, const char* data) = 0;
		virtual siXmlCDATASection createCDATASection(iCtx* ctx, const char* data) = 0;
		// TODO: createProcessingInstruction()
        virtual siXmlAttr createAttribute(iCtx* ctx, const char* name) = 0;
		// TODO: createEntityReference()
        virtual siXmlNodeList getElementsByTagName(iCtx* ctx, const char* tagName) = 0;        
        
        virtual xmlDocPtr docPtr() = 0;
		virtual const tMap<Str,Str>& nameSpaces() = 0;
    };

    struct iXmlElement : iUnk {
        virtual Str tagName() = 0;
        
        virtual Str getAttribute(iCtx* ctx, const char* name) = 0;
        virtual void setAttribute(iCtx* ctx, const char* name, const char* value) = 0;
        virtual void removeAttribute(iCtx* ctx, const char* name) = 0;
        
        virtual siXmlAttr getAttributeNode(iCtx* ctx, const char* name) = 0;
        virtual siXmlAttr setAttributeNode(iCtx* ctx, iXmlAttr* new_attr) = 0;
        virtual siXmlAttr removeAttributeNode(iXmlAttr* old_attr) = 0;
        
		virtual siXmlNodeList getElementsByTagName(const char* name) = 0;
        virtual void normalize() = 0;

        virtual xmlAttrPtr firstAttr() = 0;
    };

    struct iXmlNamedNodeMap : iUnk {
		virtual siXmlNode getNamedItem(iCtx* ctx, const char* name) = 0;
		virtual siXmlNode setNamedItem(iCtx* ctx, iXmlNode* arg) = 0;
		virtual siXmlNode removeNamedItem(iCtx* ctx, const char* name) = 0;

		virtual siXmlNode item(iCtx* ctx, int index) = 0;
		virtual int length() = 0;
    };

    struct iXmlNode : iUnk {        

        enum Type {
            TYPE_ELEMENT = 1,
            TYPE_ATTRIBUTE = 2,
            TYPE_TEXT = 3,
            TYPE_CDATA_SECTION = 4,
            TYPE_COMMENT = 8,
            TYPE_DOCUMENT = 9
        };

        virtual Str nodeName() = 0;
        virtual Str nodeValue() = 0;
        virtual void setNodeValue(const char* value) = 0;
        virtual Type nodeType() = 0;
        
		virtual siXmlNode parentNode(iCtx* ctx) = 0;
        virtual siXmlNodeList childNodes() = 0;
        virtual siXmlNode firstChild(iCtx* ctx) = 0;
        virtual siXmlNode lastChild(iCtx* ctx) = 0;
        virtual siXmlNode previousSibling(iCtx* ctx) = 0;
        virtual siXmlNode nextSibling(iCtx* ctx) = 0;
        virtual siXmlNamedNodeMap attributes() = 0;
        virtual siXmlDocument ownerDocument(iCtx* ctx) = 0;
        
		virtual siXmlNode insertBefore(iCtx* ctx, iXmlNode* ref_child, iXmlNode* new_child) = 0;
        virtual siXmlNode replaceChild(iCtx* ctx, iXmlNode* new_child, iXmlNode* old_child) = 0;
        virtual siXmlNode removeChild(iXmlNode* old_child) = 0;
        virtual siXmlNode appendChild(iCtx* ctx, iXmlNode* new_child) = 0;
        
		virtual bool hasChildNodes() = 0;
        virtual siXmlNode cloneNode(iCtx* ctx, bool deep = false) = 0;     
        virtual siXmlNode ownerNode() = 0;
    };

    struct iXmlNodeArray : iUnk {
		virtual siXmlNode item(iCtx* ctx, int index) = 0;
		virtual int length() = 0;
    };

    struct iXmlNodeList : iUnk {
		virtual siXmlNode item(iCtx* ctx, int index) = 0;
		virtual int length(iCtx* ctx) = 0;
    };

    struct iXmlText : iUnk {
		virtual siXmlText splitText(int offset) = 0; 
	};

    siXmlNode wrapNode(iCtx* ctx, xmlNodePtr node, iXmlNode* owner_node);
	void swapNode(iCtx* ctx, iXmlNode* old_node, xmlNodePtr new_node, iXmlNode* new_owner_node);
}