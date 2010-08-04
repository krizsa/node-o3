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
#ifndef O3_C_PROTOCOL1_WIN32_H
#define O3_C_PROTOCOL1_WIN32_H

namespace o3 {    
    o3_iid(iProtocol, 0xb9c1a43a, 0x4cab, 0x4d80, 0x9b, 0xce, 0x6e, 0x3a, 0x92, 0x26, 
        0x7e, 0x5f);

    struct iProtocol : iUnk {
        virtual Str processURL(const char* url) = 0;

        virtual siStream start(const char* url, Delegate progress_cb = Delegate()) = 0;
    };
    
    struct cProtocol1 : cScr, iProtocol
    {
        cProtocol1(iCtx* ctx)
            :   m_ctx(ctx)
        {
            // by default let's use the attached resource as the source
            // NOTE: the o3 object should be available as a value... 
            Var v = ctx->value("o3");
            siScr o3 = v.toScr();
            if (o3) {
                int id = o3->resolve(siCtx(m_ctx),"resources", false);
                o3->invoke(siCtx(m_ctx), ACCESS_GET, id, 0, 0, &v);
                m_source = v.toScr();
            }            
        }

        virtual ~cProtocol1()
        {
        
        }


        o3_begin_class(cScr)
            o3_add_iface(iProtocol)
        o3_end_class()

        siScr           m_source;
        siWeak          m_ctx;

        o3_glue_gen()

        // some custom url magic if needed...
        virtual Str processURL(const char* url)
        {
            Str processed(url);
            if (!strCompare(processed.ptr(),"o3://"),5)
                processed.remove(0,5);
            if (processed[processed.size() - 1] == '/')
                processed.remove(processed.size()-1,1);

            return processed;
        }

        // starting a request and sending the necessary notifications
        virtual siStream start(const char* url, Delegate progress_cb = Delegate())
        {
            if (!m_source)
                return siStream();

            Var ret,path = url;
            int id = m_source->resolve(siCtx(m_ctx),"protocolOpen", false);
            m_source->invoke(siCtx(m_ctx), ACCESS_CALL, id, 1, &path, &ret);
            return ret.toScr();
        }       

        o3_fun bool addSource(iScr* source)
        {
            m_source = source;
            return true;
        }

        o3_ext("cO3") o3_get static siScr protocolHandler(iCtx* ctx)
        {
            // the protocol handler needs to be created and registered
            // by the real protocol class it is used by...
            Var v = ctx->value("protocolHandler");
            return v.toScr();
        }
    };
}

#endif // O3_C_PROTOCOL1_WIN32_H
