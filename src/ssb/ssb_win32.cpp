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

#include <core/o3_core.h>

//#include "xml/o3_xml.h"
#include <shared/o3_tools_win32.h>
#include <shared/o3_glue_idispatch.h>

#include "fs/o3_fs.h"
#include "blob/o3_blob.h"
//#include "http/o3_http.h"
#include "window/o3_window.h"
#include "resource/o3_resource.h"
#include "protocol/o3_protocol.h"
#include "screen/o3_screen.h"

#include "o3_HostIE.h"

namespace o3 {

struct cCtx : cUnk, iCtx, iCtx1 {

    cCtx(iMgr* mgr) 
     : m_mgr(mgr)
     , m_track(0)
     , m_loop(g_sys->createMessageLoop())
    {        
    }

    virtual ~cCtx()
    {        
    }

    o3_begin_class(cUnk)
        o3_add_iface(iAlloc)
        o3_add_iface(iCtx)
        o3_add_iface(iCtx1)
    o3_end_class()

    siMgr                       m_mgr;
    ComTrack*                   m_track;
    siMessageLoop               m_loop;
    tMap<Str, Var>              m_values;
    HANDLE                      m_app_window;

    // iAlloc
    void* alloc(size_t size)
    {
        return g_sys->alloc(size);
    }

    void free(void* ptr)
    {
        return g_sys->free(ptr);
    }

    // iCtx
    siMgr mgr()
    {
        return m_mgr;
    }

    virtual ComTrack** track() 
    {
       return &m_track;
    }

    virtual siMessageLoop loop()
    {
        return m_loop;
    }

    virtual Var value(const char* key) 
    {
        return m_values[key];
    }

    virtual Var setValue(const char* key, const Var& val)
    {
        return m_values[key] = val;
    }

    virtual void tear() 
    {
        for (ComTrack *i = m_track, *j = 0; i; i = j) {
            j = i->m_next;
            i->m_phead = 0;
            i->tear();
        }
    }

    virtual Str fsRoot()
    {
        return Str();
    }

    virtual Var eval(const char* name, siEx* ex = 0)
    {
        return Var();    
    }

    virtual void setAppWindow(void* handle)
    {
        m_app_window = handle;
    }
    
    virtual void* appWindow() 
    {
        return (void*) m_app_window;
    }
}; 

}

int WINAPI WinMain(HINSTANCE hi, HINSTANCE hp, LPSTR arg, int show)
{
    using namespace o3;

    if (OleInitialize(NULL) != S_OK)
        return -1;

    cSys sys;

    siMgr mgr = o3_new(cMgr)();
    siCtx ctx = o3_new(cCtx)(mgr);        

    //mgr->addExtTraits(cO3::extTraits());
    mgr->addExtTraits(cFs1::extTraits());
    //mgr->addExtTraits(cHttp1::extTraits());
    mgr->addExtTraits(cBlob1::extTraits());
    //mgr->addExtTraits(cXml_v1::extTraits());abrakadabra
    mgr->addExtTraits(cWindow1::extTraits());
    mgr->addExtTraits(cResource1::extTraits());
    mgr->addExtTraits(cProtocol1::extTraits());
    mgr->addExtTraits(cScreen1::extTraits());

	ssb_setIE8mode("ssb.exe");

    HostIE* host = o3_new(HostIE)(ctx);
    host->AddRef();

    Str err;

    if (!host->createWindow())
        err = getLastError();
    
    if (!host->initProtocol())
        err = getLastError(); 
    

    Str url = "file:///";
    if (arg && *arg){
        Str path(arg);
        path.findAndReplaceAll("\\", "/");
        url.append(path.ptr());
    } else {
        Str cwd = cwdPath();
        cwd.findAndReplaceAll("\\", "/");
        url.append(cwd);
        url.append("/ssb_start.html");
    }

    host->displayURL(url);
        
    host->showWindow();
    host->start();
    OleUninitialize();
    return 0;
}