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
#include <js/o3_js.h>

#include "fs/o3_fs.h"
#include "blob/o3_blob.h"
#include "resource/o3_resource.h"
#include "screen/o3_screen.h"
#include "window/o3_window.h"
#include "tools/o3_tools.h"
#include "process/o3_process.h"

#include "commctrl.h"

int WINAPI WinMain(HINSTANCE hi, HINSTANCE hp, LPSTR arg, int show)
//int main(int argc, char **argv) {
{
    using namespace o3;  

    cSys sys;


    //CoInitializeEx(NULL, COINIT_APARTMENTTHREADED); 


    INITCOMMONCONTROLSEX cc = {sizeof( INITCOMMONCONTROLSEX ), ICC_WIN95_CLASSES /*| ICC_STANDARD_CLASSES*/};
    InitCommonControlsEx(&cc); 



    int ret = 0;
    {// scope the local vars
        
        siMgr mgr = o3_new(cMgr)();    
       
        mgr->addExtTraits(cFs1::extTraits());
        mgr->addExtTraits(cBlob1::extTraits());
        mgr->addExtTraits(cJs1::extTraits());
        mgr->addExtTraits(cResource1::extTraits());
        mgr->addExtTraits(cResourceBuilder1::extTraits());
        mgr->addExtTraits(cScreen1::extTraits());
        mgr->addExtTraits(cWindow1::extTraits());
        mgr->addExtTraits(cButton1::extTraits());
        mgr->addExtTraits(cStaticCtrl1::extTraits());
        mgr->addExtTraits(cTools1::extTraits());
        mgr->addExtTraits(cProcess1::extTraits());

        char x[MAX_PATH];
        GetModuleFileNameA(0, x, MAX_PATH);
        
        char* args[3];
        args[0] = x;
        // TODO: need a command line parser...
        args[1] = arg;
		args[2] = 0;

        siCtx js = o3_new(cJs1)(mgr, 2, args, 0, true);

        Buf buf = ((cSys*)g_sys)->resource("installer.js");
        Str script(buf);
        Var rval;
        rval = js->eval(script);  

        Str err = rval.toStr();

		Var exitCode = js->value("exitCode");
        ret = exitCode.toInt32();
		((cJs1*)js.ptr())->tear();
    }
    
    //CoUninitialize(); 
    return ret;
}  