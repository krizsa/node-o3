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
//#include <test/o3_proto_v1.h>
#include "xml/o3_xml.h"
#include "socket/o3_socket.h"
#include "fs/o3_fs.h"
#include "blob/o3_blob.h"
#include "console/o3_console.h"
#include "http/o3_http.h"
#include "process/o3_process.h"
#include "protocol/o3_protocol.h"
#include "resource/o3_resource.h"
#include "screen/o3_screen.h"
#include "window/o3_window.h"
#include "tools/o3_tools.h"
#include "process/o3_process.h"
#include "test/o3_test.h" 

#include "image/o3_image.h"
//#include "scanner/o3_scan.h"
//#include "barcode/o3_barcode.h"

#include "rsa/o3_rsa.h"
#include "sha1/o3_sha1.h"
#include "md5/o3_md5.h"
//#include "canvas/o3_cCanvas1_win32.h"

#include "zip/o3_zip.h"
#include "socket/o3_socket.h"

//int WINAPI WinMain(HINSTANCE hi, HINSTANCE hp, LPSTR arg, int show){
int main(int argc, char **argv) {

    using namespace o3;  

    //CoInitializeEx(NULL, COINIT_APARTMENTTHREADED); 

    cSys sys;

    siMgr mgr = o3_new(cMgr)();
    siCtx ctx = o3_new(cJs1)(mgr, --argc, ++argv,0,true);
   
    
    //mgr->addExtTraits(cCanvas1::extTraits());
    mgr->addExtTraits(cFs1::extTraits());
    mgr->addExtTraits(cHttp1::extTraits());
    mgr->addExtTraits(cBlob1::extTraits());
    mgr->addExtTraits(cConsole1::extTraits());
    mgr->addExtTraits(cXml1::extTraits());
    //mgr->addExtTraits(cJs1::extTraits());
    mgr->addExtTraits(cSocket1::extTraits());
    mgr->addExtTraits(cResource1::extTraits());
    mgr->addExtTraits(cResourceBuilder1::extTraits());
    mgr->addExtTraits(cScreen1::extTraits());
	mgr->addExtTraits(cProcess1::extTraits());
	mgr->addExtTraits(cTest1::extTraits());

	mgr->addExtTraits(cImage1::extTraits());
	//mgr->addExtTraits(cBarcode1::extTraits());
	//mgr->addExtTraits(cScan1::extTraits());

	mgr->addExtTraits(cRSA1::extTraits());
	mgr->addExtTraits(cSHA1Hash1::extTraits());
	mgr->addExtTraits(cMD5Hash1::extTraits());
	mgr->addExtTraits(cZip1::extTraits());

	mgr->addFactory("fs", &cFs1::rootDir);
	mgr->addFactory("http", &cHttp1::factory);

    WSADATA wsd;
    WSAStartup(MAKEWORD(2,2), &wsd);
	int ret = 0;
    bool wait = true;
    {// scope the local vars        
        for(int i = 0; i < argc;i++){
            if(strEquals(argv[i],"-w")) wait = true;
        }	

		HANDLE prelude_file = CreateFileA("prelude.js",GENERIC_READ,
			FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
			NULL,OPEN_EXISTING,0,NULL);			

		if (INVALID_HANDLE_VALUE != prelude_file) {
			unsigned long size,high,read;
			Str prelude(size = GetFileSize(prelude_file, &high));
			ReadFile(prelude_file,prelude.ptr(), size, &read, 0);
			prelude.resize(read);
			//ctx->eval(prelude);
			//if (((cJs1*)ctx.ptr())->scriptError())
			//	return -1;
		}

		HANDLE script_file = CreateFileA(argv[0],GENERIC_READ,
			FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
			NULL,OPEN_EXISTING,0,NULL);		

		if (INVALID_HANDLE_VALUE == script_file) {
			return -1;
		}
	
		unsigned long size,high,read;		
		Str script(size = GetFileSize(script_file, &high));
		ReadFile(script_file,script.ptr(), size, &read, 0);
		script.resize(read);
		ctx->eval(script);
		if (((cJs1*)ctx.ptr())->scriptError())
			ret = -1;
		

        siCtx1(ctx)->tear();
	}
    
    //CoUninitialize(); 

    // if(wait)
	// getc(stdin);

    WSACleanup();
    return ret;
}  

