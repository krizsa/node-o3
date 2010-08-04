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
#define O3_STATIC
//#define O3_WITH_LIBEVENT

#include <core/o3_core.h>
#include <js/o3_js.h>
#ifdef O3_STATIC
#include <console/o3_console.h>
//#include <socket/o3_socket.h>
#include <fs/o3_fs.h>
#include <process/o3_process.h>
#include <xml/o3_xml.h>
#endif // O3_STATIC

using namespace o3;

void run(iCtx* ctx, const char* path)
{
    FILE*   stream;
    siEx    ex;

    stream = fopen(path, "r");
    if (!stream)
        exit(-1);
    ctx->eval(Str(Buf(siStream(o3_new(cStream)(stream)).ptr())), &ex);
    if (ex) {
        fprintf(stderr, "%s\n", ex->message().ptr());
        exit(-1);
    }
}

int main(int argc, char** argv, char** envp)
{
#ifdef O3_WITH_LIBEVENT
	event_init();
#endif

    cSys    sys;
    siMgr   mgr = o3_new(cMgr)();
    siCtx   ctx = o3_new(cJs1)(mgr, argc - 1, argv + 1, envp);

	sys.v8inited();

    mgr->addExtTraits(cConsole1::extTraits());
    //mgr->addExtTraits(cSocket1::extTraits());
	mgr->addExtTraits(cFs1::extTraits());
    mgr->addExtTraits(cProcess1::extTraits());
	mgr->addExtTraits(cXml1::extTraits());
    if (argc < 2)
        return -1;
    //run(ctx, "/bin/prelude.js");
    run(ctx, argv[1]);

	return 0;
}
