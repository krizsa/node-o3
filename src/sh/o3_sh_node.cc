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

#include <v8.h>
#include <core/o3_core.h>
#include <js/o3_js.h>
#include <console/o3_console.h>
#include <xml/o3_xml.h>
//#include <socket/o3_socket.h>
#include <fs/o3_fs.h>
#include <image/o3_image.h>

using namespace o3;
using namespace v8;



extern "C" void
init (Handle<Object> target) 
{
  HandleScope scope;
  
  g_sys = new cSys();
  siMgr mgr = o3_new(cMgr)(); // will be referenced by cJs1
  mgr->addExtTraits(cConsole1::extTraits());
  mgr->addExtTraits(cXml1::extTraits());
  mgr->addExtTraits(cFs1::extTraits());
  mgr->addExtTraits(cImage1::extTraits());	

  iCtx* ctx = o3_new(cJs1)(target, mgr, 0, 0, 0);
  ctx->addRef(); // will be released by a clenup callback in the cJs1
  
}

