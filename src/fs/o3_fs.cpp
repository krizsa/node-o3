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
#include <fs/o3_fs.h>

using namespace o3;

extern "C" {

o3_mod bool o3_init(iSys* sys)
{
    g_sys = sys;
    return true;
}

o3_mod void o3_reg(iMgr* mgr)
{
    mgr->addExtTraits(cFs1::extTraits());
}

}
