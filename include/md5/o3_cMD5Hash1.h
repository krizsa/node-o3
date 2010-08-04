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
#ifndef O3_CMD5HASH1_H
#define O3_CMD5HASH1_H

#include "core/o3_crypto.h"

namespace o3 {
    struct cMD5Hash1 : cScr {
        cMD5Hash1() 
		{
        }

        o3_begin_class(cScr)
        o3_end_class();

		o3_glue_gen();

		static o3_ext("cO3") o3_get siScr md5(iCtx* ctx)
		{
			Var var = ctx->value("md5");
			siScr md5 = var.toScr();
			if (md5)
				return md5;
			else 
				return ctx->setValue(
					"md5",Var(o3_new(cMD5Hash1)())).toScr();
		}

        o3_fun Buf hash(const Buf& buf) 
		{			
			return md5((const uint8_t*)buf.ptr(), buf.size());
        }

		o3_fun Buf hash(const Str& str) 
		{			
			return md5((const uint8_t*)str.ptr(), str.size());
		}

		Buf md5( const uint8_t* in, size_t in_len ) 
		{
			size_t size = MD5_SIZE;
			Buf out;

			out.reserve(size);
			hashMD5(in, in_len, (uint8_t*)out.ptr());
			out.resize(size);
			return out;
		}
	};
}

#endif // O3_CMD5HASH1_H
