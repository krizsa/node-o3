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
#ifndef O3_CSHA1HASH_H
#define O3_CSHA1HASH_H

#include "core/o3_crypto.h"

namespace o3 {
    struct cSHA1Hash1 : cScr {
        cSHA1Hash1() 
		{            
        }

        o3_begin_class(cScr)
        o3_end_class();

		o3_glue_gen();

		static o3_ext("cO3") o3_get siScr sha1(iCtx* ctx)
		{
			Var var = ctx->value("sha1");
			siScr sha1 = var.toScr();
			if (sha1)
				return sha1;
			else 
				return ctx->setValue(
				"sha1",Var(o3_new(cSHA1Hash1)())).toScr();
		}

        o3_fun Buf hash(const Buf& buf)
		{                
			return sh1((const uint8_t*)buf.ptr(), buf.size());
        }

		o3_fun Buf hash(const Str& str)
		{
			return sh1((const uint8_t*)str.ptr(), str.size());
		}

		Buf sh1( const uint8_t* in, size_t in_len ) 
		{
			size_t size = SHA1_SIZE;
			Buf out;

			out.reserve(size);
			hashSHA1(in, in_len, (uint8_t*)out.ptr());
			out.resize(size);
			
			return out;
		}
	};
}

#endif // O3_CSHA1HASH_H
