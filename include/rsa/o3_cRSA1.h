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
#ifndef O3_CRSA1_H
#define O3_CRSA1_H

#include "core/o3_crypto.h"

namespace o3 {
    struct cRSA1 : cScr {
        cRSA1()
		{
        }

        o3_begin_class(cScr)
		o3_end_class();

		o3_glue_gen();

        static o3_ext("cO3") o3_get siScr rsa(iCtx* ctx)
		{
			Var var = ctx->value("rsa");
			siScr rsa = var.toScr();
			if (rsa)
				return rsa;
			else 
				return ctx->setValue(
					"rsa",Var(o3_new(cRSA1)())).toScr();
        }

        o3_fun Buf encrypt( const Buf& in, const Buf& mod, const Buf& exp, bool prv=true )
		{
            size_t size = (in.size() / mod.size() + 1) * mod.size();
            Buf out;

            out.reserve(size);
            size = encryptRSA((uint8_t*)in.ptr(), in.size(), (uint8_t*)out.ptr(), (uint8_t*)mod.ptr(),
                              mod.size(), (uint8_t*)exp.ptr(), exp.size(), prv);
            out.resize(size);
			return out;
        }

        o3_fun Buf decrypt( const Buf& in, const Buf& mod, const Buf& exp, bool prv=true )
		{
            size_t size = (in.size() / mod.size() + 1) * mod.size();
            Buf out;

            out.reserve(size);
            size = decryptRSA((const uint8_t*) in.ptr(), in.size(), (uint8_t*) out.ptr(), (uint8_t*) mod.ptr(),
                              mod.size(),(const uint8_t*) exp.ptr(), exp.size(), prv);
            if (size == (size_t)-1)
                return Buf();
            out.resize(size);
			return out;
		}
    };
}

#endif // O3_CRSA1_H
