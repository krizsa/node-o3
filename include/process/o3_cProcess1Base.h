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
#ifndef O3_C_PROCESS1_BASE_H
#define O3_C_PROCESS1_BASE_H

namespace o3 {

o3_cls(cProcess1Base);

struct cProcess1Base : cScr {
    siCtx m_ctx;
    siScr m_onterminate;

    o3_begin_class(cScr)
    o3_end_class()

	o3_glue_gen()

    virtual o3_get siStream stdIn() = 0;

    virtual o3_set siStream setStdIn(iStream* in) = 0;

    virtual o3_get siStream stdOut() = 0;

    virtual o3_set siStream setStdOut(iStream* out) = 0;

    virtual o3_get siStream stdErr() = 0;

    virtual o3_set siStream setStdErr(iStream* err) = 0;

    virtual o3_get siScr onterminate()
    {
        return m_onterminate;
    }

    virtual o3_set siScr setOnterminate(iCtx* ctx, iScr* onterminate)
    {
        m_ctx = ctx;
        if (m_onterminate = onterminate)
            startListening();
        else
            stopListening();
        return m_onterminate;
    }

    virtual o3_fun void exec(iCtx* ctx, const char* args) = 0;

    virtual o3_get int exitCode() = 0;

    virtual void startListening() = 0;

    virtual void stopListening() = 0;
};

}

#endif // O3_C_PROCESS1_BASE_H
