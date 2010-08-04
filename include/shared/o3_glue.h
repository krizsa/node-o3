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

#ifndef O3_C_GLUE_H
#define O3_C_GLUE_H

namespace o3 {

struct ComTrack {
    ComTrack(ComTrack** phead) {
    	m_phead = phead;
	    m_prev  = 0;
        if (*phead)
            (*phead)->m_prev = this, m_next = *phead;
        else
            m_next = 0;
        *phead = this;   
    }

    virtual ~ComTrack() {
	    if (m_phead && *m_phead == this)
            *m_phead = m_next;
	    if (m_next)
            m_next->m_prev = m_prev;
	    if (m_prev)
            m_prev->m_next = m_next;
    }

    virtual void tear() = 0;

    ComTrack** m_phead;
    ComTrack*  m_next;
    ComTrack*  m_prev;
};

o3_iid(iCtx1, 0xc0dfdab2, 0x1004, 0x42e6, 0xb4, 0xd5, 0xdf, 0xf3, 0x76, 0x7b, 0x59, 0x4b);

struct iCtx1 : iUnk {
    virtual ComTrack** track() = 0;

    virtual void tear() = 0;
};

struct ScrInfo {
    bool    func;
    Str     name;
};

inline void scrInfo(iScr* pthis, iCtx* ctx, int index, ScrInfo* info) {
    cScr* pthis1 = (cScr*) pthis;

    if (index < O3_VALUE_OFFSET) {
        Trait*  traits;
        Trait*  ptrait;

        traits = pthis1->select();
        while (index >= O3_TRAIT_COUNT) {
            traits  = (Trait*) traits->ptr;
            index  -= O3_TRAIT_COUNT;
        }
        if (index < O3_EXT_TRAIT_COUNT) 
            ptrait = traits + 1;
        else {
            ptrait   = ctx->mgr()->extTraits(traits->cls_name) + 1;
            index   -= O3_EXT_TRAIT_COUNT;
        }
        for (; ptrait->type != Trait::TYPE_END; ++ptrait)
            if (ptrait->index == index)
                break;
        info->func = ptrait->type == Trait::TYPE_FUN;
        info->name = ptrait->fun_name;
    } else {
        info->func = pthis1->m_values[index].type() == Var::TYPE_SCR;
        info->name = pthis1->name(ctx, index);
    }
}

static const char* ex_invalid_op = "Operation on this property not supported";
static const char* ex_invalid_argc = "Invalid argument count";
static const char* ex_invalid_ret = "Invalid return valuse";
static const char* ex_invalid_trait_id = "Invalid trait id";
static const char* ex_array_conversion = "Autoconvert mode on the o3 array must be turned on for this operation";
static const char* ex_cleanup = "Operation not supported in the clean-up phase.";
static const char* ex_invalid_value = "Invalid argument.";
static const char* ex_invalid_node = "File node invalid.";
static const char* ex_invalid_filename = "Invalid file name.";
static const char* ex_file_exists = "File already exists.";
static const char* ex_path_access = "File path is not accessible.";
static const char* ex_self_elevation = "Self elevation failed.";
static const char* ex_file_not_found = "File not found";

#define o3_set_ex(msg) \
    if (ex) \
        *ex = o3_new(cEx)(msg);
}

#endif