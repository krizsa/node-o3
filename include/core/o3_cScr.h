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
#ifndef O3_C_SCR_H
#define O3_C_SCR_H

#define O3_TRAIT_COUNT (O3_CLS_TRAIT_COUNT + O3_EXT_TRAIT_COUNT)

#define o3_fun
#define o3_get
#define o3_set
#define o3_prop
#define o3_name(T)
#define o3_ext(T)
#define o3_tgt
#define o3_enum(...)

#define O3_WITH_GLUE

#ifdef O3_WITH_GLUE
#define o3_glue_gen() \
    Trait* select(); \
    static Trait* clsTraits();\
    static Trait* extTraits();\
    static siEx clsInvoke(iScr* pthis, iCtx* ctx, \
        int index, int argc,const Var* argv, Var* rval);\
    static siEx extInvoke(iScr* pthis, iCtx* ctx, \
        int index, int argc,const Var* argv, Var* rval);
#else // !O3_WITH_GLUE
#define o3_glue_gen() \
    static Trait* extTraits(){return 0;}
#endif // O3_WITH_GLUE

namespace o3 {

o3_cls(cEx);

struct cEx : cUnk, iEx {
    o3_begin_class(cUnk)
        o3_add_iface(iEx)
    o3_end_class()

	cEx(){}

    cEx(const char* msg) : m_message(msg)
    {
    }

	static iEx* fmt(siEx *ex, const char *fmt, ...){
		if(!ex) return 0;
		cEx *x = o3_new(cEx);
		x->addRef();
		va_list ap;
        va_start(ap, fmt);
        x->m_message.appendfv(fmt, ap);
        va_end(ap);
		return *ex = (iEx*)x;
	}

    Str message()
    {
        return m_message;
    }

	Str m_message;
};

o3_cls(cScr);

struct cScr : cUnk, iScr {
    tMap<Str, int>  m_indices;
    tMap<int, Var>  m_values;
    int m_index;

    cScr() : m_index(O3_VALUE_OFFSET)
    {
        o3_trace2 trace;
    }

    o3_begin_class(cUnk)
        o3_add_iface(iScr)
    o3_end_class();

    int enumerate(iCtx* ctx, int index)
    {
        o3_trace2 trace;
        Trait* ptrait = select();
        int base = 0;

        ++index;
        for (; ptrait && index >= O3_TRAIT_COUNT;
             index -= O3_TRAIT_COUNT) { 
            ptrait = (Trait*) ptrait->ptr;
            base += O3_TRAIT_COUNT;
        }
        if (ptrait) {
            if (index >= O3_EXT_TRAIT_COUNT) {
                ptrait = ctx->mgr()->extTraits(ptrait->cls_name);
                base += O3_EXT_TRAIT_COUNT;
                index -= O3_EXT_TRAIT_COUNT;
            }
            ++ptrait;
            for (; ptrait->type != Trait::TYPE_END; ++ptrait)
                if (ptrait->offset == index)
                    return base + index;
        }
        for (tMap<int, Var>::ConstIter i = m_values.begin();
             i != m_values.end(); ++i)
            if (i->key >= index)
                return i->key;
        return NOT_FOUND;
    }

    Str name(iCtx* ctx, int index)
    {
        o3_trace2 trace;

        if (index < O3_VALUE_OFFSET) {
            Trait* ptrait = select();

            for (; index >= O3_TRAIT_COUNT; index -= O3_TRAIT_COUNT) 
                ptrait = (Trait*) ptrait->ptr;
            if (index >= O3_EXT_TRAIT_COUNT) {
                ptrait = ctx->mgr()->extTraits(ptrait->cls_name);
                index -= O3_EXT_TRAIT_COUNT;
            }
            ++ptrait;
            for (; ptrait->type != Trait::TYPE_END; ++ptrait)
                if (ptrait->index == index)
                    return ptrait->fun_name;
        } else 
            for (tMap<Str, int>::ConstIter i = m_indices.begin();
                 i != m_indices.end(); ++i)
                if (i->val == index)
                    return i->key;
        return Str();
    }

    int resolve(iCtx* ctx, const char* name, bool set)
    {
        o3_trace2 trace;
		//printf("resolve %s\n",name);
        tMap<Str, int>::ConstIter iter;
        int base;	
        iter = m_indices.find(name);
        if (iter != m_indices.end())
            return iter->val;
        base = 0;
        for (Trait* traits = select(); traits;
             traits = (Trait*) traits[0].ptr) {
            for (Trait* ptrait = traits + 1;
                 ptrait->type != Trait::TYPE_END; ++ptrait)
                if (o3::strEquals(ptrait->fun_name, name))
                    return m_indices[name] = base + ptrait->offset;
            base += O3_TRAIT_COUNT;
        }
        base = O3_EXT_TRAIT_COUNT;
        for (Trait* traits = select(); traits;
             traits = (Trait*) traits[0].ptr) {
            for (Trait* ptrait = ctx->mgr()->extTraits(traits[0].cls_name) + 1;
                 ptrait->type != Trait::TYPE_END; ++ptrait)
                if (o3::strEquals(ptrait->fun_name, name)) 
                    return m_indices[name] = base + ptrait->offset;
            base += O3_TRAIT_COUNT;
        }
        if (set) {
            m_values[m_index] = Var(ctx);
            return m_indices[name] = m_index++;
        }
        return NOT_FOUND;        
    }

    siEx invoke(iCtx* ctx, Access access, int index, int argc, const Var* argv,
                Var* rval)
    {
        o3_trace2 trace;
        tMap<int, Var>::Iter iter;
        Trait* ptrait;

        iter = m_values.find(index);
        if (iter != m_values.end()) {
            switch (access) {
            case ACCESS_CALL:
                if (siScr scr = iter->val.toScr()) {
                    int index = scr->resolve(ctx, "__self__");

                    scr->invoke(ctx, access, index, argc, argv, rval);
                    return 0;
                }
            case ACCESS_GET:
                *rval = iter->val;
                return 0;
            case ACCESS_SET:
                *rval = iter->val = argv[0];
                return 0;
            case ACCESS_DEL:
                m_values.remove(iter);
                *rval = true;
                return 0;
            }
        }
        if (index >= O3_VALUE_OFFSET)
            return 0;
        ptrait = select();
        for (; index >= O3_TRAIT_COUNT; index -= O3_TRAIT_COUNT) 
            ptrait = (Trait*) ptrait->ptr;
        if (index >= O3_EXT_TRAIT_COUNT) {
            ptrait = ctx->mgr()->extTraits(ptrait->cls_name);
            index -= O3_EXT_TRAIT_COUNT;
        }
        do
            ++ptrait;
        while (ptrait->offset != index);
        for (; ptrait->type != Trait::TYPE_END && ptrait->offset == index;
             ++ptrait) {

            switch (access) {
            case ACCESS_CALL:
                switch (ptrait->type) {
                case Trait::TYPE_GET:
                    if (siEx ex = (*ptrait->invoke)(this, ctx, ptrait->index, 0,
                                                    0, rval))
                        return ex;
                    if (siScr scr = rval->toScr()) {
                        int self = scr->resolve(ctx, "__self__");

                        if (self >= 0)
                            return scr->invoke(ctx, ACCESS_CALL, self, argc,
                                               argv, rval);
                    }
                    break;
                case Trait::TYPE_FUN:
                    return (*ptrait->invoke)(this, ctx, ptrait->index, argc,
                                             argv, rval);
                default:
                    break;
                };
                break;
            case ACCESS_GET:
                switch (ptrait->type) {
                case Trait::TYPE_GET:
                    return (*ptrait->invoke)(this, ctx, ptrait->index, argc,
                                             argv, rval);
                case Trait::TYPE_FUN:
                    *rval = createFun(this, ptrait->invoke, ptrait->index).ptr();
                    return 0;
                default:
                    break;
                }
                break;
            case ACCESS_SET:
                switch (ptrait->type) {
                case Trait::TYPE_SET:
                    return (*ptrait->invoke)(this, ctx, ptrait->index, argc,
                                             argv, rval);
                case Trait::TYPE_FUN:
                    *rval = m_values.find(index)->val = argv[0];
                    return 0;
                default:
                    break;
                }
                break;
            case ACCESS_DEL:
                *rval = false;
                return 0;
            }
        }
        return 0;
    } 

    virtual Trait* select()
    {
        return clsTraits();
    }

    static Trait* clsTraits()
    {
        static Trait TRAITS[] = {
            {   0,  Trait::TYPE_BEGIN,  "cScr", 0,  0,  0,  0   },
            {   0,  Trait::TYPE_END,    "cScr", 0,  0,  0,  0   },
        };

        return TRAITS;
    }

    siScr createFun(iScr* scr, Trait::invoke_t invoke, int index);

	void setProperty(iCtx* ctx, const Str& name, const Var& value)
	{
		Var ret;
		int i = resolve(ctx,name,true);
		invoke(ctx, ACCESS_SET, i, 1,&value,&ret);
	}

	Var property(iCtx* ctx, const Str& name)
	{
		int i = resolve(ctx,name,false);
		Var ret;
		if (i<0)
			return Var();

		invoke(ctx, ACCESS_GET, i, 0,0,&ret);
		return ret;
	}

};

o3_iid(iScrFun, 0x9ff368d, 0x29f, 0x426b, 0x9d, 
	   0xe2, 0xfa, 0x3e, 0xac, 0xc6, 0x73, 0x34);

struct iScrFun : iUnk {
};

struct cScrFun : cScr, iScrFun {
    siScr m_scr;
    Trait::invoke_t m_invoke;
    int m_index;

    o3_begin_class(cScr)
		o3_add_iface(iScrFun)
    o3_end_class()

    cScrFun(iScr* scr, Trait::invoke_t invoke, int index) : m_scr(scr),
        m_invoke(invoke), m_index(index)
    {
        o3_trace2 trace;
    }

    Trait* select()
    {
        return clsTraits();
    }

    static Trait* clsTraits()
    {
        static Trait TRAITS[] = {
            {   0,  Trait::TYPE_BEGIN,  "cScrFun",  0,          0,          0,  cScr::clsTraits()   },
            {   0,  Trait::TYPE_FUN,    "cScrFun",  "__self__", clsInvoke,  0,  0                   },
            {   0,  Trait::TYPE_END,    "cScrFun",  0,          0,          0,  0                   }
        };

        return TRAITS;
    }

    static siEx clsInvoke(iScr* pthis, iCtx* ctx, int, int argc,
                          const Var* argv, Var* rval)
    {
        cScrFun* pthis1 = (cScrFun*) pthis;

        return (*pthis1->m_invoke)(pthis1->m_scr, ctx, pthis1->m_index, argc,
                                   argv, rval);
    }
};

inline siScr cScr::createFun(iScr* scr, Trait::invoke_t invoke, int index)
{
    o3_trace2 trace;

    return o3_new(cScrFun)(scr, invoke, index);
}

}

#endif // O3_I_SCR_H
