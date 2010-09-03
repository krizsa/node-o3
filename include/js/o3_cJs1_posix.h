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
#ifndef O3_C_JS1_POSIX_H
#define O3_C_JS1_POSIX_H

#include <v8.h>

using namespace v8;

namespace o3 {

struct cJs1 : cJs1Base {
    struct cScrObj : cUnk, iScr {
        cJs1* m_pthis;
        Persistent<Object> m_object;

        cScrObj(cJs1* pthis, Handle<Object> object) : m_pthis(pthis),
            m_object(Persistent<Object>::New(object))
        {
        }

        ~cScrObj()
        {
            m_object.Dispose();
        }

        o3_begin_class(cUnk)
            o3_add_iface(iScr)
        o3_end_class()        

        virtual int enumerate(iCtx* ctx, int index)
        {
            return -1;
        }

        virtual Str name(iCtx* ctx, int index)
        {
            return Str();
        }

        virtual int resolve(iCtx* ctx, const char* name, bool set)
        {
            if (strEquals(name, "__self__"))
                return 0;
            return -1;
        }

        virtual siEx invoke(iCtx* ctx, Access access, int index, int argc,
                            const Var* argv, Var* rval)
        {
            if (index == 0) {
                Local<Object> object = Local<Object>::New(m_object);
                Local<Function> function = Local<Function>::Cast(object);
                tVec<Handle<Value> > args;

                for (int i = 0; i < argc; ++i)
                    args.push(m_pthis->toValue(argv[i]));
                function->Call(object, argc, args);
            }
            return 0;
        }
    };

    static void* cast(Local<Value> value)
    {
        return Local<External>::Cast(value)->Value();
    }

    static Handle<Value> invocation(const Arguments& args)
    {
        cJs1* pthis = (cJs1*) cast(args.Data());
        iScr* scr = (iScr*) cast(args.This()->GetInternalField(0));
        int self = scr->resolve(pthis, "__self__");
        int argc = args.Length();
        tVec<Var> argv(argc);
        Var rval((iAlloc*) pthis);

        if (self < 0)
            return Handle<Value>();
        for (int i = 0; i < argc; ++i)
            argv.push(pthis->toVar(args[i]));
        if (siEx ex = scr->invoke(pthis, ACCESS_CALL, self, argc, argv, &rval))
            return ThrowException(String::New(ex->message()));
        return pthis->toValue(rval);
    }

    static Handle<Value> namedGetter(Local<String> property,
                                     const AccessorInfo& info)
    {
        cJs1* pthis = (cJs1*) cast(info.Data());
        iScr* scr = (iScr*) cast(info.Holder()->GetInternalField(0));
        Str name = *String::Utf8Value(property);
        int index = scr->resolve(pthis, name);
        Var rval((iAlloc*) pthis);

        if (index < 0)
            return Handle<Value>();
        if (siEx ex = scr->invoke(pthis, ACCESS_GET, index, 0, 0, &rval))
            return ThrowException(String::New(ex->message()));
        return pthis->toValue(rval);
    }

    static Handle<Value> namedSetter(Local<String> property,
                                     Local<Value> value,
                                     const AccessorInfo& info)
    {
        cJs1* pthis = (cJs1*) cast(info.Data());
        iScr* scr = (iScr*) cast(info.Holder()->GetInternalField(0));
        Str name = *String::Utf8Value(property);
        int index = scr->resolve(pthis, name, true);
        Var arg(pthis->toVar(value));
        Var rval((iAlloc*) pthis);

        if (index < 0)
            return Handle<Value>();
        if (siEx ex = scr->invoke(pthis, ACCESS_SET, index, 1, &arg, &rval))
            return ThrowException(String::New(ex->message()));
        return pthis->toValue(rval);
    }

    static Handle<Integer> namedQuery(Local<String> property,
                                      const AccessorInfo& info)
    {
        cJs1* pthis = (cJs1*) cast(info.Data());
        iScr* scr = (iScr*) cast(info.Holder()->GetInternalField(0));
        Str name = *String::Utf8Value(property);
        int index = scr->resolve(pthis, name);

        //return index >= 0 ? True() : False();
		return index >= 0 ? v8::Integer::New(v8::DontEnum)
			: v8::Handle<v8::Integer>();		
	}

    static Handle<Boolean> namedDeleter(Local<String> property,
                                        const AccessorInfo& info)
    {
        cJs1* pthis = (cJs1*) cast(info.Data());
        iScr* scr = (iScr*) cast(info.Holder()->GetInternalField(0));
        Str name = *String::Utf8Value(property);
        int index = scr->resolve(pthis, name);
        Var rval((iAlloc*) pthis);

        if (index < 0)
            return False();
        if (siEx ex = scr->invoke(pthis, ACCESS_DEL, index, 0, 0, &rval)) {
            ThrowException(String::New(ex->message()));
	    return Handle<Boolean>();
	}
        return Boolean::New(rval.toBool());
    }

    static Handle<Array> namedEnumerator(const AccessorInfo& info)
    {
        cJs1* pthis = (cJs1*) cast(info.Data());
        iScr* scr = (iScr*) cast(info.Holder()->GetInternalField(0));
        Handle<Array> names;
        size_t length;
        int key;

        length = 0;
        for (int index = scr->enumerate(pthis, -1); index >= 0;
             index = scr->enumerate(pthis, index))
            ++length;
        names = Array::New(length);
        key = 0;
        for (int index = scr->enumerate(pthis, -1); index >= 0;
             index = scr->enumerate(pthis, index))
            names->Set(Number::New(key++),
                       String::New(scr->name(pthis, index)));
        return names;
    }

    static Handle<Value> indexedGetter(uint32_t index,
                                       const AccessorInfo& info)
    {
        cJs1* pthis = (cJs1*) cast(info.Data());
        iScr* scr = (iScr*) cast(info.Holder()->GetInternalField(0));
        int getter = scr->resolve(pthis, "__getter__");
        Var arg((int) index, pthis);
        Var rval((iAlloc*) pthis);

        if (getter < 0)
            return Handle<Value>();
        if (siEx ex = scr->invoke(pthis, ACCESS_CALL, getter, 1, &arg, &rval))
            return ThrowException(String::New(ex->message()));
        return pthis->toValue(rval);
    }

    static Handle<Value> indexedSetter(uint32_t index, Local<Value> value,
                                       const AccessorInfo& info)
    {
        cJs1* pthis = (cJs1*) cast(info.Data());
        iScr* scr = (iScr*) cast(info.Holder()->GetInternalField(0));
        int setter = scr->resolve(pthis, "__setter__");
        tVec<Var> argv(2);
        Var rval((iAlloc*) pthis);

        if (setter < 0)
            return Handle<Value>();
        argv.push(Var((int) index, pthis));
        argv.push(pthis->toVar(value));
        if (siEx ex = scr->invoke(pthis, ACCESS_CALL, setter, 2, argv, &rval))
            return ThrowException(String::New(ex->message()));
        return pthis->toValue(rval);
    }

    static Handle<Integer> indexedQuery(uint32_t index,
                                        const AccessorInfo& info)
    {
        cJs1* pthis = (cJs1*) cast(info.Data());
        iScr* scr = (iScr*) cast(info.Holder()->GetInternalField(0));
        int query = scr->resolve(pthis, "__query__");
        Var arg((int) index, (iAlloc*) pthis);
        Var rval((iAlloc*) pthis);

        if (query < 0)
            return Local<Integer>();
        if (siEx ex = scr->invoke(pthis, ACCESS_CALL, query, 1, &arg, &rval)) {
            ThrowException(String::New(ex->message()));
            return Handle<Integer>();
        }
        //return Boolean::New(rval.toBool());
		return rval.toBool() ? v8::Integer::New(v8::None)
			: v8::Handle<v8::Integer>();
	}

    static Handle<Boolean> indexedDeleter(uint32_t index,
                                          const AccessorInfo& info)
    {
        cJs1* pthis = (cJs1*) cast(info.Data());
        iScr* scr = (iScr*) cast(info.Holder()->GetInternalField(0));
        int deleter = scr->resolve(pthis, "__deleter__");
        Var arg((int) index, (iAlloc*) pthis);
        Var rval((iAlloc*) pthis);

        if (deleter < 0)
            return Local<Boolean>();
        if (siEx ex = scr->invoke(pthis, ACCESS_CALL, deleter, 1, &arg,
                                  &rval)) {
            ThrowException(String::New(ex->message()));
            return Handle<Boolean>();
        }
        return Boolean::New(rval.toBool());
    }

    static Handle<Array> indexedEnumerator(const AccessorInfo& info)
    {
        cJs1* pthis = (cJs1*) cast(info.Data());
        iScr* scr = (iScr*) cast(info.Holder()->GetInternalField(0));
        int enumerator = scr->resolve(pthis, "__enumerator__");
        Var arg((iAlloc*) pthis);
        Var rval((iAlloc*) pthis);
        Handle<Array> names;
        size_t length;
        int key;

        length = 0;
        arg = -1;
        if (siEx ex = scr->invoke(pthis, ACCESS_CALL, enumerator, 1, &arg,
                                  &rval)) {
            ThrowException(String::New(ex->message()));
            return Handle<Array>();
        }
        for (int index = rval.toInt32(); index >= 0; index = rval.toInt32()) {
            ++length;
            arg = index;
            scr->invoke(pthis, ACCESS_CALL, enumerator, 1, &arg, &rval);
        }
        names = Array::New(length);
        key = 0;
        arg = -1;
        if (siEx ex = scr->invoke(pthis, ACCESS_CALL, enumerator, 1, &arg,
                                  &rval)) {
            ThrowException(String::New(ex->message()));
            return Handle<Array>();
        }
        for (int index = rval.toInt32(); index >= 0; index = rval.toInt32()) {
            names->Set(Number::New(key++),
                       String::New(scr->name(pthis, index)));
            arg = index;
            scr->invoke(pthis, ACCESS_CALL, enumerator, 1, &arg, &rval);
        }
        return names;
    }

    static void finalize(Persistent<Value> value, void* parameter)
    {
		if (!value.IsNearDeath())
			return;

        cJs1* pthis = (cJs1*) parameter;
        Handle<Object> object = value->ToObject();
        iScr* scr = (iScr*) cast(object->GetInternalField(0));
        if (scr) {
            scr->release();
			pthis->m_wrappers[scr].Clear();
			pthis->m_wrappers.remove(scr);
            //pthis->m_objects.remove(*object);        
		}
    }

	static void cleanup(Persistent<Value> value, void *parameter)
	{
		cJs1* pthis = (cJs1*) parameter;
		pthis->release();
	}

	void weakReferenceCallback(Persistent<Value> value, void *parameter)
	{
		cJs1* pthis = (cJs1*) parameter;
		Handle<Object> object = value->ToObject();
		iScr* scr = (iScr*) cast(object->GetInternalField(0));
		if (scr) {
			pthis->m_wrappers[scr].Clear();
			pthis->m_wrappers.remove(scr);
		}
	}

    Persistent<ObjectTemplate> m_template;
    tMap<Object*, Object*> m_objects;
    tMap<iScr*, Handle<Object> > m_wrappers;
	siMgr m_mgr;
    siMessageLoop m_loop;
    tMap<Str, Var> m_values;

    Handle<Object> createObject(iScr* scr)
    { 
        Persistent<Object> object;

		tMap<iScr*, Handle<Object> >::Iter it = 
			m_wrappers.find(scr);

		if (it != m_wrappers.end() && !it->val.IsEmpty()) {			
			return it->val;
		}	
        scr->addRef();
        object = Persistent<Object>::New(m_template->NewInstance());
        object.MakeWeak(this, finalize);
        object->SetInternalField(0, External::New(scr));
        //m_objects[*object] = *object;
        m_wrappers[scr] = object;
		return object;
    }

    Var toVar(Handle<Value> value)
    {
        if (value->IsUndefined())
            return Var((iAlloc*) this);
        else if (value->IsNull())
            return Var((iScr*) 0, this);
        else if (value->IsBoolean())
            return Var(value->ToBoolean()->Value(), this);
        else if (value->IsInt32())
            return Var(value->ToInt32()->Value(), this);
        else if (value->IsNumber())
            return Var(value->ToNumber()->Value(), this);
        else if (value->IsObject()) {
            Handle<Object> object = value->ToObject();

            if (object->InternalFieldCount() > 0) 
                return (iScr*) cast(object->GetInternalField(0));
            else
                return o3_new(cScrObj)(this, object);
        } else if (value->IsString())
            return Var(*String::Utf8Value(value->ToString()), this);
        return Var((iAlloc*) this);
    }
 
    Handle<Value> toValue(const Var& val)
    {
        switch (val.type()) {
        case Var::TYPE_VOID:
            return Undefined();
        case Var::TYPE_NULL:
            return Null();
        case Var::TYPE_BOOL:
            return Boolean::New(val.toBool());
        case Var::TYPE_INT32:
            return Int32::New(val.toInt32());
        case Var::TYPE_INT64:
        case Var::TYPE_DOUBLE:
            return Number::New(val.toDouble());
        case Var::TYPE_SCR:
            return createObject(val.toScr());
        case Var::TYPE_STR:
            return String::New(val.toStr());
        case Var::TYPE_WSTR:
            return String::New((::uint16_t*) val.toWStr().ptr());
        default:
            return Undefined();
        }
    }

public:
    cJs1(Handle<Object> target, iMgr* mgr, int argc, char** argv, char** envp)
    {
        o3_trace2 trace;
        HandleScope handle_scope;
        Local<ObjectTemplate> handle;
        Local<External> data;       
		Handle<Object> object;
		Persistent<Object> object2;

        handle = ObjectTemplate::New();
        m_template = Persistent<ObjectTemplate>::New(handle);
        m_mgr = mgr;
        m_loop = g_sys->createMessageLoop();
        m_template->SetInternalFieldCount(1);
        data = External::New(this);
        m_template->SetCallAsFunctionHandler(invocation, data);
        m_template->SetNamedPropertyHandler(namedGetter, namedSetter,
                                            namedQuery, namedDeleter,
                                            namedEnumerator, data);
        m_template->SetIndexedPropertyHandler(indexedGetter, indexedSetter,
                                              indexedQuery, indexedDeleter,
                                              indexedEnumerator, data);

		object = createObject(o3_new(cO3)(this, argc, argv, envp));		
		object2 = Persistent<Object>::New(object);
		object2.MakeWeak(this, cleanup);
		target->Set(String::New("root"), object2);
	}


    ~cJs1()
    {
        o3_trace2 trace;
        HandleScope handle_scope;

        for (tMap<Object*, Object*>::Iter i = m_objects.begin();
             i != m_objects.end(); ++i) {
            Local<Object> object = i->val;
            iScr* scr = (iScr*) cast(object->GetInternalField(0));

            scr->release();
            object->SetInternalField(0, Null());
        }

        m_template.Dispose();
    }

    o3_begin_class(cJs1Base)
    o3_end_class()

    o3_glue_gen()

    static o3_ext("cO3") o3_get siScr js(iCtx* ctx)
    {
        o3_trace3 trace;
        //Var js = ctx->value("js");

        //if (js.type() == Var::TYPE_VOID)
        //    js = ctx->setValue("js", (iScr*) o3_new(cJs1)(ctx->mgr(), 0, 0, 0));
        //return js.toScr();
		return siScr();
    }

    void* alloc(size_t size)
    {
        o3_trace2 trace;

        return memAlloc(size);
    }

    void free(void* ptr)
    {
        o3_trace2 trace;

        memFree(ptr);
    }

    siMgr mgr()
    {
        o3_trace2 trace;

        return m_mgr;
    }

    siMessageLoop loop()
    {
        o3_trace2 trace;

        return m_loop;
    }

    Var value(const char* key)
    {
        o3_trace2 trace;

        return m_values[key];
    }

    Var setValue(const char* key, const Var& val)
    {
        o3_trace2 trace;

        return m_values[key] = val;
    }

    o3_fun Var eval(const char* str, siEx* ex)
    {
        o3_trace3 trace;
        HandleScope handle_scope;
        TryCatch try_catch;
        v8::Handle<Script> script;
        v8::Handle<Value> result;

        if (*str == '#')
            while (*str)
                if (*str++ == '\n')
                    break;
        script = Script::New(String::New(str));
        if (script.IsEmpty())
            goto error;
        result = script->Run();
        if (result.IsEmpty())
            goto error;
        return toVar(result);
    error:
        Str msg = *String::Utf8Value(try_catch.Exception());
        Str line = Str::fromInt32(try_catch.Message()->GetLineNumber());

        if (ex)
            *ex = o3_new(cEx)(msg + " on line " + line);
        return Var((iAlloc*) this);
    }

	virtual void setAppWindow(void*)
	{
		
	}

	virtual void* appWindow()
	{

	}
};

}

#endif // O3_C_JS1_POSIX_H
