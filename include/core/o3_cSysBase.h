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
#ifndef O3_C_SYS_BASE_H
#define O3_C_SYS_BASE_H

#ifdef O3_WIN32
#define o3_mod __declspec(dllexport)
#else
#define o3_mod
#endif // O3_WIN32

namespace o3 {

class Delegate {
    siWeak m_ctx;
    siWeak m_unk;
    void (cUnk::*m_pmf)(iUnk*);
    void (*m_pf)(iUnk*);

public:
    Delegate() : m_pf(0)
    {
        o3_trace1 trace;
    }

    Delegate(iCtx* ctx, iScr* scr) : m_ctx(ctx), m_unk(scr), m_pf(0)
    {
        o3_trace1 trace;
    }

    template<typename T>
    Delegate(T* unk, void (T::*pmf)(iUnk*))
        : m_unk(o3_cast unk), m_pmf((void (cUnk::*)(iUnk*)) pmf), m_pf(0)
    {
        o3_trace1 trace;
    }

    Delegate(void (*pf)(iUnk*)) : m_pf(pf)
    {
        o3_trace1 trace;
    }

    bool valid() const
    {
        o3_trace1 trace;

        return siCtx(m_ctx) || siUnk(m_unk) || m_pf;
    }

    operator bool() const
    {
        o3_trace1 trace;

        return valid();
    }

    template<typename T>
    void call(T* src)
    {
        o3_trace1 trace;

        if (siCtx ctx = m_ctx) {
            if (siScr scr = m_unk) {
                int self = scr->resolve(ctx, "__self__");
                Var arg(siScr(src), ctx);
                Var rval((iAlloc *) ctx);

                scr->invoke(ctx, iScr::ACCESS_CALL, self, 1, &arg, &rval);
            }
        } else if (siUnk unk = m_unk)
            (((cUnk*) unk.ptr())->*m_pmf)(o3_cast src);
        else if (m_pf)
            (*m_pf)(o3_cast src);
    }

    template<typename T>
    void call(const tSi<T>& src)
    {
        return call(src.ptr());
    }

    template<typename T>
    void operator()(T* src)
    {
        o3_trace1 trace;

        call(src);
    }

    template<typename T>
    void operator()(const tSi<T>& src)
    {
        return operator()(src.ptr());
    }
};

o3_cls(cThreadPool);

struct cThreadPool : cUnk, iThreadPool {
    struct Command {
        Delegate fun;
        siUnk src;

        Command()
        {
        }

        Command(Delegate fun, iUnk* src) : fun(fun), src(src)
        {
        }
    };

    tVec<siThread> m_threads;
    siMutex m_mutex;
    siEvent m_event;
    tList<Command> m_commands;

    cThreadPool()
    {
        o3_trace2 trace;
        m_mutex = g_sys->createMutex();
        m_event = g_sys->createEvent();
    }

    o3_begin_class(cUnk)
        o3_add_iface(iThreadPool)
    o3_end_class()

    void deinit()
    {
        o3_trace2 trace;

        for (size_t i = 0; i < m_threads.size(); ++i)
            m_threads[i]->cancel();
        m_event->broadcast();
        for (size_t i = 0; i < m_threads.size(); ++i) {
            while (m_threads[i]->running())

#ifdef O3_WIN32
                m_event->signal();
#else
                ;
#endif
            m_threads[i] = 0;
        }
    }

    void post(const Delegate& fun, iUnk* src)
    {
        o3_trace2 trace;
        Lock lock(m_mutex);

        m_commands.pushBack(Command(fun, src));
        if (m_commands.size() == 1)
            m_event->signal();
    }

    void run(iUnk* src)
    {
        siThread thread = src;

        while (!thread->cancelled()) {
            Command front;

            {
                Lock lock(m_mutex);

                while (m_commands.empty()) {
                    m_event->wait(m_mutex);
                    if (thread->cancelled())
                        return;
                }
                front = m_commands.front();
                m_commands.popFront();
            }
            front.fun(front.src);
        }
    }
};

o3_cls(cSysBase);

struct cSysBase : cUnk, iSys {
    cSysBase()
    {
        g_sys = this;
        g_sys->addRef();
    }

    ~cSysBase()
    {
        g_sys = 0;
    }

    o3_begin_class(cUnk)
        o3_add_iface(iAlloc)
        o3_add_iface(iSys)
    o3_end_class()

    void traceEnter(const char*, int)
    {
    }

    void traceLeave()
    {
    }

    siThreadPool createThreadPool(int count)
    {
        scThreadPool pool = o3_new(cThreadPool)();

        while (count-- > 0) {
            siThread thread = createThread(Delegate(pool.ptr(), &cThreadPool::run));

            if (!thread)
                return siThreadPool();
            pool->m_threads.push(thread);
        }
        return pool;
    }
};

}

#endif // O3_C_SYS_BASE_H
