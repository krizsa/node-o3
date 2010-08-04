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
#ifndef O3_I_SYS_H
#define O3_I_SYS_H

namespace o3 {

class Delegate;

o3_iid(iModule, 0x18710501,
                0xFEDE,
                0x4EE5,
                0xB7, 0x35, 0x12, 0xDF, 0x45, 0x2C, 0x3C, 0xBD);

struct iModule : iUnk {
    virtual void* symbol(const char* name) = 0;
};

o3_iid(iThread, 0x3EBD25DE,
                0x7080,
                0x46DC,
                0x9A, 0x73, 0x82, 0x63, 0xB1, 0x5D, 0x3D, 0xEC);

struct iThread : iUnk {
    virtual bool running() = 0;

    virtual bool cancelled() = 0;

    virtual void cancel() = 0;

    virtual void join() = 0;
};

o3_iid(iMutex, 0xFF57D501,
               0x07C3,
               0x4989,
               0x8F, 0x49, 0xF2, 0x9C, 0xC9, 0x11, 0x2D, 0x99);

struct iMutex : iUnk {
    virtual void lock() = 0;

    virtual void unlock() = 0;
};

o3_iid(iEvent, 0x046E7AAD,
               0xC4A2,
               0x41F0,
               0x8B, 0x16, 0xF1, 0x37, 0xA5, 0xD3, 0xCF, 0x58);

struct iEvent : iUnk {
    virtual void wait(iMutex* mutex) = 0;

    virtual void signal() = 0;

    virtual void broadcast() = 0;
};

o3_iid(iListener, 0x76DD8A99,
                  0x1E04,
                  0x4B39,
                  0x94, 0x6E, 0xB6, 0x1F, 0x0F, 0x03, 0xEC, 0xFA);

struct iListener : iUnk {
    virtual void* handle() = 0;
};

o3_iid(iTimer, 0x1B9D8917,
               0xDFB8,
               0x4756,
               0xAC, 0xB9, 0xEB, 0xF1, 0xE7, 0xF7, 0x00, 0x82);

struct iTimer : iUnk {
    virtual void restart(int timeout) = 0;
};

o3_iid(iThreadPool, 0xAF4D0DC7,
                    0x3704,
                    0x4B37,
                    0x9E, 0x55, 0x20, 0x05, 0xE4, 0xD6, 0x1A, 0xBB);

struct iThreadPool : iUnk {
    virtual void deinit() = 0;

    virtual void post(const Delegate& fun, iUnk* src) = 0;
};

o3_iid(iMessageLoop, 0x5831B732,
                     0xE79C,
                     0x4079, 0x9E, 0x05, 0x5B, 0x80, 0xFA, 0xC1, 0x74, 0xD3);

struct iMessageLoop : iUnk {
    virtual siListener createListener(void* handle, unsigned oflag,
                                      const Delegate& fun) = 0;

    virtual siTimer createTimer(int timeout, const Delegate& fun) = 0;

    virtual void post(const Delegate& fun, iUnk* src) = 0;

    virtual void wait(int timeout = -1) = 0;
};

o3_iid(iSys, 0xE8E2E0B6,
             0x148D,
             0x4EA6,
             0x92, 0xDC, 0xEB, 0x1F, 0xFB, 0xAB, 0x2F, 0x23);

struct iSys : iAlloc {
    virtual void traceEnter(const char* file, int line) = 0;

    virtual void traceLeave() = 0;

    virtual void o3assert(const char* pred, const char* file, int line) = 0;

    virtual void logfv(const char* format, va_list ap) = 0;

    virtual siModule loadModule(const char* name) = 0;

    virtual siThread createThread(const Delegate& run) = 0;

    virtual siMutex createMutex() = 0;

    virtual siEvent createEvent() = 0;

    virtual siThreadPool createThreadPool(int count = 10) = 0;

    virtual siMessageLoop createMessageLoop() = 0;

	virtual bool approvalBox(const char* msg, const char* caption) = 0;
};

class Lock {
public:
    Lock(iMutex* mutex) : m_mutex(mutex)
    {
        o3_trace1 trace;

        m_mutex->lock();
    }

    ~Lock()
    {
        o3_trace1 trace;

        m_mutex->unlock();
    }

private:
    siMutex m_mutex;
};

iSys* g_sys;

inline void traceEnter(const char* file, int line)
{
    return g_sys->traceEnter(file, line);
}

inline void traceLeave()
{
    return g_sys->traceLeave();
}

inline void o3assert(const char* pred, const char* file, int line)
{
    return g_sys->o3assert(pred, file, line);
}

inline void log(const char* format, ...)
{
    va_list ap;

    va_start(ap, format);
    g_sys->logfv(format, ap);
    va_end(ap);
}

inline void* memAlloc(size_t size)
{
    o3_trace0 trace;

    return g_sys->alloc(size);
}

inline void memFree(void* ptr)
{
    o3_trace0 trace;

    return g_sys->free(ptr);
}

}

#endif // O3_I_SYS_H
