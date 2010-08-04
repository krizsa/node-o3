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
#ifndef O3_C_SYS_WIN32_H
#define O3_C_SYS_WIN32_H

#include <stdlib.h>
#include <dlfcn.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>

#include <v8.h>
namespace o3 {

inline bool operator<(const timeval& a, const timeval& b)
{
    return a.tv_sec < b.tv_sec
        || (a.tv_sec == b.tv_sec
            && a.tv_usec < b.tv_usec);
}

o3_cls(cModule);

struct cModule : cUnk, iModule {
    void* m_handle;

    cModule(void* handle)
    {
        o3_trace2 trace;
        m_handle = handle;
    }

    ~cModule()
    {
        typedef void (*o3_deinit_t)();

        o3_trace2 trace;
        o3_deinit_t deinit = (o3_deinit_t) dlsym(m_handle, "deinit");

        if (deinit)
            deinit();
        dlclose(m_handle);
    }

    o3_begin_class(cUnk)
        o3_add_iface(iModule)
    o3_end_class()

    void* symbol(const char* name)
    {
        o3_trace2 trace;

        return dlsym(m_handle, name);
    }
};

o3_cls(cThread);

struct cThread : cUnk, iThread {
    static void* start_routine(void* arg)
    {
        o3_trace2 trace;
        cThread* pthis = (cThread*) arg;

        pthis->m_run(pthis);
        pthis->m_running = false;
        return 0;
    }

    Delegate m_run;
    bool m_running;
    bool m_cancelled;
    bool  m_joined;
    pthread_t m_thread;

    cThread(const Delegate& run) : m_run(run), m_running(false),
        m_cancelled(false), m_joined(true)
    {
        o3_trace2 trace;
    }

    ~cThread()
    {
        o3_trace2 trace;

        if (!m_joined) {
            if (!m_cancelled) {
                if (!m_running)
                    return;
                m_cancelled = true;
            }
            pthread_detach(m_thread);
        }
    }
       
    o3_begin_class(cUnk)
        o3_add_iface(iThread)
    o3_end_class()

    bool running()
    {
        o3_trace2 trace;

        return m_running;
    }

    bool cancelled()
    {
        o3_trace2 trace;

        return m_cancelled;
    }

    void run()
    {
        o3_trace2 trace;

        if (pthread_create(&m_thread, 0, start_routine, this) == 0)
            m_running = true;
    }

    void cancel()
    {
        o3_trace2 trace;

        m_cancelled = true;
    }

    void join()
    {
        o3_trace2 trace;

        m_cancelled = true;
        pthread_join(m_thread, 0);
        m_joined = true;
    }
};

o3_cls(cMutex);

struct cMutex : cUnk, iMutex {
    pthread_mutex_t m_mutex;

    cMutex(const pthread_mutex_t& mutex) : m_mutex(mutex)
    {
        o3_trace2 trace;
    }

    virtual ~cMutex()
    {
        o3_trace2 trace;

        pthread_mutex_destroy(&m_mutex);
    }

    o3_begin_class(cUnk)
        o3_add_iface(iMutex)
    o3_end_class()

    void lock()
    {
        o3_trace2 trace;

        pthread_mutex_lock(&m_mutex);
    }

    void unlock()
    {
        o3_trace2 trace;

        pthread_mutex_unlock(&m_mutex);
    }
};

o3_cls(cEvent);

struct cEvent : cUnk, iEvent {
    pthread_cond_t m_cond;

    cEvent(const pthread_cond_t& cond) : m_cond(cond)
    {
        o3_trace2 trace;
    }

    ~cEvent()
    {
        o3_trace2 trace;

        pthread_cond_destroy(&m_cond);
    }

    o3_begin_class(cUnk)
        o3_add_iface(iEvent)
    o3_end_class()

    void wait(iMutex* mutex)
    {
        o3_trace2 trace;

        pthread_cond_wait(&m_cond, &((cMutex*) mutex)->m_mutex);
    }

    void signal()
    {
        o3_trace2 trace;

        pthread_cond_signal(&m_cond);
    }

    void broadcast()
    {
        o3_trace2 trace;

        pthread_cond_broadcast(&m_cond);
    }
};

o3_cls(cMessageLoop);

struct cMessageLoop : cUnk, iMessageLoop {
    o3_cls(cListener);

    struct cListener : cUnk, iListener {
        siWeak m_weak;
        int m_fd;
        unsigned m_oflag;
        Delegate m_fun;

        cListener(cMessageLoop* loop, const Delegate& fun, int fd,
                  unsigned oflag) : m_weak(loop), m_fun(fun)
        {
            m_fd = fd;
            m_oflag = oflag;
            loop->addListener(this);
        }

        ~cListener()
        {
            if (siMessageLoop loop = m_weak)
                ((cMessageLoop*) loop.ptr())->removeListener(this);
        }

        o3_begin_class(cUnk)
            o3_add_iface(iListener)
        o3_end_class()

        void* handle()
        {
            return &m_fd;
        }

        void signal()
        {
            m_fun(this);
        }
    };

    struct cTimer : cUnk, iTimer {
        siWeak m_weak;
        Delegate m_fun;
        struct timeval m_tv;
        tList<cTimer*>::Iter m_iter;

        cTimer(cMessageLoop* loop, const Delegate& fun, int timeout)
            : m_weak(loop), m_fun(fun)
        {
            start(timeout);
        }

        ~cTimer()
        {
            stop();
        }

        o3_begin_class(cUnk)
            o3_add_iface(iTimer)
        o3_end_class()

        void restart(int timeout)
        {
            stop();
            start(timeout);
        }

        void start(int timeout)
        {
            if (siMessageLoop loop = m_weak) {
                time_t tv_msec;

                gettimeofday(&m_tv, 0);
                tv_msec = m_tv.tv_usec / 1000 + timeout * O3_TICK_SIZE;
                m_tv.tv_sec = m_tv.tv_sec + tv_msec / 1000;
                m_tv.tv_usec = tv_msec % 1000 * 1000;
                m_iter = ((cMessageLoop*) loop.ptr())->addTimer(this);
            }
        }

        void stop()
        {
            if (!m_iter.valid())
                return;
            if (siMessageLoop loop = m_weak)
                ((cMessageLoop*) loop.ptr())->removeTimer(this);
            m_iter = 0;
        }

        void signal()
        {
            stop();
            m_fun(this);
        }
    };

    struct Message {
        unsigned seq;
        Delegate fun;
        siUnk src;

        Message()
        {
        }

        Message(int seq, const Delegate& fun, iUnk* src) : seq(seq), fun(fun),
                                                           src(src)
        {
        }
    };

    siMutex m_mutex;
    int m_nfds;
    fd_set m_readfds;
    fd_set m_writefds;
    fd_set m_errorfds;
    tMap<int, cListener*> m_listeners;
    tList<cTimer*> m_timers;
    volatile unsigned m_seq;
    int m_in;
    int m_out;
    Message m_front;

    cMessageLoop(int in, int out) : m_mutex(g_sys->createMutex())
    {
        m_nfds = in + 1;
        FD_ZERO(&m_readfds);
        FD_ZERO(&m_writefds);
        FD_ZERO(&m_errorfds);
        FD_SET(in, &m_readfds);
        m_seq = 0;
        m_in = in;
        m_out = out;
    }

    ~cMessageLoop()
    {
        close(m_out);
        close(m_in);
    }

    o3_begin_class(cUnk)
        o3_add_iface(iMessageLoop)
    o3_end_class();

    siListener createListener(void* handle, unsigned oflag, const Delegate& fun)
    {
        return o3_new(cListener)(this, fun, *(int*) handle, oflag);
    }

    siTimer createTimer(int timeout, const Delegate& fun)
    {
        return o3_new(cTimer)(this, fun, timeout);
    }

    void post(const Delegate& fun, iUnk* src)
    {
        uint8_t msg[sizeof(Message)];

        new (msg) Message(m_seq, fun, src);
        write(m_out, msg, sizeof(Message));
    }

    void wait(int timeout)
    {
        do {
            bool empty;
            int  nfds;
            fd_set readfds;
            fd_set writefds;
            fd_set errorfds;
            struct timeval tv;
            struct timeval pt;
            struct timeval ct;
            uint8_t msg[sizeof(Message)];

            {
                Lock lock(m_mutex);

                empty = m_timers.empty();
                nfds = m_nfds;
                readfds = m_readfds;
                writefds = m_writefds;
                errorfds = m_errorfds;
            }
            m_front.fun(m_front.src);
            m_front.fun = Delegate();
            if (empty) {
                tv.tv_sec = timeout * O3_TICK_SIZE / 1000;
                tv.tv_usec = timeout * O3_TICK_SIZE % 1000 * 1000;
            } else {
                tv.tv_sec = O3_TICK_SIZE / 1000;
                tv.tv_usec = O3_TICK_SIZE % 1000 * 1000;
                gettimeofday(&pt, 0);
            }
            int sum = select(nfds, &readfds, &writefds, &errorfds,
                          !empty || timeout >= 0 ? &tv : 0);
            atomicInc((volatile int&) m_seq);
            while (nfds-- > 0)
                if (FD_ISSET(nfds, &readfds) || FD_ISSET(nfds, &writefds)) {
					cListener* listener;

                    {
                        Lock lock(m_mutex);

                        listener = m_listeners[nfds];
                    }
                    listener->signal();
                }
            while (read(m_in, msg, sizeof(Message)) == sizeof(Message)) {
                Message* front = (Message*) msg;

                if (front->seq == m_seq) {
                    m_front = *front;
                    break;
                }
                front->fun(front->src);
            }
            if (timeout < 0 && nfds == 0)
                continue;
            if (empty) 
                timeout = 0;
            else {
                gettimeofday(&ct, 0);
                do {
                    cTimer* timer; {
                        Lock lock(m_mutex);

                        timer = m_timers.front();
                    }
                    if (ct < timer->m_tv)
                        break;
                    timer->signal();
                    {
                        Lock lock(m_mutex);

                        empty = m_timers.empty();
                    }
                } while (!empty);
                if (timeout > 0)
                    timeout -= ((ct.tv_sec - pt.tv_sec) * 1000
                             + (ct.tv_usec - pt.tv_usec) / 1000) / O3_TICK_SIZE;
            }
        } while (timeout > 0);
    }

    tList<cTimer*>::Iter addTimer(cTimer* timer)
    {
        Lock lock(m_mutex);
        tList<cTimer*>::Iter i;

        for (i = m_timers.begin(); i != m_timers.end(); ++i)
            if (timer->m_tv < (*i)->m_tv)
                break;
        return m_timers.insert(i, timer);
    }

    void removeTimer(cTimer* timer)
    {
        Lock lock(m_mutex);

        m_timers.remove(timer->m_iter);
    }

    void addListener(cListener* listener)
    {
        Lock lock(m_mutex);
        int fd = listener->m_fd;
        unsigned oflag = listener->m_oflag;

        m_nfds = max(m_nfds, fd + 1);
        if (oflag == O_RDONLY || oflag == O_RDWR)
            FD_SET(fd, &m_readfds);
        if (oflag == O_WRONLY || oflag == O_RDWR)
            FD_SET(fd, &m_writefds);
        FD_SET(fd, &m_errorfds);
        m_listeners[fd] = listener;
    }

    void removeListener(cListener* listener)
    {
        Lock lock(m_mutex);
        int fd = listener->m_fd;

        FD_CLR(fd, &m_readfds);
        FD_CLR(fd, &m_writefds);
        FD_CLR(fd, &m_errorfds);
        if (m_nfds == fd + 1) {
            while (fd-- > 0)
                if (FD_ISSET(fd, &m_errorfds))
                    break;
            m_nfds = fd + 1;
        }
        m_listeners.remove(fd);
    }
};

o3_cls(cSys);

struct cSys : cSysBase {
    o3_begin_class(cSysBase)
    o3_end_class()

	cSys():cSysBase()
		,m_v8inited(false)
		,m_overall(0)
	{

	}

	bool m_v8inited;
	int64_t m_overall;	

	void v8inited(bool init=true){
		m_v8inited = init;
	}

    void* alloc(size_t size)
    {
		size += sizeof (size_t);
		void *ptr = ::malloc(size);
		*(size_t *) ptr = size;
		m_overall+=size;
		if (m_v8inited)
			v8::V8::AdjustAmountOfExternalAllocatedMemory(size);
		
		//printf("alloc %p %d \n",((size_t *) ptr) + 1,size);
		//printf("overall: %lld", m_overall);
		return ((size_t *) ptr) + 1;		
        //return ::malloc(size);		
    }

    void free(void* ptr)
    {
		//printf("free %p\n",ptr);
		if (m_v8inited)
			v8::V8::AdjustAmountOfExternalAllocatedMemory(- * (((size_t *) ptr) - 1));
		
		m_overall-= *(((size_t *) ptr)-1);
		ptr = (void *) (((size_t *) ptr) - 1);
		::free(ptr);			
        //::free(ptr);		
    }

    void o3assert(const char* pred, const char* file, int line)
    {
        o3::log("Assertion %s failed in file %s on line %d\n", pred, file, line);
        abort();
    }

    void logfv(const char* format, va_list ap)
    {
        vfprintf(stderr, format, ap);
    }
    
    siModule loadModule(const char* name)
    {
        typedef bool (*o3_init_t)(iSys*);

        void* handle;
        o3_init_t o3_init;

#ifdef O3_APPLE
        handle = dlopen(Str("lib") + name + ".dylib", RTLD_NOW);
#endif // O3_APPLE
#ifdef O3_LINUX
        handle = dlopen(Str("lib") + name + ".so", RTLD_NOW);
#endif // O3_LINUX
        if (!handle) 
            return siModule();
        o3_init = (o3_init_t) dlsym(handle, "o3_init");
        if (!o3_init || !o3_init(this)) {
            dlclose(handle);
            return siModule();
        }
        return o3_new(cModule)(handle);
    }
    
    siThread createThread(const Delegate& run)
    {
        o3_trace2 trace;
        scThread thread = o3_new(cThread)(run);

        thread->run();
        if (!thread->running())
            return siThread();
        return thread;
    }

    siMutex createMutex()
    {
        o3_trace2 trace;
        pthread_mutex_t mutex;

        if (pthread_mutex_init(&mutex, 0) < 0)
            return siMutex();
        return o3_new(cMutex)(mutex);
    }

    siEvent createEvent()
    {
        o3_trace2 trace;
        pthread_cond_t cond;

        if (pthread_cond_init(&cond, 0) < 0)
            return siEvent();
        return o3_new(cEvent)(cond);
    }

    siMessageLoop createMessageLoop()
    {
        o3_trace2 trace;
        int fd[2];

        if (pipe(fd) < 0)
            return siMessageLoop();
        fcntl(fd[0], F_SETFL, O_NONBLOCK);
        return o3_new(cMessageLoop)(fd[0], fd[1]);
    }
	
	bool approvalBox(const char* msg, const char* caption)
	{
		o3_assert(false);
		return true;
	}
};

}

#endif // O3_C_SYS_WIN32_H
