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

#include <shared/o3_tools_win32.h>
#include <lib_zlib.h>

namespace o3 {
    o3_cls(cModule);

    struct cModule : cUnk, iModule {
        HMODULE m_handle;

        cModule(void* handle)
        {
            m_handle = (HMODULE)handle;
        }

        ~cModule()
        {
            typedef void (*o3_deinit_t)();
            
            o3_deinit_t deinit = (o3_deinit_t) GetProcAddress(m_handle, "deinit");

            if (deinit)
                deinit();
            FreeLibrary(m_handle);
        }

        o3_begin_class(cUnk)
            o3_add_iface(iModule)
        o3_end_class()

        void* symbol(const char* name)
        {
            return GetProcAddress(m_handle, name);
        }
    };

    struct cThread : cUnk, iThread 
    {    
        cThread(Delegate run) 
            : m_run(run)
            , m_cancelled(false)
            , m_joined(false)
            , m_handle(0)
        {
        }

        ~cThread()
        {
            if ((m_running || m_cancelled) && !m_joined) {
                m_cancelled = true;
                if (m_handle)
                    CloseHandle(m_handle);
            }
        }

        o3_begin_class(cUnk)
            o3_add_iface(iThread)
        o3_end_class()

        bool        m_cancelled;
        bool        m_joined;
        bool        m_running;
        Delegate    m_run;
        HANDLE      m_handle;
        DWORD       m_thread_id;

        static DWORD WINAPI startRoutine(LPVOID lpParameter) {
            cThread* t = (cThread*) lpParameter;

            t->m_run(o3_cast t);
            t->m_running = false;
            return 0;
        }
        
        bool running()
        {
            return m_running;
        }

        bool cancelled()
        {
            return m_cancelled;
        }

        void run()
        {            
            m_handle = CreateThread(0, 0, startRoutine, (LPVOID) this,
                0, &m_thread_id);
            m_running = true;
        }

        void cancel()
        {
            m_cancelled = true;
        }

        void join()
        {
            if (running() || m_cancelled) {
                m_cancelled = true;
                WaitForSingleObject(m_handle, INFINITE);
                m_joined = true;
            }
        }
    };

    o3_cls(cThread);

    struct cMutex : cUnk, iMutex 
    {
        cMutex()
        {
            InitializeCriticalSection(&m_section);
        }

        ~cMutex()
        {
            DeleteCriticalSection(&m_section);
        }        

        o3_begin_class(cUnk)
            o3_add_iface(iMutex)
        o3_end_class()

        CRITICAL_SECTION m_section;

        void lock()
        {
            EnterCriticalSection(&m_section);
        }

        void unlock()
        {
            LeaveCriticalSection(&m_section);
        }
    };

    o3_cls(cMutex);

    interface iHandle : iUnk {
        virtual void* handle() = 0;
    };

    o3_iid(iHandle, 0x3d3427f3, 0x48ed, 0x4075, 0x9c, 0xa4, 0x86, 0xdc, 0x9f,
        0x51, 0xff, 0x26);

    struct cHandle : cUnk, iHandle 
    {
        enum Type {TYPE_DEFAULT, TYPE_DIRCHANGE, TYPE_SOCKET};        

        cHandle(HANDLE handle, Type type = TYPE_DEFAULT)
            : m_handle(handle)
            , m_type(type)
        {
            
        }

        virtual ~cHandle()
        {
            if (m_handle)
                switch (m_type) {
                    case TYPE_DEFAULT:
                        CloseHandle(m_handle);
                        break;
                    case TYPE_DIRCHANGE:
                        FindCloseChangeNotification(m_handle);
                        break;
                    case TYPE_SOCKET:
                        //WSACloseEvent(m_handle);
                        break;
                    default:
                        //db_assert(false);
                        //!TODO: impl
                        break;
                }
        }

        o3_begin_class(cUnk)
            o3_add_iface(iHandle)
        o3_end_class()

        HANDLE      m_handle;
        Type        m_type;

        inline HANDLE handle()
        {
            return m_handle;
        }
    };

    o3_cls(cHandle);

    struct cEvent : cUnk, iEvent, iHandle 
    {
        cEvent(HANDLE event)
            : m_event(event)
        {
            
        }
        
        virtual ~cEvent()
        {
            if (m_event)
                CloseHandle(m_event);
        }

        o3_begin_class(cUnk)
            o3_add_iface(iEvent)
            o3_add_iface(iHandle)
        o3_end_class()

        HANDLE m_event;

        virtual void wait(iMutex* mutex)
        {
            if(mutex)
                mutex->unlock();
            
            WaitForSingleObject(m_event, INFINITE);
            
            if(mutex)
                mutex->lock();
        }

        virtual void signal()
        {
            if (m_event)
                SetEvent(m_event);
        }

        virtual void broadcast()
        {
            signal();
        }

        HANDLE handle() {
            return m_event;
        }        
    };

    o3_cls(cEvent);

    static const int    O3_MESSAGE_LIMIT = 10000;
    struct cMessageLoop : cUnk, iMessageLoop 
    {
    
        o3_cls(cListener);
        struct cListener : cUnk, iListener 
        {
            cListener(iMessageLoop* loop, iHandle* handle) 
                : m_handle(handle)
                , m_loop(loop)
            {
            }

            ~cListener()
            {
                if (siMessageLoop loop = m_loop)
                    ((cMessageLoop*) loop.ptr())->removeListener(siHandle(m_handle));
            }
        
            o3_begin_class(cUnk)
                o3_add_iface(iListener)
            o3_end_class()

            siWeak      m_handle;
            siWeak      m_loop;

            virtual void* handle() 
            {
                return m_handle;    
            }
        };        

        o3_cls(cTimer);
        struct cTimer : cUnk, iTimer
        {
            o3_begin_class(cUnk)
                o3_add_iface(iTimer)
            o3_end_class()
        };        

        struct Message 
        {
            Delegate fun;
            siUnk    arg;

            Message()
            {
            }

            Message(Delegate fun, iUnk* arg) : fun(fun), arg(arg)
            {
            }
        };

        struct cEventHandler 
        {
            cEventHandler(iMessageLoop* loop, iEvent* event) 
                : m_msgloop(loop)
                , m_event(event)
                , m_reset(true)
                , m_mutex(g_sys->createMutex())                
            {
                m_handles[0] = siHandle(m_event)->handle();
                m_nhandles = 1;
            }
            
            struct EventObject {
                EventObject()
					: handle(0)
					, delegate()
					, manual(false)
                {
                }

                EventObject(iHandle* h, Delegate d, bool m)
                    : handle(h)
                    , delegate(d)
					, manual(m)
                {
                
                }

				EventObject& operator=(const EventObject& that)
				{
					if (this != &that) {
						handle = that.handle;
						delegate = that.delegate;
						manual = that.manual;
					}
					return *this;
				}

                siHandle handle;
                Delegate delegate;
				bool manual;
            };

            siWeak                      m_msgloop;
            siMutex                     m_mutex;
            siEvent                     m_event;            
            //mutex protected members:
            bool                        m_reset;                                        
            tMap<HANDLE,EventObject>    m_event_objs;
            size_t                      m_nhandles;
            HANDLE                      m_handles[64];

            siListener createListener(iHandle* ihandle, Delegate fun, bool manual=false) 
            {				
                if (!ihandle)
                    return siListener();

                {
                    Lock lock(m_mutex);
                    m_event_objs[ihandle->handle()] = EventObject(ihandle,fun,manual);
					m_reset = true;
                }
                m_event->signal();
                return o3_new(cListener(siMessageLoop(m_msgloop), ihandle));
            }

            void removeListener(iHandle* ihandle)
            {
                if (!ihandle)
                    return;

                tMap<HANDLE,EventObject>::Iter it;
                HANDLE h = ihandle->handle();
                {
                    Lock lock(m_mutex);
					it = m_event_objs.find(h);
                    if (it != m_event_objs.end()) {
                        m_event_objs.remove(it);
                        m_reset = true;
                    }
                }
                m_event->signal();                
            }

            bool wait(int timeout)
            {
                {
                    Lock lock(m_mutex);
                    if (m_reset)
                        reset();
                }    				
                DWORD index = MsgWaitForMultipleObjectsEx ( m_nhandles, 
                        m_handles, timeout, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
                switch (index) {
                    case WAIT_FAILED:
                        // !TODO: report error
                        //db_assert(false);
						break;
                    case WAIT_TIMEOUT:
                        // nothing happend (again...)
                        return false;
                    case 0:
                        // incomming msg, add/remove listener
                        return handleEventsFrom(index);                                             
                                                
                    default:
                        // new windows message to be dispatched
                        if (index == m_nhandles) {

#ifdef O3_PLUGIN
    // let's not mess with the messages in the browser...
    return false;
#endif
                             MSG msg;
                             while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
                             {
                               TranslateMessage (&msg);
                               DispatchMessage(&msg);
                             }
                             break;                            
                        }

                        // handles were signaled
                        else                        
                            handleEventsFrom(index);                                             
                        
                        return true;
                }
                return false; // Added this to suppress warning C4175
            }

            bool handleEventsFrom(DWORD index)
            {
                if (index == 0) {
                    index = WaitForMultipleObjects( m_nhandles-1, 
                        m_handles + 1, FALSE, 0) + 1;
                }

                bool ret(false);                
                siMessageLoop loop(m_msgloop);
                tMap<HANDLE,EventObject>::Iter it;
				EventObject copy;

                while (index > 0 && index < m_nhandles) {                    
                    bool removed(false);
                    {
                        Lock lock(m_mutex);
                        // copy the delegate:
                        it = m_event_objs.find(m_handles[index]);
						if (it == m_event_objs.end()) {
                            removed = true;
						}
						else
							copy = (*it).val;
                    }

                    if (!removed) {
                        // do the call
                        loop->post(copy.delegate, 0);
                        ret = true;    
						// reset if its is a manual reset event handle like socket events
						if (copy.manual)
							ResetEvent(copy.handle);
					}
                    // if no more handles break
                    if (index == m_nhandles-1)
                        break;

                    // if there are more handles after index c, check them as well
                    index = WaitForMultipleObjects( m_nhandles-index-1, 
                        m_handles + index + 1, FALSE, 0) + index + 1;                        
                }  
                return ret;
            }

            void reset()
            {
                tMap<HANDLE,EventObject>::Iter it;
                {
                    Lock lock(m_mutex);                    
                    m_nhandles = 1;
                    for (it=m_event_objs.begin(); it != m_event_objs.end(); ++it){
                        m_handles[m_nhandles] = ((*it).key);
                        m_nhandles++;    
                    }
					m_reset = false;
                }
            }
        };

        #pragma warning(disable:4355) // passing 'this' here is legal in this case 
        cMessageLoop() 
            : m_mutex(g_sys->createMutex())
            , m_event(g_sys->createEvent())
            , m_event_handler(this, m_event)
        {
            
        }
        #pragma warning(default:4355)

        o3_begin_class(cUnk)
            o3_add_iface(iMessageLoop)
        o3_end_class()

        siMutex              m_mutex;
        siEvent              m_event;   
        cEventHandler        m_event_handler;   
        tList<Message*>      m_queue;

		// flag is an os specific flag here, on windows if its first bit set, 
		// the handle needs to be reset manually
        siListener createListener(void* handle, unsigned flag, const Delegate& fun)
        {
            return m_event_handler.createListener((iHandle*) handle, fun, flag&1);
        }

        void removeListener(iHandle* handle)
        {
            m_event_handler.removeListener(handle);
        }

        siTimer createTimer(int timeout, const Delegate& fun)
        {
            timeout;
            fun;
            return siTimer();
        }

        void post(const Delegate& fun, iUnk* arg)
        {
            Lock lock(m_mutex);
			if (m_queue.size() <  O3_MESSAGE_LIMIT)
                m_queue.pushBack(o3_new(Message(fun, arg)));
            m_event->signal();
        }

        void wait(int timeout)
        {
            DWORD before(0);            
            Message* last_to_handle, *to_send;
            size_t nmessages;
            bool sent = false;
            
            while(timeout>=0 && !sent)
            {                       
                before = GetTickCount();                
                m_event_handler.wait(0);

                {
                    Lock lock(m_mutex);
                    nmessages = m_queue.size();
                }                                    
                
                if ( !nmessages && !m_event_handler.wait(timeout)) {                    
                    //no messages                    
                    timeout -= (GetTickCount() - before) / 10;
                    continue;
                }

                {
                    Lock lock(m_mutex);               
                    last_to_handle = m_queue.back();
					// update message count...
					nmessages = m_queue.size();
                }
                    
                size_t left = nmessages;
                bool last=false;
                
				while (left > 0 && !last) {
                    {
                        Lock lock(m_mutex);
                        to_send = *m_queue.begin();						
						o3_assert(m_queue.size());
						m_queue.popFront();
						left--;
                    }
                    to_send->fun(to_send->arg);
                    last = (to_send == last_to_handle);
                    o3_delete(to_send);                                        
                    sent = true;
                } 
             
                timeout -= (GetTickCount() - before) / 10;
            }
            
        }
    };

    o3_cls(cMessageLoop);

    struct cSys : cSysBase 
    {
        struct Position 
        {
            Position()
                : start(0), size(0)
            {        
            }

            Position(size_t start, size_t size)
                : start(start), size(size)
            {        
            }
            size_t start;
            size_t size;
        };

        static const size_t     rsc_magic_num = 12344321;
        tMap<Str, Position>     m_files;        // map of appended resource files
        size_t                  m_start_addr;   // start of the resource section in the file
        siStream                m_stream;       // self

        cSys()
        {
			CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
            g_sys = this;
            g_sys->addRef();
            initResource();
        }

        virtual ~cSys()
        {
            // every o3 component must be deleted before this destructor returns
            m_files.clear();
            m_stream = 0;
            m_weak = 0;
			CoUninitialize(); 
        }

        o3_begin_class(cSysBase)
        o3_end_class()

        void* alloc(size_t size)
        {
            return ::operator new(size);
        }

        void free(void* ptr)
        {
            ::operator delete(ptr);
        }

        void o3assert(const char* pred, const char* file, int line)
        {
			o3::log("Assertion %s failed in file %s on line %d\n", pred, file, line);
			abort();
        }

        siModule loadModule(const char* name)
        {
            typedef bool (*o3_init_t)(iSys*);

            HMODULE handle;
            o3_init_t o3_init;

			WStr wname = Str(name);
#ifdef O3_PLUGIN
			WStr v = Str(O3_VERSION_STRING);			
            WStr wpath = tmpPath() + L"o3_" + v + L"\\components\\" + wname + v + L".dll";			
#else
			WStr wpath = wname + L".dll";
#endif
			handle = LoadLibraryW( wpath );
            if (!handle) 
                return siModule();

            o3_init = (o3_init_t) GetProcAddress(handle, "o3_init");
            if (!o3_init || !o3_init(this)) {
                FreeLibrary((HMODULE)handle);
                return siModule();
            }
            return o3_new(cModule)(handle);
        }
    
        siThread createThread(const Delegate& run)
        {
            scThread thread = o3_new(cThread)(run);

            thread->run();
            if (!thread->running())
                return siThread();
            return siThread(thread.ptr());
        }

        siMutex createMutex()
        {
            return o3_new(cMutex)();
        }

        siEvent createEvent()
        {            
            return o3_new(cEvent)(CreateEvent(NULL, FALSE, FALSE, NULL)) ;
        }

        siMessageLoop createMessageLoop()
        {
            return o3_new(cMessageLoop)();
        }

		bool removeLogFile()
		{
			DeleteFileW(L"c:\\Users\\Gabor\\AppData\\Local\\Temp\\Low\\o3_v0_9\\o3log.txt");
			return true;
		}

		virtual void logfv(const char* format, va_list ap)
		{
			//format;ap;
			//vfprintf(stderr, format, ap);
			//static bool rem = removeLogFile();
			//FILE* file = fopen("c:\\Users\\Gabor\\AppData\\Local\\Temp\\Low\\o3_v0_9\\o3log.txt", "a");
			//vfprintf(file, format, ap);
			//fclose(file);
		}

		virtual bool approvalBox(const char* msg, const char* caption)
		{
			int ret = MessageBoxA(NULL, msg, caption, MB_OKCANCEL);
			return (ret == IDOK);
		}

         // reading self as a stream and finding/reading the resource header in it        
        void initResource()
        {           
            // TODO: openSelf is not implemented for posix yet
            if ( !(m_stream = openSelf()) )
                goto error;          

            if (!readStartAddr())
                goto error;

            if (!readHeader())
                goto error;
            
            return;

          error:
            m_files.clear();
            m_start_addr = 0;
            m_stream = 0;
        }

        virtual tVec<Str> resourcePaths() 
        {
            tVec<Str> list;
            tMap<Str, Position>::Iter it = m_files.begin();
            for (; it != m_files.end(); ++it) {
                list.push((*it).key);
            }
            return list;        
        }

        virtual Buf resource(const char* path)
        {
            Buf data;
            Position pos;
            tMap<Str, Position>::Iter it = m_files.find(path);
            if (it != m_files.end()) {
                pos = (*it).val;
                data = unzipResource(pos.start, pos.size);
            }
            return data;        
        }

        private: 

        // unzipping a resource file into a Buf
        Buf unzipResource(size_t from, size_t size)
        {
            Buf zipped, unzipped;
            m_stream->setPos(from);
            zipped.reserve(size);
            m_stream->read(zipped.ptr(), size);            
            zipped.resize(size);
            ZLib::unzip(zipped, unzipped);
            return unzipped;
        }
           

        // finding and validating the start address of the resource section if there is any
        bool readStartAddr()
        {
            size_t buf(0), magic_num(0);

            // TODO: stream::size() is not implemented for posix yet
            // read the last size_t data from the end of the file
            m_stream->setPos(m_stream->size() - sizeof(size_t));
            if (sizeof(size_t) != m_stream->read((void*)&buf, sizeof(size_t)))
                return false;

            // check if its a valid position
            if (buf != m_stream->setPos(buf))
                return false;

            // check if it points to a rsc_magic_num
            if (sizeof(size_t) != m_stream->read((void*)&magic_num, sizeof(size_t))
                || rsc_magic_num != magic_num)
                    return false;
            
            // if it does then we have just found our rsc
            m_start_addr = buf+sizeof(size_t);
            return true;
        }

        // reads in the header section of the resource and build up the file map
        bool readHeader()
        {
            size_t name_len(0), start, size;
            Str name;

            m_stream->setPos(m_start_addr);                        

            for(;;) 
            {
                // read in size of tnext path
                if ( sizeof(size_t) != m_stream->read( (void*)&name_len, sizeof(size_t)) )
                    return false;

                // if we found the trailing 0 then no more files left
                if (!name_len)
                    return true;

                // read in next path
                name.reserve(name_len);
                if (name_len != m_stream->read( name.ptr(), name_len ))
                    return false;
                name.resize(name_len);

                // read in next start pos
                if ( sizeof(size_t) != m_stream->read( (void*)&start, sizeof(size_t)) )
                    return false;            

                // read in the corresponding size
                if ( sizeof(size_t) != m_stream->read( (void*)&size, sizeof(size_t)) )
                    return false;            

                // add entry to the map
                m_files[name] = Position(start, size);    

            } 
        }
    };

    o3_cls(cSys);
}

#endif // O3_C_SYS_WIN32_H
