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
#ifndef O3_C_PROCESS1_WIN32_H
#define O3_C_PROCESS1_WIN32_H

#include <shared/o3_tools_win32.h>

namespace o3{

    struct cProcess1 : cProcess1Base {

        cProcess1(iCtx* ctx)
            : m_stdin_r(0)
            , m_stdin_w(0)
            , m_stdout_r(0)
            , m_stdout_w(0)
			, m_stderr_r(0)
			, m_stderr_w(0)
            , m_terminated(0) 
            , m_exitcode(0) 
        {
            m_p_info.hProcess = 0;   
			Var value = ctx->value("out");

			if (value.type() == Var::TYPE_VOID)
				value = ctx->setValue("out", o3_new(cStream)(GetStdHandle(STD_OUTPUT_HANDLE)));				 
		
			m_stdout_default = value.toScr();

			value = ctx->value("err");
			if (value.type() == Var::TYPE_VOID)
				value = ctx->setValue("err", o3_new(cStream)(GetStdHandle(STD_ERROR_HANDLE)));				 

			m_stderr_default = value.toScr();
		}
	
        virtual ~cProcess1() 
        {
        }

        o3_begin_class(cProcess1Base)
        o3_end_class();

		o3_glue_gen()

        HANDLE                  m_stdin_r; 
        HANDLE                  m_stdin_w; 
        HANDLE                  m_stdout_r; 
        HANDLE                  m_stdout_w;
        HANDLE                  m_stderr_r; 
        HANDLE                  m_stderr_w;
        OVERLAPPED              m_overlapped_out;
		OVERLAPPED              m_overlapped_err;
        siEvent                 m_event_out;
        siEvent                 m_event_err;
		siHandle                m_hprocess;
        PROCESS_INFORMATION     m_p_info;
        DWORD                   m_av;
        siStream				m_stdout_default;
		siStream				m_stdout_custom;
        siStream				m_stderr_default;
		siStream				m_stderr_custom;

		siWeak                  m_ctx;
        siListener              m_listener_out;
		siListener              m_listener_err;
        siListener              m_listener_term;
        //o3_get_imm() Str        m_output;
        Str				        m_name;
        char                    m_first_out;
		char                    m_first_err;
        bool			        m_terminated;
        int						m_exitcode;

        
        o3_fun int run(iCtx* ctx, const char* app) 
        {
            WStr wapp = Str(app);
            m_ctx = ctx;
            return run(ctx, wapp, 0); 
        }

        o3_fun int runSelf(iCtx* ctx)
        {
            m_ctx = ctx;
            DWORD error = run(ctx, getSelfPath(), 0);
            return ((int)error);    
        }

        o3_fun void runSelfElevated(iCtx* ctx, const Str& args) 
        {
            m_ctx = ctx;			
            WStr wargs = args;
			runElevated( ctx, getSelfPath(), wargs );
        }

        o3_fun void runSimple(const char* cmd) 
        {
            WStr wcmd = Str(cmd);
            runSimple(wcmd);
        }

        o3_get bool valid() 
        {
            return m_hprocess ? true : false;
        }

        o3_get int pid() 
        {
            return (int) m_p_info.dwProcessId;
        }

        //o3_fun Str receive(iCtx* ctx, int timeout = 0) 
        //{            
        //    ctx->loop()->wait( timeout );
        //    if (!m_onreceive) {
        //        Str ret = m_output;                
        //        m_output.clear();
        //        return ret;
        //    }
        //    return Str();
        //}

        o3_fun void send(const char* input, size_t size)
        {
            unsigned long bread;
            WaitForInputIdle( m_p_info.hProcess, 1000);
            WriteFile(m_stdin_w,input,(DWORD)size,&bread,NULL); 
        }
        
        o3_fun void kill() 
        {
            if(m_hprocess){
                TerminateProcess((HANDLE)m_hprocess->handle(), 0 );
                closeHandles();
                m_hprocess = 0;
                m_p_info.hProcess = 0;
                m_p_info.hThread = 0;
                //m_output.clear();
            }            
        }

        static o3_ext("cO3") o3_fun siScr process(iCtx* ctx, const char* name = 0, int pid = 0) 
        {
            cProcess1* ret = o3_new(cProcess1)(ctx) ;
            ret->m_p_info.dwProcessId = (DWORD) pid;
            ret->m_name = name;
            if(pid) {
                ret->m_p_info.hProcess = OpenProcess(SYNCHRONIZE|PROCESS_TERMINATE, FALSE,pid);
                ret->m_hprocess = o3_new(cHandle)(ret->m_p_info.hProcess);
            }
            return ret;
        }

        DWORD run(iCtx* ctx, const wchar_t* app, const wchar_t* currdir=0) 
        {
            STARTUPINFOW si;
            SECURITY_ATTRIBUTES sa;
            SECURITY_DESCRIPTOR sd;
            OSVERSIONINFO osv;
            osv.dwOSVersionInfoSize = sizeof(osv);
            GetVersionEx(&osv);
            // DWORD retval = 0;
            if (osv.dwPlatformId == VER_PLATFORM_WIN32_NT) {
                //initialize security descriptor (Windows NT)
                InitializeSecurityDescriptor(&sd,SECURITY_DESCRIPTOR_REVISION);
                SetSecurityDescriptorDacl(&sd, true, NULL, false);
                sa.lpSecurityDescriptor = &sd;
            }
            else sa.lpSecurityDescriptor = NULL;
            sa.nLength = sizeof(SECURITY_ATTRIBUTES);
            //allow inheritable handles
            sa.bInheritHandle = TRUE;         
            
			setupPipe(&sa, &m_stdout_r, &m_stdout_w, STD_OUTPUT_HANDLE);
			setupPipe(&sa, &m_stderr_r, &m_stderr_w, STD_ERROR_HANDLE);
			setupPipe(&sa, &m_stdin_w, &m_stdin_r, STD_INPUT_HANDLE);

            ZeroMemory( &si, sizeof(STARTUPINFO) );
            si.cb = sizeof(STARTUPINFO);  

            //The dwFlags member tells CreateProcess how to make the process.
            //STARTF_USESTDHANDLES validates the hStd* members. STARTF_USESHOWWINDOW
            //validates the wShowWindow member.

            si.dwFlags = STARTF_USESTDHANDLES|STARTF_USESHOWWINDOW;
            si.wShowWindow = SW_SHOWDEFAULT;//showflags;
            //set the new handles for the child process
            si.hStdOutput    = m_stdout_w;
            si.hStdError    = m_stderr_w;     
            si.hStdInput    = m_stdin_r;
 
            //spawn the child process
            WStr cmd = app;
            if ( ! CreateProcessW(  NULL,cmd.ptr(),
                                    NULL,
                                    NULL,
                                    TRUE,
                                    CREATE_NO_WINDOW,
                                    NULL,
                                    currdir,
                                    &si,
                                    &m_p_info ) ) {
                closeHandles();
                return (DWORD) -1;
            }
			int e = GetLastError();            
            ZeroMemory( &m_overlapped_out, sizeof(OVERLAPPED) );
            m_overlapped_out.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
			m_event_out = o3_new(cEvent)(m_overlapped_out.hEvent);                       
            e = GetLastError();

			ZeroMemory( &m_overlapped_err, sizeof(OVERLAPPED) );
            m_overlapped_err.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
			m_event_err = o3_new(cEvent)(m_overlapped_err.hEvent);                       
			e = GetLastError();

            m_listener_out = ctx->loop()->createListener(siHandle(m_event_out).ptr(),0,
                    Delegate(this, &cProcess1::onReceive));
			m_listener_err = ctx->loop()->createListener(siHandle(m_event_err).ptr(),0,
                    Delegate(this, &cProcess1::onError));
            m_hprocess = o3_new(cHandle)(m_p_info.hProcess);
            m_listener_term = ctx->loop()->createListener(m_hprocess.ptr(), 0, 
                    Delegate(this, &cProcess1::onTerminate));

			ReadFile(m_stdout_r, &m_first_out, 1,&m_av,&m_overlapped_out);
			e = GetLastError();
			ReadFile(m_stderr_r, &m_first_err, 1,&m_av,&m_overlapped_err);
			e = GetLastError();
            //SetStdHandle(STD_INPUT_HANDLE, hSaveStdin);
            //SetStdHandle(STD_OUTPUT_HANDLE, hSaveStdout);
			//SetStdHandle(STD_ERROR_HANDLE, hSaveStderr);
         
            return m_p_info.dwProcessId;
        }

		int setupPipe( SECURITY_ATTRIBUTES* sa, HANDLE* toDup, HANDLE* toFetch, DWORD pipeid ) 
		{
			HANDLE hSave = GetStdHandle(pipeid);
			//create stdout pipe
			if (!createPipeEx(toDup,toFetch,sa,0,FILE_FLAG_OVERLAPPED,0)) {
				closeHandles();
				return (DWORD) -1;
			}

			if( !SetStdHandle(pipeid, toFetch) ) {
				closeHandles();
				return (DWORD) -1;
			}

			HANDLE rddup;
			if( ! DuplicateHandle(    GetCurrentProcess(), 
				*toDup,
				GetCurrentProcess(),
				&rddup,
				0,
				FALSE,
				DUPLICATE_SAME_ACCESS ) ) {
					/*int e =*/ GetLastError();
					closeHandles();
					return (DWORD) -1;
			}

			CloseHandle( *toDup );*toDup = rddup;
			SetStdHandle(pipeid, hSave);
			return 0; //! is this OK?
		}

        bool runElevated( iCtx* ctx, const wchar_t* path, const wchar_t* parameters = NULL, const wchar_t* dir = NULL ) 
        {
            m_name = path;

            SHELLEXECUTEINFOW shex;

            memset( &shex, 0, sizeof( shex) );

            shex.cbSize        = sizeof( SHELLEXECUTEINFOW );
            shex.fMask        = SEE_MASK_NOCLOSEPROCESS;
            shex.hwnd        = 0;
            shex.lpVerb        = L"runas";
            shex.lpFile        = path;
            shex.lpParameters  = parameters;
            shex.lpDirectory    = dir;
            shex.nShow        = SW_NORMAL;

            ::ShellExecuteExW( &shex );
            m_p_info.hProcess = shex.hProcess;
            // DWORD e = GetLastError();

			if (m_p_info.hProcess ) {
				m_hprocess = o3_new(cHandle)(m_p_info.hProcess);
                m_listener_term = ctx->loop()->createListener(m_hprocess.ptr(), 0, 
                        Delegate(this, &cProcess1::onTerminate));
			}

            m_p_info.dwProcessId = (DWORD) -1;
            //TODO: process ID? 
            return (int)shex.hInstApp > 32;
        } 

        void runSimple(wchar_t* cmd) 
        {
            STARTUPINFOW si;
            PROCESS_INFORMATION pi;

            ZeroMemory( &si, sizeof(si) );
            si.cb = sizeof(si);
            ZeroMemory( &pi, sizeof(pi) );

            // Start the child process. 
            CreateProcessW( NULL,   // No module name (use command line)
                cmd,        // Command line
                NULL,           // Process handle not inheritable
                NULL,           // Thread handle not inheritable
                FALSE,          // Set handle inheritance to FALSE
                0,              // No creation flags
                NULL,           // Use parent's environment block
                NULL,           // Use parent's starting directory 
                &si,            // Pointer to STARTUPINFO structure
                &pi );           // Pointer to PROCESS_INFORMATION structure

            //m_p_info = pi;
        }

        void closeHandles() 
        {
            if (m_stdin_r)  CloseHandle(m_stdin_r);
            if (m_stdin_w)  CloseHandle(m_stdin_w);
            if (m_stdout_r) CloseHandle(m_stdout_r);
            if (m_stdout_w) CloseHandle(m_stdout_w);
            if (m_stderr_r) CloseHandle(m_stderr_r);
            if (m_stderr_w) CloseHandle(m_stderr_w);

            m_stdin_r = m_stdin_w = m_stdout_r = m_stdout_w = m_stderr_r = m_stderr_w = 0;
        }

        void onReceive(iUnk*)
        {
			o3_assert(m_stdout_default);
            unsigned long b_read;   //bytes read
            unsigned long avail;   //bytes available			
			siStream out = m_stdout_custom ? m_stdout_custom : m_stdout_default;
			
			// did we read the first char from the stdout of the child process successfully?
			if(GetOverlappedResult( m_stdout_r, &m_overlapped_out,
				&b_read, FALSE) && b_read)
			{
				// write out the first char
				out->write(&m_first_out, 1);
				PeekNamedPipe(m_stdout_r,0,0,0,&avail,NULL);
				// read/write out the rest
				Str buf(avail);
				if (avail != 0) {
					ReadFile(m_stdout_r, buf.ptr(), avail, &b_read,NULL);
					buf.resize(b_read);
					out->write(buf.ptr(),b_read);
				}
				
				// continue listening
				siHandle h = m_event_out;
				ZeroMemory( &m_overlapped_out, sizeof(OVERLAPPED) );
				ResetEvent(h->handle());                
				m_overlapped_out.hEvent = h->handle();
				ReadFile(m_stdout_r, &m_first_out, 1,&m_av,&m_overlapped_out);
			}									
        }

        void onError(iUnk*)
        {
			o3_assert(m_stderr_default);
            unsigned long b_read;   //bytes read
            unsigned long avail;   //bytes available			
			siStream err = m_stderr_custom ? m_stderr_custom : m_stderr_default;
			
			// did we read the first char from the stderr of the child process successfully?
			if(GetOverlappedResult( m_stderr_r, &m_overlapped_err,
				&b_read, FALSE) && b_read)
			{
				// write out the first char
				err->write(&m_first_err, 1);
				PeekNamedPipe(m_stderr_r,0,0,0,&avail,NULL);
				// read/write out the rest
				Str buf(avail);
				if (avail != 0) {
					ReadFile(m_stderr_r, buf.ptr(), avail, &b_read,NULL);
					buf.resize(b_read);
					err->write(buf.ptr(),b_read);
				}
				
				// continue listening
				siHandle h = m_event_err;
				ZeroMemory( &m_overlapped_err, sizeof(OVERLAPPED) );
				ResetEvent(h->handle());                
				m_overlapped_err.hEvent = h->handle();
				ReadFile(m_stderr_r, &m_first_err, 1,&m_av,&m_overlapped_err);
			}									
        }


        void onTerminate(iUnk*) 
        {
            DWORD outcode;
            int32_t ret = GetExitCodeProcess(m_hprocess->handle(),&outcode); 
            m_exitcode = (int) outcode;

            if (!ret || m_exitcode != STILL_ACTIVE) {
                m_terminated = m_exitcode < 0;
                m_listener_out = 0;
                m_listener_term = 0;
                //m_listener_in = 0;
				m_p_info.hProcess = 0;
                m_hprocess = 0;
                // DWORD error = GetLastError();                                       
            }     

            if (m_onterminate)
                Delegate(siCtx(m_ctx), m_onterminate)(this);

        }

        // NOTE: These functions are only here to that the process component
        // will compile against the current base. We will have to decide later
        // on a common interface for the component on each platform.
        siStream stdIn()
        {
			return siStream();
            //return m_stdin_custom ? m_stdin_custom : m_stdin_default; 
        }

        siStream setStdIn(iStream* in)
        {
			return in;
            //return m_stdin_custom = in; 
        }

        siStream stdOut()
        {
			return m_stdout_custom ? m_stdout_custom : m_stdout_default; 
        }

        siStream setStdOut(iStream* out)
        {
            return m_stdout_custom = out; 
        }

        siStream stdErr()
        {
            return m_stderr_custom ? m_stderr_custom : m_stderr_default; 
        }

        siStream setStdErr(iStream* err)
        {
            return m_stderr_custom = err; 
        }

        void exec(iCtx* ctx, const char* args)
        {
			WStr wargs = Str(args);
			m_ctx = ctx;
			run(ctx, wargs, 0);         }

		o3_get int exitCode() {
			return m_exitcode;
		}

		// currently the windows implementation is always listening for termination
		virtual void startListening() {

		}

		virtual void stopListening() {

		}
    };

}

#endif // O3_C_PROCESS1_WIN32_H
