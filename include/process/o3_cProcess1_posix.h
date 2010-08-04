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
#ifndef O3_C_PROCESS1_POSIX_H
#define O3_C_PROCESS1_POSIX_H

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

namespace o3 {

struct cProcess1 : cProcess1Base {
    siStream m_in;
    siStream m_out;
    siStream m_err;
    Str m_name;
    siTimer m_timer;
    pid_t m_pid;
    int m_stat;
	
	Str m_cmd;

    cProcess1() 
    {
    }

    o3_begin_class(cProcess1Base)
    o3_end_class()

	o3_glue_gen()

    static o3_ext("cO3") o3_fun siScr process()
    {
        o3_trace3 trace;

        return o3_new(cProcess1)();
    }

    static o3_ext("cO3") o3_fun int system(const char* command)
    {
        return ::system(command);
    }

    siStream stdIn()
    {
        return m_in;
    }

    siStream setStdIn(iStream* in)
    {
        return m_in = in;
    }

    siStream stdOut()
    {
        return m_out;
    }

    siStream setStdOut(iStream* out)
    {
        return m_out = out;
    }

    siStream stdErr()
    {
        return m_err;
    }

    siStream setStdErr(iStream* err)
    {
        return m_err = err;
    }

	void task(iUnk*)
	{
		int i = ::system(m_cmd.ptr());
	}

    void exec(iCtx* ctx, const char* args)
    {
		m_cmd = args;
		ctx->mgr()->pool()->post(Delegate(this, &cProcess1::task),
			o3_cast this);
	
	
	/*
        tVec<Str> argv;
        tVec<char*> argv1;

        if (args) {
            Str args1 = args;
            char* start;
            char* end;

            start = args1.ptr();
            while (chrIsSpace(*start))
                ++start; 
            end = start;
            while (*end) {
                if (chrIsSpace(*end)) {
                    *end = 0;
                    argv.push(start);
                    start = end + 1;
                    while (chrIsSpace(*start))
                        ++start; 
                    end = start;
                } else
                    ++end;
            }
            argv.push(start);
            for (size_t i = 0; i < argv.size(); ++i)
                argv1.push(argv[i].ptr());
        }
        argv1.push(0);
        m_pid = fork();
        if (m_pid == 0) {
            if (m_in)
                dup2(fileno((FILE*) m_in->unwrap()), 0);            
            if (m_out) 
                dup2(fileno((FILE*) m_out->unwrap()), 1);            
            if (m_err)
                dup2(fileno((FILE*) m_err->unwrap()), 2);           
            execvp(argv1[0], argv1.ptr());
        }
		*/
    }

    int exitCode()
    {
        if (m_pid && waitpid(m_pid, &m_stat, WNOHANG) == m_pid) 
            m_pid = 0;
        return WIFEXITED(m_stat) ? WEXITSTATUS(m_stat) : -1;
    }

    void startListening()
    {
        m_timer = m_ctx->loop()->createTimer(10,
                                             Delegate(this,
                                                      &cProcess1::listen));
    }

    void stopListening()
    {
        m_timer = 0;
    }

    void listen(iUnk*)
    {
        if (m_pid && waitpid(m_pid, &m_stat, WNOHANG) == m_pid) 
            m_pid = 0;
        if (m_pid)
            m_timer = m_ctx->loop()->createTimer(10,
                                                 Delegate(this,
                                                          &cProcess1::listen));
        else
            Delegate(siCtx(m_ctx), m_onterminate)(this);
    }
};

}

#endif // O3_C_PROCESS1_POSIX_H
