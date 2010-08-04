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
#ifndef O3_C_FS1_POSIX_H
#define O3_C_FS1_POSIX_H

#include <sys/stat.h>
#include <dirent.h>

namespace o3 {

struct cFs1 : cFs1Base {
    bool m_valid;
    Str m_root_path;
    Str m_rel_path;
    int64_t m_time;
    siTimer m_timer;

    cFs1(const char* root_path = "/", const char* rel_path = "/")
        : m_root_path(root_path), m_rel_path(rel_path)
    {
        m_valid = parsePath();
    }

    o3_begin_class(cFs1Base)
    o3_end_class()

    o3_glue_gen();

    static o3_ext("cO3") o3_get siScr fs(iCtx* ctx)
    {
        o3_trace3 trace;
        Var fs = ctx->value("fs");

        if (fs.type() == Var::TYPE_VOID)
            fs = ctx->setValue("fs", o3_new(cFs1)());
        return fs.toScr();
    }

    static o3_ext("cO3") o3_get siScr cwd(iCtx* ctx)
    {
        o3_trace3 trace;
        Var cwd = ctx->value("cwd");

        if (cwd.type() == Var::TYPE_VOID) {
            char* buf = getcwd(0, 0);

            cwd = ctx->setValue("cwd", o3_new(cFs1)("/", buf));
            free(buf);
        }
        return cwd.toScr();
    }

    bool valid()
    {
        o3_trace3 trace;

        return m_valid;
    }

    bool exists()
    {
        o3_trace3 trace;
        struct stat buf;

        return stat(localPath(), &buf) == 0;
    }

    Type type()
    {
        struct stat buf;

        if (stat(localPath(), &buf) >= 0) {
            if (buf.st_mode & S_IFDIR)
                return TYPE_DIR;
            else if (buf.st_mode & S_IFREG)
                return TYPE_FILE;
            else if (buf.st_mode & S_IFLNK)
                return TYPE_LINK;
        }
        return TYPE_INVALID;
    }

    int64_t accessedTime()
    {
        o3_trace3 trace;
        struct stat buf;

        if (stat(localPath(), &buf) < 0) 
            return 0;
#ifdef O3_APPLE
        return buf.st_atimespec.tv_sec;
#endif // O3_APPLE
#ifdef O3_LINUX
        return buf.st_atime;
#endif // O3_LINUX
    }

    int64_t modifiedTime()
    {
        o3_trace3 trace;
        struct stat buf;

        if (stat(localPath(), &buf) < 0) 
            return 0;
#ifdef O3_APPLE
        return buf.st_mtimespec.tv_sec;
#endif // O3_APPLE
#ifdef O3_LINUX
        return buf.st_mtime;
#endif // O3_LINUX
    }

    int64_t createdTime()
    {
        o3_trace3 trace;
        struct stat buf;

        if (stat(localPath(), &buf) < 0) 
            return 0;
#ifdef O3_APPLE
        return buf.st_ctimespec.tv_sec;
#endif // P_APPLE
#ifdef O3_LINUX
        return buf.st_ctime;
#endif // P_LINUX
    }

    size_t size()
    {
        o3_trace3 trace;
        struct stat buf;

        if (stat(localPath(), &buf) < 0) 
            return 0;
        return buf.st_size;
    }

    Str path()
    {
        o3_trace trace;

        return m_rel_path;
    }

    siFs get(const char* path)
    {
        o3_trace3 trace;

        return o3_new(cFs1)(m_root_path,
                            *path == '/' ? Str(path)
							: (m_rel_path.size() > 1 ? m_rel_path : Str()) + "/" + path);
    }

    bool hasChildren()
    {
        o3_trace3 trace;
        DIR* dir = opendir(localPath());

        if (!dir)
            return false;
        closedir(dir);
        return true;
    }

    tVec<Str> scandir(const char* path = 0)
    {
        o3_trace3 trace;
        tVec<Str> names;
        DIR* dir;

        if (*path)
            return ((cFs1*) get(path).ptr())->scandir();
        dir = opendir(localPath());
        if (!dir)
            return names;
        while (struct dirent* ent = readdir(dir))
            if (!strEquals(ent->d_name, ".") &&
                !strEquals(ent->d_name, ".."))
                names.push(ent->d_name);
        closedir(dir);
        return names;
    }

    tVec<siFs> children()
    {
        o3_trace3 trace;
        tVec<siFs> names;
        DIR* dir = opendir(localPath());     

        if (!dir)
            return names;
		while (struct dirent* ent = readdir(dir))
            if (!strEquals(ent->d_name, ".") &&
				!strEquals(ent->d_name, "..")) 
                names.push(get(ent->d_name));				
		
        closedir(dir);
        return names;
    }

    bool createDir()
    {
        o3_trace trace;
		
        if (!exists()) 
            if ( ! ((cFs1*) parent().ptr())->createDir())
				return false;
        mkdir(localPath(), 0777);
        return isDir();
    }

    bool createFile()
    {
        o3_trace trace;

        if (!exists()) 
            if ( ! ((cFs1*) parent().ptr())->createDir())
				return false;
		
		FILE *file = ::fopen(localPath(), "a");	
		
		if (file)
			::fclose(file);
        
		return isFile();
    }

    bool createLink(iFs* to)
    {
        o3_trace trace;

        if (!exists()) 
            if ( ! ((cFs1*) parent().ptr())->createDir())
				return false;
        link(localPath(), ((cFs1*) to)->localPath());
        return isLink();
    }

    bool remove(bool deep)
    {
        o3_trace trace;

        if (deep) {
            tVec<siFs> nodes = children();

            for (size_t i = 0; i < nodes.size(); ++i) 
                ((cFs1*) nodes[i].ptr())->remove(deep);
        }
        if (exists()) {
            if (isDir())
                rmdir(localPath());
            else
                unlink(localPath());
        }
        return !exists();
    }

    siStream open(const char* mode, siEx* ex)
    {
        o3_trace3 trace;
        FILE* stream;

        createFile();
        stream = ::fopen(localPath(), mode);
        if (!stream)
            return siStream();
        return o3_new(cStream)(stream);
    }

    void startListening()
    {
        m_time = modifiedTime();
        m_timer = siCtx(m_ctx)->loop()->createTimer(10,
                                                    Delegate(this, &cFs1::listen));
    }

    void stopListening()
    {
        m_timer = 0;
    }

    void listen(iUnk*)
    {
        int64_t time = modifiedTime();

        if (m_time != time) {
            m_time = time;
            Delegate(siCtx(m_ctx), m_onchange)(this);
        }
        m_timer->restart(10);
    }

    bool parsePath()
    {		
        Str path = m_rel_path;
        const char* src = path.ptr();
        char* dst = path.ptr();

        while (*src) {
            if (*src == '/') {
                *dst++ = *src++; 
                if (*src == '.') {
                    *dst++ = *src++;
                    if (*src == '.') {
                        *dst++ = *src++;
                        if (!*src || *src == '/') {
                            dst -= 3;
                            if (dst-- == path.ptr())
                                return false;
                            while (dst != path.ptr() && *dst != '/')
                                --dst;
                        }
                    } else if (!*src || *src == '/')
                        dst -= 2;
                } else if (*src == '/')
                    return false;
            } else
                *dst++ = *src++;
        }
        path.resize(dst - path.ptr());
        m_rel_path = path;
		return true;

    }

    Str localPath()
    {
        return (m_root_path.size() > 1 ? m_root_path : Str()) + m_rel_path;
    }
	
	void openDoc() 
	{

	}
};

}

#endif // O3_C_FS1_POSIX_H
