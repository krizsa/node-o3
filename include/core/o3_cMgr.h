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
#ifndef O3_C_MGR_H
#define O3_C_MGR_H


namespace o3 {

struct Trait {
    enum Type {
        TYPE_BEGIN,
        TYPE_END,
        TYPE_FUN,
        TYPE_GET,
        TYPE_SET
    };

    typedef siEx (*invoke_t)(iScr* pthis, iCtx* ctx, int index, int argc,
                             const Var* argv, Var* rval);

    int offset;
    Type type;
    const char* cls_name;
    const char* fun_name;
    invoke_t invoke;
    int index;
    void* ptr;

    static Trait begin()
    {
        static const Trait BEGIN = { 0, TYPE_BEGIN, 0, 0, 0, 0, 0 };
        o3_trace1 trace;

        return BEGIN;
    }

    static Trait end()
    {
        static const Trait END = { 0, TYPE_END, 0, 0, 0, 0, 0 };
        o3_trace1 trace;

        return END;
    }
};

Str hostFromURL( const Str& url );

o3_cls(cMgr);

struct cMgr : cUnk, iMgr {

    tMap<Str, siModule> m_modules;
	tMap<Str, factory_t> m_factories;
    tMap<Str, tVec<Trait> > m_traits;
    siThreadPool m_pool;
    Str m_root;
	Str m_current_url;
	siMutex m_mutex;

#ifdef O3_WIN32
    #define O3_FS_ROOT ""
#else
    #define O3_FS_ROOT "/"
#endif

    cMgr(const char* root = O3_FS_ROOT)		
	{
        o3_trace2 trace;

        //m_pool = g_sys->createThreadPool();
        m_root = root;
		m_mutex = g_sys->createMutex();
    }

    ~cMgr()
    {
		if (m_pool)
			m_pool->deinit();
    }

    o3_begin_class(cUnk)
        o3_add_iface(iMgr)
    o3_end_class()

    Trait* extTraits(const char* name)
    {
        o3_trace2 trace;        
		static Trait TRAITS[] = { Trait::begin(), Trait::end() };
        Trait* traits = TRAITS;
		{
			Lock lock(m_mutex);
			tMap<Str, tVec<Trait> >::Iter iter = m_traits.find(name);
			if (iter != m_traits.end())
				traits = iter->val;
		}        
		return traits;
    }

    bool loadModule(const char* name)
    {
        typedef void (*o3_reg_t)(iMgr*);

        o3_trace2   trace;
        siModule    module;
        o3_reg_t    o3_reg;

        module = g_sys->loadModule(name);
        if (!module) 
            return false;
        o3_reg = (o3_reg_t) module->symbol("o3_reg");
        if (!o3_reg)
            return false;
        o3_reg(this);
		{
			Lock lock(m_mutex);
			m_modules[name] = module;
		}
        return true;
    }

    void addExtTraits(Trait* traits)
    {
        o3_trace2 trace;

        for (Trait* ptrait = traits + 1; ptrait->type != Trait::TYPE_END;
             ++ptrait) {

			Lock lock(m_mutex);
            tVec<Trait>& traits = m_traits[ptrait->cls_name];
            Trait* prev; 
            Trait* curr;

            if (traits.empty()) {
                traits.push(Trait::begin());
                traits.push(Trait::end());
            }
            prev = &traits[traits.size() - 2];
            curr = &(traits.back() = *ptrait);
            curr->offset = prev->offset;
            if (prev->fun_name && !strEquals(curr->fun_name, prev->fun_name))
                ++curr->offset;
            traits.push(Trait::end());
        }
    }

	void addFactory(const Str& name, factory_t factory) 
	{
		m_factories[name] = factory;
	}

	factory_t factory(const Str& name) 
	{
		return m_factories[name];
	}

    siThreadPool pool()
    {
        o3_trace2 trace;

        return m_pool;
    }

    Str root()
    {
        o3_trace2 trace;
#ifdef O3_PLUGIN
        return m_root + "/" + hostFromURL(m_current_url);
#else
		return m_root;
#endif
    }

	Buf downloadComponent(iCtx* ctx, const Str& name,  
		Delegate onreadystatechange, Delegate onprogress)
	{
		Str url = Str(O3_BASE_URL) + O3_VERSION_STRING + "/" + name 
			+ O3_VERSION_STRING + ".zip";		
		
		return downloadFile(url, ctx, onreadystatechange, onprogress);
	}

	Buf downloadUpdateInfo(iCtx* ctx)
	{
		Str url = Str(O3_BASE_URL) + O3_VERSION_STRING + "/hash.zip";
		return downloadFile(url, ctx, Delegate(), Delegate());
	}

	Buf downloadFile( Str url, iCtx* ctx, Delegate onreadystatechange, 
		Delegate onprogress ) 
	{
		siHttp http = m_factories["http"](0);
		http->setOnreadystatechange(onreadystatechange);
		http->setOnprogress(onprogress);
		http->open("GET", url, true);
		http->send(ctx, url, true);	
		return http->responseBlob();
	}

	Str currentUrl()
	{
		return m_current_url;
	}	

	void setCurrentUrl(const Str& url)
	{
		m_current_url = url;
	}

	siFs settingFile(const Str& url)
	{
		siFs root = m_factories["fs"](0);
		Str host = hostFromURL(url);
		if (host.empty())
			return siFs();
		Str path("settings/");
		path.append(host);
		siFs file = root->get(path);
		return file;
	}	

	tMap<Str, int> readSettings()
	{
		siFs file = settingFile(currentUrl());
		if (!file || !file->exists())
			return tMap<Str,int>();
		
		return parseSettings(file->data());
	}

	void writeSettings(const tMap<Str, int>& settings)
	{
		siFs file = settingFile(currentUrl());
		if (file)
			file->setData(serializeSettings(settings));
	}

	tMap<Str, int> parseSettings( const Str& data) 
	{
		tMap<Str, int> settings;
		const char *p = data.ptr();
		const char *k,*v,*e, *i = p;
		while(*i) {
			// key starts here
			k=i; 			
			// equation mark
			while(*i && *i != '=')
				i++;
			o3_assert(i);			
			e = i; 			
			// last char of val
			while(*i && *i != '\n')
				i++;
			o3_assert(i);
			v = i; 
			// add new entry
			settings[Str(k,e-k)] = Str(e+1, v-e-1).toInt32();
			++i;
		}
		return settings;
	}

	Str serializeSettings( const tMap<Str, int>& settings) 
	{
		Str data;
		tMap<Str,int>::ConstIter it;
		for(it=settings.begin(); it!=settings.end(); ++it) {
			data.append((*it).key);
			data.appendf("=%i\n", (*it).val);
		}
		
		return data;
	}

	Str allSettings()
	{
		siFs root = m_factories["fs"](0);
		siFs settings_dir = root->get("settings");
		if (!settings_dir || !settings_dir->exists())
			return Str();
		Str settings;
		tVec<siFs> children = settings_dir->children();
		for (size_t i=0; i<children.size(); i++) {
			settings.append(children[i]->name());
			Str data = children[i]->data();
			data.findAndReplaceAll("\n",",");
			settings.appendf("(%s);", data.ptr());
		}
		return settings;
	}

	bool writeAllSettings(const Str& settings)
	{
		if (!safeLocation())
			return false;

		siFs root = m_factories["fs"](0);
		siFs settings_dir = root->get("settings");
		if (!settings_dir || !settings_dir->exists())
			return false;
		
		const char* src = settings.ptr();
		while(src && *src) {
			const char *e1,*e2,*s=src;
			while(*src && *src!='(')
				src++;
			o3_assert(*src);
			e1=src;
			while(*src && *src!=')')
				src++;
			o3_assert(*src);
			e2=src;
			Str fileName(s,e1-s);
			Str data(e1+1, e2-e1-1);
			data.findAndReplaceAll(",","\n");
			siFs file = settings_dir->get(fileName);
			if (file && file->exists())
				file->setData(data);
			
			src++; // skipping '('
			if (*src) 
				src++; // skipping ';'
		}

		return true;
	}

	Str pathFromURL( const Str& url )
	{		
		size_t i;
		i=url.find("//");
		o3_assert(i!=NOT_FOUND);	
		i=url.find(i+2,"/");
		o3_assert(i!=NOT_FOUND);

		return Str(url.ptr()+i+1);
	}	

	// should return true only for files in the o3 root folder
	bool safeLocation()
	{
		Str o3path,path("/"),host =  hostFromURL(m_current_url);
		if (!strEquals(host.ptr(),"localhost"))
			return false;
		
		path.append(pathFromURL(m_current_url));
		siFs file,root = m_factories["fs"](0);
		
		file = root->get(path.ptr());

		return (file && file->valid() && file->exists());
	}

	Str latestVersion(iCtx* ctx)
	{
		Str url(O3_BASE_URL);
		url.append("version");
		Buf data = downloadFile(url,
			ctx, Delegate(), Delegate());
		
		size_t l = data.size();
		//max: v99.9999 min: v0.9
		if (l > 8 || l < 4) 
			return Str();
		else
			return data;
	}
	
};

// temporary utility function, this should be replaced with some
// more serious url parser and put somewhere else...
Str hostFromURL( const Str& url ) 
{
	size_t s,i;
	i=url.find("//");
	o3_assert(i!=NOT_FOUND);	
	i = s = i+2;

	// TODO: there should be a distinction between files on 
	// the localhost and files on the 'temp' directory where 
	// the fs component has access 
	if (url.ptr()[i] == '/') {
		return Str("localhost");
	}
	//end of domain name
	i=url.find(i,"/");
	o3_assert(i!=NOT_FOUND);
	const char* p=url.ptr();
	return Str(p+s, i-s);
}

}

#endif // O3_C_MGR_H
