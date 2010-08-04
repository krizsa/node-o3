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
#ifndef O3_C_O3_H
#define O3_C_O3_H

#include "o3_pub_key.h"
#include "o3_crypto.h"
#include "shared/o3_zip_tools.h"

#ifdef O3_WITH_LIBEVENT
	#include<event.h>    
#endif	

namespace o3 {

o3_cls(cLoadProgress);
struct cLoadProgress : cScr {
	typedef iHttp::ReadyState ReadyState;

	cLoadProgress()
		: m_state(iHttp::READY_STATE_UNINITIALIZED)
		, m_bytes_received(0)
	{
		m_mutex = g_sys->createMutex();
	}

	virtual ~cLoadProgress()
	{

	}
	
	o3_begin_class(cScr)
	o3_end_class()

	o3_glue_gen()

	Str m_file_name;
	size_t m_bytes_received;
	ReadyState m_state;
	siMutex m_mutex;

	o3_enum("ReadyState",
		READY_STATE_UNINITIALIZED,
		READY_STATE_LOADING,
		READY_STATE_LOADED,
		READY_STATE_INTERACTIVE,
		READY_STATE_COMPLETED);

	o3_get size_t bytesReceived()
	{
		o3_trace3 trace;
		Lock lock(m_mutex);

		return m_bytes_received;
	}

	o3_get ReadyState readyState()
	{
		o3_trace3 trace;
		Lock lock(m_mutex);

		return m_state;
	}

	o3_get Str fileName()
	{
		o3_trace3 trace;
		Lock lock(m_mutex);

		return m_file_name;
	}

	void setFileName(const Str& name)
	{
		o3_trace3 trace;
		Lock lock(m_mutex);

		m_file_name = name;
	}

	void setState(ReadyState state)
	{
		o3_trace3 trace;
		Lock lock(m_mutex);

		m_state = state;
	}

	void setBytesReceived(size_t bytes_received)
	{
		o3_trace3 trace;
		Lock lock(m_mutex);

		m_bytes_received = bytes_received;
	}
};

struct cO3 : cScr {
    tVec<Str>   m_args;
    tVec<Str>   m_envs;
	siWeak		m_ctx;	
	tList<Str>  m_to_approve;
	tList<Str>  m_to_load;
	siScr		m_onload;
	siScr		m_onprogress;
	siScr		m_onfail;
	siScr		m_onnotification;
	bool		m_loading;
	scLoadProgress	m_load_progress;

    cO3(iCtx* ctx, int /*argc*/, char** argv, char** envp)
		: m_loading(false)
	{
        if (argv)
            while (*argv)
                m_args.push(*argv++);
        if (envp)
            while (*envp)
                m_envs.push(*envp++);

		m_load_progress = o3_new(cLoadProgress);		
		m_ctx = ctx;
		ctx->setValue("cO3", this);
	}

    ~cO3()
    {
    }

    o3_begin_class(cScr)
    o3_end_class()

	o3_glue_gen()

    o3_get tVec<Str> args()
    {
        return m_args;
    }

    o3_get tVec<Str> envs()
    {
        return m_envs;
    }

	o3_fun void wait(iCtx* ctx, int timeout = -1)
	{
		o3_trace3 trace;
#ifdef O3_WITH_LIBEVENT
	event_dispatch();    
#endif	
		ctx->loop()->wait(timeout);
	}

	o3_fun void exit(int status = 0)
	{
		o3_trace3 trace;

		::exit(status);
	}

	o3_get Str versionInfo()
	{		
		o3_trace3 trace;

		Str version(O3_VERSION_STRING);
		version.findAndReplaceAll("_", ".");
		return version;    
	}

	o3_get Str settings(iCtx* ctx)
	{
		return ctx->mgr()->allSettings();
	}		

	o3_set Str setSettings(iCtx* ctx, const Str& settings, siEx* ex)
	{
		if (!ctx->mgr()->writeAllSettings(settings) && ex)
			*ex = o3_new(cEx)("O3 settings could not be saved.");
		return settings;
	}

	o3_get Str settingsURL()
	{
#ifdef O3_WIN32	
		Str base = installDirPath();
		base.findAndReplaceAll("\\", "/");
		return "File:///" + base + "/settings.html";
#else
		return Str();
#endif	
	}

    o3_fun bool loadModule(iCtx* ctx, const char* name) 
    {
        o3_trace3 trace;

        return ctx->mgr()->loadModule(name);
    }

	o3_fun void require(iCtx* ctx, const Str& module)
	{
#ifdef O3_PLUGIN
		m_to_approve.pushBack(module);
#else
		ctx->mgr()->loadModule(module);
#endif
	}

	// checks if the required modules are aproved, and download them
	// asynch if they are not yet downloaded on the client pc
	o3_fun void loadModules(iCtx* ctx, iScr* onload, iScr* onprogress=0, iScr* onfail=0) 
	{
		o3_trace3 trace;
		// if a load is in progress the second call should fail
		if (m_loading)
			return;

		m_onload = onload;
		if (onprogress)
			m_onprogress = onprogress;
		if (onfail)
			m_onfail = onfail;
		m_ctx = ctx;
		approveModules();

		// start loading
		m_to_approve.clear();
		ctx->mgr()->pool()->post(Delegate(this, &cO3::moduleLoading),
			o3_cast this);
	}

	o3_set siScr setOnUpdateNotification(iCtx* ctx, iScr* scr)
	{
		m_ctx = ctx;
		return m_onnotification = scr;
	}

	void onLoad(iUnk*)
	{
		m_loading = false;
		Delegate(siCtx(m_ctx),m_onload)(this);
	}

	void onStateChange(iUnk* http)
	{
		siHttp ihttp = http;
		m_load_progress->setState(
			ihttp->readyState());
		Delegate(siCtx(m_ctx), m_onprogress)(
			siScr(m_load_progress.ptr()));
	}

	void onProgress(iUnk* http)
	{
		siHttp ihttp = http;
		m_load_progress->setBytesReceived(
			ihttp->bytesReceived());
		Delegate(siCtx(m_ctx), m_onprogress)(
			siScr(m_load_progress.ptr()));
	}

	void onNotification(iUnk* http)
	{
		http;
		Delegate(siCtx(m_ctx), m_onprogress)(
			siScr(this));
	}

	void onFail(iUnk*)
	{
		Delegate(siCtx(m_ctx), m_onfail)(
			siScr(this));	
	}	

	// warn the user if a site tries to use an o3 component 
	// and its not approved by the user yet for that site 
	void approveModules() 
	{
		siMgr mgr = siCtx(m_ctx)->mgr();

		// read settings
		tMap<Str, int> settings = mgr->readSettings();

		// approval
		Str name;
		tList<Str> to_approve, approved;		
		for (tList<Str>::Iter it = m_to_approve.begin(); 
			it != m_to_approve.end(); ++it) {				
				if ( settings[*it])
					m_to_load.pushBack(*it);
				else
					to_approve.pushBack(*it);
		}

		approved = approveByUser(to_approve);
		m_to_load.append(approved.begin(), approved.end()); 

		// save settings
		for (tList<Str>::Iter it = approved.begin(); 
			it != approved.end(); ++it) 
			settings[*it]=1;		
		mgr->writeSettings(settings);
	}

	// message box for user aproval
	tList<Str> approveByUser( const tList<Str>& to_approve ) 
	{				
		tList<Str> approved;
		if(to_approve.empty())
			return approved;

		Str msg("The current site is trying to use the following o3 components:\n\n");		
		for (tList<Str>::ConstIter it = to_approve.begin(); 
			it != to_approve.end(); ++it) {	
				msg.append(*it);
				msg.append(", ");
		}
		msg.append("\n\nDo you allow access to those components?");

		if (g_sys->approvalBox(msg, "O3 approval"))		
			approved = to_approve;

		return approved;
	}

	// loading the approved modules, downloading/unpacking/validating 		
	// them if they are missing repeating the process if it failed 
	// and the user selected retry in the pop-up warning window
	// NOTE: this method is executed from a worker thread async
	// NOTE2: after it has finished it will launch the updating asynch
	void moduleLoading(iUnk*)
	{
#ifdef O3_WIN32	
		siCtx ctx = siCtx(m_ctx);
		siMgr mgr = ctx->mgr();		
		tList<Str>::Iter
			it = m_to_load.begin(),
			end = m_to_load.end();	
		
		bool success = true;
		for (; it != end; ++it) {
			if ( ! mgr->loadModule(*it)) {					
				m_load_progress->setFileName(*it);
				Buf downloaded = mgr->downloadComponent(ctx,*it,
					Delegate(this, &cO3::onStateChange), 
					Delegate(this, &cO3::onProgress));
				siStream stream = o3_new(cBufStream)(downloaded);
				if (!unpackModule(*it, stream) 
					|| !mgr->loadModule(*it))
						success = false;
			}
		}

		if (success) {			
			m_to_load.clear();
			ctx->loop()->post(Delegate(this, &cO3::onLoad),o3_cast this);
		} else
			ctx->loop()->post(Delegate(this, &cO3::onFail),o3_cast this);

		// starting the component update in the bg... 
		ctx->mgr()->pool()->post(Delegate(this, &cO3::moduleUpdating),
			o3_cast this);
#endif			
	}

	// unzip the downloaded module, validates it and put the dll in place
	bool unpackModule(const Str& name, iStream* zipped, bool update=false ) 
	{
#ifdef O3_WIN32	
		using namespace zip_tools;
		bool ret = false;
		siCtx ctx(m_ctx);		
		if (!ctx || !zipped)
			return false;

		Str sigName, fileName = name + O3_VERSION_STRING;		
		sigName = fileName;
		fileName.append(".dll");
		Str path("tmp/");
		path.appendf("%s%i",name.ptr(), ctx.ptr());
		siFs fs = ctx->mgr()->factory("fs")(0),
			components = fs->get("components"),
			tmpFolder = fs->get(path),
			unzipped = tmpFolder->get(fileName),
			signature = tmpFolder->get("signature");

		if (!components->exists())
			components->createDir();

		siStream unz_stream = unzipped->open("w");
		siStream sign_stream = signature->open("w");
		if (!unz_stream || !sign_stream)
			return false;

		// unzipping
		siEx error;
		CentralDir central_dir;
		if (error = readCentral(zipped, central_dir))
			goto error;
		
		if (error = readFileFromZip(fileName,zipped,unz_stream,central_dir))
			goto error;
	
		if (error = readFileFromZip(sigName,zipped,sign_stream,central_dir))
			goto error;

		// validating
		unz_stream = unzipped->open("r");
		sign_stream = signature->open("r");
		if (!validateModule(unz_stream,sign_stream))
			goto error;		

		if (update) {
			// rename original dll...
			siFs original = components->get(fileName);
			Str prefix = "tmp/toRem";
			prefix.appendf("%d", ctx.ptr());
			if (original && original->exists())
				original->move(fs->get(prefix + fileName));
		}
		// move validated dll file to the root folder of o3
		unz_stream->close();
		unzipped->move(components, &error);
		ret = !error.valid();

		// if the move failed check if the file is there already 
		// (other process can finish it earlier)
		if (components->get(fileName)->exists() && !update)
			ret = true;

error:
		if (unz_stream)
			unz_stream->close();
		if (sign_stream)
			sign_stream->close();
		fs->get("tmp")->remove(true);
		return ret;
#endif		
	}

	// checks the signiture comes with the dll for validation
	bool validateModule(iStream* data, iStream* signature)
	{
		using namespace Crypto;
		if (!data || !signature)
			return false;

		Buf hash(SHA1_SIZE),encrypted,decrypted;
		if (!hashSHA1(data, (uint8_t*) hash.ptr())) 
			return false;

		hash.resize(SHA1_SIZE);
		size_t enc_size = signature->size();
		encrypted.reserve(enc_size);
		if (enc_size != signature->read(encrypted.ptr(), enc_size))
			return false;
		encrypted.resize(enc_size);
				
		size_t size = (encrypted.size() / mod_size + 1) * mod_size;
		decrypted.reserve(size);
		size = decryptRSA((const uint8_t*) encrypted.ptr(), enc_size, 
			(uint8_t*) decrypted.ptr(), (uint8_t*) &mod, mod_size,
			(const uint8_t*) &pub_exp, pub_exp_size, true);
		
		if ((size_t)-1 == size)
			return false;
		decrypted.resize(size);
		return (size == hash.size() &&
			memEquals(decrypted.ptr(), hash.ptr(), size));
	}

	// checks if there is a new root available, then for the modules
	// downloads a zipped file containing hash files for the latest version of each components
	// we check the local versions hash against these values and update the component if needed
	void moduleUpdating(iUnk*)
	{
#ifdef O3_WIN32	
		using namespace zip_tools;
		siCtx ctx = siCtx(m_ctx);
		siMgr mgr = ctx->mgr();
		checkForMajorUpdate();

		Buf zipped = mgr->downloadUpdateInfo(ctx);
		siStream stream = o3_new(cBufStream)(zipped);
		Buf b(SHA1_SIZE);
		siStream hash = o3_new(cBufStream)(b);
		Str ver(O3_VERSION_STRING); 
		// unzipping
		siEx error;
		CentralDir central_dir;
		if (error = readCentral(stream, central_dir))
			return;
	
		// now let's check all the hash files in the zipped file
		// against the hash of the local version of the components
		siFs fs = ctx->mgr()->factory("fs")(0),
			components = fs->get("components");

		tMap<Str, CentralHeader>::ConstIter 
			it = central_dir.headers.begin(),
			end = central_dir.headers.end();

		for (; it!=end; ++it){
			hash->setPos(0);
			if (error = readFileFromZip((*it).key,stream,hash,central_dir))
				// zip file is corrupted, reporting error or restart?
				return; 
			siFs local = components->get((*it).key + ".dll");
			// update only components the user already have
			if (!local || !local->exists())
				continue;

			siStream stream_local = local->open("r");
			if (!stream_local)
				continue;

			Buf hash_local(SHA1_SIZE);
			hash_local.reserve(SHA1_SIZE);			
			if (SHA1_SIZE != hashSHA1(stream_local, (uint8_t*)hash_local.ptr()))
				continue;
			
			hash_local.resize(SHA1_SIZE);

			if (!memEquals(hash_local.ptr(), ((cBufStream*)hash.ptr())->m_buf.ptr(),
				SHA1_SIZE)) {
					Str name = (*it).key;
					name.findAndReplaceAll(ver.ptr(), "");
					updateComponent(name);
			}
		}
#endif
	}

	// if we already know that a component should be updated..,
	// downloading the latest version with some validation, 
	// rename the original, replace it with the new one
	// mark the original to be deleted, remove the temp folder
	void updateComponent( const Str& name ) 
	{
#ifdef O3_WIN32
		siCtx ctx = siCtx(m_ctx);
		Buf downloaded = ctx->mgr()->downloadComponent(ctx,name,
			Delegate(), Delegate());
		siStream stream = o3_new(cBufStream)(downloaded);
		unpackModule(name, stream, true);
#endif		
	}			

	void checkForMajorUpdate() 
	{
		siCtx ctx = siCtx(m_ctx);
		siMgr mgr = ctx->mgr();
		Str latest = mgr->latestVersion(ctx);
		if (latest.empty())
			return;
		latest.findAndReplaceAll(".", "_");
		if (!strEquals(O3_VERSION_STRING, latest.ptr()))
			ctx->loop()->post(Delegate(this, &cO3::onNotification),o3_cast this);
	}


};

}

#endif // O3_C_O3_H
