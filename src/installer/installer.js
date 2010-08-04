var appInfo = {
    GUID    : "8A66ECAC-63FD-4AFA-9D42-3034D18C88F4",
    baseApp : "O3Demo",
    version : "0.9",
    locale  : "en_US"
};
appInfo.fullId    = appInfo.baseApp + "-" + appInfo.GUID;
appInfo.fancyName = "O3 plugin";

var i18n = {
    en_US: {
        instTitle      : appInfo.fancyName + " Installation",
        instWelcome    : "Welcome to the Installation of Ajax.org O3 Browser extension for advanced webapps.",
        instStory      : "It is strongly recommended that you close all browsers before pressing the Install button.\n\nPlease choose if you would like to install " + appInfo.fancyName + " for all users,\nor just for yourself. Then press Install. Press Cancel to exit.",
        instUsage      : "Who will use " + appInfo.fancyName + " on this computer?",
        instSuccess    : "Installation Completed",
        instSuccess2   : "The Installation Wizard has successfully installed " + appInfo.fancyName + ". Click Finish to exit the wizard.",
        instFailed     : "Installation failed.",
        instFailedError: "Installation failed. Error code: ",
        uninstTitle    : appInfo.fancyName + " Maintenance",
        uninstWelcome  : "Program Maintenance",
        uninstStory    : "Modify, repair or remove the program",
        uninstQuestion : "Are you sure you want to remove " + appInfo.fancyName + " from your computer?",
        uninstReinst   : "Reinstall " + appInfo.fancyName + ". This option first removes the program and then installs it again.\nChoose this option if you are experiencing problems with the software.",
        uninstRemove   : "Remove " + appInfo.fancyName + " from your computer",
        uninstSuccess  : "Install Wizard Completed",
        uninstFailed   : appInfo.fancyName + " could not been uninstalled properly.",
        removeTitle    : "Remove the Program",
        removeWelcome  : "You have chosen to remove the program from your system.",
        removeStory    : "Click Remove to remove " + appInfo.fancyName + " from your computer. After removal, this program will no longer be available for use.\n\nIf you want to review or change any settings, click Back.",
        removeSuccess  : "The Install Wizard successfully uninstalled " + appInfo.fancyName + ". Click Finish to exit the wizard.",
        reinstWelcome  : "You have chosen to reinstall the program on your system.",
        reinstStory    : "Click Reinstall to remove and install " + appInfo.fancyName + " on your computer.\n\nIf you want to review or change any settings, click Back.",
        reinstSuccess  : "The Install Wizard successfully reinstalled " + appInfo.fancyName + ". Click Finish to exit the wizard.",
        exitwarning    : "Are you sure you want exit " + appInfo.fancyName + " Install Wizard?",
        update         : appInfo.fancyName + " is already installed on your machine. Do you want to update the installed version with this one?",
        install        : "Install",
        cancel         : "Cancel",
        allusers       : "All Users",
        justforme      : "Just me (Recommended)",
        finish         : "Finish",
        retry          : "Retry",
        reinstall      : "Reinstall",
        remove         : "Remove",
        uninstall      : "Uninstall",
        back           : "< Back",
        next           : "Next >",
        errDelete      : "Could not remove files. Please close all browsers and retry.",
        errReg         : "Could not modify windows registries.",
        errWrite       : "Could not create files. ",
        errFind        : "Could not find file. ",
        errUnknown     : "Unknown error, error code: ",
        errAdmin       : "Could not obtain Adminstrator priviliges. ",
        exit           : "Exit"
    },
    nl_NL: {
        instTitle      : appInfo.fancyName + " Installatie",
        instWelcome    : "Welkom bij de Installatie Wizard van " + appInfo.fancyName + "\nDe makkelijskte manier om online documenten te bewerken",
        instStory      : "Wij raden u aan alle Internet browsers af te sluiten, alvorens u de Installeer knopt klikt.\n\nKies a.u.b. of u " + appInfo.fancyName + " wilt installeren voor alle gebruikers,\nof enkel voor uzelf. Klik daarna op Installeer. Klik op Annuleren om af te sluiten.",
        instUsage      : "Wie zal " + appInfo.fancyName + " op deze computer gebruiken?",
        instSuccess    : "De installatie is afgerond",
        instSuccess2   : "De installatie van " + appInfo.fancyName + " is succesvol afgerond. Klik op Afronden om de wizard af te sluiten.",
        instFailed     : "De installatie is mislukt.",
        instFailedError: "De installatie is mislukt. Error code: ",
        uninstTitle    : appInfo.fancyName + " Onderhoud",
        uninstWelcome  : "Software Onderhoud",
        uninstStory    : "Onderhoud, repareer of verwijder de software",
        uninstQuestion : "Weet u zeker dat u " + appInfo.fancyName + " volledig van uw computer wilt verwijderen?",
        uninstReinst   : "Installeer " + appInfo.fancyName + " opnieuw. Deze optie verwijdert de software, alvorens het opnieuw te installeren.\nKies deze optie als u problemen ondervind met het gebruik van de software.",
        uninstRemove   : "Verwijder " + appInfo.fancyName + " van uw computer",
        uninstSuccess  : appInfo.fancyName + " is verwijderd van uw computer.",
        uninstFailed   : appInfo.fancyName + " kon niet worden verwijderd.",
        removeTitle    : "Verwijder de Software",
        removeWelcome  : "You have chosen to remove the program from your system.",
        removeStory    : "Click Remove to remove " + appInfo.fancyName + " from your computer. After\nremoval, this program will no longer be available for use.\n\nIf you want to review or change any settings, click Back.",
        reinstWelcome  : "You have chosen to reinstall the program on your system.",
        reinstStory    : "Click Reinstall to remove and install " + appInfo.fancyName + " on your computer.\n\nIf you want to review or change any settings, click Back.",
        exitwarning    : "Weet u zeker dat u de " + appInfo.fancyName + " Installatie Wizard wilt afsluiten?",
        update         : "Ajax.org " + appInfo.fancyName + " is reeds geinstalleerd op deze computer. Wilt u de huidige versie overschrijven met een nieuwere versie?",
        install        : "Installeren",
        cancel         : "Annuleren",
        allusers       : "Alle Gebruikers",
        justforme      : "Alleen deze gebruiker",
        finish         : "Afronden",
        retry          : "Opnieuw",
        reinstall      : "Herinstalleren",
        remove         : "Verwijderen",
        uninstall      : "Verwijderen",
        back           : "< Vorige",
        next           : "Volgende >",
        errDelete      : "Niet alle bestanden kunnen worden verwijderd. Sluit alle Internet browsers af en probeer het opnieuw.",
        errReg         : "De Windows Registry kon niet worden aangepast.",
        errWrite       : "Niet alle bestanden konden weggeschreven worden. ",
        errFind        : "Het bestand kon niet worden gevonden. ",
        errUnknown     : "Onbekende foutmelding met code: ",
        errAdmin       : "Kon geen adminstratieve rechten verkrijgen. ",
        exit           : "Afsluiten"
    },

    get: function(key) {
        return (i18n[appInfo.locale] && i18n[appInfo.locale][key])
            ? i18n[appInfo.locale][key]
            : i18n.en_US[key] ? i18n.en_US[key] : key;
    }
};
// alias for i18n.get
var _ = i18n.get;

function getErrMsg(code) {
    switch(code) {
        case (-1) : return _("errWrite");
        case (-2) : return _("errDelete");
        case (-3) : return _("errReg");
        case (-4) : return _("errFind");
        case (-5) : return _("errAdmin");
        //this should not happen...
        default: return _("errUnknown") + code;
    }
}

var dialogRunning = true;
function dialogStart(){
    while (dialogRunning) {
        o3.wait(1);
    }
    dialogRunning = true;
}

var processRunning = true;
function runProcess(proc_to_run, command, mode) {
    proc_to_run.onterminate = function(procref) {
        processRunning = false;
    }
	switch(mode) {
		case 'selfElevated' : 
			proc_to_run.runSelfElevated(command); 
			break;
		case 'simple' : 			
			break;	
		default:	
			proc_to_run.exec(command);
	}
    
    while (processRunning) {
        o3.wait(5);
    }
    processRunning = true;
}

var installer = {
    win   : null,
    result: -1,
    setUp : function() {
        var m, p,
            all = false,
            o = {
                wnd   : m = o3.createWindow(_("instTitle"), 320, 240, 500, 385),                
                page1 : {
                    wnd    : p = m.createWindow("page1", 0, 0, 500, 385),
                    ib1    : p.createImgbox("o3_installer_left.bmp", 0, 0, 144, 315),
                    b      : p.createBlank(144, 0, 352, 315),
                    sep    : p.createSeparator(0, 315, 500),
                    tb1    : p.createTextbox(_("instWelcome"), 166, 24, 315, 40, 16, p.BOLD),
                    tb2    : p.createTextbox(_("instStory"), 166, 66, 315, 100, 13),
                    tb3    : p.createTextbox(_("instUsage"), 166, 168, 315, 20, 13, p.BOLD),
                    rb1    : p.createRButton(_("allusers"), 166, 190, 315, 20, 13),
                    rb2    : p.createRButton(_("justforme"), 166, 212, 315, 20, 13),
                    bt1    : p.createButton(_("install"), 300, 330, 90, 20, 13),
                    bt2    : p.createButton(_("cancel"), 395, 330, 90, 20, 13)
                },
                page2 : {
                    wnd : p = m.createWindow("page2", 0, 0, 500, 385),
                    ib1 : p.createImgbox("o3_installer_left.bmp", 0, 0, 144, 315),
                    b   : p.createBlank(144, 0, 352, 315),
                    sep : p.createSeparator(0, 315, 500),
                    tb1 : p.createTextbox(_("instSuccess"), 166, 24, 315, 40, 14, p.BOLD),
                    tb2 : p.createTextbox(_("instSuccess2"), 166, 66, 315, 100, 13),
                    bt1 : p.createButton(_("finish"), 395, 330, 90, 20, 13)
                },
                page3 : {
                    wnd    : p = m.createWindow("page3", 0, 0, 500, 420),
                    ib1    : p.createImgbox("o3_installer_left.bmp", 0, 0, 144, 315),
                    b      : p.createBlank(144, 0, 352, 315),
                    sep    : p.createSeparator(0, 315, 500),
                    tb1    : p.createTextbox(_("instFailed"), 166, 24, 315, 40, 14, p.BOLD),
                    tb2    : p.createTextbox(_("instFailedError"), 166, 66, 315, 100, 13),
                    bt     : p.createButton(_("exit"), 395, 330, 90, 20, 13),
                    all    : false
                },
                reset : false
            };            
        o.wnd.icon = "install.ico";
        o.page1.bt1.onclick = function(){
            //installation call
            installer.result = all;
            dialogRunning = false;
        };
        o.page1.bt2.onclick = function(){
            dialogRunning = o3.alertBox(appInfo.fancyName, _("exitwarning")) != 1;
        };
        o.page1.rb1.onclick = function(){
            o.page1.bt1.enabled = true;
			all = true;
        };
        o.page1.rb2.onclick = function(){
            o.page1.bt1.enabled = true;
            all = false;
        };
        o.page2.bt1.onclick = function(){
            dialogRunning = false;
        };
        o.page3.bt.onclick = function(){
            dialogRunning = false;
        };
        o.reset = function(){
            o.page1.bt1.onclick =
            o.page1.bt2.onclick =
            o.page1.rb1.onclick =
            o.page1.rb2.onclick =
            o.page2.bt1.onclick =
            o.page3.bt.onclick  = 0;
        };
        this.win = o;
        
        return o;
    },
    
    run: function(all_usr) {
		var base = all_usr ? o3.programFiles : o3.appData;
        //copy dll
        //O3/onedit-guid/np-onedit-guid.dll
        var basepath = "O3/" + appInfo.fullId;
        var npdll    = base.get(basepath + "/np-" + appInfo.fullId + ".dll");  
        var dlldata = o3.resources.get("npplugin.dll");
        //patch:    
        installer.patchFile(dlldata);
        while (true) {
            try {npdll.blob = dlldata; break;}                
            catch(e){ if (!o3.alertBox(appInfo.fancyName, getErrMsg(-1), "retrycancel"))  return -1;}               
        }

        var uninst,
            nppath = npdll.path.replace(/\//g,"\\");
        nppath = nppath.substring(1);
        //self copy for uninstaller
        var thisFile = o3.fs.get("/" + o3.selfPath);
        while (true){
            uninst = thisFile.copy(base.get(basepath));
            if (uninst.exists)
                break;
            else if (!o3.alertBox(appInfo.fancyName, getErrMsg(-1), "retrycancel"))
                return -1;
        }
        path = uninst.path.replace(/\//g,"\\");
        path = path.substring(1);
        var uninstargs = "-u" + (all_usr ? " -a" : "");
        //reg uninstaller
        while (true){
            if (o3.regUninstaller(all_usr, {
                name       : appInfo.fullId,
                exe        : path,
                args       : uninstargs,
                icon       : path,
                path       : path,
                displayname: "Ajax.org " + appInfo.fullId,
                helplink   : "",
                publisher  : "Ajax.org BV",
                major      : "1",
                minor      : "1"
            }))
                break;
            else if (!o3.alertBox(appInfo.fancyName, getErrMsg(-3), "retrycancel"))
                return -3;
        }
        //reg activexplugin    
        while (true) {
            if (o3.regDll(nppath, all_usr))
                break;
            else if (!o3.alertBox(appInfo.fancyName, getErrMsg(-3), "retrycancel"))
                return -3;
        } 
        //mozilla registration 
        while (true) {
            if (o3.regMozillaPlugin(all_usr, {
                company    : "Ajax.org BV",
                appname    : appInfo.fullId,
                version    : "1",
                path       : nppath,
                product    : appInfo.fullId,
                description: "NP plugin for Ajax.org " + appInfo.fancyName + " guid: " + appInfo.GUID,
                mimetype   : "application/" + appInfo.fullId
            }))
                break;
            else if (!o3.alertBox(appInfo.fancyName, getErrMsg(-3), "retrycancel"))
                return -3;
        }
        
        return 1;
    },
    
    start: function(elevate) {
        if (elevate) {
            if (o3.adminUser && o3.winVersionMajor == 5)
                return this.run(true);
            var proc = o3.process();            
            runProcess(proc, "-a", "selfElevated");
            return proc.exitCode;
        }
        else {
            return this.run(false);
        }
    },
    
    runPrevious: function() {
        var idx, uninargs, uninpath,
            uninfull = o3.getUninstPath(appInfo.fullId);
        //split uninstaller string to path / args
        idx = uninfull.indexOf(" -");
        if (idx == -1) {
            //some problem with the installed version...
            o3.exitcode = -1;
            cancel = true;
        }
        else {
            var proc = o3.process();
            uninargs = uninfull.substring(idx);
            uninpath = uninfull.substring(0, idx);
            //tmp copy of the uninstaller and run it (otherwise we can not delete the files)
            var thisFile = o3.fs.get(uninpath),
                tmppath  = o3.tempPath + appInfo.fullId + ".exe";
            thisFile.copy(o3.fs.get(tmppath));
            runProcess(proc, tmppath + uninargs + " -f -t");
            //if the uninstall process returns with an error code it means that the
            //uninstall failed and the user gave up retrying it...
            if (proc.exitCode < 0)
                cancel = true;
        }
    },
	    
    patchFile: function(data) {
    var stemGUIDstr = "AAAAAAAA-1111-BBBB-1111-CCCCCCCCCCCC",
            stemGUID    = this.getGUIDBlob(stemGUIDstr),
            stemAppName = "O3Stem";
        GUID = this.getGUIDBlob(appInfo.GUID);
        //PATCHING
        data.replace(stemGUID, GUID);
        data.replaceUtf16(stemGUIDstr, appInfo.GUID);
        data.replaceUtf16(stemAppName, appInfo.baseApp);        
    },
    
    getGUIDBlob: function(guidStr) {
        var hexstr = guidStr.replace(/-/g, ""),
            b      = o3.blob.fromHex(hexstr);
        this.swapBytes(b, 0, 3);
        this.swapBytes(b, 1, 2);
        this.swapBytes(b, 4, 5);
        this.swapBytes(b, 6, 7);
        return b;
    },
    
    swapBytes: function(blob, idx1, idx2) {
        var tmp    = blob[idx1];
        blob[idx1] = blob[idx2];
        blob[idx2] = tmp;
    },
    
    showErrorPage: function(error) {
        var inst = this.win || this.setUp();
        inst.page1.wnd.visible = false;
        inst.page2.wnd.visible = false;
        inst.page3.wnd.visible = true;

        inst.page3.tb2.text    = _("instFailedError") + error;
        //inst.wnd.doModal();
        dialogStart();
        return inst;
    },
    
    showWelcomePage: function() {
        // reset state first:
        this.result = -1;
        var inst = this.win || this.setUp();
        inst.page1.wnd.visible = true;
        inst.page2.wnd.visible = false;
        inst.page3.wnd.visible = false;

        inst.page1.rb2.checked = true;
        //inst.wnd.doModal();
        dialogStart();
        return inst;
    },
    
    showSuccessPage: function() {
        var inst = this.win || this.setUp();
        inst.page1.wnd.visible = false;
        inst.page2.wnd.visible = true;
        inst.page3.wnd.visible = false;

        //inst.wnd.doModal();
        dialogStart();
        return inst;
    }
};

var uninstaller = {
    win      : null,
    result   : false,
    confirmed: -1,
    exit: false,
    setUp    : function() {
        var m, p,
            reinstall = false,
            o = {
                wnd : m = o3.createWindow(_("uninstTitle"), 320, 240, 500, 385),
                page1 : {
                    wnd    : p = m.createWindow("page1", 0, 0, 500, 385),
                    b      : p.createBlank(0, 62, 500, 385, 1),
					ib1    : p.createImgbox("o3_installer_header.bmp", 0, 0, 496, 61),
					tb1    : p.createTextbox(_("uninstWelcome"), 16, 16, 315, 14, 14, p.BOLD, 0xFFFFFF),
                    tb2    : p.createTextbox(_("uninstStory"), 30, 35, 315, 14, 13, 0, 0xFFFFFF),
                    sep    : p.createSeparator(0, 61, 500),
                    tb3    : p.createTextbox(_("uninstReinst"), 105, 105, 340, 55, 13),
                    tb4    : p.createTextbox(_("uninstRemove"), 105, 190, 340, 55, 13),
                    rb1    : p.createRButton(_("reinstall"), 30, 80, 315, 20, 13, p.BOLD, 1),
                    rb2    : p.createRButton(_("remove"), 30, 160, 315, 20, 13, p.BOLD, 1),
                    bt1    : p.createButton(_("next"), 300, 330, 90, 20, 13),
                    bt2    : p.createButton(_("cancel"), 395, 330, 90, 20, 13),
					sep    : p.createSeparator(0, 315, 500)
                },
                page2 : {
                    wnd : p = m.createWindow("page2", 0, 0, 522, 420),
                    ib1 : p.createImgbox("o3_installer_left.bmp", 0, 0, 144, 315),
                    b   : p.createBlank(144, 0, 352, 315),
                    sep : p.createSeparator(0, 315, 500),
                    sep : p.createSeparator(0, 315, 500),
                    tb2 : p.createTextbox(_("uninstSuccess"), 166, 24, 315, 40, 14, p.BOLD),
                    tb2 : p.createTextbox(_("removeSuccess"), 166, 66, 315, 100, 13),
                    bt1 : p.createButton(_("finish"), 395, 330, 90, 20, 13)
                },
                page3 : {
                    wnd    : p = m.createWindow("page3", 0, 0, 522, 420),
                    ib1    : p.createImgbox("o3_installer_left.bmp", 0, 0, 144, 315),
                    b      : p.createBlank(144, 0, 352, 315),
                    sep    : p.createSeparator(0, 315, 500),
                    tb1    : p.createTextbox(_("uninstFailed"), 166, 24, 315, 40, 14, p.BOLD),
                    tb2    : p.createTextbox("", 166, 66, 315, 100, 13),
                    bt     : p.createButton(_("exit"), 395, 330, 90, 20, 13)
                    //all : false
                },
                page4 : {
                    wnd    : p = m.createWindow("page4", 0, 0, 522, 385),
					b      : p.createBlank(0, 62, 500, 385, 1),
                    ib1    : p.createImgbox("o3_installer_header.bmp", 0, 0, 496, 61),
                    tb1    : p.createTextbox(_("uninstTitle"), 16, 16, 315, 14, 14, p.BOLD,0xFFFFFF),
                    tb2    : p.createTextbox(_("removeWelcome"), 30, 35, 315, 14, 13, 0, 0xFFFFFF),
                    sep    : p.createSeparator(0, 61, 500),
                    tb3    : p.createTextbox(_("removeStory"), 30, 70, 400, 55, 13),
                    bt1    : p.createButton(_("back"), 210, 330, 90, 20, 13),
                    bt2    : p.createButton(_("remove"), 300, 330, 90, 20, 13),
                    bt3    : p.createButton(_("cancel"), 395, 330, 90, 20, 13),
					sep    : p.createSeparator(0, 315, 500)
                },
                reset : false
            };
        o.page1.bt1.onclick = function(){
            //installation call
            uninstaller.result = reinstall;
            dialogRunning = false;
        };
        o.page1.bt2.onclick = function(){
            uninstaller.exit = true;
            dialogRunning = o3.alertBox(appInfo.fancyName, _("exitwarning")) != 1;
        };
        o.page1.rb1.onclick = function() {
            o.page1.bt1.enabled = true;
            reinstall = true;
        };
        o.page1.rb2.onclick = function() {
            o.page1.bt1.enabled = true;
            reinstall = false;
        };
        o.page2.bt1.onclick = function(){
            dialogRunning = false;
        };
        o.page3.bt.onclick = function(){
            dialogRunning = false;
        };
        o.page4.bt1.onclick = function(){
            uninstaller.confirmed = 0;
            dialogRunning = false;
        };
        o.page4.bt2.onclick = function(){
            // remove confirmed
            uninstaller.confirmed = 1;
            dialogRunning = false;
        };
        o.page4.bt3.onclick = function(){
            uninstaller.exit = true;
            dialogRunning = o3.alertBox(appInfo.fancyName, _("exitwarning")) != 1;
        };
        o.reset = function(){
            o.page1.bt1.onclick =
            o.page1.bt2.onclick =
            o.page1.rb1.onclick =
            o.page1.rb2.onclick =
            o.page2.bt1.onclick =
            o.page3.bt.onclick  = 
            o.page4.bt1.onclick =
            o.page4.bt3.onclick =
            o.page4.bt2.onclick = 0;
        };
        o.wnd.onclose = function(){ 
            uninstaller.exit = true;
            dialogRunning = false;
        }
        this.win = o;
            
        return o;
    },
    
    runFromTemp: function(args) {
        // tmp copy of the uninstaller and run it 
        // (otherwise we can not delete the files)
        var thisFile = o3.fs.get(o3.selfPath),
            tmppath  = o3.tempPath + appInfo.fullId  + ".exe",
            tmpcpy   = o3.fs.get(tmppath);
        tmpcpy.remove(true);
        tmpcpy = thisFile.copy(tmpcpy);
        o3.process().runSimple(tmppath + " " + args + " -t");
    },
    
    run: function(all_usr, update) {
        while (true) {
            if (o3.unregMozillaPlugin(all_usr, "Ajax.org BV", appInfo.fullId, "1"))
                break;
            else if (!o3.alertBox(appInfo.fancyName, getErrMsg(-3) + " unregNP", "retrycancel"))
                return -3;
        }        
        var base = all_usr ? o3.programFiles : o3.appData;
        
        if (base.exists) {        
            var basepath = "O3/" + appInfo.fullId;
            var dll = base.get(basepath + "/np-" + appInfo.fullId + ".dll");
            
            if (dll.exists) {                
                var path = dll.path.replace(/\//g,"\\");
                path = path.substring(1);
                while (true) {
                    if (o3.unregDll(path, all_usr))
                        break;
                    else if (!o3.alertBox(appInfo.fancyName, getErrMsg(-3) + " unregDll", "retrycancel"))
                        return -3;
                }
                while (true) {				
					var installDir = base.get(basepath);
					if (installDir.remove(true))
                        break;
					if (update) {
						// if we can not remove the files it probably means they are being used
						// let's try to move them into a temp folder so we can install the new version
						var succeded=true, 							 
							files = installDir.children,
							tmpFolder = o3.tmpDir.get("o3trash");
						// if there are any remaining files from the previous update let's remove those first
						tmpFolder.remove(true);
						if (!tmpFolder.createDir())
							succeded = false;
						 										
						for(var i=0; i<files.length; i++) {
							try {
								files[i].move(tmpFolder);
							} catch(e){succeded = false;}
						}		
						
						if (succeded)
							break;
					}
							
                    if (!o3.alertBox(appInfo.fancyName, getErrMsg(-2), "retrycancel"))
                        return -2;
                }
            }
        }
        while (true) {
            if (o3.unregUninstaller(all_usr, appInfo.fullId))
                break;
            else if (!o3.alertBox(appInfo.fancyName,getErrMsg(-3) + " unregUninstaller","retrycancel"))
                return -3;
        }
        if (reinstall)
            return installer.start(all_usr);
        return 1;
    },
    
    start: function(elevate, reinstall, update) {
        if (elevate) {
            if (o3.adminUser && o3.winVersionMajor == 5)
                return this.run(true, update);

            var proc = o3.process();
            runProcess(proc, "-u -a -t -e" + (reinstall ? " -r" : ""), "selfElevated");
            return proc.exitCode;
        }
        else {
            return this.run(false, update);
        }
    },
    
    showErrorPage: function(error) {
        var uinst = this.win || this.setUp();
        uinst.page1.wnd.visible = false;
        uinst.page2.wnd.visible = false;
        uinst.page3.wnd.visible = true;
        uinst.page4.wnd.visible = false;

        uinst.page3.tb2.text = getErrMsg(error);
        //uinst.wnd.doModal();
        dialogStart();
        return uinst;
    },
    
    showWelcomePage: function() {
        // reset states first:
        this.confirmed = -1;
        var uinst = this.win || this.setUp();
        uinst.page1.wnd.visible = true;
        uinst.page2.wnd.visible = false;
        uinst.page3.wnd.visible = false;
        uinst.page4.wnd.visible = false;

        //uinst.page1.bt1.enabled = false;
        uinst.page1[this.result ? "rb1" : "rb2"].checked = true;
        //uinst.wnd.doModal();
        dialogStart();
        return uinst;
    },
    
    showConfirmPage: function(bReinstall) {
        var uinst = this.win || this.setUp();
        
        uinst.page4.tb2.text = _(bReinstall ? "reinstWelcome" : "removeWelcome");
        uinst.page4.tb3.text = _(bReinstall ? "reinstStory" : "removeStory");
        uinst.page4.bt2.text = _(bReinstall ? "reinstall" : "remove");
        
        uinst.page1.wnd.visible = false;
        uinst.page2.wnd.visible = false;
        uinst.page3.wnd.visible = false;
        uinst.page4.wnd.visible = true;

        //uinst.wnd.doModal();
        dialogStart();
        uinst.page4.wnd.visible = false;
        return uinst;
    },
    
    showSuccessPage: function(bReinstall) {
        var uinst = this.win || this.setUp();
        
        uinst.page2.tb2.text = _(bReinstall ? "reinstSuccess" : "removeSuccess");
        
        uinst.page1.wnd.visible = false;
        uinst.page2.wnd.visible = true;
        uinst.page3.wnd.visible = false;
        uinst.page4.wnd.visible = false;

        //uinst.wnd.doModal();
        dialogStart();
        return uinst;
    }
};

//Parsing command line arguments
var i,
    flatargs = "",
    args     = o3.args,
    l        = args.length;
for (i = 1; i < l; i++) {
    flatargs += args[i];
}

var error, uninst,
    alluser   = flatargs.indexOf('-a') != -1,  //install / uninstall in all user mode
    uninstall = flatargs.indexOf('-u') != -1,  //uninstall mode
    reinstall = flatargs.indexOf('-r') != -1,  //reinstall mode
    temp      = flatargs.indexOf('-t') != -1,  //running the uninstaller from a temp copy
    elevated  = flatargs.indexOf('-e') != -1,  //the installer is already in an elevated mode
    fast      = flatargs.indexOf('-f') != -1;  //uninstalling in silent mode
o3.exitcode = 0;

//START:
//UNINSTALL
if (uninstall && !temp) {
    uninstaller.runFromTemp(flatargs);
}
else if (uninstall && temp && elevated) {
    //if all user was selected, we had to restart the uninstaller in elevated mode
    //let's do the uninstall and return the result as the exit code
    o3.exitcode = uninstaller.run(true, reinstall);
}
else if (uninstall && temp && fast) {
    error = uninstaller.start(alluser, reinstall, true);
    if (error < 0) {
        uninstaller.showErrorPage(error);
    }
    o3.exitcode = error;
}
else if (uninstall && temp) {
    var uinst;    
    while (true) {
        uinst = uninstaller.showWelcomePage();
        if (uninstaller.exit)
            break;
            
        if (uninstaller.result != -1) {
            //Next button has been pressed
            var bReinstall = uninstaller.result;
            //remove/ reinstall (confirm first!)
            uninstaller.showConfirmPage(bReinstall);
            if (uninstaller.exit) 
                break;
                
            if (uninstaller.confirmed === 1) {
                // Remove button clicked...
                error = uninstaller.start(alluser, bReinstall);
                if (error < 0) {
                    //There was an error show error window
                    uninstaller.showErrorPage(error);
                    break;
                }
                else {
                    //well done... installation completed, show the end page then exit
                    uninstaller.showSuccessPage(bReinstall);
                    break;
                }
            }            
        }            
    }
    uinst.reset();
}
//INSTALL
else if (alluser) {
    //if all user was selected, we had to restart the installer in elevated mode
    //let's do the install and return the result as the exit code
    o3.exitcode = installer.run(true);
}
else {
    //Before running the installer let's see if a previous version of onedit has been already installed
    if (o3.checkIfInstalled(appInfo.fullId)) {
        var cancel = false;
        if (!o3.alertBox(appInfo.fancyName, _("update"))) {
            //if the user choose to leave...
            cancel = true;
        }
        else {
            //if the user wants to update let's uninstall the previous version
            //note: this function call will set the cancel flag internally if needed:
            installer.runPrevious();
        }
    }
    if (!cancel) {
        //Installation dialog
        var inst = installer.showWelcomePage();
        if (installer.result != -1) {
            //Install button was pressed
            error = installer.start(installer.result);
            if (error < 0) {
                //An error occured during installation, so warn the user again and say goodbye
                installer.showErrorPage(error);
            }
            else {
                //installation finished succesfully, show last page
                installer.showSuccessPage();
            }
        }
        inst.reset();
    }
}
