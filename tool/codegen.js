#!/bin/o3
o3.loadModule('console');
o3.loadModule('fs');

var includeTrace = false, 
    immLog = false, log = false;
    logFileName = 'codegenLog.txt',
    errFileName = 'codegenErr.txt';	
var i,l,files = [], arguments = o3.args, arg, 
	scriptFile, scriptFolder, scriptFilePath = arguments[0].replace(/\\/g, '/'); 
if ((scriptFile = o3.cwd.get(scriptFilePath)) && scriptFile.exists){
	scriptFolder = o3.cwd.get(scriptFilePath).parent;
}
else {
	scriptFolder = o3.fs.get(scriptFilePath).parent;	
}

for (i=1, l=arguments.length; i<l; i++) {
	arg = arguments[i];
	switch(arg) {
		case '-h':
			// TODO:
			break;
		case '-l':
			log = true;
			break;
		case '-v':
			immLog = true;
			break;
		case '-trace':
			includeTrace = true;
			break;
		default:
			files.push(arg);			
	}	
}	

function include(file) {
    if (includeTrace)
        o3.print('include: ' + file + '\n');
        
	data = scriptFolder.get(file).data;
	if (!data.length && includeTrace)
        o3.print('open file failed!');
    else        
        eval(data);        
    
    if (includeTrace)    
        o3.print('include finished: ' + file + '\n');
}

function fileWriter(fileName) {
    return function(str) {
        o3.cwd.get(fileName).data = str;
    }
} 

include("o3_Reporter.js");
include("o3_FileHandler.js");
include("o3_Parser.js");
include("o3_Generator.js");
Reporter.immediate = immLog;
Reporter.logWriter = o3.print;
Reporter.errorWriter = o3.print;

// by default it generates all glue in ../include
if (files.length == 0) {
	FileHandler.scanFiles(
		scriptFolder.get('../include'));
}

// if there were files/folders specified, let's traverse them
var f,f2;
for (i=0; i<files.length; i++) {    
	f = o3.cwd.get(files[i]);
    f2 = o3.fs.get(files[i]);
	if (f.exists) 
        FileHandler.scanFiles(f);
    else if(f2.exists)
		FileHandler.scanFiles(f2);
	else
        Reporter.globalError('File does not exists: ', files[i], '.\n');        
} 

// if -l was specified let's log to stdout 
// (if -i was specified then it is already logged out at this point no need to log it again)
if (!Reporter.immediate && log) 
    Reporter.dumpLogs();

// in any case, let's save the logs and errors into files
Reporter.logWriter = fileWriter(logFileName);
Reporter.errorWriter = fileWriter(errFileName);    
Reporter.dumpLogs();
Reporter.dumpErrors();

