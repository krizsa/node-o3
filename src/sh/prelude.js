o3.loadModule('fs');
o3.loadModule('console');

function include(file, includeTrace) {	
	var scriptFile, scriptFolder, scriptFilePath = o3.args[0].replace(/\\/g, '/'); 
	if ((scriptFile = o3.cwd.get(scriptFilePath)) && scriptFile.exists){
		scriptFolder = o3.cwd.get(scriptFilePath).parent;
	}
	else {
		scriptFolder = o3.fs.get(scriptFilePath).parent;	
	}    
	
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

this.include = include;