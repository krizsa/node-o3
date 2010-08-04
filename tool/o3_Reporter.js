this.Reporter = {
    errors : {general:[]},
    logs   : {general:[]},        
    currentFile   : 'general',    
    logWriter : o3.print,//o3.stdOut.write,
    errorWriter: o3.print,//o3.stdOut.write,
    immediate: false,    
    
    reset : function() {
        with(this) { 
            errors = {general:[]};
            logs = {general:[]};
            currentFile = 'general';
        }
    },
    newFile : function(fileName) {
        with(this) { 
            currentFile = fileName;
            if (!logs[fileName]) 
                logs[fileName] = [];
            if (!errors[fileName])
                errors[fileName]= [];           
        }        
    },
    error : function(){
        with(this) {
            var i,l;
            errorWriter('ERROR: ' + currentFile + ': ');
            for (i=0,l=arguments.length; i<l; i++) {
                errors[currentFile].push(arguments[i]);
                errorWriter(arguments[i]);
            }    
        }
    },
    globalError : function() {
        with(this) {
            var i,l;
                errorWriter('GLOBAL ERROR: ');
            for (i=0,l=arguments.length; i<l; i++) {
                errors['general'].push(arguments[i]);
                errorWriter(arguments[i]);
            }    
        }
    },
    log : function(){
        with(this) {
            var i,l;
            if (immediate)
                    //o3.stdOut.write(currentFile + ': ');
					logWriter(currentFile + ': ');
            for (i=0,l=arguments.length; i<l; i++) {
                logs[currentFile].push(arguments[i]);
                if (immediate)
                    //o3.stdOut.write(arguments[i]);
					logWriter(arguments[i]);
            }    
        }
    },
    dumpLogs : function(){
        with(this) {
            dump("Logs", logs, logWriter);
        }    
    },
    dumpErrors : function(){
        with(this) { 
            dump("Errors", errors, errorWriter);            
        }    
    },
    dump : function(title, dataObj, writeMethod) {
        var file, data, t=[];
        t.push(title, ': \n================================\n');
        for (file in dataObj) {
            data = dataObj[file];
            if (data.length) 
                t.push('\n',file,': \n','-----------------\n',data.join(''));            
        }
        t.push('\n\n');
        writeMethod(t.join(''));
    } 
};

/*
Reporter.newFile('file1');
Reporter.log('first log: ', 'blah1 ', 'blah2', '\n');
Reporter.log('second log: \n');
Reporter.error('shit hit the fan', ' please press the panic button!!!\n');
Reporter.newFile('file2');
Reporter.log('fuck sake');
Reporter.error('fuck sake\n');
Reporter.error('fuck sake2\n');
Reporter.error('WTF??\n');
Reporter.error('Panic button broken\n');
Reporter.newFile('file3');
Reporter.dumpLogs();
Reporter.dumpErrors();
*/