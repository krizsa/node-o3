var files = [
				'test_blob.js',
				'test_fs.js',
				'test_xml.js'			
			],
	o3RootPath = 'c:/Development/scannerMergeTo/',
	o3TestsPath = o3RootPath + 'tests/',
	o3ExePath = o3RootPath + 'build/o3_dbg.exe',		
	argv = o3.args,
	argc = argv.length,
	testFile = null,
	tests = [],			
	testCase = '',	
	outFile = o3.cwd.get("outFile.txt"),
	outStream = outFile.open("w"),
	errFile = o3.cwd.get("errFile.txt"),
	errStream = errFile.open("w"),
	command;

function launchProcess(cmd) {
	var running = true,
		proc = o3.process();
	
	proc.stdOut = outStream;
	proc.stdErr = errStream;
	proc.onterminate = function(){running = false};
	proc.exec(cmd);
	while(running)
		o3.wait(1);
	return proc.exitCode;	
}	
	
if (argc > 1)
	testFile = argv[1];
		
if (argc > 2)		
	testCase = argv[2];
	
tests = testFile ? [testFile] : files;

for (var i=0, l=tests.length; i<l; i++) {
	var moreTests = true, res, last, exitCode, startedCases;
	command = o3ExePath + ' ' + o3TestsPath + tests[i];
	o3.print('Starting: ' + tests[i] + ':\n');
	o3.print('===================================\n')
	while(moreTests) {
		exitCode = launchProcess(command + (testCase ? (' ' + testCase + ' -s') : ''));			
		errStream.close();
		outStream.close();
		o3.print(errFile.data);

		if (exitCode<0) {			
			res = outFile.data;
			startedCases = res.split('\n');
			testCase = startedCases[startedCases.length-2];				
		} 
		else {
			testCase = '';
			moreTests = false;
		}
		errFile.data = '';
		outFile.data = '';
		errStream = errFile.open('w');
		outStream = outFile.open('w');		
	}
	o3.print('\n(finished: ' + tests[i] + ')\n\n');
}
	
	