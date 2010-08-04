o3.loadModule('console');

this.assert = function(condition, message) {
	return condition ? false : ("ERROR: " + message + '\n');
}

this.runScript = function(tests, setup, teardown) {	
	var v,
		result, 
		last = o3.args.length > 1 ? o3.args[1] : null, 
		foundLast = last ? false : true, 
		stdErr = o3.stdErr,
		stdOut = o3.stdOut;
	
	for (v in tests) {
		if(foundLast) {
			stdOut.write(v + '\n');	
			stdErr.write('\n-' + v + ':\n');
			if (setup) 
				if( ! (result = setup()))				
					stdErr.write(result);
			
			result = tests[v]();
			if ( result != true )
				stdErr.write(result);
			
			if (teardown)
				if ( ! (result = teardown()))
					stdErr.write(result);
		}
		else {
			foundLast = (v == last);
		}
	}
}

