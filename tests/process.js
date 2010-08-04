var proc = o3.process(),
	running = true;
	
//proc.stdOut = outStream;
//proc.stdErr = errStream;
proc.onterminate = function(){running = false};
proc.exec("gedit test_xml.js");
while(running)
	o3.wait(1);
	
o3.print("process exited with code: " + proc.exitCode + "\n");