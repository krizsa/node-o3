for (var i=0; i<2; i++) {
	var fs = o3.fs;
	var dir = fs.get('/mnt/dev/node/node/deps/o3/tests');
	o3.print(dir.path);
	var children = dir.children;
	var data = '';
	for (var j=0; j<children.length; j++) {
		o3.print(i + '.' + j + '\n');
		if (children[j].isFile) 
			data = children[j].data;
	} 
}