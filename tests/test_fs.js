#!/bin/o3
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
o3.loadModule('console');
o3.loadModule('fs');

include("test_prelude.js");

var setup = function(){
	root = o3.cwd;
	wd_name = "testfs";
	wd = root.get(wd_name);
	wd.createDir();
	return	assert(wd.valid && wd.exists,"can not create test folder")
			|| true;
}

var teardown = function(){
	wd.remove(true);
	return 	assert(wd.exists, "can not delete test folder") 
			|| true;
}


var tests = {		
	'validity':function(){	
		var node;
		return (
				assert(wd.valid, "working dir is not valid")
			||	assert(wd.exists, "working dir does not exist")
			||( 
				node = wd.get("notafile.ext"),
				assert(node.valid, "new node is not valid")	
			)
			||	assert(!node.exists, "new node seems to exist but it should not") 
			||(
				node = root.parent,
				assert(!node.valid, "parent of the root node should not be valid")
			)			
			||	true 
		);
	},
	'type':function(){
		var node;
		return (
				assert(wd.isDir,"working dir is not a folder?")
			||	assert(!wd.isFile,"working dir is is a file but should be a folder")
			||	assert(!wd.isLink,"working dir is is a link but should be a folder")
			|| (
				node = wd.get("filea.ext"),
				node.createFile(),
				assert(node.exists,"could not create filea.ext")
			)
			||	assert(node.isFile,"filea.ext is not a file")
			||	assert(!node.isDir,"filea.ext should not be a folder")
			||	assert(!node.isLink,"filea.ext should not be a link")
			|| (
				node.remove(),
				assert(!node.exists,"could not delete filea.ext")
			)
			||	assert(!node.isFile,"filea.ext should have been deleted")
			||	assert(!node.isDir,"filea.ext should not be a folder, and should have been deleted")
			||	assert(!node.isLink,"filea.ext should not be a link, and should have been deleted")
			||	true
		);		
	},
	'datamanip':function(){
		//str mode
		var node = wd.get("content.txt"),
			content = "This is some content.",
			readback,
			stream,
			res;
		node.data = content;
		
		function readStream(stream) {
			var ret='';
			while (!stream.eof)	{
					ret += stream.read(1);
					stream.flush();
			}
			return ret;
		}
		
		return (
				assert(node.exists,"could not create content.txt")
			|| (
				readback = node.data,
				assert( readback == content, "(str mode) string written to the file was: [" + content + "] but the result of reading the file is : [" + readback + "]")			
			)
			|| (
				//stream mode read
				stream = node.open("r"),
				res = stream.read(3),
				stream.flush(),
				assert(res == content.substring(0,3), "first 3 chars of the data in the file are not correct. Content was : " + content + " Read back result: " + res)
			)
			|| (
				stream.pos = 3,
				res = stream.read(1),
				stream.flush(),
				assert(res == content.charAt(3), "4th character of the data in the file is not correct. Content was: " + content + " Read back result: " + res)
			)
			|| (
				stream.pos = 0,
				res = readStream(stream),				
				stream.close(),
				assert(res == content, "(stream mode) string written to the file was: [" + content + "] but the result of reading the file is : [" + res + "]")
			)
			|| (
				//stream mode write
				stream = node.open("w"),
				stream.pos = 1,
				stream.write("H"),
				stream.flush(),
				stream.close(),
				stream = node.open("r"),
				stream.pos = 1,
				res = stream.read(1),
				stream.flush(),
				stream.close(),
				assert(res == "H","after modification the 2nd char should be: [H] , but it is :[" + res + "]")
			)
			|| true
		);	
	},
	'creation':function(){
		var node = wd.get("file.ext");
		node.createFile();
		
		return (
				assert(node.valid,"could not create file.ext, node is invalid")			
			||	assert(node.exists,"could not create file.ext, node does not exist")
			||	assert(node.isFile,"could not create file.ext, node is not a file")
			|| (
				node = wd.get("Fodler"),
				node.createDir(),
				assert(node.valid,"could not create Folder, node is invalid")
			)
			||	assert(node.exists,"could not create Folder, node does not exist")
			||	assert(node.isDir,"could not create Folder, node is not a folder")
			|| (
				node = wd.get("Fodler2/Folder3/Folder4/tmp.txt"),				
				node.data = "this is a test",
				assert(node.data == "this is a test","recursive directory creation has failed")
			)
			|| true
		);						
		/*
				try{ var a = node.data; return failed("Reading from a file that does not exist should report an error! " + a)}
				catch(e){}
				*/
	},	
	'reposition':function(){
		var folder,dest,dest2,node = wd.get("torep.txt");
		node.data = "content";
		folder = wd.get("folder");
		dest = folder.get("destfile");
		folder.createDir();
		node.copy(dest);
		return (
				assert(dest.data == "content", "could not copy file")
			|| (
				dest2 = dest.move(wd),
				assert(dest2.data == "content", "could not move file")
			)
			|| true	
		);
	},
	'list':function(){
		var children,length,error,node = wd.get("tolist");
		function traverse(folder) {
			for (var i = 0; i < length; i++){
				if ( ! folder[i].exists ) return("child " + i + " does not exist");
				if (folder[i].isDir) children[i].get("childFile").createFile();		
			}		
			return false;
		}
		node.createDir();
		node.get("first").createFile();
		node.get("second").createFile();
		node.get("third").createFile();
		node.get("fourth").createDir();
		children = node.children;
		length = children.length;

		return (
				assert(length == 4, "\'tolist\' folder should contain 4 children but has: " + length)	
			||	assert(! ( error = traverse(children)), error)
			||	assert(node.get("fourth/childFile").isFile,"childFile could not be created")
			||	true
		);
	},
	'navigation':function(){
		//dir tree to create and traverse
		//		dir1
		//			dir1_1
		//				dir1_1_1
		//					file1_1_1a
		//					file1_1_1b
		//				file1_1a
		//				file1_1b
		//			dir1_2
		//				file1_2a
		//				file1_2b
		//			file1a
		//			file1b
		//		dir2
		//			.file2a
		//			.file2b
		wd.get("dir1").createDir();
		wd.get("dir1/dir1_1").createDir();
		wd.get("dir1/dir1_1/dir1_1_1").createDir();
		wd.get("dir1/dir1_1/dir1_1_1/file1_1_1a").createFile();
		wd.get("dir1/dir1_1/dir1_1_1/file1_1_1b").createFile();
		wd.get("dir1/dir1_1/file1_1a").createFile();
		wd.get("dir1/dir1_1/file1_1b").createFile();
		wd.get("dir1/dir1_2").createDir();
		wd.get("dir1/dir1_2/file1_2a").createFile();
		wd.get("dir1/dir1_2/file1_2b").createFile();		
		wd.get("dir1/file1a").createFile();
		wd.get("dir1/file1b").createFile();		
		wd.get("dir2").createDir();
		wd.get("dir2/.file2a").createFile();
		wd.get("dir2/.file2b").createFile();
		
		var tmpname, wd_path = wd.path;
		return (		
				assert(root.get(wd_path).name == wd_name,"getting to the working directory by its path failed from root")
			||	assert(wd.get("/" + wd_path).name == wd_name,"getting to the working directory by its path failed from itself")
			||	assert(wd.get("dir1/dir2/../../dir2/.file2a").name == ".file2a","path test1 failed")
			||	assert(wd.get("./dir1/./dir2/../../dir2/.file2a").name == ".file2a","path test2 failed")
			||	assert(wd.get("dir1").get("..").name == wd_name,"path \"..\" does not work")
			|| (
				tmpname = wd.get("dir1/dir1_1/dir1_1_1").children[0].parent.name,
				assert( tmpname == "dir1_1_1", "path test3 failed: " + tmpname)
			)
			||	assert(root.parent.valid,"the parent folder of the file root node should not be reachable (by prop parent)")
			//if (root.get("..").valid) return failed ("the parent folder of the file root node should not be reachable (by get(\"..\"))");			
			||	true
		);
	},
    'onchange' : function(){
        var changed,node = wd.get("changeme.txt");        
        node.data = "unchanged"
        changed = false, dirchanged = false;
        node.onchange = function(this_node) {
            changed = true;
            this_node.onchange = 0;
        }
        
        wd.onchange = function(this_node) {
            dirchanged = true;
            this_node.onchange = 0;
        }
        o3.wait(10);
        node.data = "changed";
        o3.wait(1);
		
        return (
				assert(changed, "File change notification failed.")
			||	assert(dirchanged, "Dir change notification failed.")	
			|| (
				changed = false,
				node.data = "changed2",
				o3.wait(1),
				assert(!changed,"onchange notification reset failed")	
			)
			|| true
		);                    
    }    	
}

var wd,
	test = 'fs';

runScript(tests,setup,teardown);	