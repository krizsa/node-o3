this.FileHandler = {
   scanFiles : function(topNode) {
        function checkFile(file) {
            var name = file.name,
                glueName,
                glueFile,            
                data = file.data;

            if (data.indexOf('o3_glue_gen()') == -1) {
				//Reporter.log('no need to gen glue for: ',name,'\n');
                return 0;
            }
            
            glueName = name.substr(0,3) + 'scr_' + name.substr(3);
            glueFile = file.parent.get(glueName);
			if (!glueFile.exists) {
                if (glueFile.createFile()) {
                    Reporter.log('created glue file: ',glueName,'\n');
                    return glueFile;
                }
                else {
                    Reporter.error('glue file could not be created: ' 
                        ,glueName,'\n');
                    return 0;    
                }
            }

            // TODO: this should be '<'
            if (glueFile.modifiedTime != file.modifiedTime) {
                Reporter.log('found old glue file, old time: '
                    ,glueFile.modifiedTime,'\n');
                return glueFile;
            } 
            
            Reporter.log('glue file was uptodate: ',topNode.name,'\n');
            return 0;    
        }            

        var glueFile;
        if (!topNode.exists) {
            Reporter.globalError('filehandler: file does not exist? ', 
                topNode.name, '\n');
            return;
        }
        if (topNode.isFile) {
            Reporter.newFile(topNode.name);
            if (glueFile = checkFile(topNode))
                this.genGlue(topNode, glueFile);

        }
        else if(topNode.isDir) {
			var i,l,children = topNode.children;
			for (i=0, l=children.length; i<l; i++) {
                this.scanFiles(children[i]);
            }
        }
        else {
            Reporter.globalError('invalid file node: ',topNode.name,'\n');
        }            
   },
   genGlue : function(srcFile, glueFile) {     
        var i,l,result = [],
            scanned = Lexer.scan(srcFile.data),
            classes = Parser.parse(scanned.tree);
        
        for (i=0, l=classes.length; i<l; i++) {
            if (classes[i].gen)
				result.push(Generator.run(classes[i].struct, classes[i].traits));
        }
        try{    
            glueFile.data = result.join('');   
        } catch(e) {
            Reporter.error('Could not write to the file: ', glueFile.name, '\n');
        }     
   }
};
