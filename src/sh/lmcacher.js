o3.cwd.path;
/*
 * Note: The files that are scanned are converted to xml format and must be properly constructed, if not script will fail.
 * HTML files starting with "<!doctype html>\n" are processed properly
 */

// keep reference to all fileChanges for undo
var fileChanges = [];
var o3Obj;
apf.onload = function() {
    // init o3 plugin
    o3Obj = o3.create("8A66ECAC-63FD-4AFA-9D42-3034D18C88F4", { 
        oninstallprompt : function(){ alert("can't find o3 plugin"); },
        product : 'O3Demo'
    });

    var fPath       = "/C:/development/javeline/prj_nbd/trunk/bezoekers/";
    //var fPath       = "/C:/development/javeline/lmcacher/output/";
    //var apfFile     = fPath + "js/apf_release_nbd.js";
    var apfFile     = "/C:/development/javeline/prj_nbd/trunk/bezoekers/js/apf_release_nbd.js";
    
    var files     = [
        fPath + "bezoekers.html", 
        fPath + "pages/home.xml", 
        fPath + "pages/news.xml",
        fPath + "pages/news_item.xml",
        fPath + "pages/compare.xml",
        fPath + "pages/page.xml",
        fPath + "pages/product.xml",
        fPath + "pages/productenoverzicht.xml",
        fPath + "pages/request_info.xml",
        fPath + "pages/search.xml",
        fPath + "pages/supplier.xml"
    ];
    
    // Warning: setting outputPath to same page as original files will overwrite them, backup will be created
    // set outputPath to "." to save changes to original file
    var outputPath    = "/C:/development/javeline/lmcacher/output/";
    //var outputPath    = "/C:/development/javeline/prj_nbd/trunk/bezoekers/pages/";
    //var outputPath  = ".";
    
    var hasChanges  = false;
    var lmObjs = [];
    for (var lmChanged, incChanged, doctype, fileName, filePath, fileOutput, lmNodes, fileStr, fileXml, fi = 0, fl = files.length; fi < fl; fi++) {
        // rename <?lm to <lm><![CDATA[ and ?> to ]]></lm> to make them available in the xml
        if (!o3Obj.fs.get(files[fi]).exists) {
            ////document.getElementById("lstOutput").innerHTML += "<li>file not found: " + files[fi] + "</li>";
            continue;
        }
        fileStr = o3Obj.fs.get(files[fi]).data.split("<?lm").join("<lm><![CDATA[").split("?>").join("]]></lm>");

        // ignore doctype declaration in html files
        if (fileStr.indexOf("<!doctype html>") == 0) {
            doctype = "<!doctype html>\n";
            fileStr = fileStr.substr("<!doctype html>".length);
        } 
        else {
            doctype = "";
        }
            
        fileXml = (fileStr.trim().indexOf("<html") != 0) 
            ? apf.getXml("<script xmlns:a='http://www.ajax.org/2005/aml'>" + fileStr + "</script>")
            : apf.getXml(fileStr);
        if (!fileXml) debugger;
        /*
         * start process lm script
         */
        // get lm nodes, for some reason this doesn't work in bezoekers.html
        lmNodes = fileXml.getElementsByTagName("lm");
        if (lmNodes.length) {
            for (var oId, ni = 0, nl = lmNodes.length; ni < nl; ni++) {
                oId = "c" + parseInt(lmObjs.length+1);

                // lm node starting with ~~ is already cached
                if (lmNodes[ni].firstChild.nodeValue.substr(0, 2) == "~~")
                    continue;
                    
                // generate code from lm
                var f = apf.lm.compile(lmNodes[ni].text, {parsecode: 1});
                // save reference to lm code for saving later
                lmObjs.push({
                    id          : oId,
                    attributes  : f,
                    compiled    : apf.lm.lastCode().replace("var _f", "this." + oId)
                })
                
                // replace script with reference to generated code
                lmNodes[ni].firstChild.nodeValue = "~~" + oId + "~~"
                lmChanged = true;
            }
            
        }
        else {
            lmChanged = false;
        }
        /*
         * end process lm scripts
         */
        
        /*
         * start process include files
         */
        incNodes = fileXml.getElementsByTagName("a:include");

        if (incNodes.length) {
            for (var includeXml = null, fileSrc = "", filePath = "", fileContent = "", ni = 0, nl = incNodes.length; ni < nl; ni++) {
                fileSrc = incNodes[ni].getAttribute("src");

                var filePath = files[fi].split("/");
                filePath.pop();
                filePath = filePath.join("/");

                if (!fileSrc) debugger;
                fileStr = o3Obj.fs.get(filePath + "/" + fileSrc).data;
                // load filecontent, relative to path of current file
                // replace incNodes[ni] with fileContent
                includeXml = apf.getXml(fileStr);
                
                // if top node is a:application, only add it's childNodes to file 
                if (includeXml.tagName == "a:application") {
                    // insert all nodes from the include file
                    for (var ci = 0, cl = includeXml.childNodes.length; ci < cl; ci++) {
                        incNodes[ni].parentNode.insertBefore(includeXml.childNodes[ci].cloneNode(true), incNodes[ni])
                        //incNodes[ni].appendChild(includeXml.childNodes[ci]);
                    }
                    // remove include node
                    incNodes[ni].parentNode.removeChild(incNodes[ni]);
                }
                else {
                    debugger;
                    apf.xmldb.replaceNode(apf.getXml(fileStr), incNodes[ni]);
                }
            }
            incChanged = true;
        }
        else {
            incChanged = false;
        }

        if (lmChanged || incChanged) {
            hasChanges = true;
            // rename back from <lm><![CDATA[ to <?lm
            fileOutput = doctype + fileXml.xml.split("<lm><![CDATA[").join("<?lm").split("]]></lm>").join("?>");
            
            // remove script node
            if (fileOutput.indexOf('<script xmlns:a="http://www.ajax.org/2005/aml">') > -1) {
                fileOutput = fileOutput.replace('<script xmlns:a="http://www.ajax.org/2005/aml">', "");
                fileOutput = fileOutput.replace("</script>", "");
            }
            
            // write to file
            fileName = files[fi].split("/")[files[fi].split("/").length-1];
            filePath = (outputPath != ".") ? outputPath : files[fi].substr(0, files[fi].indexOf(fileName));
                
            // outputfile is same as original file, create backup
            if (files[fi] == filePath + fileName && o3Obj.fs.get(filePath + "backup_" + fileName).data != o3Obj.fs.get(files[fi]).data) {
                fileChanges.push({
                    file    : filePath + "backup_" + fileName,
                    before  : o3Obj.fs.get(filePath + "backup_" + fileName).data,
                    after   : o3Obj.fs.get(files[fi]).data
                });
                o3Obj.fs.get(filePath + "backup_" + fileName).data = o3Obj.fs.get(files[fi]).data;
                //document.getElementById("lstOutput").innerHTML += "<li>backup created: " + filePath + "backup_" + fileName + "</li>";
            }

            if (o3Obj.fs.get(filePath + fileName).data != fileOutput) {
                fileChanges.push({
                    file    : filePath + fileName,
                    before  : o3Obj.fs.get(filePath + fileName).data,
                    after   : fileOutput
                });
                o3Obj.fs.get(filePath + fileName).data = fileOutput;
/*
                if (lmChanged && !incChanged)
                    document.getElementById("lstOutput").innerHTML += "<li>lm cache processed:<br/>source: " + files[fi] + "<br/>output: " + filePath + fileName + "</li>";
                else if (incChanged && !lmChanged)
                    document.getElementById("lstOutput").innerHTML += "<li>include processed:<br/>source: " + files[fi] + "<br/>output: " + filePath + fileName + "</li>";
                else if (incChanged && lmChanged)
                    document.getElementById("lstOutput").innerHTML += "<li>lm cache and include processed:<br/>source: " + files[fi] + "<br/>output: " + filePath + fileName + "</li>";
*/
            }
            else {
                //document.getElementById("lstOutput").innerHTML += "<li>no changes made:<br/>source: " + files[fi] + "<br/>output: -</li>";
            }            
        } 
        else {
            //document.getElementById("lstOutput").innerHTML += "<li>no lm to cache or include found:<br/>source: " + files[fi] + "<br/>output: -</li>";
        }   
    }

    // create lm cache code
    if (lmObjs.length) {
        var lmCache = "";
        for (var numReplace, i = 0, l = lmObjs.length; i < l; i++) {
            // replace _f in _async method with generated function name
            if (lmObjs[i].compiled.indexOf("_async\(_n,_c,_a,_w,_f,this,") > -1)
                lmObjs[i].compiled = lmObjs[i].compiled.replace(/(_async\(_n,_c,_a,_w,_f,this,)+/g, "_async\(_n,_c,_a,_w," + lmObjs[i].id + ",this,");
            
            lmCache += lmObjs[i].compiled + "\n";
            for (var attr in lmObjs[i].attributes) {
                if (typeof lmObjs[i].attributes[attr] == "string" || typeof lmObjs[i].attributes[attr] == "number") {
                    lmCache += "this." + lmObjs[i].id + "." + attr + " = " + lmObjs[i].attributes[attr] + ";\n";
                }
                /*
                else    
                    debugger;
                */
            }
            lmCache += "\n";
        }
    
        // read apf file
        var apfFileStr = o3Obj.fs.get(apfFile).data;
        
        // replace lmcache code
        var startIdx = apfFileStr.indexOf("var LMBEGINCACHE;");
        var endIdx = apfFileStr.indexOf("var LMENDCACHE;");
        if (startIdx && endIdx)
            var length = endIdx - startIdx;

        var toReplace = apfFileStr.substr(startIdx, length);
        //if (toReplace.indexOf("var LMBEGINCACHE;") == -1 || toReplace.indexOf("var LMENDCACHE;") == -1) debugger;
        apfFileStr = apfFileStr.replace(toReplace, "var LMBEGINCACHE;\n" + lmCache + "\n")
        
        // save new apf file
        apfFileName = apfFile.split("/")[apfFile.split("/").length-1];
        apfFilePath = (outputPath != ".") ? outputPath : apfFile.substr(0, apfFile.indexOf(apfFileName));
    
        // outputfile is same as original file, create backup
        if (apfFile == apfFilePath + apfFileName && o3Obj.fs.get(apfFilePath + "backup_" + apfFileName).data != o3Obj.fs.get(apfFile).data) {
            fileChanges.push({
                file    : apfFilePath + "backup_" + apfFileName,
                before  : o3Obj.fs.get(apfFilePath + "backup_" + apfFileName).data,
                after   : o3Obj.fs.get(apfFile).data
            });
            o3Obj.fs.get(apfFilePath + "backup_" + apfFileName).data = o3Obj.fs.get(apfFile).data;
            //document.getElementById("lstOutput").innerHTML += "<li>backup created:<br/>source: " + apfFile + "<br/>output: " + apfFilePath + "backup_" + apfFileName + "</li>";
        }
        
        // check for changes
        if (o3Obj.fs.get(apfFilePath + apfFileName).data != apfFileStr) {
            hasChanges = true;
            fileChanges.push({
                file    : apfFilePath + apfFileName,
                before  : o3Obj.fs.get(apfFilePath + apfFileName).data,
                after   : apfFileStr
            });
            o3Obj.fs.get(apfFilePath + apfFileName).data = apfFileStr;
            
            // success message
            //document.getElementById("lstOutput").innerHTML += "<li>lm cache processed:<br/>source: " + apfFile + "<br/>output: " + apfFilePath + apfFileName + "</li>";
        }
        else {
            //document.getElementById("lstOutput").innerHTML += "<li>no changes made:<br/>source: " + apfFile + "<br/>output: -</li>";
        }    
    }
    else {
        //document.getElementById("lstOutput").innerHTML += "<li>no lm cache to add:<br/>source: " + apfFile + "<br/>output: -</li>";
    }    
    
    if (hasChanges)
        btnUndo.setProperty("visible", true);
}

function undoChanges() {
    for (var i = 0, l = fileChanges.length; i < l; i++) {
        // file didn't exists before changes, delete
        if (fileChanges[i].before == "") {
            o3Obj.fs.get(fileChanges[i].file).remove();
            //document.getElementById("lstOutput").innerHTML += "<li>file removed: " + fileChanges[i].file + "</li>";
        }
        else {
            o3Obj.fs.get(fileChanges[i].file).data = fileChanges[i].before;
            //document.getElementById("lstOutput").innerHTML += "<li>file reverted: " + fileChanges[i].file + "</li>";
        }
    }
    //debugger;
    //fileChanges
}
