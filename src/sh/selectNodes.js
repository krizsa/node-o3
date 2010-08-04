/**
 * @author Admin
 */
var fileStr = o3.cwd.get("bezoekers.html").data;
//var fileStr = o3.cwd.get("bezoekers.html").data.split("<?lm").join("<lm><![CDATA[").split("?>").join("]]></lm>");

var fileXml = o3.xml.parseFromString(fileStr).firstChild;
var res = fileXml.selectNodes("/blah:html");
o3.print(res[0].nodeName);