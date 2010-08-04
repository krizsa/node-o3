var xmlData = [];
xmlData.push(
'<?xml version=\"1.0\" encoding=\"utf-8\"?>\n',
'<bookList xmlns=\"http://site1.com\" xmlns:pub=\"http://site2.com\">\n',
'    <book isbn=\"0471777781\">\n',
'        <title>Professional Ajax</title>\n',
'        <author>Nicholas C. Zakas, Jeremy McPeak, Joe Fawcett</author>\n',
'        <pub:name>Wrox</pub:name>\n',
'    </book>\n',
'    <book isbn=\"0764579088\">\n',
'        <title>Professional JavaScript for Web Developers</title>\n',
'        <author>Nicholas C. Zakas</author>\n',
'        <pub:name>Wrox</pub:name>\n',
'    </book>\n',
'    <book isbn=\"0764557599\">\n',
'        <title>Professional C#</title>\n',
'        <author>Simon Robinson, et al</author>\n',
'        <pub:name>Wrox</pub:name>\n',
'    </book>\n',
'    <book isbn=\"1861006314\">\n',
'        <title>GDI+ Programming: Creating Custom Controls Using C#</title>\n',
'        <author>Eric White</author>\n',
'        <pub:name>Wrox</pub:name>\n',
'    </book>\n',
'    <book isbn=\"1861002025\">\n',
'        <title>Professional Visual Basic 6 Databases</title>\n',
'        <author>Charles Williams</author>\n',
'        <pub:name>Wrox</pub:name>\n',
'    </book>\n',
'</bookList>\n');

var mxml = o3.xml.parseFromString(xmlData.join(""));
var node = mxml.firstChild;

var sNameSpace = "xmlns:na='http://site1.com' xmlns:pub='http://site2.com'";
mxml.setProperty("SelectionNamespaces", sNameSpace);
var oRoot = mxml.documentElement;
var sXPath = "na:book/pub:name";
var cPublishers = oRoot.selectNodes(sXPath);
o3.print(cPublishers.length);

