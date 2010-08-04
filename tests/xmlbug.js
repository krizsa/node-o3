//var doc = o3.xml.parseFromString("<juhu />", "text/xml")
//doc.documentElement.selectSingleNode("juhu/foo[@name='bar']")

//var doc = o3.xml.parseFromString('<a:application xmlns:a="http://ajax.org/2005/aml" />', "text/xml");
//o3.print(doc.documentElement.namespaceURI + "\n");

var doc = o3.xml.parseFromString('<doc><child1 /><child2 /></doc>');
var docElem = doc.documentElement;
var child = docElem.firstChild;
var removed = docElem.removeChild(child);
removed.setAttribute("blah", "blih");

o3.print(removed.getAttribute("blah"));
