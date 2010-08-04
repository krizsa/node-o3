var str = 
"<?xml version=\"1.0\" encoding=\"utf-8\" ?>" +
"<a:application xmlns:a=\"http://ajax.org/2005/aml\" xmlns=\"blahblah\">" +
"    <a:script>alert('hello world!');</a:script>" +
"</a:application>";

var dom = o3.xml.parseFromString(str);

var attr = dom.firstChild.getAttribute("xmlns:a");
o3.print(attr);