function assert(condition, message) {
	return condition ? false : ("ERROR: " + message + '\n');
}

function runScript(ctor) {	
	var obj = new ctor(),
		tests = obj.tests,
		setup = obj.setup,
		teardown = obj.teardown,
		v,
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
 


var testXml2 = function() {		

	diffNodes = function(node1, node2){
		return (
				assert(node1.nodeName == node2.nodeName,	"different nodeNames: " + node1.nodeName + " != " + node2.nodeName)
			||	assert(node1.nodeValue == node2.nodeValue,	"different nodeValues")
			||	assert(node1.nodeType == node2.nodeType,	"different nodeTypes")
			//||	assert(node1.parentNode.nodeName == node2.parentNode.nodeName,	"different parent nodeNames")
			||	assert(node1.childNodes.length == node2.childNodes.length,		"different child numbers")
			||	assert((!node1.firstChild && !node2.firstChild) || node1.firstChild.nodeName == node2.firstChild.nodeName,	"different firstChild names")
			||	assert((!node1.attributes && !node2.attributes) || node1.attributes.length == node2.attributes.length,		"different attribute numbers")
			||	false
		);
	}
	
	function diffNodesRec(node1, node2) {
		var l, i, a, b;
		function compareChildren(node1, node2) {
			var i,l,e;
			for (i=l-1; i>=0; i--) {
				if (e = diffNodesRec(node1.childNodes[i], node2.childNodes[i])) 
					return e;
			}
			return false;
		}
		
		function compareAttributes(node1, node2) {
			if (node1.attributes || node2.attributes) {
				l = node1.attributes.length
				if (l != node2.attributes.length) 
					return "different attribute numbers"
				for (i=l-1; i>=0; i--) {
					a = node1.attributes[i];
					b = node2.attributes[i];
					if ( (a.name != b.name) || (a.value != b.value) ) 
						return "different attributes: [" + a.name + ", " + a.value + "] != [" + b.name + ", " + b.value + "].";
				}
			}
			return false;	
		}
		
		return (
				assert(node1.nodeName 	== node2.nodeName,	"different nodeNames: " + node1.nodeName + " != " + node2.nodeName)
			||	assert(node1.nodeValue 	== node2.nodeValue,	"different nodeValues")
			||	assert(node1.nodeType 	== node2.nodeType,	"different nodeTypes")	
			|| 
			(			
				l = node1.childNodes.length,
				assert(l == node2.childNodes.length,"different child numbers")
			)
			||	compareChildren(node1, node2)
			||	compareAttributes(node1,node2)	
			||	true
		);	
	} 
	
	function diffNodeListsRec(list1, list2) {
		var ret;
		if (ret = assert(list1.length == list2.length, "the two node lists differs in length")) 
			return ret;
			
		for (var v=0; v<list1.length; v++) {
			if (ret = diffNodesRec(list1[v], list2[v]))
				return ret;	
		}
		return true;
	}
	
	function diffAttrLists(list1, list2) {
		var ret;
		if (ret = assert(list1.length == list2.length, "the two attribute lists differs in length"))
			return ret;
			
		for (var v=0; v<list1.length; v++) {
			if (ret = (assert(list1[v].name != list2[v].name) 
				|| (list1[v].value !=list2[v].value)))
					return ret;	
		}
		return true;
		
	}
	
	function selectNodeWithAttr(node, name, value) {
		return node.selectSingleNode("descendant-or-self::node()[@"
            + name + "='" + value + "']");
	}
	
	function addTextNode(node, text) {
		return nodode.appendChild(node.ownerDocument.createTextNode(text));	
	}
	
	function removeAttr(node,xpath,name) {
		return node.selectSingleNode(xpath).removeAttribute(name);
	}
	
	function removeAttrAll(node,xpath,name) {
		var found = node.selectNodes(xpath);
		for (var v=0; v<found.length; v++) {
			found[v].removeAttribute(name);
		}
	}
	
	function rippAllAttr(node) {
		var l = node.attributes.length,
			name,
			out = [];
			
		for (var v=0; v<l; v++) {
			out.push(node.removeAttributeNode(node.attributes[0]));
		}	
		
		return out;
	}
/*
	function rippAllAttr2(node) {
		var attrs = node.attributes,
			name,
			out = [];
			
		for (var v=0; v<attrs.length; v++) {
			name = attrs[v].name;
			out.push(node.removeAttributeNode(node.getAttributeNode(name)));
		}	
		
		return out;
	}
*/		
	function appendAttrList(node,attrs) {
		for (var v=0; v<attrs.length; v++) {
			node.appendChild(attrs[v]);
		}	
	}
	
	function setAttrList(node,attrs) {
		for (var v=0; v<attrs.length; v++) {
			node.appendChild(attrs[v].name,attrs[v].value);
		}		
	}
	
	function insertAttrList(node,attrs) {
		for (var v=0; v<attrs.length; v++) {
			node.appendChild(attrs[v].name,attrs[v].value);
		}		
	}
	
	function rippNode(node,xpath) {
		var torem = node.selectSingleNode(xpath),
			parent = node.parentNode;
			
		return parent.removeChild(torem);	
	}
	
	function rippAllNodes(node) {
		var l = node.childNodes.length,
			parent = node.parentNode,
			out = [];
			
		for (var v=0; v<l; v++) {
			out.push(parent.removeChild(node.firstChild));
		}	

		return out;	
	}
	
	function appendNodeAll(node, children) {
		for(var v=0; v<children.length; v++) {
			node.appendChild(children[v]);
		}
	}
	
	function revertChildNodes(node) {
		var l = node.childNodes.length/2-1,
			detached, s=[];

		for(var v=0; v<=l; v++) {
			detached = node.removeChild(node.firstChild);
			s.push(node.replaceChild(detached, node.lastChild));
		}
		
		for(var v=s.length-1; v>=0; v--) {
			node.insertBefore(s[v], node.firstChild);
		}
		
		return node;
	}
	
	function cloneNodes(nodes) {
		var clones = [];
		for (var v=0; v<nodes.length; v++) 
			clones.push(nodes[v].cloneNode(true));
		return clones;	
	}
	
	var mxml; 	

	this.setup = function(){
		mxml = o3.xml.parseFromString(o3.cwd.get("xml/profile.xml").data);
		mxml.setProperty("SelectionNamespaces",
			"xmlns:a='www.ajax.org'");
		return true;
	}

	this.teardown = function(){
		return true
	}
	
	this.tests = {	
		'attrRipCloneReappend':function(){
			var node;
			node = selectNodeWithAttr(mxml.documentElement, "model", "mdlProfile");
			var cloned,ripped = rippAllAttr(node);
			cloned = cloneNodes(ripped);
			appendAttrList(node, cloned);
			return diffNodeListsRec(ripped, node.attributes);		
		},
		'childNodesRipCloneReappend':function(){
			var node;
			node = selectNodeWithAttr(mxml.documentElement, "model", "mdlProfile");
			var cloned,ripped = rippAllNodes(node);
			cloned = cloneNodes(ripped);
			appendNodeAll(node, cloned);
			return diffNodeListsRec(ripped, node.childNodes);		
		},
		'revertChildren':function(){
			var node;
			node = selectNodeWithAttr(mxml.documentElement, "model", "mdlProfile");		
			var cloned;
			cloned = cloneNodes(node.childNodes);
			revertChildNodes(node);
			var rev = node.childNodes,
				ret;

			if (ret = assert(cloned.length == rev.length, "the two node lists differs in length"))
				return ret;
				
			for (var v=0; v<cloned.length; v++) {
				if (ret = diffNodes(cloned[v], rev[cloned.length-v-1]))
					return ret;	
			}
			return true;
		},
	}
	
};



var sys=require('sys'),
	count = 0,
	id,rounds;

sys.debug("Starting ...");


function runTest(num) {
	o3.print("round: " + count + "-----------------\n");
	if (num)
		rounds = num;
	count = count+1; 
	if (count == 1) {
		id = setInterval(runTest, 1000);
	}
	else if (count === rounds) {
		clearInterval(id);
	}	
	runScript(testXml2);
}

runTest(2000);




