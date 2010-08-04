var clients = [];
function createSocket()
{
	var socket= o3.socketTCP();	
	socket.id = clients.length;
	clients.push(socket);
	
	socket.onconnect = function(_this){
		var id = _this.id;
		o3.print("connected: " + id + "\n");
		_this.receive();
		//_this.send("first message from socket: " + id);		
	}
	
	socket.onreceive = function(_this){
		var id = _this.id;
		o3.print("client " + id + " recevied: " + 
			_this.receivedText + "\n");
		_this.clearBuf();
		_this.send("message from socket: " + id);
	}
	
	socket.connect('127.0.0.1', 4321);	
}
for(var i=0; i<1000; i++)
	createSocket();

o3.wait(1);