var serv = o3.socketTCP(),
accepted = [];

serv.onaccept = function(socket)
{
	o3.print("accepted \n");
	accepted.push(socket);		
	socket.onreceive = function(_this)
	{
		o3.print('server: received text: ' 
			+ _this.receivedText + '\n');				
		_this.clearBuf();
		_this.send("reply from server");
	};
	socket.send("accepted");
	socket.receive();		
}

serv.bind('127.0.0.1', 4321)
serv.accept(); 
o3.wait(1);