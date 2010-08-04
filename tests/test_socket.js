
var serv = o3.socketTCP(),
	client = o3.socketTCP(),
	connected = false,
	accepted = false,
	sent = false;

function testTCP() {  
    	
	serv.onaccept = function(socket)
	{
		o3.print("accepted \n");
		accepted = socket;		
		socket.onsend = function()
		{
		   o3.print("sending \n");
		}
		socket.send('This is a message');
		socket.onreceive = function()
		{
			o3.print('received text: ' 
				+ accepted.receivedText + '\n');				
		};
		socket.receive();		
    }
	
    client.onreceive = function()
	{
        o3.print('received text: ' + client.receivedText + '\n');
		client.send('this is a response text');
	}

    client.onconnect = function()
	{
        o3.print("connected \n");
        connected = true;
		client.receive();
    }
    
    serv.bind('127.0.0.1', 4321)
    serv.accept();    
    client.connect('127.0.0.1', 4321);
o3.wait(1);
}	

testTCP();