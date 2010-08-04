/*
 * Copyright (C) 2010 Ajax.org BV
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this library; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

o3.loadModule("image");
o3.loadModule("console");

o3.print("**** o3 image tests ****\n\n");

function RunTypeTests()
{
	var bitmaptypes = new Array("bw", "argb");
	for (var testcount=0;testcount<2;testcount++)
	{
		var bmpt = bitmaptypes[testcount];
		
		o3.print("**** doing a test run for format: "+ bmpt + "\n");
		
		for (var test = 3;test<5;test++)
		{   
			var w = test * 30;
			var h = (10-test) * 30;
			o3.print("**** testing image with dimensions "+w+","+h+ "\n");
			
			var x = o3.image(w,h, bmpt);
			var color = 0xffffffff;
			x.clear(color);
			var R = Math.min(w,h)/3;
			for (var p = 0;p<w;p++)
			{
				var xx = p;
				var yy = h/2+Math.sin((p*6.283)/w)*h/3;
			
				x.setPixel(xx,yy, 0xff000000)
				x.line(0,0,w-p*5,p*5, 0x80000000)
			}
			
			o3.print("setting a pixel to 0xff607090 and getting it back...\n")
			x.setPixel(2,2, 0xff607090)
			color = x.getPixel(2,2);
			o3.print("result color: "+ color + "\n");

			x.rect(20,20,50,50,0xff000080);
	 

			o3.print("width: " + x.width + "\n");
			o3.print("height: " + x.height + "\n");
			o3.print("x: " + x.x + "\n");
			o3.print("y: " + x.y + "\n");
			o3.print("mode: " + x.mode + "\n");
			
			o3.print("writing to disk..\n");
			x.savePng(o3.cwd.get("test.png"));
			o3.print("writing done..\n");
			
			o3.print("\n");
		}
	}
}

function SpeechBubble(ctx, x,y,ang)
{
	ctx.save();
	ctx.translate(x,y);
	ctx.rotate(ang);
	ctx.strokeStyle= "rgb(255,128,0)";
	ctx.beginPath();  
	ctx.moveTo(75,25);  
	ctx.quadraticCurveTo(25,25,25,62.5);  
	ctx.quadraticCurveTo(25,100,50,100);  
	ctx.quadraticCurveTo(50,120,30,125);  
	ctx.quadraticCurveTo(60,120,65,100);  
	ctx.quadraticCurveTo(125,100,125,62.5);  
	ctx.quadraticCurveTo(125,25,75,25);  
	ctx.stroke();  
	ctx.restore();
}

function drawtocontext(ctx)
{
	//ctx.strokeStyle = "rgb(0,0,0)";
	roundedRect(ctx,112,112,150,150,15);
	roundedRect(ctx,119,119,136,136,9);
    ctx.fillStyle = "rgb(200,0,0)";
    ctx.fillRect (10, 10, 55, 50);

    ctx.fillStyle = "rgba(0, 0, 200, 0.5)";
    ctx.fillRect (30, 30, 55, 50);
	
   
	ctx.beginPath();
	ctx.moveTo(20,40);
    ctx.lineTo(260,280);
    ctx.lineTo(20,280);
    ctx.closePath();
	
    ctx.moveTo(10,10);
    ctx.lineTo(300,300);
    ctx.lineTo(10,300);
    ctx.closePath();
    
    ctx.fill();
//    ctx.stroke();
	ctx.beginPath();
	SpeechBubble(ctx,50,50,0);
	ctx.strokeWidth = 2;
	SpeechBubble(ctx,100,100,90);
	
	ctx.strokeWidth = 3;
	SpeechBubble(ctx,20,200,45);
}


        
		
function drawpac(ctx) 
{
  roundedRect(ctx,12,12,150,150,15);
  roundedRect(ctx,19,19,150,150,9);
  roundedRect(ctx,53,53,49,33,10);
  roundedRect(ctx,53,119,49,16,6);
  roundedRect(ctx,135,53,49,33,10);
  roundedRect(ctx,135,119,25,49,10);

  ctx.beginPath();
  ctx.arc(37,37,13,Math.PI/7,-Math.PI/7,true);
  ctx.lineTo(31,37);
  ctx.fill();
  for(var i=0;i<8;i++)
  {
    ctx.fillRect(51+i*16,35,4,4);
  }
  for(i=0;i<6;i++){
    ctx.fillRect(115,51+i*16,4,4);
  }
  for(i=0;i<8;i++){
    ctx.fillRect(51+i*16,99,4,4);
  }
  
  ctx.beginPath();
  ctx.moveTo(83,116);
  ctx.lineTo(83,102);
  ctx.bezierCurveTo(83,94,89,88,97,88);
  ctx.bezierCurveTo(105,88,111,94,111,102);
  ctx.lineTo(111,116);
  ctx.lineTo(106.333,111.333);
  ctx.lineTo(101.666,116);
  ctx.lineTo(97,111.333);
  ctx.lineTo(92.333,116);
  ctx.lineTo(87.666,111.333);
  ctx.lineTo(83,116);
  ctx.fill();

  
  ctx.fillStyle = "white";
  ctx.beginPath();
  ctx.moveTo(91,96);
  ctx.bezierCurveTo(88,96,87,99,87,101);
  ctx.bezierCurveTo(87,103,88,106,91,106);
  ctx.bezierCurveTo(94,106,95,103,95,101);
  ctx.bezierCurveTo(95,99,94,96,91,96);
  ctx.moveTo(103,96);

  ctx.bezierCurveTo(100,96,99,99,99,101);
  ctx.bezierCurveTo(99,103,100,106,103,106);
  ctx.bezierCurveTo(106,106,107,103,107,101);
  ctx.bezierCurveTo(107,99,106,96,103,96);
  ctx.fill();
  ctx.fillStyle = "black";
  ctx.beginPath();
  ctx.arc(101,102,2,0,Math.PI*2,true);
  ctx.fill();
  
  ctx.beginPath();
  ctx.arc(89,102,2,0,Math.PI*2,true);
  ctx.fill();
}

function roundedRect(ctx,x,y,width,height,radius){
  ctx.beginPath();
  ctx.moveTo(x,y+radius);
  ctx.lineTo(x,y+height-radius);
  ctx.quadraticCurveTo(x,y+height,x+radius,y+height);
  ctx.lineTo(x+width-radius,y+height);
  ctx.quadraticCurveTo(x+width,y+height,x+width,y+height-radius);
  ctx.lineTo(x+width,y+radius);
  ctx.quadraticCurveTo(x+width,y,x+width-radius,y);
  ctx.lineTo(x+radius,y);
  ctx.quadraticCurveTo(x,y,x,y+radius);
  ctx.stroke();
}		

function drawcol(ctx) 
{
	ctx.fillStyle="red";
	ctx.fillRect(0,0,10,10);
	ctx.fillStyle="BLUE";
	ctx.fillRect(10,0,10,10);
	ctx.fillStyle="GrEeN";
	ctx.fillRect(10,10,10,10);
	ctx.fillStyle="Yellow";
	ctx.fillRect(0,10,10,10);

	ctx.fillStyle="not a color";
	ctx.fillRect(0,20,10,10);

	ctx.fillStyle="#888";
	ctx.fillRect(10,20,10,10);

	ctx.fillStyle="#347680";
	ctx.fillRect(0,30,10,10);
	
	ctx.fillStyle="rgb(60,20,20)";
	ctx.fillRect(10,30,10,10);

	ctx.fillStyle="rgb(20,60,20,0.5)";
	ctx.fillRect(0,40,10,10);

	ctx.fillStyle="rgba(20,20,60,0.3)";
	ctx.fillRect(10,40,10,10);

	ctx.fillStyle="rgba(bullshit)";
	ctx.fillRect(0,50,10,10);
	
	ctx.fillStyle="rgb(bullshit)";
	ctx.fillRect(10,50,10,10);
	
	
	ctx.fillStyle="!@#$^!#$&";
	ctx.fillStyle="blah blue red";
	ctx.fillStyle="pink with blue dots";
}

function drawstars(ctx) 
{
	ctx.fillStyle="gray";
	ctx.fillRect(0,0,150,150);
	ctx.translate(75,75);

	// Create a circular clipping path        
	ctx.beginPath();
	ctx.arc(0,0,50,0,Math.PI*2,true);
	ctx.clip();

	// draw background
	//  var lingrad = ctx.createLinearGradient(0,-75,0,75);
	// lingrad.addColorStop(0, '#232256');
	// lingrad.addColorStop(1, '#143778');

	ctx.fillStyle = "#143778";
	ctx.fillRect(-75,-75,150,150);
	// draw stars

  for (j=1;j<50;j++)
  {
    ctx.save();
    ctx.fillStyle = '#fff';
    ctx.translate(75-Math.floor(Math.random()*150),75-Math.floor(Math.random()*150));
    drawStar(ctx,Math.floor(Math.random()*4)+2);
    ctx.restore();
  }
}

function drawStar(ctx,r){
  ctx.save();
  ctx.beginPath()
  ctx.moveTo(r,0);
  for (i=0;i<9;i++){
    ctx.rotate(Math.PI/5);
    if(i%2 == 0) {
      ctx.lineTo((r/0.525731)*0.200811,0);
    } else {
      ctx.lineTo(r,0);
    }
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

if (1)
{
	var img4 = o3.image(150,150, "argb");
	img4.clear(0xff808080);
	drawstars(img4);
	img4.savePng(o3.cwd.get("clippedstars.png"));
};

if (1)
{
	var img = o3.image(300,300, "argb");
	img.clear(0xffffffff);
	drawtocontext(img);
	img.savePng(o3.cwd.get("canvastest.png"));
}

if (1)
{
	var img3 = o3.image(150,150, "argb");
	img3.clear(0xffffe0f0);
	drawcol(img3);
	img3.savePng(o3.cwd.get("colortest.png"));
}

if (1)
{		
	var img2 = o3.image(150,150, "argb");
	img2.clear(0xffffffff);
	drawpac(img2);
	img2.savePng(o3.cwd.get("pactest.png"));
}

if (1)
{
	 RunTypeTests();
}


if (1)
{
	var img4 = o3.image(150,150, "argb");
	img4.clear(0xffffffff);
	img4.strokeStyle = "black";
	img4.lineWidth = 1;
	for (var i = 0;i<150;i+=3.05)
	{
		img4.moveTo(i,0);
		img4.lineTo(i,150);
	};
	img4.stroke();
	img4.savePng(o3.cwd.get("aa-test1.png"));

	var img4 = o3.image(150,150, "argb");
	img4.clear(0xffffffff);
	img4.strokeStyle = "black";
	img4.lineWidth = 1;
	for (var i = 0;i<150;i+=3.05)
	{
		img4.moveTo(i*2,0);
		img4.lineTo(i,150);
	};
	img4.stroke();
	img4.savePng(o3.cwd.get("aa-test2.png"));

};

'';


