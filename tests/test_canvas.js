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

function drawtocontext(ctx)
{
    ctx.fillStyle = "rgb(200,0,0)";
    ctx.fillRect (10, 10, 55, 50);

    ctx.fillStyle = "rgba(0, 0, 200, 0.5)";
    ctx.fillRect (30, 30, 55, 50);
    
    ctx.moveTo(20,40);
    ctx.lineTo(260,280);
    ctx.lineTo(20,280);
    ctx.closePath();
	
    ctx.moveTo(10,10);
    ctx.lineTo(300,300);
    ctx.lineTo(10,300);
    ctx.closePath();
    

    ctx.fill();
    ctx.stroke();
	
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

}

function draw() 
{
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    drawtocontext(ctx);
}
  
if(o3 != "no")
{
    var img = o3.image(300,300, "argb");
    drawtocontext(ctx);
}  