var A = [];
for (j = 0;j<8;j++)
{
	y = 100 + j*20;
	off = j*365;
	for (i =0 ;i<365;i++)
	{
		A[i+off]=y+Math.sin(i*0.04)*20;
	}
};

var t = (new Date).getTime();
var totalcount = 40;
var img = o3.image(400,300,"argb");

for (count = 0;count<totalcount;count++)
{
	img.clear(0xffffffff)

	img.fillStyle = "rgba(0,0,0,0.1)";
	img.strokeStyle = "black";
	img.lineWidth = 1;
	for (j = 0;j<8;j++)
	{
		off = j*365;
		img.moveTo(-5,y);
		for (i =0 ;i<365;i++)
		{
			img.lineTo(i, A[i+off]);
		}
		img.lineTo(365,400);
		img.lineTo(0,400);
		img.closePath();
		img.fill();
		img.moveTo(-5,y);
		for (i =0 ;i<365;i++)
		{
			img.lineTo(i, A[i+off]);
		}
		img.stroke();

	};

	img.savePng("speeddump"+count+".png");
}

t2 = (new Date).getTime();

dt = t2-t;
o3.print("tstart: " + t+ " t-end: " + t2 + "\n")
o3.print(dt + " msec\n")
o3.print((dt/totalcount) + " msec per png\n")
""
