all:
	c++ -I../../include -I../../external/include -L../../external/lib/linux -Wall -g -o o3 o3_sh_posix.cpp -ldl -lpthread -lv8 -lxml2
	mv o3 /bin

clean:
	rm /bin/o3
