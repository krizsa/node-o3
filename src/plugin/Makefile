all:
	c++ -I../../include -I../../external/include -L../../external/lib/apple -Wall -g -o o3 npplugin o3_plugin.mm -ldl -lpthread -lv8
	mv o3 /bin

clean:
	rm /bin/o3
