all:
	c++ -I../../include -I../../external/include -L../../external/lib/linux -Wall -o libconsole.so -shared *.cpp 
	mv libconsole.so /usr/lib

clean:
	rm /usr/lib/libconsole.so
