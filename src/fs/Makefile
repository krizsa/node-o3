all:
	c++ -I../../include -I../../external/include -L../../external/lib/linux -Wall -o libfs.so -shared *.cpp 
	mv libfs.so /usr/lib

clean:
	rm /usr/lib/libfs.so
