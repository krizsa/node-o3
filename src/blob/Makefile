all:
	c++ -I../../include -I../../external/include -L../../external/lib/linux -Wall -o libblob.so -shared *.cpp 
	mv libblob.so /usr/lib

clean:
	rm /usr/lib/libblob.so
