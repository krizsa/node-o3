all:
	c++ -I../../include -I../../external/include -L../../external/lib/linux -Wall -o libprocess.so -shared *.cpp -lcurl
	mv libprocess.so /usr/lib

clean:
	rm /usr/lib/libprocess.so
