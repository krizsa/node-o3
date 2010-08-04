all:
	c++ -I../../include -I../../external/include -L../../external/lib/linux -Wall -o libhttp.so -shared *.cpp -lcurl
	 mv libhttp.so /usr/lib

clean:
	 rm /usr/lib/libhttp.so
