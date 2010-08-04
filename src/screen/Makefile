all:
	g++ -I../../include -I../../external/include -L../../external/lib/apple -Wall -dynamiclib -g -o libscreen.dylib *.mm -framework Cocoa
	sudo mv libscreen.dylib /usr/lib

clean:
	sudo rm /usr/lib/libscreen.dylib
