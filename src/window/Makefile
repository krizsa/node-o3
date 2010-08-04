all:
	g++ -I../../include -I../../external/include -L../../external/lib/apple -Wall -dynamiclib -g -o libwindow.dylib *.mm -framework Cocoa
	sudo mv libwindow.dylib /usr/lib

clean:
	sudo rm /usr/lib/libwindow.dylib
