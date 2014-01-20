all:
	g++ vecAdd.cpp -o vecAdd.x -framework OpenCL

clean:
	rm *.x 
