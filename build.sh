mkdir build
cd build
cmake ..
make

valgrind --track-origins=yes --leak-check=yes ./bin/alu_demo
