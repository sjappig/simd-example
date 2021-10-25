CXX = clang++
CXXFLAGS = -march=native -g -O3

.PHONY: clean

simd-example: src/data.o src/main.o
	$(CXX) $(LDFLAGS) src/data.o src/main.o $(LOADLIBES) $(LDLIBS) -o simd-example

clean:
	rm -f src/*.o
	rm -f simd-example
