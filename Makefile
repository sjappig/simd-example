OPTIMIZATION_LEVEL = 0
CXX = clang++
CXXFLAGS = -march=native -g -O${OPTIMIZATION_LEVEL}

.PHONY: clean

simd-example: src/data.o src/main.o
	$(CXX) $(LDFLAGS) src/data.o src/main.o $(LOADLIBES) $(LDLIBS) -o simd-example

clean:
	rm -f src/*.o
	rm -f simd-example
