CC=g++
CFLAGS=-std=c++11
LDFLAGS=-larmadillo -lmlpack
IFLAGS=-I/usr/include/libxml2/

all: test

test: test.cc
	$(CC) $(CFLAGS) $(LDFLAGS) $(IFLAGS) -o test.out test.cc

clean:
	rm test
