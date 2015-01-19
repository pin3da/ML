CC=g++
CFLAGS=-std=c++11
LDFLAGS=-larmadillo -lmlpack
IFLAGS=-I/usr/include/libxml2/

all: test tutorial

test: test.cc
	$(CC) $(CFLAGS) $(LDFLAGS) $(IFLAGS) -o test.out test.cc

tutorial: GP_tutorial.cc
	$(CC) $(CFLAGS) $(LDFLAGS) $(IFLAGS) -o tutorial.out GP_tutorial.cc

clean:
	rm test


