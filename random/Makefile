CC=g++
CFLAGS=-std=c++11
LDFLAGS=-larmadillo -lmlpack -lxml2
IFLAGS= -I/usr/include/libxml2/

all: test tutorial noise

test: test.cc
	$(CC) $(CFLAGS) $(LDFLAGS) $(IFLAGS) -o test.out test.cc

generator: data_generator.cc
	$(CC) $(CFLAGS) $(LDFLAGS) $(IFLAGS) -o generator.out data_generator.cc

tutorial: GP_tutorial.cc
	$(CC) $(CFLAGS) $(LDFLAGS) $(IFLAGS) -o tutorial.out GP_tutorial.cc

noise: GP_noisy_observations.cc
	$(CC) $(CFLAGS) $(LDFLAGS) $(IFLAGS) -o noise.out GP_noisy_observations.cc

clean:
	rm *.out


