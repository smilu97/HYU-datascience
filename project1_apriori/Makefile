CC := g++
CFLAGS := -O2 -I src/headers/

build:
	$(CC) $(CFLAGS) -c -o src/apriori.o src/apriori.cc
	$(CC) $(CFLAGS) -c -o src/main.o src/main.cc
	$(CC) $(CFLAGS) -c -o src/utils.o src/utils.cc
	$(CC) $(CFLAGS) -c -o src/itemset.o src/itemset.cc
	$(CC) $(CFLAGS) -c -o src/itemsetlist.o src/itemsetlist.cc
	$(CC) -o apriori src/apriori.o src/main.o src/utils.o src/itemset.o src/itemsetlist.o

clean:
	rm -rf src/*.o

all: build
