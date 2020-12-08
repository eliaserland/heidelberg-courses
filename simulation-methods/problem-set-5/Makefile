# Makefile for Problem Set 5, Fundamentals of Simulation Methods
# 2020-11-10
#
# Useage:
# make 		- compile all binaries
# make clean		- clean all binaries 
# make memtest		- test for memory leaks with valgrind

EXE = test

SRC = pcg_basic.c
OBJ = $(SRC:.c=.o)

CC = gcc
CFLAGS = -std=c99 -Wall -g -lm	#-I../datastructures-v1.0.8.2/include

all:	$(EXE)

# Object file for library
obj:	$(OBJ)

# Test for memory leaks
memtest: memtest1

# Clean up
clean:
	-rm -f $(EXE) $(OBJ)
	
test: test.c $(SRC)
	gcc $^ -o $@ $(CFLAGS)

memtest1: test
	valgrind --leak-check=full --show-reachable=yes ./$<

