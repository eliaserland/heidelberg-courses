# Makefile for Problem Set 8, Fundamentals of Simulation Methods
# Author: Elias Olofsson
# Date: 2021-01-20
#
# Useage:
# make 		- compile binaries
# make clean		- clean binaries 
# make memtest		- test for memory leaks with valgrind

EXE = tree

OBJ = $(SRC:.c=.o)

CC = gcc
CFLAGS = -std=c99 -Wall -g -lm

all:	$(EXE)

# Object file for library
obj:	$(OBJ)

# Clean up
clean:
	-rm -f $(EXE) $(OBJ)

# Compile executable
tree: tree.c $(SRC)
	gcc $^ -o $@ $(CFLAGS)

# Test for memory leaks
memtest: tree
	valgrind --leak-check=full --show-reachable=yes ./$<
