# Makefile for Problem Set 5, Fundamentals of Simulation Methods
# Author: Elias Olofsson
# Date: 2020-12-09
#
# Useage:
# make 			- compile binaries
# make clean		- clean binaries 
# make memtest		- test for memory leaks with valgrind

EXE = md

SRC = pcg_basic.c
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
md: md.c $(SRC)
	gcc $^ -o $@ $(CFLAGS)

# Test for memory leaks
memtest: md
	valgrind --leak-check=full --show-reachable=yes ./$<