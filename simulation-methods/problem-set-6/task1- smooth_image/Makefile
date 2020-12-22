# Makefile for Problem Set 6, Fundamentals of Simulation Methods
# Author: Elias Olofsson
# Date: 2020-12-16
#
# Useage:
# make 		- compile binaries
# make clean		- clean binaries 
# make memtest		- test for memory leaks with valgrind

EXE = smooth_image

#SRC = pcg_basic.c
OBJ = $(SRC:.c=.o)

CC = gcc
CFLAGS = -std=c99 -Wall -g -lm -I/home/elias/Documents/fftw-3.3.9/include -L/home/elias/Documents/fftw-3.3.9/lib -lfftw3

all:	$(EXE)

# Object file for library
obj:	$(OBJ)

# Clean up
clean:
	-rm -f $(EXE) $(OBJ)

# Compile executable
smooth_image: smooth_image.c
	gcc $^ -o $@ $(CFLAGS)

# Test for memory leaks
memtest: smooth_image
	valgrind --leak-check=full --show-reachable=yes ./$<
