# Makefile for Problem Set 6, TASK 2, Fundamentals of Simulation Methods
# Author: Elias Olofsson
# Date: 2020-12-23
#
# Useage:
# make 		- compile binaries
# make clean		- clean binaries 
# make memtest		- test for memory leaks with valgrind

EXE = particle_mesh

SRC = /home/elias/Documents/pcg-c-basic-master/pcg_basic.o
OBJ = $(SRC:.c=.o)

CC = gcc
CFLAGS = -std=c99 -Wall -g -lm -I/home/elias/Documents/fftw-3.3.9/include -L/home/elias/Documents/fftw-3.3.9/lib -lfftw3 -I/home/elias/Documents/pcg-c-basic-master

all:	$(EXE)

# Object file for library
obj:	$(OBJ)

# Clean up
clean:
	-rm -f $(EXE) $(OBJ)

# Compile executable
particle_mesh: particle_mesh.c $(SRC)
	gcc $^ -o $@ $(CFLAGS)

# Test for memory leaks
memtest: particle_mesh
	valgrind --leak-check=full --show-reachable=yes ./$<
