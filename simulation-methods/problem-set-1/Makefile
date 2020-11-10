# Makefile for Problem Set 1, Fundamentals of Simulation Methods
# 2020-11-10
#
# Useage:
# make 		- compile all binaries
# make clean		- clean all binaries 
# make memtest		- test for memory leaks with valgrind

EXE = fsm_ex1_task2 fsm_ex1_task3 fsm_ex1_task4

#SRC = ../datastructures-v1.0.8.2/src/stack/stack.c
#OBJ = $(SRC:.c=.o)

CC = gcc
CFLAGS = -std=c99 -Wall -g -lm	#-I../datastructures-v1.0.8.2/include

all:	$(EXE)

# Object file for library
obj:	$(OBJ)

# Test for memory leaks
memtest: memtest2 memtest3 memtest4

# Clean up
clean:
	-rm -f $(EXE) $(OBJ)
	
fsm_ex1_task2: fsm_ex1_task2.c
	gcc $^ -o $@ $(CFLAGS)
	
fsm_ex1_task3: fsm_ex1_task3.c
	gcc $^ -o $@ $(CFLAGS)
	
fsm_ex1_task4: fsm_ex1_task4.c
	gcc $^ -o $@ $(CFLAGS) 

memtest2: fsm_ex1_task2
	valgrind --leak-check=full --show-reachable=yes ./$<

memtest3: fsm_ex1_task3
	valgrind --leak-check=full --show-reachable=yes ./$<
	
memtest4: fsm_ex1_task4
	valgrind --leak-check=full --show-reachable=yes ./$<
	
