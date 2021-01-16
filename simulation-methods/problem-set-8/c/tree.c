/*
 * 	Fundamentals of Simulation Methods - WiSe 2020/2021
 * 	Problem Set 8 
 *
 *	Tree algorithm for gravitational N-body system using hierarchical 
 *	multiplole expansion.
 *
 *   	Author: Elias Olofsson
 * 	Date: 2021-01-20
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#define MAX_POINTS 5000      			/* Max no. of particles used. */
#define MAX_NODES (5 * MAX_POINTS)		/* Max no. of nodes in tree.  */

static double opening_threshold = 0.8;      	/* Tree opening angle. */
static double eps               = 0.001;    	/* Gravitational softening length. */

/* Let's define two types of structures, one for the particles, 
   one for the tree nodes */
typedef struct particle_data
{
	double pos[3];
 	double mass;
	double acc_tree[3];
	double acc_exact[3];
} particle;

typedef struct node_data
{
	double center[3];
	double len; 
	double cm[3];
	double mass;
	struct node_data *children[8];
	particle *p;
} node;

/* Lets create, for simplicity, some static global arrays which will hold our data */
static node      tree[MAX_NODES];
static particle  star[MAX_POINTS];

/**
 * get_empty_node() - Return pointer to an empty tree node.
 *
 * Returns: Pointer to an empty node.
 */
node *get_empty_node(void)
{
	node *no;
	static int count_nodes = 0;

	if (count_nodes < MAX_NODES) {
		no = &tree[count_nodes++];
		memset(no, 0, sizeof(node));
	} else {
		printf("sorry, we are out of tree nodes.\n");
		exit(1);
	}
	return no;
}

/**
 * get_subnode_index() - Determine subnode index for a particle within a given node.
 * @current:	Pointer to the node in question.
 * @p:		Pointer to the particle.
 *
 * Determines the subnode index (0-7) in which the given particle falls within 
 * a given node. Index ordering is in binary enumeration of the octant with + as
 * 1, big-endian order.
 *
 * Returns: Subnode index of the given particle.
 */ 
int get_subnode_index(node *current, particle *p)
{
	int index = 0;

	/* Binary enumeration (xyz) = {000,...,111} of the octant with + as 1, 
	   big-endian order. */ 
	if (p->pos[0] > current->center[0]) {  
		index += 4;
	}	
	if (p->pos[1] > current->center[1]) { 
		index += 2;
	}	
	if (p->pos[2] > current->center[2]) {
		index += 1;
	}	
	return index;
}

/**
 * insert_particle() - Insert a particle in the given node of the tree.
 * @current: 	Pointer to the node in question.
 * @pnew: 	Pointer to the particle to insert.
 *
 * Recursively checks if the current node is containing old particles, and to 
 * accommodate the new particle, divides the node into octants of subnodes, 
 * until all nodes each contains a single particle.
 *
 * Returns: Nothing.
 */ 
void insert_particle(node *current, particle *pnew)
{
	node *child;
	int n, p_subnode, pnew_subnode;

	if (current->p) {
		/* The node contains a particle already. 
		   Need to create a new set of 8 subnodes, and then move this particle to one of them */
		n = 0
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					// Get pointer to a new empty node.
					child = get_empty_node();
					
					// Insert node as a child to the current node 
					current->children[n++] = child;
					
					// Set length and geometric center of child node.
					child->len = 0.5 * current->len;
					child->center[0] = current->center[0] + 0.25 * (2*i-1) * current->len;
					child->center[1] = current->center[1] + 0.25 * (2*j-1) * current->len;
					child->center[2] = current->center[2] + 0.25 * (2*k-1) * current->len;
				}
			}
		}

		/* Determine in which subnode the old particle sits. */
		p_subnode = get_subnode_index(current, current->p);

		/* Move the old particle to this subnode. */
		current->children[p_subnode]->p = current->p;
		current->p = NULL;

		/* Determine in which subnode the new particle sits. */
		pnew_subnode = get_subnode_index(current, pnew);

		/* Try to insert the new particle there. */
		insert_particle(current->children[pnew_subnode], pnew);
	
	} else {
	
		/* The current node does not contain a particle. 
		   Check in which subnode the new particle would fall in. */
		pnew_subnode = get_subnode_index(current, pnew);

		/* If the corresponding subnode exists, we try to insert the particle there,
		   otherwise we know there are no subnodes in the node, so we can put the particle into the current node */
		if (current->children[pnew_subnode]) {
			insert_particle(current->children[pnew_subnode], pnew);
		} else {
			current->p = pnew;
		}
	}
}

/**
 * calc_multipole_moments() - Calculate the multipole moments for the current node.
 * @current: Pointer to the node in question.
 *
 * Recursively calculates the moments of the multipole expansion of the current
 * node. (Only monopole moments in this case.)
 *
 * Returns: Nothing.
 */ 
void calc_multipole_moments(node *current)
{
	int n, j;
	node *child;

	/* Do we have subnodes? */
	if(current->children[0]) {
		/* Yes, so let's first calculate their multipole moments. */
		for (n = 0; n < 8; n++) {
			calc_multipole_moments(current->children[n]);
		}
		
		/* Initialize the node multipole moments to zero. */
		current->mass  = 0;
		for (j = 0; j < 3; j++) {
			current->cm[j] = 0;
		}
		
		/* Now calculate the moment of the current node from those of its children */
		
		/* Monopole moment: 
		   For each of the children, sum up the total mass from the 
		   children and the calculate the combined center of mass. */
		for (n = 0; n < 8; n++) {
			child = current->children[n];
			current->mass += child->mass;
			for (j = 0; j < 3; j++) {
				current->cm[j] += child->mass * child->cm[j];
			} 
		}		
		// Normalize the weighted average for the combined center of mass.
		for (j = 0; j < 3; j++) {
			current->cm[j] /= current->mass
		}
	
	} else {
	
		/* Do we at least have a particle? */
		if (current->p) {
			/* Yes, so let's copy this particle to the multipole moments of the node. */
			current->mass = current->p->mass;
			for (j = 0; j < 3; j++) {
				current->cm[j] = current->p->pos[j];
			}	
		} else {
			/* Nothing in here at all; let's initialize the multipole moments to zero. */
			current->mass  = 0;
			for (j = 0; j < 3; j++) {
				current->cm[j] = 0;
			}
		}
	}
}

/**
 * get_opening_angle() - Calculate the opening angle for the current node from 
 *			  the given position.
 * @current:	Pointer to the node in question.
 * @pos: 	Position of the reference point.
 *		 
 * Approximates the opening angle theta as the fraction l/r, where l is the 
 * length of the current node and r is the distance from the reference point to 
 * the center of mass of the node.
 *
 * Returns: The opening angle.
 */
double get_opening_angle(node *current, double pos[3])
{
	double r2 = 0;

	for(int j = 0; j < 3; j++) {
		r2 += (current->cm[j] - pos[j]) * (current->cm[j] - pos[j]);
	}
	
	return current->len / (sqrt(r2) + 1.0e-35);
}

/**
 * walk_tree() - Calculate the acceleration at a reference point, resulting from 
 *		  the particles in the given node. 
 * @current: 	Pointer to the node in question.
 * @pos: 	Position of the reference point.
 * @acc:	Total acceleration at the reference point.
 *
 * Walk the tree and recursively sum up all contributions to the acceleration at
 * the reference point, due to the particles contained in the current node. 
 *
 * Returns: Nothing.
 */
void walk_tree(node *current, double pos[3], double acc[3])
{
	int n;
	double theta;

	/* Only do something if there is mass in this branch of the tree (i.e. if it is not empty). */
	if(current->mass) {
		
		theta = get_opening_angle(current, pos);

		/* If the node is seen under a small enough angle or contains a single particle,
		 * we take its multipole expansion, and we're done for this branch.
		 * NOTE: Avoid self-attraction of a particle. */
		if (theta < opening_threshold || current->p) {
			
			/**/ 
			
			
			
			acc[0] += 
			acc[1] += 
			acc[2] += 
			
			/*
			 * ..... TO BE FILLED IN ....
			 *
			 *     acc[0] += ....
			 *     acc[1] += ....
			 *     acc[2] += ....
			 */
		} else {
			/* Otherwise we have to open the node and look at all children nodes in turn */

			/* Make sure that we actually have subnodes. */
			if (current->children[0]) {             
				for (n = 0; n < 8; n++) {
					walk_tree(current->children[n], pos, acc);
				}
			}
		}
	}
}



int main(int argc, char **argv)
{
	double t0, t1;			// Start/stop times.
	const int N = MAX_POINTS; 	// Number of particles in the simulation.
	
	srand48(42);			// Set a random number seed

	/* Create a random particle set, uniformly distributed in a box of unit 
	   sidelength. Total mass of the N particles is unity. */
	for (int i = 0; i < N; i++) {
		star[i].mass = 1.0 / N;
		for (int j = 0; j < 3; j++) {
			star[i].pos[j] = drand48();      
		}
	}

	/* Create an empty root node for the tree. */
	node *root = get_empty_node();

	/* Set the dimension and position of the root node. */
	root->len = 1.0;
	for(int j = 0; j < 3; j++) {
		root->center[j] = 0.5;
	}

	/* Insert the particles into the tree. */
	for (int i = 0; i < N; i++) {
		insert_particle(root, &star[i]);
	}
	
	/* Calculate the multipole moments. */
	calc_multipole_moments(root);


	/* Start the timer. */
	t0 = (double) clock();

	/* Calculate the accelerations with the tree algorithm. */
	for (int i = 0; i < N; i++) {
		// Set accelerations to zero.
		for (int j = 0; j < 3; j++) {
			star[i].acc_tree[j] = 0;
		}
		// Calculate the acceleration for the current particle.
		walk_tree(root, star[i].pos, star[i].acc_tree);
	}

	/* Stop the timer. */
	t1 = (double) clock();
	printf("\nForce calculation with tree took:        %8g sec\n", (t1 - t0) / CLOCKS_PER_SEC);

	/* Start the timer. */
	t0 = (double) clock();

	/* Calculate the accelerations with direct summation, for comparison. */
	for(int i = 0; i < N; i++) {
		
		/*
		 * ..... TO BE FILLED IN ....
		 *
		 *     star[i].acc_exact[0] = ....
		 *     star[i].acc_exact[0] = ....
		 *     star[i].acc_exact[0] = ....
   		 */
	}
	
	/* Stop the timer. */
	t1 = (double) clock();
	printf("\nCalculation with direct summation took:  %8g sec\n", (t1 - t0) / CLOCKS_PER_SEC);



	/* Now do the calculation of the mean relative error. */

	double err_sum = 0;

	/*
	* ..... TO BE FILLED IN ....
	*
	*/

	err_sum /= N;

	printf("\nAverage relative error:  		    %8g\n", err_sum);

	return 0;
}
