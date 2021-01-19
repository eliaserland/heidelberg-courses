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

#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

// =========================== Global variables ==============================

static int N = 5000;      		  // Default max no. of particles used. 
static int MAX_NODES;			  // Max no. of nodes in tree.	 

static double opening_threshold = 0.8;   // Deafult Tree opening angle.       
const static double eps	 = 0.001; // Gravitational softening length. 	 
const static double G 		 = 1.0;   // Newton's gravitational const.

static bool quadrupoles = false;    	  // Use quadrupole moments. (Off by default)
static int node_counter = 0; 		  // Counter of particle-node interactions.	

// ========================== Internal datatypes ==============================

/* Let's define two types of structures, one for the particles, 
   one for the tree nodes */
typedef struct particle {
	double pos[3];
 	double mass;
	double acc_tree[3];
	double acc_exact[3];
} particle;

typedef struct node {
	double center[3];
	double len; 
	double cm[3];	
	double qm[9];
	double mass;
	struct node *children[8];
	particle *p;
} node;

// Global pointers to the tree structure, ie arrays of nodes and particles.
static node     *tree;
static particle *star;

// ========================== Internal functions ==============================

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
		/* The node contains a particle already. Need to create a new set
		   of 8 subnodes, and then move this particle to one of them */
		n = 0;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					// Get pointer to a new empty node.
					child = get_empty_node();
					
					// Insert node as a child to the current node 
					current->children[n++] = child;
					
					// Set length and geometric center of child node.
					child->len = 0.5 * current->len;
					child->center[0] = current->center[0] + 0.25*(2*i-1)*current->len;
					child->center[1] = current->center[1] + 0.25*(2*j-1)*current->len;
					child->center[2] = current->center[2] + 0.25*(2*k-1)*current->len;
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

		/* If the corresponding subnode exists, we try to insert the 
		   particle there, otherwise we know there are no subnodes in the
		   node, so we can put the particle into the current node */
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
	node *child;
	
	double  dr2, dr[3], drdr[9];  

	/* Do we have subnodes? */
	if(current->children[0]) {
		/* Yes, so let's first calculate their multipole moments. */
		for (int n = 0; n < 8; n++) {
			calc_multipole_moments(current->children[n]);
		}
		
		/* Initialize the node multipole moments to zero. */
		current->mass  = 0;
		for (int j = 0; j < 3; j++) {
			current->cm[j] = 0;
		}
		if (quadrupoles) {
			for (int i = 0; i < 9; i++) {
				current->qm[i] = 0; 
			}
		}
		
		/* Now calculate the moment of the current node from those of its children */
		//--------------------------------------------------------------
		/* Monopole moment: 
		   For each of the children, sum up the total mass from the 
		   children and the calculate the combined center of mass. */
		for (int n = 0; n < 8; n++) {
			child = current->children[n];
			current->mass += child->mass;
			for (int j = 0; j < 3; j++) {
				current->cm[j] += child->mass * child->cm[j];
			} 
		}		
		// Normalize the weighted average to get the combined center of mass.
		for (int j = 0; j < 3; j++) {
			current->cm[j] /= current->mass;
		}
		//--------------------------------------------------------------
		/* Quadrupole moment: 
		   If user has chosen so, also calculate the quadrupole-moments. */
		if (quadrupoles) {
			// For each of the chilren, get the resulting quad-moment.
			for (int n = 0; n < 8; n++) {
				// Get pointer to the child. 
				child = current->children[n];	
				
				// Find the displacement vector, and it's square.
				dr2 = 0;
				for (int d = 0; d < 3; d++) {
					dr[d] = current->cm[d] - child->cm[d]; 
					dr2 += dr[d] * dr[d];
				}
				
				// Form outer product dr * dr, and multiply by 3.
				// Subtract dr2 from the diagonal elements. 
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						drdr[3*i+j] = 3.*dr[i]*dr[j];
						if (i == j) {
							drdr[3*i+j] -= dr2;  
						} 
					}
				}
				
				// Multiply with the total mass of the subnode.
				for (int i = 0; i < 9; i++) {
					drdr[i] *= child->mass;  
				}
				
				// Include old quad-moment from subnode.
				for (int i = 0; i < 9; i++) {
					drdr[i] += child->qm[i]; 
				}
				
				/* Add total contribution to the quad-moment of 
				   the current node. */
				for (int i = 0; i < 9; i++) {
					current->qm[i] += drdr[i]; 	
				}
			}
		}
		
	
	} else {
	
		/* Do we at least have a particle? */
		if (current->p) {
			/* Yes, so let's copy this particle to the multipole 
			   moments of the node. */
			current->mass = current->p->mass;
			for (int j = 0; j < 3; j++) {
				current->cm[j] = current->p->pos[j];
			}
			/* Quadrupole moments in node with a single particle are 
			   always zero. */
			for (int i = 0; i < 9; i++) {
				current->qm[i] = 0; 
			}	
		} else {
			/* Nothing in here at all; let's initialize the multipole
			   moments to zero. */
			current->mass  = 0;
			for (int j = 0; j < 3; j++) {
				current->cm[j] = 0;
			}
			for (int i = 0; i < 9; i++) {
				current->qm[i] = 0; 
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
	double y[3], y1, y2, y3, y5, M;
	
	double yQy, yQ[3];
	double *Q; 

	/* Only do something if there is mass in this branch of the tree 
	   (i.e. if it is not empty). */
	if(current->mass) {
		
		theta = get_opening_angle(current, pos);

		/* If the node is seen under a small enough angle or contains a 
		 * single particle, we take its multipole expansion, and we're 
		 * done for this branch. NOTE: Avoid self-attraction of a particle. */
		if (theta < opening_threshold || current->p) {
			
			/* Vector y from CM of node to the reference position. */
			y2 = 0;
			for (int j = 0; j < 3; j++) {
				y[j] = pos[j] - current->cm[j]; 
				y2 += y[j] * y[j]; 
			}
			
			/* Ensure no self-interaction, ie distance to the reference 
			   point from the node CM is small (~0). */
			if (y2 > 10e-20) {
				// Increment particle-node interaction counter.
				node_counter++; 
				
				// Include the Plummer softening parameter.
				y2 += eps*eps; 
				
				// Get |y|^3 and the total mass of node.	
				y1 = sqrt(y2);
				y3 = y1 * y2; 
				M = current->mass;
				
				// Calculate acceleration due monopole moments.
				for (int i = 0; i < 3; i++) {
					acc[i] -= G*M*y[i]/y3;
				}
				
				/* If user has chosen so, also calculate acceleration
				   using quad-moments. */
				if (quadrupoles) {
					// Get |y|^5
					y5 = y3 * y2;
					 
					// Pointer to node's quadrupole moment. 
					Q = current->qm; 
					
					// Form matrix product y_i * Q_ij * y_j
					yQy = 0;
					for (int i = 0; i < 3; i++) {
						for (int j = 0; j < 3; j++) {
							yQy += y[i]*Q[3*i+j]*y[j];	
						}
					}
					
					// Form matrix product 2 * y^T * Q
					for (int i = 0; i < 3; i++) {
						yQ[i] = 0;
						for (int j = 0; j < 3; j++) {
							yQ[i] += y[j] * Q[3*i+j]; 
						}
						yQ[i] *= 2.0; 
					}
		
					// Add quad-contributions to the acceleration.
					for (int i = 0; i < 3; i++) {
						acc[i] -= G*0.5*(5.*yQy*y[i]/y2 - yQ[i])/y5; 
					}
				}
			}
		} else {
			/* Otherwise we have to open the node and look at all 
			   children nodes in turn */

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
	int opt;
	bool use_default_N = true; 
	bool use_default_t = true;
	bool machine_table = false;
	
	// Parse all arguments.	
	while((opt = getopt(argc, argv, "n:t:qm")) != -1) {  
		switch(opt) {  
            		case 'n':
            			/* Interpret the number of particles to use. */
            			N = atoi(optarg);  
                		use_default_N = false; 
				break;
			case 't': 
				/* Interpret the opening angle threshold to use. */
				opening_threshold = atof(optarg);
				use_default_t = false;
				break; 			
			case 'q': 
				/* Toggle calculation of acceleration using 
				   quadrupole moments. */ 
				quadrupoles = true; 
				break;
			case 'm':
				// Print machine-readable table of data.
				machine_table = true;
				break; 
        	}  
    	}
    	if (use_default_N && use_default_t && !quadrupoles) {
    		fprintf(stderr, "Usage:\n\t [-n] N [-t] theta [-q] [-m]\n"
    			"\t where N is an integer, giving the numer of particles, \n"
    			"\t and theta is the opening angle threshold. \n"
    			"\t Use -q to estimate acceleration up to quadrupole moments.\n"
    			"\t Use -m to output machine-readable table.\n\n"); 
    		return 1;
    	} 
	    	
	if (machine_table) {
		printf("%d %f %d ", N, opening_threshold, quadrupoles);
	} else {
	    	if (quadrupoles) {
	    		fprintf(stderr, "Using monopole & quadrupole moments.\n\n");	
	    	} else {
	    		fprintf(stderr, "Using only monopole moments.\n\n");	
	    	}
	    	if (use_default_N) {
	    		fprintf(stderr, "Using default size:        N     = %-d\n", N); 
	    	} else {
	    		fprintf(stderr, "Number of particles:       N     = %-d\n", N);
	    	}
	    	if (use_default_t) {
	    		fprintf(stderr, "Using default threshold:   theta = %-.2f\n", opening_threshold);
	    	} else {
	    		fprintf(stderr, "Opening angle threshold:   theta = %-.4f\n", opening_threshold); 
	    	}
    	}
    	
	MAX_NODES = 5*N;	// Max no. of nodes in tree.
	
	/* Dynamically allocate arrays for the tree structure. */
	tree = calloc(MAX_NODES, sizeof(node));	// Array of tree nodes.
	star = calloc(N, sizeof(particle));		// Array of particles.
	
	double t0, t1;			// Start/stop times.
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
	if (machine_table) {
		printf("%9g ", (t1 - t0) / CLOCKS_PER_SEC);
	} else {
		printf("\nForce calculation with tree:         %-9g sec\n", (t1 - t0) / CLOCKS_PER_SEC);
	}

	double dx[3], dx2, m;
	double eps2 = eps*eps;  

	/* Start the timer. */
	t0 = (double) clock();
	
	/* Calculate the accelerations with direct summation, for comparison. */	
	for (int i = 0; i < N; i++) {
		/* Set accelerations to zero. */
		for (int d = 0; d < 3; d++) {
			star[i].acc_exact[d] = 0;
		}		
		
		/* Direct summation. For each particle, sum over all particles.*/
		for (int j = 0; j < N; j++) {
			/* Ensure no self-interaction. */
			if (i != j) {
				/* Find displacement vector. */
				dx2 = 0;
				for (int d = 0; d < 3; d++) {
					dx[d] = star[i].pos[d] - star[j].pos[d]; 
					dx2 += dx[d] * dx[d]; 
				}
				m = star[j].mass; 
				
				/* Add contribution to acceleration. */
				for (int d = 0; d < 3; d++) {
					star[i].acc_exact[d] -= G*m*dx[d] / pow(dx2+eps2, 1.5);
				}	
			} 	
		}
	}
	
	/* Stop the timer. */
	t1 = (double) clock();
	if (machine_table) {
		printf("%9g ", (t1 - t0) / CLOCKS_PER_SEC);
	} else {
		printf("Calculation with direct summation:   %-9g sec\n", (t1 - t0) / CLOCKS_PER_SEC);
	}
	double err_sum = 0;
	double diff[3], diff2, dir_acc, tree_acc, direct2;  
	
	/* Calculate of the mean relative error. */
	for (int i = 0; i < N; i++) {
		diff2   = 0;
		direct2 = 0;
		for (int d = 0; d < 3; d++) {
			dir_acc  = star[i].acc_tree[d]; 
			tree_acc = star[i].acc_exact[d];
			
			diff[d] = dir_acc - tree_acc; 
			
			diff2   += diff[d] * diff[d]; 
			direct2 += dir_acc * dir_acc; 
		}
		err_sum += sqrt(diff2) / sqrt(direct2); 
	}
	err_sum /= N;

	if (machine_table) {
		printf("%g %g\n", err_sum, (double) node_counter/N);
	} else {
		printf("\nAverage relative error:  		 	   %-9g\n", err_sum);
		printf("Average particle-node interactions per particle:   %-9g\n", (double) node_counter/N);
	}
	
	free(tree);
	free(star);

	return 0;
}
