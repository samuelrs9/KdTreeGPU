#include <stdbool.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <math.h>
#include <iostream>
#include <iomanip>

#include "src/Gpu.h"
#include "src/KdNode.h"

using std::setprecision;
using namespace std;

//Gpu *gpu;

/* Create a simple k-d tree and print its topology for inspection. */
sint main(sint argc, char **argv)
{
	// Set the defaults then parse the input arguments.
	sint numPoints = 102400;
	sint extraPoints = 0;
	sint numDimensions = 3;
	sint numThreads = 512;
	sint numBlocks = 32;
	sint searchDistance = 0.05*RAND_MAX;
	sint maximumNumberOfNodesToPrint = 5;

	for (sint i = 1; i < argc; i++) {
		if ( 0 == strcmp(argv[i], "-n") || 0 == strcmp(argv[i], "--numPoints") ) {
			numPoints = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-x") || 0 == strcmp(argv[i], "--extraPoints") ) {
			extraPoints = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-d") || 0 == strcmp(argv[i], "--numDimensions") ) {
			numDimensions = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-t") || 0 == strcmp(argv[i], "--numThreads") ) {
			numThreads = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-b") || 0 == strcmp(argv[i], "--numBlocks") ) {
			numBlocks = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-s") || 0 == strcmp(argv[i], "--searchDistance") ) {
			searchDistance = atol(argv[++i]);
			continue;
		}
		if ( 0 == strcmp(argv[i], "-p") || 0 == strcmp(argv[i], "--maximumNodesToPrint") ) {
			maximumNumberOfNodesToPrint = atol(argv[++i]);
			continue;
		}
		cout << "Unsupported command-line argument: " <<  argv[i] << endl;
		exit(1);
	}

	sint i = maximumNumberOfNodesToPrint + numDimensions + extraPoints;
	// Declare the two-dimensional coordinates array that contains (x,y,z) coordinates.
	/*
    sint coordinates[NUM_TUPLES][DIMENSIONS] = {
    {2,3,3}, {5,4,2}, {9,6,7}, {4,7,9}, {8,1,5},
    {7,2,6}, {9,4,1}, {8,4,2}, {9,7,8}, {6,3,1},
    {3,4,5}, {1,6,8}, {9,5,3}, {2,1,3}, {8,7,6},
    {5,4,2}, {6,3,1}, {8,7,6}, {9,6,7}, {2,1,3},
    {7,2,6}, {4,7,9}, {1,6,8}, {3,4,5}, {9,4,1} };
	 */
	//  gpu = new Gpu(numThreads,numBlocks,0,numDimensions);
	Gpu::gpuSetup(2, numThreads,numBlocks,numDimensions);
	if (Gpu::getNumThreads() == 0 || Gpu::getNumBlocks() == 0) {
		cout << "KdNode Tree cannot be built with " << numThreads << " threads or " << numBlocks << " blocks." << endl;
		exit(1);
	}
	cout << "Points = " << numPoints << " dimensions = " << numDimensions << ", threads = " << numThreads << ", blocks = " << numBlocks << endl;
	
	//auto max = RAND_MAX;

	srand(0);
	KdCoord (*coordinates) = new KdCoord[numPoints*numDimensions];
	for ( i = 0; i<numPoints; i++) {
		for (sint j=0; j<numDimensions; j++) {
		
			coordinates[i*numDimensions+j] = rand();
			//cout << coordinates[i*numDimensions+j] << " ";	
			//coordinates[i*numDimensions+j] = (KdCoord)rand();
			//coordinates[i*numDimensions+j] = (j==1)? (numPoints-i) : i;
			//coordinates[i*numDimensions+j] =  i;
		}
	}

	// Imprime as coordenadas de alguns pontos
	/*
	for ( i = 0; i<10; i++) {
		cout << "point " << i << ": (";
		for (sint j=0; j<numDimensions; j++) {
			cout << coordinates[i*numDimensions+j];
			if (j<numDimensions-1) { 
				cout << ", ";
			}
		}
		cout << ")" << endl;
	}*/

	// Create the k-d tree.  First copy the data to a tuple in its kdNode.
	// also null out the gt and lt references
	// create and initialize the kdNodes array
	KdNode *kdNodes = new KdNode[numPoints];
	if (kdNodes == NULL) {
		printf("Can't allocate %d kdNodes\n", numPoints);
		exit (1);
	}

	KdNode *root = KdNode::createKdTree(kdNodes, coordinates, numDimensions, numPoints);

	// Print the k-d tree "sideways" with the root at the left.
	cout << endl;

	if (searchDistance == 0){
		return 0;
	}
	TIMER_DECLARATION();

	// read the KdTree back from GPU
	Gpu::getKdTreeResults( kdNodes,  coordinates, numPoints, numDimensions);
#define VERIFY_ON_HOST
#ifdef VERIFY_ON_HOST
	sint numberOfNodes = root->verifyKdTree( kdNodes, coordinates, numDimensions, 0);
	cout <<  "Number of nodes on host = " << numberOfNodes << endl;
#endif

	TIMER_START();	

	for (sint i = 0; i < numPoints; i++) {
	
		//cout << "----------- point " << i << "------------\n";
		//Search the k-d tree for the k-d nodes that lie within the cutoff distance of the first tuple.
		
		KdCoord* query = (KdCoord *)malloc(numDimensions * sizeof(KdCoord));
		for (sint j = 0; j < numDimensions; j++) {
			query[j] = coordinates[i*numDimensions+j];
		}

		// Imprime ponto de consulta
		/*
		cout << "point query " << i << ": (";
		for (sint j=0; j<numDimensions; j++) {
			cout << query[j];
			if (j<numDimensions-1) { 
				cout << ", ";
			}
		}
		cout << ")" << endl;
		*/
			

		// KdCoord (*query) = new KdCoord[numPoints*numDimensions];
		// for ( i = 0; i<numPoints; i++) {
		// 	for (sint j=0; j<numDimensions; j++) {
		// 		query[i*numDimensions+j] = (KdCoord)rand();
		// 	}
		// }		

		list<KdNode> kdList = root->searchKdTree(kdNodes, coordinates, query, searchDistance, numDimensions, 0);

		//cout << " --> " << kdList.size() << " nodes within " << searchDistance << " units of ";

		//KdNode::printTuple(query, numDimensions);

		//cout << " in all dimensions." << endl << endl;
		/*
		if (kdList.size() != 0) {
			cout << " --> List of k-d nodes within " << searchDistance << "-unit search distance follows:" << endl;
			cout << " ----> ";
			list<KdNode>::iterator it;
			for (it = kdList.begin(); it != kdList.end(); it++) {
				KdNode::printTupleOriginal(coordinates+it->getTuple()*numDimensions, numDimensions);
				cout << " ";
			}
			cout << endl << endl;
		}*/
	}
	TIMER_STOP(double searchTime);
	cout << "searchTime = " << fixed << setprecision(4) << searchTime << " seconds" << endl << endl;
	return 0;
}
