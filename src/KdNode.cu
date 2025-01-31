#include <stdbool.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <math.h>
#include <iostream>
#include <iomanip>

#include "Gpu.h"
#include "KdNode.h"

using std::setprecision;
using namespace std;

/*
 * The superKeyCompare method compares two sint arrays in all k dimensions,
 * and uses the sorting or partition coordinate as the most significant dimension.
 *
 * calling parameters:
 *
 * a - a int*
 * b - a int*
 * p - the most significant dimension
 * dim - the number of dimensions
 *
 * returns: +1, 0 or -1 as the result of comparing two sint arrays
 */
 KdCoord KdNode::superKeyCompare(const KdCoord *a, const KdCoord *b, const sint p, const sint dim)
{
	KdCoord diff = 0;
	for (sint i = 0; i < dim; i++) {
		sint r = i + p;
		r = (r < dim) ? r : r - dim;
		diff = a[r] - b[r];
		if (diff != 0) {
			break;
		}
	}
	return diff;
}

/*
 * Walk the k-d tree and check that the children of a node are in the correct branch of that node.
 *
 * calling parameters:
 *
 * dim - the number of dimensions
 * depth - the depth in the k-d tree
 *
 * returns: a count of the number of kdNodes in the k-d tree
 */
sint KdNode::verifyKdTree( const KdNode kdNodes[], const KdCoord coords[], const sint dim, const sint depth) const
{
	sint count = 1 ;

	// The partition cycles as x, y, z, w...
	sint axis = depth % dim;

	if (ltChild != -1) {
		if (superKeyCompare(coords+kdNodes[ltChild].tuple*dim, coords+tuple*dim, axis, dim) >= 0) {
			cout << "At Depth " << depth << " LT child is > node on axis " << axis << "!" << endl;
			printTuple(coords+tuple*dim, dim);
			cout << " < [" << ltChild << "]";
			printTuple(coords+kdNodes[ltChild].tuple*dim, dim);
			cout << endl;
			exit(1);
		}
		count += kdNodes[ltChild].verifyKdTree(kdNodes, coords, dim, depth + 1);
	}
	if (gtChild != -1) {
		if (superKeyCompare(coords+kdNodes[gtChild].tuple*dim, coords+tuple*dim, axis, dim) <= 0) {
			cout << "At Depth " << depth << " GT child is < node on axis " << axis << "!" << endl;
			printTuple(coords+tuple*dim, dim);
			cout << " > [" << gtChild << "]";
			printTuple(coords+kdNodes[gtChild].tuple*dim, dim);
			cout << endl;
			exit(1);
		}
		count += kdNodes[gtChild].verifyKdTree(kdNodes, coords, dim, depth + 1);
	}
	return count;
}

/*
 * The createKdTree function performs the necessary initialization then calls the buildKdTree function.
 *
 * calling parameters:
 *
 * coordinates - a vector<int*> of references to each of the (x, y, z, w...) tuples
 * numDimensions - the number of dimensions
 *
 * returns: a KdNode pointer to the root of the k-d tree
 */
KdNode* KdNode::createKdTree(KdNode kdNodes[], KdCoord coordinates[],  const sint numDimensions, const sint numTuples)
{
	TIMER_DECLARATION();

	TIMER_START();
	Gpu::initializeKdNodesArray(coordinates, numTuples, numDimensions);
	cudaDeviceSynchronize();
	TIMER_STOP (double initTime);

	// Sort the reference array using multiple threads if possible.

	TIMER_START();
	sint end[numDimensions]; // Array used to collect results of the remove duplicates function
	Gpu::mergeSort(end, numTuples, numDimensions);
	TIMER_STOP (double sortTime);

	// Check that the same number of references was removed from each reference array.
	for (sint i = 0; i < numDimensions-1; i++) {
		if (end[i] < 0) {
			cout << "removeDuplicates failed on dimension " << i << endl;
			cout << end[0];
			for (sint k = 1;  k<numDimensions; k++) cout << ", " << end[k] ;
			cout << endl;
			exit(1);
		}
		for (sint j = i + 1; j < numDimensions; j++) {
			if ( end[i] != end[j] ) {
				cout << "Duplicate removal error" << endl;
				cout << end[0];
				for (sint k = 1;  k<numDimensions; k++) cout << ", " << end[k] ;
				cout << endl;
				exit(1);
			}
		}
	}
	cout << numTuples-end[0] << " equal nodes removed. "<< endl;

	// Build the k-d tree.
	TIMER_START();
	//  refIdx_t root = gpu->startBuildKdTree(kdNodes, end[0], numDimensions);
	refIdx_t root = Gpu::buildKdTree(kdNodes, end[0], numDimensions);
	TIMER_STOP (double kdTime);

	// Verify the k-d tree and report the number of KdNodes.
	TIMER_START();
	sint numberOfNodes = Gpu::verifyKdTree(kdNodes, root, numDimensions, numTuples);
	// sint numberOfNodes = kdNodes[root].verifyKdTree( kdNodes, coordinates, numDimensions, 0);
	cout <<  "Number of nodes = " << numberOfNodes << endl;
	TIMER_STOP (double verifyTime);

	cout << "totalTime = " << fixed << setprecision(4) << initTime + sortTime + kdTime + verifyTime
			<< "  initTime = " << initTime << "  sortTime + removeDuplicatesTime = " << sortTime
			<< "  kdTime = " << kdTime << "  verifyTime = " << verifyTime << endl << endl;

	// Return the pointer to the root of the k-d tree.
	return &kdNodes[root];
}

/*
 * Search the k-d tree and find the KdNodes that lie within a cutoff distance
 * from a query node in all k dimensions.
 *
 * calling parameters:
 *
 * query - the query point
 * cut - the cutoff distance
 * dim - the number of dimensions
 * depth - the depth in the k-d tree
 *
 * returns: a list that contains the kdNodes that lie within the cutoff distance of the query node
 */
list<KdNode> KdNode::searchKdTree(const KdNode kdNodes[], const KdCoord coords[], const KdCoord* query, const KdCoord cut,
		const sint dim, const sint depth) const {

	// The partition cycles as x, y, z, w...
	sint axis = depth % dim;

	// If the distance from the query node to the k-d node is within the cutoff distance
	// in all k dimensions, add the k-d node to a list.
	list<KdNode> result;
	bool inside = true;
	for (sint i = 0; i < dim; i++) {
		if (abs(query[i] - coords[tuple*dim+i]) > cut) {
			inside = false;
			break;
		}
	}
	if (inside) {
		result.push_back(*this); // The push_back function expects a KdNode for a call by reference.
	}

	// Search the < branch of the k-d tree if the partition coordinate of the query point minus
	// the cutoff distance is <= the partition coordinate of the k-d node.  The < branch must be
	// searched when the cutoff distance equals the partition coordinate because the super key
	// may assign a point to either branch of the tree if the sorting or partition coordinate,
	// which forms the most significant portion of the super key, shows equality.
	if ( ltChild != -1 && (query[axis] - cut) <= coords[tuple*dim+axis] ) {
		list<KdNode> ltResult = kdNodes[ltChild].searchKdTree(kdNodes, coords, query, cut, dim, depth + 1);
		result.splice(result.end(), ltResult); // Can't substitute searchKdTree(...) for ltResult.
	}

	// Search the > branch of the k-d tree if the partition coordinate of the query point plus
	// the cutoff distance is >= the partition coordinate of the k-d node.  The < branch must be
	// searched when the cutoff distance equals the partition coordinate because the super key
	// may assign a point to either branch of the tree if the sorting or partition coordinate,
	// which forms the most significant portion of the super key, shows equality.
	if ( gtChild != -1 && (query[axis] + cut) >= coords[tuple*dim+axis] ) {
		list<KdNode> gtResult = kdNodes[gtChild].searchKdTree(kdNodes, coords, query, cut, dim, depth + 1);
		result.splice(result.end(), gtResult); // Can't substitute searchKdTree(...) for gtResult.
	}

	return result;
}

/*
 * Print one tuple.
 *
 * calling parameters:
 *
 * tuple - the tuple to print
 * dim - the number of dimensions
 */
void KdNode::printTuple(const KdCoord* tuple, const sint dim)
{
	cout << "(";
	for (sint j=0; j<dim; j++) {
		cout << tuple[j];
		if (j<dim-1) { 
			cout << ", ";
		}
	}
	cout << ")" << endl;
}

void KdNode::printTupleOriginal(const KdCoord* tuple, const sint dim)
{
	cout << "(" << tuple[dim] << ",";
	for (sint i=1; i<dim-1; i++) cout << tuple[i] << ",";
	cout << tuple[dim-1] << ")";
}

/*
 * Print the k-d tree "sideways" with the root at the ltChild.
 *
 * calling parameters:
 *
 * dim - the number of dimensions
 * depth - the depth in the k-d tree
 */
void KdNode::printKdTree(KdNode kdNodes[], const KdCoord coords[], const sint dim, const sint depth) const
{
	if (gtChild != -1) {
		kdNodes[gtChild].printKdTree(kdNodes, coords, dim, depth+1);
	}
	for (sint i=0; i<depth; i++) cout << "       ";
	printTuple(coords+tuple*dim, dim);
	cout << endl;
	if (ltChild != -1) {
		kdNodes[ltChild].printKdTree(kdNodes, coords, dim, depth+1);
	}
}
