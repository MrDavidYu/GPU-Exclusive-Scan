# GPU-Exclusive-Scan
Perform exclusive scan on a randomly generated input array (default 1 million). The GPU performs the exlusive scan and the results are copied back to the host and verified to check the final answer. Exclusive scan is the opposite of inclusive scan (or prefix sum) in that instead of calculating the total sum of all previous terms including the current term at a given position k (for all k) for an input array A_1, A_2, ..., A_k, ... A_n, we are interested in the total sum of all the terms excluding the current term.

Note there are 3 kernel functions:

work_efficient_inclusive_scan --> This kernel is responsible for loading elements from input array X into working array XY. Each thread loads 2 elements from X in a coalesced manner. This kernen then performs both the reduction step by producing the partial sums at the internal nodes of the tree as well as the distribution (reverse) step by traversing back up the tree to finish building the final output from the elements that have not been included in the partial sum calculation. Once exclusive scan is completed, the results are copied to the output array Y to be processed by the next kernel.

work_efficient_inclusive_scan2 -->  This kernel is based off the assumption that the GRID_DIM is 1024, or exactly twice the BLOCK_DIM. This way, one thread block in this exclusive scan stage will be able to exactly handle all the block outputs from the previous exclusive scan stage. The purpose of this kernel is to perform secondary exclusive scan. Owing to the large size of the input, the first kernel is only able to perform exclusive scan on an input as large as BLOCK_SIZE * 2. The final sum from each section in the previous kernel is now an individual element in the exlusive scan operation of the current kernel. The results are copied again from the working array XY to the output array Y.

work_efficient_inclusive_scan3 --> This kernel is responsible for taking the result from the previous kernel (containing the exclusive scan sum of each section of the original input) and adding them individually to each element of each section of the original input. The entire exclusive scan operation is thus complete, and the outputs are copied back to the host.

Complexity of each work_efficient_inclusive_scan kernel: Reduction step performs n/2, n/4, ... , 1 adds, which is O(n-1) or O(n). Reverse step performs 2-1, 4-1, ... , n/(2-1) add, or (n-2) - (log(n)-1) = O(n). Hence total complexity is O(n).

N.B. Block size is currently fixed at 512. This should be changed according to GPU specification.


# Makefile
In order to make, please ensure the SM architecture matches your GPU. 
