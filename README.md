# GPU-Prefix-Sum
Perform prefix sum (also known as inclusive scan) on a randomly generated input array (default 1 million). The GPU performs the prefix sum and the results are copied back to the host and verified to check the final answer.

Note there are 3 kernel functions:

work_efficient_inclusive_scan --> This kernel is responsible for loading elements from input array X into working array XY. Each thread loads 2 elements from X in a coalesced manner. This kernen then performs both the reduction step by producing the partial sums at the internal nodes of the tree as well as the distribution (reverse) step by traversing back up the tree to finish building the final output from the elements that have not been included in the partial sum calculation. Once prefix scan is completed, the results are copied to the output array Y to be processed by the next kernel.

work_efficient_inclusive_scan2 -->  This kernel is based off the assumption that the GRID_DIM is 1024, or exactly twice the BLOCK_DIM. This way, one thread block in this prefix-scan stage will be able to exactly handle all the block outputs from the previous prefix-scan stage. The purpose of this kernel is to perform secondary prefix-scan. Owing to the large size of the input, the first kernel is only able to perform prefix scan on an input as large as BLOCK_SIZE * 2. The final sum from each section in the previous kernel is now an individual element in the prefix-scan operation of the current kernel. The results are copied again from the working array XY to the output array Y.

work_efficient_inclusive_scan3 --> This kernel is responsible for taking the result from the previous kernel (containing the prefix-scan sum of each section of the original input) and adding them individually to each element of each section of the original input. The entire prefix-scan operation is thus complete, and the outputs are copied back to the host.

Complexity of each work_efficient_inclusive_scan kernel: Reduction step performs n/2, n/4, ... , 1 adds, which is O(n-1) or O(n). Reverse step performs 2-1, 4-1, ... , n/(2-1) add, or (n-2) - (log(n)-1) = O(n). Hence total complexity is O(n).

N.B. Block size is currently fixed at 512. This should be changed according to GPU specification.
