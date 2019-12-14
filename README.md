# EC527_MMM_Final_Project
Optimizing MMM with scalar and vector techniques alongside multithreading. Final project for EC527

### Done
- Basic shell script
- Base serial code
- Base GPU code
- Slideshow
- Vectorization
- AVX (256)
- OMP

### To do
- More shell automation (block size, register blocks, GPU grid/block sizes - 16 threads^2 per block)
- Don't use powers of 2 for matrix sizes, will be very poorly optimized in comparison to non powers of 2.
- Loop unrolling and multiple accumulators
- PThreads

### Future
- Paper/analysis
- AVX (512-bit registers)
- using icc compiler for optimization

### Extra
- Other MMM algorithms (Canon's algorithm, etc.)
- Trying ATLAS
- Compare professional MMM libraries (CPU vs GPU)


## Idea for vector code
I found this: https://github.com/richardstartin/cppavxbenchmarks/blob/master/mmul.cpp

We may be able to adapt it to work with C instead of C++. I tried putting it directly into the code we already have but I'm getting segfaults and I don't know why, so I'm thinking about starting from the ground up with this as a basis. The C++ version compiles and runs lighting fast, so I'd love to get it working in our code.

Original blog post: http://richardstartin.uk/mmm-revisited/
