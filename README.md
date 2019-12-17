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
- Loop unrolling and multiple accumulators
- PThreads

### Future (after Tuesday)
- Paper/analysis

### Extra
- AVX (512-bit registers)
- using icc compiler for optimization
- Other MMM algorithms (Canon's algorithm, etc.)
- Trying ATLAS
- Compare professional MMM libraries (CPU vs GPU)


## Roofline plot ideas:

I found this roofline plot for the Tesla K-40:
https://www.researchgate.net/figure/Roofline-model-analysis-for-Two-Phase-RP-kernel-on-NVIDIA-Tesla-K40-GPU_fig1_313453394

Maybe we calculate the AI of our various programs and plot points on a roofline plot from the internet, like this one?

