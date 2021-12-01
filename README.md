# Advent of GPU Code 2021

This repo contains solutions to the 2021 [Advent of Code](https://adventofcode.com/) written for the [GPU using Python](https://numba.pydata.org/numba-doc/dev/cuda/overview.html).

## FAQ

### General

#### What is Advent of Code?

[Advent of Code](https://adventofcode.com/) is a series of computer science problems released each day throughout December.

Participants have to write code to solve each problem.

#### Who are you?

My name is [Jacob Tomlinson](https://twitter.com/_jacobtomlinson). I work for NVIDIA maintaining open source Python projects including [Dask](https://dask.org/) and [RAPIDS](https://rapids.ai/).

#### What kind of GPU are you using?

I generally SSH to a machine with a pair of NVIDIA Quadro RTX 8000s.

If you want to try it for yourself you'll need a Pascal series GPU or better. Check out the RAPIDS list of [supported GPUs](https://medium.com/dropout-analytics/which-gpus-work-with-rapids-ai-f562ef29c75f).

#### Is there a better way to solve that problem?

Quite possibly!

My goal here is to increase my skill in GPU development in Python, rather than solve the problems perfectly. 

I also do not have a formal computer science education, so I find things like Advent of Code really useful for building my computer science fundamentals. 

If you want to give me pointers and tips in the live stream chat then please do!

#### Can everything be done on the GPU?

Not all problems map onto something that can run in parallel, therefore I am not expecting to solve every challenge on the GPU. Instead I am aiming to do *as much as possible* on the GPU. Code that can be parallelised on the GPU will be faster than it's CPU counterpart. But if something just doesn't make sense to be implemented on the GPU I'll skip over it.


### Technical

#### Which Python libraries are you using?

I plan to mostly use [Numba's CUDA support](https://numba.pydata.org/numba-doc/dev/cuda/overview.html). This allows me to write Python and execute it on the GPU.

I may also use [CuPy](https://cupy.dev/), [cuDF](https://github.com/rapidsai/cudf) and other packages in the [RAPIDS ecosystem](https://github.com/rapidsai).

#### What Python environment are you using?

I am using the latest [RAPIDS Docker image](https://hub.docker.com/r/rapidsai/rapidsai/), which contains the RAPIDS packages plus some extras such as Jupyter Lab.

You can find instructions on installing RAPIDS via Docker or Conda [here](https://rapids.ai/start.html#get-rapids).

#### What is a kernel?

A CUDA kernel is a fancy name for a function which runs on the GPU.

In addition to executing on the GPU kernels also differ from regular functions in a few ways:

- When you call a kernel it will be called many times in parallel threads. Each thread will have a unique index so you can have each one do something slightly different. Often you pass an array to a kernel and each thread will read one or more items from the array based on it's thread index.
- Kernels cannot return values. Instead it is common to also pass the kernel an output array and have the kernel place its return value into the array at its corresponding thread index.
- The number of parallel threads is configurable via the thread hierarchy (threads, blocks and grids). Ultimately this is limited by your hardware.

#### What is a thread hierarchy (threads, blocks and grids)?

When you call a CUDA kernel it runs n times in n threads.

Threads are grouped into blocks, and the maximum number of threads per block is 1024. 

You can have any number of blocks and these are grouped together into a grid.

All threads in a block have accessed to some shared memory, and can synchronise which means they can wait for them all to reach a specific point in the function before continuing.

When we call our kernel we need to decide how many times it should be run. To do this we pass the thread and block sizes. So if we have an array with 1m items which need to be processed we could make our thread size 1000 and our block size 1000. The number of blocks that a GPU can process at any one time varies by model, but all blocks will be queued when you call your kernel.

Check out [this post](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) for more information.