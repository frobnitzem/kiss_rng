# KISS random number generator

This is a SYCL C++ adaptation of the KISS random number
generation algorithm.  Credit for the original (CUDA) implementation
goes to [@sleepyjack](https://github.com/sleeepyjack/kiss_rng).


Its usage inside a kernel is demonstrated by `example/example.cpp`:

    #include <kiss/kiss.hpp>

    ...

    using Rng = kiss::Kiss<std::uint64_t>;

    cgh.parallel_for(sycl::nd_range({4096, 32}),
          [=](sycl::nd_item<1> it) {
        const std::uint32_t tid = it.get_global_id(0);

        // generate initial local seed per thread
        const std::uint32_t local_seed =
            kiss::hashers::MurmurHash<std::uint32_t>::hash(seed+tid);

        Rng rng{local_seed};

        // grid-stride loop
        const auto grid_stride = it.get_global_range(0);
        for(std::uint64_t i = tid; i < n; i += grid_stride) {
            // generate random element and write to output
            out[i] = rng.template next<T>();
        }
    });

The following template combinations work:

  * `Kiss<uint32_t>` . `next<uint32_t>`, `next<float>`, and `next<double>`

  * `Kiss<uint64_t>` . `next<uint64_t>`

The float and real output types return values in the half-open
interval `[0,1)`.


# Installing

There are no dependencies for this package other than a SYCL compiler.
Since it is a header-only library, you may either copy the `include/kiss`
subdirectory into an appropriate include path, or else ask cmake
to do it for you with something like,

    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=`which syclcc` \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          ..
    make install

The example can then be built with,

    mkdir build && cd build
    cmake -DCMAKE_CXX_COMPILER=`which syclcc` \
          -DCMAKE_PREFIX_PATH=/usr/local \
          ..
    make


# Copyright and License

Copyright 2022 Daniel Jünger and UT-Battelle LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
