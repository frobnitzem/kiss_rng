#include <iostream>
#include <kiss/kiss.hpp>

template<class T, class Rng>
inline void uniform_distribution(
    sycl::queue &queue,
    sycl::buffer<T,1> &data,
    const std::uint32_t seed) noexcept {

    // execute kernel
    queue.submit([&](sycl::handler &cgh) {
      const std::uint64_t n = data.get_count();
      sycl::accessor out(data, cgh, sycl::write_only, sycl::no_init);

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
    });
}

// This example shows the easy generation of gigabytes of uniform random values
// in only a few milliseconds.

int main(int argc, char *argv[]) {
    // define the data types to be generated
    using data_t = std::uint64_t;
    using rng_t = kiss::Kiss<data_t>;

    sycl::queue queue;

    // number of values to draw
    static constexpr std::uint64_t n = 1UL << 28;

    // random seed
    static constexpr std::uint32_t seed = 42;

    // allocate memory for the result
    sycl::buffer<data_t, 1> data(n);

    double timer = omp_get_wtime();
    uniform_distribution<data_t, rng_t>(queue, data, seed);
    queue.wait();
    
    timer = omp_get_wtime() - timer;
    printf("Completed in %e seconds.\n", timer);

    {   // do something with drawn random numbers
        sycl::host_accessor data_h(data, sycl::read_only);
        for(std::uint64_t i = 0; i < 10; i++) {
            std::cout << data_h[i] << std::endl;
        }
    }
    return 0;
}
