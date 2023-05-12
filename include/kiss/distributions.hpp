#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>
#include "hashers.hpp"

namespace kiss {
namespace dist {

template <typename T>
class uniform {
  public:
    /** Returns one random number in the interval [0,1)
     */
    template <typename Generator>
    T operator()(Generator &rng) {
        return rng.template next<T>();
    }
};

template <typename T>
class normal {
    bool avail = false;
    T last;
  public:

    /** Generates two random numbers at once
     *  using Box-Muller transform. Stores results
     *  in a, b.
     */
    template <typename Generator>
    void gen2(Generator &rng, T &a, T &b) {
        using Ptr = sycl::multi_ptr<T,
                        sycl::access::address_space::private_space>;
        uniform<T> U;
        T x   = U(rng);
        T rho = sycl::sqrt(- 2.0 * sycl::log1p(-x));
        T th  = U(rng) * 2*M_PI;
        a = rho*sycl::sincos(th, Ptr(&x));
        b = rho*x;
    }

    /** Returns one random number following a standard
     *  normal distribution.
     */
    template <typename Generator>
    T operator()(Generator &rng) {
        if(avail) {
            avail = false;
            return last;
        }
        avail = true;
        T b;
        gen2(rng, last, b);
        return b;
    }
};

}}
