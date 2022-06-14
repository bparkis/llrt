#include <cmath>
#include "catch.hpp"
#include "process_link.hpp"
#include <algorithm>
#include <fstream>

template <typename T>
T sigmoid(const T x){
    return (1 / (1 + std::exp(-x)));
    //return (x / (1 + std::abs(x)) + 1)/2;
}

template <typename T>
T sigmoidDeriv(const T x){
    T ex = std::exp(-x);
    return ex / ((1 + ex) * (1 + ex));
}

void sigmoidTest();
