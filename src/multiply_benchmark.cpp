#include <benchmark/benchmark.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <volk/volk.h>
#include <volk/volk_alloc.hh>
#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <complex>
#include <limits>
#include <random>
#include <vector>
#include <span>

int maxPrimeFactors(int value)
{
    // https://www.geeksforgeeks.org/find-largest-prime-factor-number/
    int maxPrime = -1;

    while (value % 2 == 0)
    {
        maxPrime = 2;
        value /= 2;
    }

    for (int i = 3; i <= sqrt(value); i += 2)
    {
        while (value % i == 0)
        {
            maxPrime = i;
            value /= i;
        }
    }

    if (value > 2)
        maxPrime = value;

    return maxPrime;
}

std::vector<uint8_t> initialize_random_bit_vector(const unsigned size)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_int_distribution<uint8_t> dist{0, 255};
    auto gen = [&dist, &mersenne_engine]()
    { return dist(mersenne_engine); };

    std::vector<uint8_t> vec(size);
    std::generate(begin(vec), end(vec), gen);
    return vec;
}

std::vector<float> initialize_random_vector(const unsigned size,
                                            const float variance = 10.0)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::normal_distribution<float> dist{0, variance};
    auto gen = [&dist, &mersenne_engine]()
    { return dist(mersenne_engine); };

    std::vector<float> vec(size);
    std::generate(begin(vec), end(vec), gen);
    return vec;
}

std::vector<float> initialize_positive_random_vector(const unsigned size,
                                                     const float variance = 10.0)
{
    auto vec = initialize_random_vector(size, variance);
    for (auto &v : vec)
    {
        v = std::abs(v);
    }
    return vec;
}

std::vector<std::complex<float>>
initialize_random_complex_vector(const unsigned size, const float variance = 10.0)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::normal_distribution<float> dist{0, variance};
    auto gen = [&dist, &mersenne_engine]()
    {
        return std::complex<float>(dist(mersenne_engine), dist(mersenne_engine));
    };

    std::vector<std::complex<float>> vec(size);
    std::generate(vec.begin(), vec.end(), gen);
    return vec;
}

std::vector<int>
initialize_subcarrier_map(int subcarriers, int active_subcarriers, bool is_dc_free)
{
    std::vector<int> smap;
    if (subcarriers == active_subcarriers)
    {
        smap.resize(subcarriers);
        std::iota(smap.begin(), smap.end(), 0);
    }
    else if (active_subcarriers < subcarriers)
    {
        smap.resize(active_subcarriers);
        const int start_sc = is_dc_free ? 1 : 0;
        const int half_active = active_subcarriers / 2;
        std::iota(smap.begin(), smap.begin() + half_active, start_sc);
        std::iota(smap.begin() + half_active, smap.end(), subcarriers - half_active);
    }
    else
    {
        throw std::runtime_error("Invalid parameters!");
    }

    return smap;
}

template <typename T>
auto multiply_vecs_by_value(std::vector<T> vec0, std::vector<T> vec1)
{
    const auto loopsize = std::min({vec0.size(), vec1.size()});
    auto p0 = vec0.data();
    auto p1 = vec1.data();
    for (int i = 0; i < loopsize; ++i)
    {
        *p0++ *= *p1++;
    }

    return vec0;
}

template <typename T>
auto multiply_vecs_by_cref(const std::vector<T> &vec0, const std::vector<T> &vec1)
{
    const auto loopsize = std::min({vec0.size(), vec1.size()});
    auto result = std::vector<T>(loopsize);
    auto p0 = vec0.data();
    auto p1 = vec1.data();
    auto pr = result.data();
    for (int i = 0; i < loopsize; ++i)
    {
        *pr++ = *p0++ * *p1++;
    }

    return result;
}

template <typename T>
auto multiply_vecs_by_rval(std::vector<T> &&vec0, const std::vector<T> &&vec1)
{
    const auto loopsize = std::min({vec0.size(), vec1.size()});
    // auto result = std::vector<T>(loopsize);
    auto p0 = vec0.data();
    auto p1 = vec1.data();
    // auto pr = result.data();
    for (int i = 0; i < loopsize; ++i)
    {
        *p0++ *= *p1++;
    }

    return vec0;
}

template <typename T>
void multiply_vecs_by_return_param(std::span<T> result, std::span<const T> vec0, std::span<const T> vec1)
{
    // const auto loopsize = result.size();
    const auto loopsize = std::min({result.size(), vec0.size(), vec1.size()});
    auto p0 = std::assume_aligned<32>(vec0.data());
    auto p1 = std::assume_aligned<32>(vec1.data());
    auto pr = std::assume_aligned<32>(result.data());
// auto p0 = vec0.data();
// auto p1 = vec1.data();
// auto pr = result.data();
// for (int i = 0; i < loopsize; ++i){
//     *pr++ = *p0++ * *p1++;
// }
#pragma omp simd aligned(pr, p0, p1 : 32)
    for (int i = 0; i < loopsize; i++)
    {
        *pr++ = *p0++ * *p1++;
    }
}

template <typename T>
void multiply_vecs_by_return_pointer(T *result, const T *vec0, const T *vec1, const unsigned size)
{
    for (unsigned i = 0; i < size; i++)
    {
        *result++ = *vec0++ * *vec1++;
    }
}

template <typename T>
void multiply_vecs_by_return_pointer_omp(T *result, const T *vec0, const T *vec1, const unsigned size)
{
#pragma omp simd aligned(result, vec0, vec1 : 32)
    for (unsigned i = 0; i < size; i++)
    {
        *result++ = *vec0++ * *vec1++;
    }
}

static void BM_multiply_by_value(benchmark::State &state)
{
    const auto size = state.range(0);

    auto vec0 = initialize_random_vector(size);
    auto vec1 = initialize_random_vector(size);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec0);
        benchmark::DoNotOptimize(vec1);
        auto result = multiply_vecs_by_value(vec0, vec1);
        benchmark::DoNotOptimize(result);
    }

    state.counters["SymbolThr"] =
        benchmark::Counter(size * state.iterations(),
                           benchmark::Counter::kIsRate,
                           benchmark::Counter::OneK::kIs1024);
}

BENCHMARK(BM_multiply_by_value)->RangeMultiplier(2)->Range(8, 2 << 16);

static void BM_multiply_volk(benchmark::State &state)
{
    const auto size = state.range(0);

    auto v0 = initialize_random_vector(size);
    auto v1 = initialize_random_vector(size);
    auto vec0 = volk::vector<float>(v0.begin(), v0.end());
    auto vec1 = volk::vector<float>(v1.begin(), v1.end());
    auto result = volk::vector<float>(size);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec0);
        benchmark::DoNotOptimize(vec1);
        volk_32f_x2_multiply_32f_a(result.data(), vec0.data(), vec1.data(), size);
        // auto result = multiply_vecs_by_value(vec0, vec1);
        benchmark::DoNotOptimize(result);
    }

    state.counters["SymbolThr"] =
        benchmark::Counter(size * state.iterations(),
                           benchmark::Counter::kIsRate,
                           benchmark::Counter::OneK::kIs1024);
}

BENCHMARK(BM_multiply_volk)->RangeMultiplier(2)->Range(8, 2 << 16);

static void BM_multiply_by_cref(benchmark::State &state)
{
    const auto size = state.range(0);

    auto vec0 = initialize_random_vector(size);
    auto vec1 = initialize_random_vector(size);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec0);
        benchmark::DoNotOptimize(vec1);
        auto result = multiply_vecs_by_cref(vec0, vec1);
        benchmark::DoNotOptimize(result);
    }

    state.counters["SymbolThr"] =
        benchmark::Counter(size * state.iterations(),
                           benchmark::Counter::kIsRate,
                           benchmark::Counter::OneK::kIs1024);
}

BENCHMARK(BM_multiply_by_cref)->RangeMultiplier(2)->Range(8, 2 << 16);

static void BM_multiply_by_rval(benchmark::State &state)
{
    const auto size = state.range(0);

    const auto ref0 = initialize_random_vector(size);
    const auto ref1 = initialize_random_vector(size);

    for (auto _ : state)
    {
        state.PauseTiming(); // Stop timers. They will not count until they are resumed.
        auto vec0 = std::vector<float>(ref0.begin(), ref0.end());
        auto vec1 = std::vector<float>(ref1.begin(), ref1.end());
        state.ResumeTiming(); // And resume timers. They are now counting again.
        benchmark::DoNotOptimize(vec0);
        benchmark::DoNotOptimize(vec1);
        auto result = multiply_vecs_by_rval(std::move(vec0), std::move(vec1));
        benchmark::DoNotOptimize(result);
    }

    state.counters["SymbolThr"] =
        benchmark::Counter(size * state.iterations(),
                           benchmark::Counter::kIsRate,
                           benchmark::Counter::OneK::kIs1024);
}

BENCHMARK(BM_multiply_by_rval)->RangeMultiplier(2)->Range(8, 2 << 16);

static void BM_multiply_by_return_param(benchmark::State &state)
{
    const auto size = state.range(0);

    // const auto vec0 = initialize_random_vector(size);
    // const auto vec1 = initialize_random_vector(size);
    // auto result = initialize_random_vector(size);

    auto v0 = initialize_random_vector(size);
    auto v1 = initialize_random_vector(size);
    auto vec0 = volk::vector<float>(v0.begin(), v0.end());
    auto vec1 = volk::vector<float>(v1.begin(), v1.end());
    auto result = volk::vector<float>(size);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec0);
        benchmark::DoNotOptimize(vec1);
        multiply_vecs_by_return_param<float>(result, vec0, vec1);
        benchmark::DoNotOptimize(result);
    }

    state.counters["SymbolThr"] =
        benchmark::Counter(size * state.iterations(),
                           benchmark::Counter::kIsRate,
                           benchmark::Counter::OneK::kIs1024);
}

BENCHMARK(BM_multiply_by_return_param)->RangeMultiplier(2)->Range(8, 2 << 16);

static void BM_multiply_by_return_pointer(benchmark::State &state)
{
    const auto size = state.range(0);

    // const auto vec0 = initialize_random_vector(size);
    // const auto vec1 = initialize_random_vector(size);
    // auto result = initialize_random_vector(size);

    auto v0 = initialize_random_vector(size);
    auto v1 = initialize_random_vector(size);
    auto vec0 = volk::vector<float>(v0.begin(), v0.end());
    auto vec1 = volk::vector<float>(v1.begin(), v1.end());
    auto result = volk::vector<float>(size);

    auto p0 = vec0.data();
    auto p1 = vec1.data();
    auto pr = result.data();

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec0);
        benchmark::DoNotOptimize(vec1);
        multiply_vecs_by_return_pointer(pr, p0, p1, size);
        benchmark::DoNotOptimize(result);
    }

    state.counters["SymbolThr"] =
        benchmark::Counter(size * state.iterations(),
                           benchmark::Counter::kIsRate,
                           benchmark::Counter::OneK::kIs1024);
}

BENCHMARK(BM_multiply_by_return_pointer)->RangeMultiplier(2)->Range(8, 2 << 16);

static void BM_multiply_by_return_pointer_omp(benchmark::State &state)
{
    const auto size = state.range(0);

    // const auto vec0 = initialize_random_vector(size);
    // const auto vec1 = initialize_random_vector(size);
    // auto result = initialize_random_vector(size);

    auto v0 = initialize_random_vector(size);
    auto v1 = initialize_random_vector(size);
    auto vec0 = volk::vector<float>(v0.begin(), v0.end());
    auto vec1 = volk::vector<float>(v1.begin(), v1.end());
    auto result = volk::vector<float>(size);

    auto p0 = vec0.data();
    auto p1 = vec1.data();
    auto pr = result.data();

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vec0);
        benchmark::DoNotOptimize(vec1);
        multiply_vecs_by_return_pointer_omp(pr, p0, p1, size);
        benchmark::DoNotOptimize(result);
    }

    state.counters["SymbolThr"] =
        benchmark::Counter(size * state.iterations(),
                           benchmark::Counter::kIsRate,
                           benchmark::Counter::OneK::kIs1024);
}

BENCHMARK(BM_multiply_by_return_pointer_omp)->RangeMultiplier(2)->Range(8, 2 << 16);

BENCHMARK_MAIN();