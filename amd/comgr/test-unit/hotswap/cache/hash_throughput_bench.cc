//===- hash_throughput_bench.cc - Raw hash-strategy throughput ------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Binary 1 of the cache scaling harness: measures the raw per-lookup key-
// derivation cost of each hash strategy (FNV-1a, xxHash64, SHA-256) across
// source sizes, with no cache and no threads. This isolates the hash cost that
// the cache pays on every lookup. Charges the FULL digest (the honest cost);
// the cache-mode benchmark attributes the same number.
//
// CSV: experiment,hash,source_bytes,hash_ns,ns_per_byte,throughput_mibps
//
//===----------------------------------------------------------------------===//

#include "hotswap/cache/hash-registry.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace COMGR::hotswap::bench;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
  std::vector<size_t> source_sizes{65536, 262144, 1048576, 4194304, 8388608};
  size_t iterations = 5;
  // Inner repeats per timed sample: hashing once for small sizes is too fast to
  // time reliably, so we hash `reps` times and divide. Auto-scaled per size.
  size_t min_sample_ns = 200000; // aim for >=200us per timed sample
};

// Exception-free unsigned parse (-fno-exceptions: std::stoull would terminate).
size_t parseU64(const std::string &s, size_t fallback) {
  char *endp = nullptr;
  const unsigned long long v = std::strtoull(s.c_str(), &endp, 10);
  if (endp == s.c_str() || *endp != '\0')
    return fallback;
  return static_cast<size_t>(v);
}

std::vector<size_t> parseList(const std::string &value) {
  std::vector<size_t> out;
  size_t start = 0;
  while (start < value.size()) {
    const size_t comma = value.find(',', start);
    const std::string tok = value.substr(start, comma - start);
    if (!tok.empty()) { // skip empty tokens (trailing/double commas)
      char *endp = nullptr;
      const unsigned long long v = std::strtoull(tok.c_str(), &endp, 10);
      if (endp != tok.c_str() && *endp == '\0')
        out.push_back(static_cast<size_t>(v));
    }
    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }
  return out;
}

Options parseOptions(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("--source-bytes=", 0) == 0)
      o.source_sizes = parseList(a.substr(std::strlen("--source-bytes=")));
    else if (a.rfind("--iterations=", 0) == 0)
      o.iterations = parseU64(a.substr(std::strlen("--iterations=")), o.iterations);
    else if (a == "--help") {
      std::cout << "Usage: hash_throughput_bench "
                   "[--source-bytes=65536,...] [--iterations=5]\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << a << '\n';
      std::exit(2);
    }
  }
  return o;
}

// Median nanoseconds to hash `size` bytes once, using `entry.digest` (full
// digest cost). Auto-scales inner reps so each sample is long enough to time.
double medianHashNs(const HashEntry &entry, const std::vector<uint8_t> &buf,
                    const Options &o) {
  const size_t size = buf.size();
  // Calibrate reps: one warmup hash, then grow reps until a sample exceeds
  // min_sample_ns.
  size_t reps = 1;
  for (;;) {
    const auto t0 = Clock::now();
    volatile uint8_t sink = 0;
    for (size_t r = 0; r < reps; ++r) {
      Digest d = entry.digest(buf.data(), size);
      sink ^= d[0]; // prevent the call from being optimized away
    }
    (void)sink;
    const double ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0)
            .count();
    if (ns >= static_cast<double>(o.min_sample_ns) || reps >= (1u << 20))
      break;
    reps *= 2;
  }

  std::vector<double> samples;
  samples.reserve(o.iterations);
  for (size_t it = 0; it < o.iterations; ++it) {
    const auto t0 = Clock::now();
    volatile uint8_t sink = 0;
    for (size_t r = 0; r < reps; ++r) {
      Digest d = entry.digest(buf.data(), size);
      sink ^= d[0];
    }
    (void)sink;
    const double ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0)
            .count();
    samples.push_back(ns / static_cast<double>(reps));
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  const Options o = parseOptions(argc, argv);
  std::cout << "experiment,hash,source_bytes,hash_ns,ns_per_byte,"
               "throughput_mibps\n";
  for (const size_t size : o.source_sizes) {
    std::vector<uint8_t> buf(size);
    // Fill with a non-trivial pattern so nothing degenerates.
    for (size_t i = 0; i < size; ++i)
      buf[i] = static_cast<uint8_t>((i * 1103515245u + 12345u) >> 16);
    for (const HashEntry &entry : hashRegistry()) {
      const double ns = medianHashNs(entry, buf, o);
      const double ns_per_byte = ns / static_cast<double>(size);
      const double mibps =
          (static_cast<double>(size) / (1024.0 * 1024.0)) / (ns / 1e9);
      std::cout << "hash-throughput," << entry.name << ',' << size << ','
                << static_cast<uint64_t>(ns) << ',' << ns_per_byte << ','
                << mibps << '\n';
    }
  }
  return 0;
}
