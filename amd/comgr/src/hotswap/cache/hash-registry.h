//===- hash-registry.h - Hash strategies for cache scaling analysis -------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Uniform interface over the three content-hash strategies the cache could use
// for key derivation: FNV-1a and xxHash64 (64-bit) and SHA-256 (256-bit). Used
// only by the scaling-analysis benchmarks -- it lets us measure per-lookup key
// cost and swap the cache's key derivation without touching production code.
//
// The 64-bit hashes are vendored verbatim from the shipping implementations so
// the benchmark measures the real code:
//   FnvHash    <- ROCr hotswap.cpp        (disk-key FNV-1a)
//   HashContent<- ROCr hotswap_cache.cpp  (in-memory xxHash64 bucket hash)
// SHA-256 uses llvm::SHA256 (what the COMGR disk cache's sha256Hex is built on).
//
// Digest-width reconciliation: a lookup's honest cost is the FULL digest, but a
// bucket index only needs 64 bits. So we expose BOTH:
//   DigestFn -- writes the full digest (32 bytes; the 64-bit hashes zero-pad).
//               This is the per-lookup key-derivation cost the bench charges.
//   BucketFn -- returns a u64 (SHA-256 folded to its first 8 bytes). This is
//               what actually feeds a bucketed cache.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HASH_REGISTRY_H
#define HOTSWAP_TRANSPILER_HASH_REGISTRY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/SHA256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

namespace COMGR::hotswap::bench {

// ---- FNV-1a (vendored from ROCr hotswap.cpp) ----------------------------

inline uint64_t fnv1a(const void *data, size_t size) {
  constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffset;
  const auto *bytes = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= kFnvPrime;
  }
  return hash;
}

// ---- xxHash64 (vendored from ROCr hotswap_cache.cpp HashContent) --------

namespace detail {
constexpr uint64_t kHashPrime1 = 11400714785074694791ULL;
constexpr uint64_t kHashPrime2 = 14029467366897019727ULL;
constexpr uint64_t kHashPrime3 = 1609587929392839161ULL;
constexpr uint64_t kHashPrime4 = 9650029242287828579ULL;
constexpr uint64_t kHashPrime5 = 2870177450012600261ULL;

inline uint64_t rotateLeft(uint64_t value, unsigned count) {
  return (value << count) | (value >> (64 - count));
}
inline uint64_t read64(const unsigned char *bytes) {
  uint64_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}
inline uint32_t read32(const unsigned char *bytes) {
  uint32_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}
inline uint64_t hashRound(uint64_t acc, uint64_t input) {
  acc += input * kHashPrime2;
  acc = rotateLeft(acc, 31);
  return acc * kHashPrime1;
}
inline uint64_t mergeHashRound(uint64_t acc, uint64_t value) {
  acc ^= hashRound(0, value);
  return acc * kHashPrime1 + kHashPrime4;
}
} // namespace detail

inline uint64_t xxhash64(const void *data, size_t size) {
  using namespace detail;
  const auto *bytes = static_cast<const unsigned char *>(data);
  const unsigned char *cursor = bytes;
  const unsigned char *const end = bytes + size;
  constexpr uint64_t kSeed = 0x4f1bbcdc6762f36bULL;
  uint64_t hash = 0;
  if (size >= 32) {
    const unsigned char *const block_end = end - 32;
    uint64_t lane1 = kSeed + kHashPrime1 + kHashPrime2;
    uint64_t lane2 = kSeed + kHashPrime2;
    uint64_t lane3 = kSeed;
    uint64_t lane4 = kSeed - kHashPrime1;
    do {
      lane1 = hashRound(lane1, read64(cursor));
      cursor += 8;
      lane2 = hashRound(lane2, read64(cursor));
      cursor += 8;
      lane3 = hashRound(lane3, read64(cursor));
      cursor += 8;
      lane4 = hashRound(lane4, read64(cursor));
      cursor += 8;
    } while (cursor <= block_end);
    hash = rotateLeft(lane1, 1) + rotateLeft(lane2, 7) + rotateLeft(lane3, 12) +
           rotateLeft(lane4, 18);
    hash = mergeHashRound(hash, lane1);
    hash = mergeHashRound(hash, lane2);
    hash = mergeHashRound(hash, lane3);
    hash = mergeHashRound(hash, lane4);
  } else {
    hash = kSeed + kHashPrime5;
  }
  hash += size;
  while (static_cast<size_t>(end - cursor) >= 8) {
    hash ^= hashRound(0, read64(cursor));
    hash = rotateLeft(hash, 27) * kHashPrime1 + kHashPrime4;
    cursor += 8;
  }
  if (static_cast<size_t>(end - cursor) >= 4) {
    hash ^= static_cast<uint64_t>(read32(cursor)) * kHashPrime1;
    hash = rotateLeft(hash, 23) * kHashPrime2 + kHashPrime3;
    cursor += 4;
  }
  while (cursor != end) {
    hash ^= static_cast<uint64_t>(*cursor++) * kHashPrime5;
    hash = rotateLeft(hash, 11) * kHashPrime1;
  }
  hash ^= hash >> 33;
  hash *= kHashPrime2;
  hash ^= hash >> 29;
  hash *= kHashPrime3;
  return hash ^ (hash >> 32);
}

// ---- Uniform interface --------------------------------------------------

using Digest = std::array<uint8_t, 32>;
// Full-digest cost path: what an honest per-lookup key derivation pays.
using DigestFn = std::function<Digest(const void *, size_t)>;
// Bucket-index path: 64 bits (SHA-256 folded to its first 8 bytes).
using BucketFn = std::function<uint64_t(const void *, size_t)>;

struct HashEntry {
  const char *name;
  DigestFn digest;
  BucketFn bucket;
};

namespace detail {
inline Digest u64ToDigest(uint64_t v) {
  Digest d{};
  std::memcpy(d.data(), &v, sizeof(v)); // low 8 bytes, rest zero
  return d;
}
inline uint64_t digestToU64(const Digest &d) {
  uint64_t v = 0;
  std::memcpy(&v, d.data(), sizeof(v));
  return v;
}
} // namespace detail

// Hex string of a full digest -- the key form a string-keyed cache stores under
// (matches how the mem-cache keys today: hex of the SHA-256).
inline std::string digestToHex(const Digest &d) {
  static const char *k = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (uint8_t b : d) {
    out.push_back(k[b >> 4]);
    out.push_back(k[b & 0xF]);
  }
  return out;
}

// The registry. Order is stable so CSV rows sort predictably.
inline const std::array<HashEntry, 3> &hashRegistry() {
  static const std::array<HashEntry, 3> registry = {{
      {"fnv1a",
       [](const void *d, size_t n) { return detail::u64ToDigest(fnv1a(d, n)); },
       [](const void *d, size_t n) { return fnv1a(d, n); }},
      {"xxhash64",
       [](const void *d, size_t n) {
         return detail::u64ToDigest(xxhash64(d, n));
       },
       [](const void *d, size_t n) { return xxhash64(d, n); }},
      {"sha256",
       [](const void *d, size_t n) {
         std::array<uint8_t, 32> raw = llvm::SHA256::hash(
             llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(d), n));
         Digest out{};
         std::memcpy(out.data(), raw.data(), 32);
         return out;
       },
       [](const void *d, size_t n) {
         std::array<uint8_t, 32> raw = llvm::SHA256::hash(
             llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(d), n));
         Digest out{};
         std::memcpy(out.data(), raw.data(), 32);
         return detail::digestToU64(out);
       }},
  }};
  return registry;
}

// Look up an entry by name; returns nullptr if unknown.
inline const HashEntry *findHash(const std::string &name) {
  for (const HashEntry &e : hashRegistry())
    if (name == e.name)
      return &e;
  return nullptr;
}

} // namespace COMGR::hotswap::bench

#endif // HOTSWAP_TRANSPILER_HASH_REGISTRY_H
