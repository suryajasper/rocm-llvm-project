//===- cache_scaling_bench.cc - Cache-mode x hash scaling analysis --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Binary 2 of the cache scaling harness. Sweeps two cache MODES x three HASH
// strategies x source sizes x thread counts x {cold,warm}, driving the real
// COMGR in-memory cache. The producer is sleep-modeled (--producer-us), so the
// measured deltas come only from the cache machinery + key derivation.
//
//   single-flight : one process-global mem-cache; all N threads request the
//                   same key. Cold => producer runs ONCE (coalesced), warm =>
//                   all hit. The selected hash is installed as the cache's key
//                   derivation via setMemCacheKeyFnForTesting.
//   per-thread    : each thread owns a thread-local map keyed by the SAME hash
//                   string. No sharing => cold producer runs N times. Faithful
//                   because the mem-cache's identity IS the hash string (no
//                   exact-compare), so a per-thread map with the same key hits
//                   the same derivation path.
//
// --source-bytes (hashed) is swept INDEPENDENTLY of --bytes (producer output),
// since key-derivation cost scales with SOURCE size, not output size.
//
// CSV: experiment,hash,mode,phase,threads,payload_bytes,source_bytes,
//      latency_us,producer_calls,coalesced,ready_hits,hash_ns,throughput_mibps
//
//===----------------------------------------------------------------------===//

#include "hotswap/cache/hash-registry.h"
#include "hotswap/cache/mem-cache.h"
#include "hotswap/cache/pipeline.h"
#include "hotswap/cache/translation-cache.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace COMGR::hotswap;
using namespace COMGR::hotswap::bench;
using Clock = std::chrono::steady_clock;

namespace {

enum class Phase { kCold, kWarm };

struct Options {
  std::vector<size_t> thread_counts{1, 2, 4, 8, 16};
  std::vector<size_t> source_sizes{65536, 262144, 1048576, 4194304, 8388608};
  size_t payload_bytes = 1048576; // producer output, fixed
  size_t producer_microseconds = 2000;
  size_t iterations = 5;
};

struct Outcome {
  uint64_t latency_us = 0;
  uint64_t producer_calls = 0;
  uint64_t coalesced = 0;
  uint64_t ready_hits = 0;
};

// ---- Thread barrier (shared shape with MemCacheBenchmark) ---------------
// Captures the instant the gate opens (last arriver), so callers time only the
// post-gate concurrent work and exclude thread spawn + per-thread setup. This
// makes single-flight and per-thread timing regions symmetric (the review
// flagged per-thread warm was inflated by in-worker pre-populate/allocation).
class StartGate {
public:
  explicit StartGate(size_t expected) : expected_(expected) {}
  void ArriveAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (++arrived_ == expected_) {
      opened_at_ = Clock::now();
      open_ = true;
      condition_.notify_all();
      return;
    }
    condition_.wait(lock, [&] { return open_; });
  }
  Clock::time_point openedAt() const { return opened_at_; }

private:
  const size_t expected_;
  std::mutex mutex_;
  std::condition_variable condition_;
  size_t arrived_ = 0;
  bool open_ = false;
  Clock::time_point opened_at_{};
};

// ---- Fake AMDGPU ELF, padded to a target source size --------------------
std::string amdgpuMetadataBlob(llvm::StringRef KernelName) {
  llvm::msgpack::Document Doc;
  llvm::msgpack::MapDocNode Root = Doc.getRoot().getMap(true);
  llvm::msgpack::DocNode Version = Doc.getArrayNode();
  Version.getArray().push_back(Doc.getNode(static_cast<uint64_t>(1)));
  Version.getArray().push_back(Doc.getNode(static_cast<uint64_t>(2)));
  Root["amdhsa.version"] = Version;
  llvm::msgpack::DocNode Kernel = Doc.getMapNode();
  Kernel.getMap()[".name"] = Doc.getNode(KernelName, true);
  llvm::msgpack::DocNode Kernels = Doc.getArrayNode();
  Kernels.getArray().push_back(Kernel);
  Root["amdhsa.kernels"] = Kernels;
  std::string Blob;
  Doc.writeToBlob(Blob);
  return Blob;
}

std::string makeFakeAmdgpuElf(llvm::StringRef kernel, size_t targetSize) {
  using namespace llvm;
  constexpr size_t NoteOffset = 128;
  const std::string Blob = amdgpuMetadataBlob(kernel);
  constexpr StringLiteral NoteName = "AMDGPU";
  const uint32_t NameSz = NoteName.size() + 1;
  const uint32_t DescSz = Blob.size();
  const uint32_t NamePadded = alignTo(NameSz, 4);
  const uint32_t NoteSize =
      sizeof(ELF::Elf64_Nhdr) + NamePadded + alignTo(DescSz, 4);
  std::string ShStr(1, '\0');
  auto addSectionName = [&](StringRef Name) {
    uint32_t Offset = ShStr.size();
    ShStr.append(Name.begin(), Name.end());
    ShStr.push_back('\0');
    return Offset;
  };
  const uint32_t NoteNameOffset = addSectionName(".note");
  const uint32_t ShStrNameOffset = addSectionName(".shstrtab");
  const uint32_t ShStrOffset = NoteOffset + NoteSize;
  const uint32_t ShdrOffset = alignTo(ShStrOffset + ShStr.size(), 8);
  const uint32_t Total = ShdrOffset + 3 * sizeof(ELF::Elf64_Shdr);
  SmallVector<uint8_t> D(Total, 0);
  auto writeStruct = [&](size_t Offset, const auto &S) {
    std::memcpy(D.data() + Offset, &S, sizeof(S));
  };
  ELF::Elf64_Ehdr Ehdr = {};
  Ehdr.e_ident[ELF::EI_MAG0] = 0x7f;
  Ehdr.e_ident[ELF::EI_MAG1] = 'E';
  Ehdr.e_ident[ELF::EI_MAG2] = 'L';
  Ehdr.e_ident[ELF::EI_MAG3] = 'F';
  Ehdr.e_ident[ELF::EI_CLASS] = ELF::ELFCLASS64;
  Ehdr.e_ident[ELF::EI_DATA] = ELF::ELFDATA2LSB;
  Ehdr.e_ident[ELF::EI_VERSION] = ELF::EV_CURRENT;
  Ehdr.e_ident[ELF::EI_OSABI] = ELF::ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = ELF::ET_DYN;
  Ehdr.e_machine = ELF::EM_AMDGPU;
  Ehdr.e_version = ELF::EV_CURRENT;
  Ehdr.e_flags = 0x49;
  Ehdr.e_ehsize = sizeof(ELF::Elf64_Ehdr);
  Ehdr.e_shentsize = sizeof(ELF::Elf64_Shdr);
  Ehdr.e_shnum = 3;
  Ehdr.e_shstrndx = 2;
  Ehdr.e_shoff = ShdrOffset;
  writeStruct(0, Ehdr);
  ELF::Elf64_Nhdr Nhdr = {};
  Nhdr.n_namesz = NameSz;
  Nhdr.n_descsz = DescSz;
  Nhdr.n_type = ELF::NT_AMDGPU_METADATA;
  writeStruct(NoteOffset, Nhdr);
  std::memcpy(D.data() + NoteOffset + sizeof(Nhdr), NoteName.data(),
              NoteName.size());
  std::memcpy(D.data() + NoteOffset + sizeof(Nhdr) + NamePadded, Blob.data(),
              DescSz);
  std::memcpy(D.data() + ShStrOffset, ShStr.data(), ShStr.size());
  ELF::Elf64_Shdr Shdrs[3] = {};
  Shdrs[1].sh_name = NoteNameOffset;
  Shdrs[1].sh_type = ELF::SHT_NOTE;
  Shdrs[1].sh_offset = NoteOffset;
  Shdrs[1].sh_size = NoteSize;
  Shdrs[1].sh_addralign = 4;
  Shdrs[2].sh_name = ShStrNameOffset;
  Shdrs[2].sh_type = ELF::SHT_STRTAB;
  Shdrs[2].sh_offset = ShStrOffset;
  Shdrs[2].sh_size = ShStr.size();
  Shdrs[2].sh_addralign = 1;
  std::memcpy(D.data() + ShdrOffset, Shdrs, sizeof(Shdrs));
  std::string elf(reinterpret_cast<const char *>(D.data()), D.size());
  if (elf.size() < targetSize)
    elf.append(targetSize - elf.size(), kernel.empty() ? '\0' : kernel[0]);
  return elf;
}

TranslationCacheRequest makeRequest(const std::string &elf) {
  TranslationCacheRequest req;
  req.SourceObject = llvm::MemoryBufferRef(elf, "source");
  req.SourceGfx = "gfx1250";
  req.TargetGfx = "gfx950";
  req.SourceIsa = "amdgcn-amd-amdhsa--gfx1250";
  req.TargetIsa = "amdgcn-amd-amdhsa--gfx950";
  req.CodeIsa = req.TargetIsa;
  req.CacheDisabled = true;
  return req;
}

PipelineResult makeProduced(size_t bytes, size_t delay_us) {
  if (delay_us)
    std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
  PipelineResult r;
  std::unique_ptr<llvm::WritableMemoryBuffer> buf =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(bytes, "hsaco");
  std::memset(buf->getBufferStart(), 0x5a, bytes);
  r.Hsaco = std::move(buf);
  r.Success = true;
  return r;
}

// The hash the single-flight mem-cache key fn should use this run. The setter
// takes a plain function pointer, so we route the selection through a global.
const HashEntry *g_selected_hash = nullptr;

// Installed via setMemCacheKeyFnForTesting: derive the key string using
// g_selected_hash over the request's source bytes. Mirrors how the production
// key is a hex digest of the source content.
std::string benchKeyFn(const TranslationCacheRequest &request) {
  const HashEntry *h = g_selected_hash;
  // g_selected_hash is set before any threads spawn and constant for a row; a
  // null here means a programming error in main()'s sequencing.
  if (!h)
    std::abort();
  const void *data = request.SourceObject.getBufferStart();
  const size_t size = request.SourceObject.getBufferSize();
  return digestToHex(h->digest(data, size));
}

// A cached entry for the per-thread map.
struct LocalEntry {
  std::shared_ptr<llvm::MemoryBuffer> Hsaco;
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
    if (a.rfind("--threads=", 0) == 0)
      o.thread_counts = parseList(a.substr(std::strlen("--threads=")));
    else if (a.rfind("--source-bytes=", 0) == 0)
      o.source_sizes = parseList(a.substr(std::strlen("--source-bytes=")));
    else if (a.rfind("--bytes=", 0) == 0)
      o.payload_bytes = parseU64(a.substr(std::strlen("--bytes=")), o.payload_bytes);
    else if (a.rfind("--producer-us=", 0) == 0)
      o.producer_microseconds =
          parseU64(a.substr(std::strlen("--producer-us=")), o.producer_microseconds);
    else if (a.rfind("--iterations=", 0) == 0)
      o.iterations = parseU64(a.substr(std::strlen("--iterations=")), o.iterations);
    else if (a == "--help") {
      std::cout << "Usage: cache_scaling_bench [--threads=1,2,4,8,16] "
                   "[--source-bytes=...] [--bytes=1048576] "
                   "[--producer-us=2000] [--iterations=5]\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << a << '\n';
      std::exit(2);
    }
  }
  return o;
}

// ---- single-flight mode (shared process-global mem-cache) ---------------
Outcome runSingleFlight(Phase phase, size_t threads, const std::string &elf,
                        size_t payload, size_t producer_us) {
  resetMemCacheForTesting(4ull << 30); // isolate per run; roomy budget
  setMemCacheKeyFnForTesting(&benchKeyFn);
  const auto req = makeRequest(elf);

  if (phase == Phase::kWarm)
    getOrComputeTranslation(req, [&] { return makeProduced(payload, 0); });

  std::atomic<uint64_t> producer_calls{0};
  std::vector<MemCacheResult> results(threads);
  std::vector<std::thread> pool;
  StartGate gate(threads);
  for (size_t i = 0; i < threads; ++i)
    pool.emplace_back([&, i] {
      gate.ArriveAndWait();
      results[i] = getOrComputeTranslation(req, [&] {
        producer_calls.fetch_add(1, std::memory_order_relaxed);
        return makeProduced(payload, producer_us);
      });
    });
  for (auto &t : pool)
    t.join();
  const auto ended = Clock::now();

  const MemCacheMetrics m = snapshotMemCacheMetrics();
  Outcome o;
  // Time only post-gate concurrent work (excludes thread spawn/setup).
  o.latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                     ended - gate.openedAt())
                     .count();
  o.producer_calls = producer_calls.load();
  o.coalesced = m.Coalesced;
  o.ready_hits = m.Hits;
  return o;
}

// ---- per-thread mode (thread-local map, no coalescing) ------------------
Outcome runPerThread(Phase phase, size_t threads, const std::string &elf,
                     size_t payload, size_t producer_us) {
  const auto req = makeRequest(elf);

  std::atomic<uint64_t> producer_calls{0};
  std::vector<std::thread> pool;
  StartGate gate(threads);
  for (size_t i = 0; i < threads; ++i)
    pool.emplace_back([&] {
      // Each thread's own map -> no cross-thread sharing, no coalescing.
      // Pre-populate BEFORE the gate so warm setup (alloc/memset) is excluded
      // from the timed region, symmetric with single-flight's main-thread
      // pre-populate.
      std::unordered_map<std::string, LocalEntry> local;
      if (phase == Phase::kWarm) {
        const std::string warm_key = benchKeyFn(req);
        PipelineResult pre = makeProduced(payload, 0);
        local[warm_key] = LocalEntry{std::shared_ptr<llvm::MemoryBuffer>(
            pre.Hsaco.release())};
      }
      gate.ArriveAndWait();
      // Derive the key on every lookup, exactly as single-flight does (one
      // derivation per getOrComputeTranslation).
      const std::string k = benchKeyFn(req);
      auto it = local.find(k);
      if (it == local.end()) {
        producer_calls.fetch_add(1, std::memory_order_relaxed);
        PipelineResult produced = makeProduced(payload, producer_us);
        local[k] = LocalEntry{std::shared_ptr<llvm::MemoryBuffer>(
            produced.Hsaco.release())};
      }
    });
  for (auto &t : pool)
    t.join();
  const auto ended = Clock::now();

  Outcome o;
  o.latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                     ended - gate.openedAt())
                     .count();
  o.producer_calls = producer_calls.load();
  o.coalesced = 0;
  o.ready_hits = (phase == Phase::kWarm) ? threads : 0;
  return o;
}

template <typename Runner> Outcome medianOutcome(size_t iters, Runner &&run) {
  std::vector<Outcome> outs;
  outs.reserve(iters);
  for (size_t i = 0; i < iters; ++i)
    outs.push_back(run());
  std::sort(outs.begin(), outs.end(), [](const Outcome &l, const Outcome &r) {
    return l.latency_us < r.latency_us;
  });
  return outs[outs.size() / 2];
}

void printRow(const char *hash, const char *mode, Phase phase, size_t threads,
              size_t payload, size_t source, const Outcome &o, double hash_ns) {
  // Throughput of bytes ACTUALLY produced (payload * producer_calls), not
  // payload*threads: single-flight coalesces to one producer, so crediting it
  // payload*threads would fabricate an N-fold throughput. This normalization is
  // mode-fair (per-thread genuinely produces payload*threads; single-flight
  // produces payload*1). For warm (0 producers) throughput is reported as 0.
  const double produced_bytes =
      static_cast<double>(payload) * static_cast<double>(o.producer_calls);
  const double mibps =
      o.latency_us ? (produced_bytes / (1024.0 * 1024.0)) / (o.latency_us / 1e6)
                   : 0.0;
  std::cout << "cache-scaling," << hash << ',' << mode << ','
            << (phase == Phase::kCold ? "cold" : "warm") << ',' << threads << ','
            << payload << ',' << source << ',' << o.latency_us << ','
            << o.producer_calls << ',' << o.coalesced << ',' << o.ready_hits
            << ',' << static_cast<uint64_t>(hash_ns) << ',' << mibps << '\n';
}

// Measure the selected hash's per-call digest cost over `source` bytes, so the
// row can attribute hash_ns (same full-digest number binary 1 reports).
double measureHashNs(const HashEntry &h, const std::string &elf,
                     size_t iterations) {
  const size_t size = elf.size();
  // Calibrate reps so each sample clears ~200us (matches binary 1).
  size_t reps = 1;
  for (;;) {
    const auto t0 = Clock::now();
    volatile uint8_t sink = 0;
    for (size_t r = 0; r < reps; ++r)
      sink ^= h.digest(elf.data(), size)[0];
    (void)sink;
    const double ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0)
            .count();
    if (ns >= 200000.0 || reps >= (1u << 20))
      break;
    reps *= 2;
  }
  std::vector<double> samples;
  samples.reserve(iterations);
  for (size_t it = 0; it < iterations; ++it) {
    const auto t0 = Clock::now();
    volatile uint8_t sink = 0;
    for (size_t r = 0; r < reps; ++r)
      sink ^= h.digest(elf.data(), size)[0];
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
  std::cout << "experiment,hash,mode,phase,threads,payload_bytes,source_bytes,"
               "latency_us,producer_calls,coalesced,ready_hits,hash_ns,"
               "throughput_mibps\n";
  for (const size_t source : o.source_sizes) {
    for (const HashEntry &h : hashRegistry()) {
      g_selected_hash = &h;
      const std::string elf = makeFakeAmdgpuElf("bench_kernel", source);
      // Report the ACTUAL hashed size (== elf.size()); equals `source` for all
      // matrix sizes (>= the tiny base ELF), truthful if ever clamped.
      const size_t actual_source = elf.size();
      const double hash_ns = measureHashNs(h, elf, o.iterations);
      for (const size_t threads : o.thread_counts) {
        for (const Phase phase : {Phase::kCold, Phase::kWarm}) {
          printRow(h.name, "single-flight", phase, threads, o.payload_bytes,
                   actual_source,
                   medianOutcome(o.iterations,
                                 [&] {
                                   return runSingleFlight(
                                       phase, threads, elf, o.payload_bytes,
                                       o.producer_microseconds);
                                 }),
                   hash_ns);
          printRow(h.name, "per-thread", phase, threads, o.payload_bytes,
                   actual_source,
                   medianOutcome(o.iterations,
                                 [&] {
                                   return runPerThread(
                                       phase, threads, elf, o.payload_bytes,
                                       o.producer_microseconds);
                                 }),
                   hash_ns);
        }
      }
    }
  }
  return 0;
}
