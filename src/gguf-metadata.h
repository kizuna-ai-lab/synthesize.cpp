#pragma once

#include "synthesize.h"

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct gguf_context;

namespace synth {

// Reads every tensor in `weights_context` out of the package on disk and into
// the backend buffer already allocated for it.
//
// Families differ in what tensors they declare but not in how those tensors are
// filled, so this is shared. `family` only names the reporter in diagnostics.
synth_status_t stream_tensor_data(const std::string &  path,
                                  const gguf_context * gguf,
                                  ggml_context *       weights_context,
                                  const char *         family);

// Typed, fail-closed reads of GGUF key/value metadata.
//
// Every Model Family validates the same way: a key must exist, carry the exact
// declared type, and — for arrays — hold well-formed elements. A miss is a
// package contract violation rather than a defaulted value, so each accessor
// returns false and reports the offending key with the family tag supplied at
// construction.
class GgufMetadata {
  public:
    GgufMetadata(const gguf_context * gguf, std::string family) : gguf_(gguf), family_(std::move(family)) {}

    bool u32(const std::string & key, uint32_t & value) const;
    bool u64(const std::string & key, uint64_t & value) const;
    bool f32(const std::string & key, float & value) const;
    bool boolean(const std::string & key, bool & value) const;
    bool string(const std::string & key, std::string & value) const;

    // Rejects the array unless every element is strictly positive, which is the
    // only form the graph builders can consume for shapes and rates.
    bool positive_i32_array(const std::string & key, std::vector<uint32_t> & value) const;
    bool string_array(const std::string & key, std::vector<std::string> & value) const;

    // Whether a key is present at all, without reading it and without the
    // error report the typed accessors emit on a miss. Used where a key's
    // ABSENCE is the meaningful answer -- a package that declares it cannot
    // clone must not carry a speaker encoder, and asking has to be quiet.
    bool has(const std::string & key) const;

    // Reads a string and requires it to equal `expected`.
    bool require_string(const std::string & key, const char * expected) const;

  private:
    void report(const std::string & key, const char * detail) const;

    const gguf_context * gguf_ = nullptr;
    std::string          family_;
};

}  // namespace synth
