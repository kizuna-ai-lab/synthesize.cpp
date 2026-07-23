#pragma once

#include <charconv>
#include <cstdint>
#include <string_view>
#include <system_error>

inline bool parse_vits_runner_speaker_index(std::string_view value, uint32_t & output) {
    if (value.empty() || value.front() == '-' || value.front() == '+') {
        return false;
    }
    uint32_t parsed         = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc() || end != value.data() + value.size()) {
        return false;
    }
    output = parsed;
    return true;
}
