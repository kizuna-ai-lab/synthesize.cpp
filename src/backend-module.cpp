#include "backend-module.h"

#include "synthesize.h"

#if defined(SYNTH_GGML_BACKEND_DL)
#    include "ggml-backend.h"
#endif

#include <array>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>

#if defined(SYNTH_GGML_BACKEND_DL)
#    if defined(_WIN32)
#        define WIN32_LEAN_AND_MEAN
#        ifndef NOMINMAX
#            define NOMINMAX
#        endif
#        include <windows.h>
#    else
#        include <dlfcn.h>
#    endif
#endif

namespace {

struct BackendModuleState {
    std::mutex                            mutex;
    bool                                  frozen = false;
    std::map<std::string, synth_status_t> loaded_directories;
};

BackendModuleState & backend_module_state() {
    static BackendModuleState state;
    return state;
}

synth_status_t canonical_directory(const char * path, std::string & output) {
    if (path == nullptr || path[0] == '\0') {
        return SYNTH_ERR_INVALID_ARG;
    }

    std::error_code       error;
    std::filesystem::path directory = std::filesystem::u8path(path);
    if (!std::filesystem::is_directory(directory, error)) {
        return SYNTH_ERR_FILE_NOT_FOUND;
    }
    directory = std::filesystem::weakly_canonical(directory, error);
    if (error) {
        return SYNTH_ERR_IO;
    }
    output = directory.u8string();
    return SYNTH_OK;
}

#if defined(SYNTH_GGML_BACKEND_DL)

const char synth_library_anchor = 0;

std::filesystem::path library_self_directory() {
#    if defined(_WIN32)
    HMODULE     module = nullptr;
    const DWORD flags  = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&synth_library_anchor), &module)) {
        return {};
    }
    std::wstring path(1024, L'\0');
    for (;;) {
        const DWORD size = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
        if (size == 0) {
            return {};
        }
        if (size < path.size()) {
            path.resize(size);
            return std::filesystem::path(path).parent_path();
        }
        if (path.size() >= 32768) {
            return {};
        }
        path.resize(path.size() * 2);
    }
#    else
    Dl_info info{};
    if (dladdr(&synth_library_anchor, &info) == 0 || info.dli_fname == nullptr || info.dli_fname[0] == '\0') {
        return {};
    }
    std::error_code       error;
    std::filesystem::path path = std::filesystem::weakly_canonical(info.dli_fname, error);
    if (error) {
        path = std::filesystem::absolute(info.dli_fname, error);
    }
    return error ? std::filesystem::path{} : path.parent_path();
#    endif
}

int module_score(const std::filesystem::path & path) {
    using ScoreFunction = int (*)();
    ScoreFunction score = nullptr;
#    if defined(_WIN32)
    const DWORD previous_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(previous_mode | SEM_FAILCRITICALERRORS);
    HMODULE library = LoadLibraryW(path.wstring().c_str());
    SetErrorMode(previous_mode);
    if (library == nullptr) {
        return 0;
    }
    const FARPROC symbol = GetProcAddress(library, "ggml_backend_score");
    static_assert(sizeof(score) == sizeof(symbol));
    std::memcpy(&score, &symbol, sizeof(score));
    const int result = score == nullptr ? 0 : score();
    FreeLibrary(library);
#    else
    void * library = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        return 0;
    }
    void * symbol = dlsym(library, "ggml_backend_score");
    static_assert(sizeof(score) == sizeof(symbol));
    std::memcpy(&score, &symbol, sizeof(score));
    const int result = score == nullptr ? 0 : score();
    dlclose(library);
#    endif
    return result;
}

std::string module_prefix() {
#    if defined(_WIN32)
    return "ggml-";
#    else
    return "libggml-";
#    endif
}

std::string module_extension() {
#    if defined(_WIN32)
    return ".dll";
#    else
    return ".so";
#    endif
}

void load_best_module(const std::filesystem::path & directory, const char * backend_name) {
    const std::string     prefix     = module_prefix() + backend_name + "-";
    const std::string     extension  = module_extension();
    int                   best_score = 0;
    std::filesystem::path best_path;
    std::error_code       error;

    std::filesystem::directory_iterator item(directory, std::filesystem::directory_options::skip_permission_denied,
                                             error);
    const std::filesystem::directory_iterator end;
    while (!error && item != end) {
        if (item->is_regular_file(error)) {
            const std::string filename = item->path().filename().u8string();
            if (filename.rfind(prefix, 0) == 0 && item->path().extension() == extension) {
                const int score = module_score(item->path());
                if (score > best_score) {
                    best_score = score;
                    best_path  = item->path();
                }
            }
        }
        item.increment(error);
    }

    if (best_score == 0) {
        best_path = directory / (module_prefix() + backend_name + extension);
        if (!std::filesystem::is_regular_file(best_path, error)) {
            return;
        }
    }
    const std::string path = best_path.u8string();
    (void) ggml_backend_load(path.c_str());
}

synth_status_t load_dynamic_modules(const std::filesystem::path & directory) {
    static constexpr std::array<const char *, 15> backend_names = {
        "blas",   "zendnn",  "cann",   "cuda",    "hip",  "metal",    "rpc", "sycl",
        "vulkan", "virtgpu", "opencl", "hexagon", "musa", "openvino", "cpu",
    };
    for (const char * backend_name : backend_names) {
        load_best_module(directory, backend_name);
    }
    return ggml_backend_dev_count() == 0 ? SYNTH_ERR_BACKEND : SYNTH_OK;
}

#endif

}  // namespace

namespace synth {

void freeze_backend_modules() {
    BackendModuleState &        state = backend_module_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.frozen = true;
}

}  // namespace synth

synth_status_t synth_backend_load_from_dir(const char * artifact_dir) {
    try {
        std::string          directory;
        const synth_status_t path_status = canonical_directory(artifact_dir, directory);
        if (path_status != SYNTH_OK) {
            return path_status;
        }

        BackendModuleState &        state = backend_module_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.frozen) {
            return SYNTH_ERR_INVALID_ARG;
        }
        const auto existing = state.loaded_directories.find(directory);
        if (existing != state.loaded_directories.end()) {
            return existing->second;
        }

        const auto inserted = state.loaded_directories.emplace(directory, SYNTH_ERR_INTERNAL);
#if defined(SYNTH_GGML_BACKEND_DL)
        const synth_status_t status = load_dynamic_modules(std::filesystem::u8path(directory));
#else
        const synth_status_t status = SYNTH_OK;
#endif
        inserted.first->second = status;
        return status;
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    } catch (...) {
        return SYNTH_ERR_INTERNAL;
    }
}

synth_status_t synth_backend_load_default(void) {
    try {
#if defined(SYNTH_GGML_BACKEND_DL)
        const std::filesystem::path library_directory = library_self_directory();
        if (library_directory.empty()) {
            return SYNTH_ERR_BACKEND;
        }
        const std::filesystem::path module_directory = library_directory / "synthesize" / "backends";
        const std::string           path             = module_directory.u8string();
        return synth_backend_load_from_dir(path.c_str());
#else
        BackendModuleState &        state = backend_module_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        return state.frozen ? SYNTH_ERR_INVALID_ARG : SYNTH_OK;
#endif
    } catch (...) {
        return SYNTH_ERR_INTERNAL;
    }
}
