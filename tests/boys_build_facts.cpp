// The build-fact probe: what this build is, one line per fact, in the stable
// form docs/build-facts.md records and diffs.
//
//   boys-build-facts [--leg NAME]               print this build's facts
//   boys-build-facts [--leg NAME] --check <doc> print the facts, then compare
//                                               them with the row <doc> records
//
// Every fact is deterministic - a preprocessor fact, a compile-time constant, a
// binary fact read out of the library's own object code, or an operating-system
// report of the machine. No timing, and there must not be: a duration taken on a
// shared build machine is a distribution across ephemeral runners, not a fact.
//
// --leg is the key the recorded table is keyed by; CI passes the job's own check
// name. Without it the probe derives a name for a developer build that no CI leg
// can be mistaken for.
//
// --check gates on a fact and never on the runner's identity: the CPU model, the
// processor count, the compiler version and the recording date are tags, reported
// when they move and never failed - a leg draws different hardware each run.

#include <boys/boys.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// <intrin.h> declares __cpuid for x86 targets alone, so an arm64 build of this
// program that asked for it would not compile.
#if defined(_M_IX86) || defined(_M_X64)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

// Supplied at configure time: the artifact to inspect and the tool that reads its symbol
// table. Both optional - a build supplying neither reports the route as unestablished.
#ifndef BOYS_FACTS_LIB
#define BOYS_FACTS_LIB ""
#endif
#ifndef BOYS_FACTS_SYMBOL_TOOL
#define BOYS_FACTS_SYMBOL_TOOL ""
#endif
#ifndef BOYS_FACTS_SYMBOL_MODE
#define BOYS_FACTS_SYMBOL_MODE ""
#endif
#ifndef BOYS_FACTS_CONFIG
#define BOYS_FACTS_CONFIG "unknown"
#endif

namespace {

struct Fact {
    std::string key;
    std::string value;
};

/// Collapses whitespace runs and trims, so a value is one line (CPU brands arrive padded).
std::string Trim(const std::string& text) {
    std::string out;
    bool pending = false;
    for (const char c : text)
    {
        const bool space = (c == ' ' || c == '\t' || c == '\r' || c == '\n');
        if (space)
        {
            pending = !out.empty();
            continue;
        }
        if (pending)
        {
            out.push_back(' ');
            pending = false;
        }
        out.push_back(c);
    }
    return out;
}

bool StartsWith(const std::string& text, const char* prefix) {
    const std::size_t n = std::strlen(prefix);
    return text.size() >= n && text.compare(0, n, prefix) == 0;
}

/// One line as it comes back from a child process or a file.
bool ReadLine(std::istream& in, std::string& line) {
    if (!std::getline(in, line))
    {
        return false;
    }
    if (!line.empty() && line.back() == '\r')
    {
        line.pop_back();
    }
    return true;
}

#if defined(__linux__)
/// The value of a `key<sep>value` line, trimmed, or empty when absent. Only the
/// /proc reader calls it, and an unreferenced helper here is a build error.
std::string ValueAfter(const std::string& line, char separator) {
    const std::size_t at = line.find(separator);
    if (at == std::string::npos)
    {
        return {};
    }
    return Trim(line.substr(at + 1));
}
#endif

// --- the CPU the process is running on --------------------------------------

/// The value of an environment variable, trimmed, empty when it is not set.
/// The MSVC branch is not stylistic - this tree promotes warnings to errors and MSVC
/// deprecates getenv (C4996). maybe_unused: only some platforms call it.
[[maybe_unused]] std::string Environment(const char* name) {
#if defined(_MSC_VER)
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || value == nullptr)
    {
        return {};
    }
    const std::string text = Trim(value);
    std::free(value);
    return text;
#else
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : Trim(value);
#endif
}

/// The brand string of an x86 processor, from the CPUID leaves that carry it.
/// Empty where the target has no CPUID, which is every arm64 build's answer.
std::string X86BrandString() {
#if defined(_M_IX86) || defined(_M_X64)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, static_cast<int>(0x80000000));
    if (static_cast<unsigned>(regs[0]) < 0x80000004u)
    {
        return {};
    }
    char brand[49] = {};
    for (unsigned leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf)
    {
        __cpuid(regs, static_cast<int>(leaf));
        std::memcpy(brand + (leaf - 0x80000002u) * 16u, regs, 16u);
    }
    return Trim(brand);
#elif defined(__x86_64__) || defined(__i386__)
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(0x80000000u, &eax, &ebx, &ecx, &edx) || eax < 0x80000004u)
    {
        return {};
    }
    char brand[49] = {};
    for (unsigned leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf)
    {
        if (!__get_cpuid(leaf, &eax, &ebx, &ecx, &edx))
        {
            return {};
        }
        unsigned regs[4] = {eax, ebx, ecx, edx};
        std::memcpy(brand + (leaf - 0x80000002u) * 16u, regs, 16u);
    }
    return Trim(brand);
#else
    return {};
#endif
}

/// A path in the working directory no two probe runs share. The suffix is this
/// process's stack address: no clock reading is allowed in this program.
std::string ScratchPath(const char* prefix) {
    std::uintptr_t marker = 0;
    return std::string(prefix) +
           std::to_string(reinterpret_cast<std::uintptr_t>(&marker)) + ".tmp";
}

/// Runs `command` with its standard output redirected to `path`; true when it exited zero.
///
/// The Windows form is not decoration: cmd.exe strips the first and last quote off a `cmd /c`
/// line that starts with a quote and carries more than two, so a command whose tool and
/// arguments both need quoting runs as a broken path. The nested `cmd /c` with the whole line
/// quoted is what survives it.
bool RunToFile(const std::string& command, const std::string& path) {
#if defined(_WIN32)
    const std::string line = "cmd /c \"" + command + " > \"" + path + "\" 2>nul\"";
#else
    const std::string line = command + " > \"" + path + "\" 2>/dev/null";
#endif
    return std::system(line.c_str()) == 0;
}

/// The first line a command wrote to its standard output, empty on any failure.
/// It never throws: a fact this raises is reported as unestablished rather than
/// failing the leg. maybe_unused: only some platforms need a system query.
[[maybe_unused]] std::string FirstLineOfCommand(const std::string& command) {
    const std::string path = ScratchPath("boys-build-facts-out-");
    if (!RunToFile(command, path))
    {
        std::remove(path.c_str());
        return {};
    }
    std::ifstream in(path);
    std::string line;
    const bool read = ReadLine(in, line);
    in.close();
    std::remove(path.c_str());
    return read ? Trim(line) : std::string{};
}

/// What the operating system reports as this machine's processor, and where that came from:
/// a brand string is not available everywhere, and an unavailable model is reported as such.
void AddCpuModel(std::vector<Fact>& facts) {
    std::string model = X86BrandString();
    if (!model.empty())
    {
        facts.push_back({"cpu.model", model});
        facts.push_back({"cpu.model.source", "cpuid"});
        return;
    }

#if defined(__linux__)
    std::ifstream info("/proc/cpuinfo");
    std::string line;
    std::string implementer;
    std::string part;
    while (ReadLine(info, line))
    {
        if (line.rfind("model name", 0) == 0)
        {
            const std::string value = ValueAfter(line, ':');
            if (!value.empty())
            {
                facts.push_back({"cpu.model", value});
                facts.push_back({"cpu.model.source", "proc-cpuinfo-model-name"});
                return;
            }
        }
        if (line.rfind("CPU implementer", 0) == 0 && implementer.empty())
        {
            implementer = ValueAfter(line, ':');
        }
        if (line.rfind("CPU part", 0) == 0 && part.empty())
        {
            part = ValueAfter(line, ':');
        }
    }
    if (!implementer.empty() || !part.empty())
    {
        facts.push_back({"cpu.model", "implementer " + implementer + " part " + part});
        facts.push_back({"cpu.model.source", "proc-cpuinfo-implementer-part"});
        return;
    }
#elif defined(__APPLE__)
    const std::string brand =
        FirstLineOfCommand("sysctl -n machdep.cpu.brand_string");
    if (!brand.empty())
    {
        facts.push_back({"cpu.model", brand});
        facts.push_back({"cpu.model.source", "sysctl-machdep.cpu.brand_string"});
        return;
    }
    const std::string hwModel = FirstLineOfCommand("sysctl -n hw.model");
    if (!hwModel.empty())
    {
        facts.push_back({"cpu.model", hwModel});
        facts.push_back({"cpu.model.source", "sysctl-hw.model"});
        return;
    }
#else
    const std::string identifier = Environment("PROCESSOR_IDENTIFIER");
    if (!identifier.empty())
    {
        facts.push_back({"cpu.model", identifier});
        facts.push_back({"cpu.model.source", "PROCESSOR_IDENTIFIER"});
        return;
    }
#endif

    facts.push_back({"cpu.model", "unestablished"});
    facts.push_back({"cpu.model.source", "none"});
}

// --- the compiler and the target --------------------------------------------

std::string CompilerId() {
#if defined(__clang__) && defined(_MSC_VER)
    return "clang-cl";
#elif defined(__clang__)
    return "clang";
#elif defined(_MSC_VER)
    return "msvc";
#elif defined(__GNUC__)
    return "gcc";
#else
    return "unknown";
#endif
}

std::string CompilerVersion() {
#if defined(__clang__)
    return Trim(__clang_version__);
#elif defined(_MSC_VER)
    constexpr unsigned kFull = _MSC_FULL_VER;
    const unsigned major = kFull / 10000000u;
    const unsigned minor = (kFull / 100000u) % 100u;
    const unsigned build = kFull % 100000u;
    return std::to_string(major) + "." + std::to_string(minor) + "." +
           std::to_string(build);
#elif defined(__GNUC__)
    return Trim(__VERSION__);
#else
    return "unknown";
#endif
}

std::string OperatingSystem() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#elif defined(__unix__)
    return "unix";
#else
    return "unknown";
#endif
}

std::string Architecture() {
#if defined(_M_X64) || defined(__x86_64__)
    return "x86-64";
#elif defined(_M_IX86) || defined(__i386__)
    return "x86-32";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return "arm64";
#elif defined(_M_ARM) || defined(__arm__)
    return "arm-32";
#else
    return "unknown";
#endif
}

/// The packed arithmetic the target's ISA provides, and its width in bits - the target the
/// compiler was asked for, not the width this program uses. A build that names none has no
/// packed arithmetic at all (width 0). Two entries are the target's own baseline rather than
/// a macro's presence: MSVC defines neither __SSE2__ nor __ARM_NEON, and both require them.
std::pair<std::string, unsigned> SimdTarget() {
#if defined(__AVX512F__)
    return {"avx512", 512};
#elif defined(__AVX2__)
    return {"avx2", 256};
#elif defined(__AVX__)
    return {"avx", 256};
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    return {"sse2", 128};
#elif defined(__ARM_NEON) || defined(_M_ARM64)
    return {"neon", 128};
#else
    return {"none", 0};
#endif
}

std::string Sanitizers() {
    std::string found;
#if defined(__SANITIZE_ADDRESS__)
    found = "address";
#endif
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
    if (found.empty())
    {
        found = "address";
    }
#endif
#if __has_feature(undefined_behavior_sanitizer)
    found = found.empty() ? "undefined" : found + ",undefined";
#endif
#if __has_feature(memory_sanitizer)
    found = found.empty() ? "memory" : found + ",memory";
#endif
#endif
    return found.empty() ? "none" : found;
}

/// The name a developer build records under, reached only where no leg was named, and
/// spelled so that no CI leg can be mistaken for it. The target's packed arithmetic and
/// the sanitizers are part of it, both being properties of the build: builds differing
/// in either must not record over each other.
std::string DerivedLegName() {
    std::string name = "local-" + OperatingSystem() + "-" + Architecture() +
                       "-" + SimdTarget().first + " " + CompilerId() + "-" +
                       CompilerVersion() + " " + BOYS_FACTS_CONFIG;
    const std::string sanitizers = Sanitizers();
    if (sanitizers != "none")
    {
        name += " " + sanitizers;
    }
    return name;
}

// --- the library's own facts ------------------------------------------------

/// Reads the library artifact's symbol table for a reference to an out-of-line fma.
/// A binary fact, not a timing: a fused multiply-add leaves nothing to call, and one the
/// compiler did not fuse leaves an undefined reference to the runtime's fma.
void AddFmaRoute(std::vector<Fact>& facts) {
    const std::string tool = BOYS_FACTS_SYMBOL_TOOL;
    const std::string mode = BOYS_FACTS_SYMBOL_MODE;
    const std::string artifact = BOYS_FACTS_LIB;

    if (tool.empty() || mode.empty() || artifact.empty())
    {
        facts.push_back({"fma.route", "unestablished"});
        facts.push_back({"fma.route.reason",
                         artifact.empty() ? "no library artifact supplied by the build"
                                          : "no symbol tool found at configure time"});
        facts.push_back({"fma.route.tool", "none"});
        return;
    }

    const std::size_t slash = artifact.find_last_of("/\\");
    const std::string name =
        slash == std::string::npos ? artifact : artifact.substr(slash + 1);

    const std::string toolName = [&tool] {
        const std::size_t last = tool.find_last_of("/\\");
        return last == std::string::npos ? tool : tool.substr(last + 1);
    }();

    std::string command;
    if (mode == "msvc-dump")
    {
        command = "\"" + tool + "\" /dump /symbols \"" + artifact + "\"";
    } else if (mode == "nm-undefined")
    {
        command = "\"" + tool + "\" -u \"" + artifact + "\"";
    } else
    {
        command = "\"" + tool + "\" -t \"" + artifact + "\"";
    }

    const std::string path = ScratchPath("boys-build-facts-sym-");
    if (!RunToFile(command, path))
    {
        std::remove(path.c_str());
        facts.push_back({"fma.route", "unestablished"});
        facts.push_back({"fma.route.reason", "the symbol tool did not run"});
        facts.push_back({"fma.route.artifact", name});
        facts.push_back({"fma.route.tool", toolName});
        return;
    }

    // The undefined-symbol view differs per tool: msvc-dump marks the line UNDEF and names the
    // symbol after a `|`, nm marks it `U`, objdump `*UND*`; the parser looks for the marker and
    // the name, not for a layout. Every decoration is stripped: Mach-O (and some COFF symbols)
    // prefixes an underscore, and a COFF object reaching the runtime through an import library -
    // the shape a Debug build produces - names the thunk __imp_fma. Read as written, that is
    // not "fma", so a build that calls the runtime is reported as one that does not.
    bool undefinedFma = false;
    bool undefinedFmaf = false;
    {
        std::ifstream dump(path);
        std::string line;
        while (ReadLine(dump, line))
        {
            const bool isUndefined =
                line.find("UNDEF") != std::string::npos ||
                line.find("*UND*") != std::string::npos ||
                line.find(" U ") != std::string::npos ||
                StartsWith(line, "U ");
            if (!isUndefined)
            {
                continue;
            }
            for (std::size_t at = line.find_first_not_of(" \t"); at != std::string::npos;)
            {
                const std::size_t end = line.find_first_of(" \t|", at);
                std::string token = line.substr(at, end == std::string::npos
                                                       ? std::string::npos
                                                       : end - at);
                while (!token.empty() && token.front() == '_')
                {
                    token.erase(token.begin());
                }
                if (StartsWith(token, "imp_"))
                {
                    token.erase(0, 4);
                }
                if (token == "fma")
                {
                    undefinedFma = true;
                } else if (token == "fmaf")
                {
                    undefinedFmaf = true;
                }
                at = end == std::string::npos ? std::string::npos
                                              : line.find_first_not_of(" \t|", end);
            }
        }
    }
    std::remove(path.c_str());

    if (undefinedFma || undefinedFmaf)
    {
        std::string symbols;
        if (undefinedFma)
        {
            symbols = "fma";
        }
        if (undefinedFmaf)
        {
            symbols += symbols.empty() ? "fmaf" : ",fmaf";
        }
        facts.push_back({"fma.route", "out-of-line-call"});
        facts.push_back({"fma.route.symbols", symbols});
    } else
    {
        // Nothing left to call: the multiply-add is a fused instruction, or the
        // build emitted none; contract.* and the backends say which.
        facts.push_back({"fma.route", "no-call"});
        facts.push_back({"fma.route.symbols", "none"});
    }
    facts.push_back({"fma.route.artifact", name});
    facts.push_back({"fma.route.tool", toolName});
}

// --- the facts --------------------------------------------------------------

std::vector<Fact> Gather(const std::string& leg) {
    std::vector<Fact> facts;

    facts.push_back({"leg", leg});
    facts.push_back({"config", BOYS_FACTS_CONFIG});

    // The machine this ran on: a tag, never a gate, since a runner is ephemeral
    // and the same leg draws different hardware.
    AddCpuModel(facts);
    {
        const unsigned logical = std::thread::hardware_concurrency();
        facts.push_back({"cpu.logical", logical == 0 ? "unestablished"
                                                     : std::to_string(logical)});
    }

    facts.push_back({"os", OperatingSystem()});
    facts.push_back({"arch", Architecture()});
    facts.push_back({"pointer.bits", std::to_string(sizeof(void*) * 8u)});
    facts.push_back({"endian", std::endian::native == std::endian::little ? "little"
                                                                          : "big"});

    facts.push_back({"compiler.id", CompilerId()});
    facts.push_back({"compiler.version", CompilerVersion()});
    // MSVC reports __cplusplus as 199711L without /Zc:__cplusplus, understating the standard.
#if defined(_MSVC_LANG)
    facts.push_back({"compiler.standard", std::to_string(_MSVC_LANG)});
#else
    facts.push_back({"compiler.standard", std::to_string(__cplusplus)});
#endif
    facts.push_back({"sanitizers", Sanitizers()});

    // The instruction sets the target was asked for, as the preprocessor reports
    // them. Preprocessor facts, not capability facts: a compiler may enable an
    // instruction set without defining a macro for it (MSVC defines __AVX2__ for
    // /arch:AVX2 and no __FMA__), so what the build does is contract.* below.
    struct Macro {
        const char* name;
        bool defined;
    };
    const Macro macros[] = {
        {"__SSE2__",
#if defined(__SSE2__)
         true
#else
         false
#endif
        },
        {"__AVX__",
#if defined(__AVX__)
         true
#else
         false
#endif
        },
        {"__AVX2__",
#if defined(__AVX2__)
         true
#else
         false
#endif
        },
        {"__FMA__",
#if defined(__FMA__)
         true
#else
         false
#endif
        },
        {"__F16C__",
#if defined(__F16C__)
         true
#else
         false
#endif
        },
        {"__AVX512F__",
#if defined(__AVX512F__)
         true
#else
         false
#endif
        },
        {"__ARM_NEON",
#if defined(__ARM_NEON)
         true
#else
         false
#endif
        },
        {"__ARM_FEATURE_FMA",
#if defined(__ARM_FEATURE_FMA)
         true
#else
         false
#endif
        },
        {"__ARM_FEATURE_FP16_SCALAR_ARITHMETIC",
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
         true
#else
         false
#endif
        },
        {"__ARM_FEATURE_FP16_VECTOR_ARITHMETIC",
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
         true
#else
         false
#endif
        },
    };
    for (const Macro& macro : macros)
    {
        facts.push_back({std::string("macro.") + macro.name, macro.defined ? "1" : "0"});
    }

    {
        const std::pair<std::string, unsigned> simd = SimdTarget();
        facts.push_back({"simd.target", simd.first});
        facts.push_back({"simd.bits", std::to_string(simd.second)});
        facts.push_back({"simd.lanes.fp64", std::to_string(simd.second / 64u)});
        facts.push_back({"simd.lanes.fp32", std::to_string(simd.second / 32u)});
        facts.push_back({"simd.lanes.fp16", std::to_string(simd.second / 16u)});
    }

    // The multiply-add route in force, from the library's own report, which answers per backend:
    // contraction belongs to a translation unit's flags, the packed arithmetic under others.
    const std::span<const boys::backend::BackendInfo> backends =
        boys::backend::BoysBackends();
    for (const boys::backend::BackendInfo& backend : backends)
    {
        facts.push_back({std::string("backend.") + backend.name + ".contracts",
                         backend.contracts ? "1" : "0"});
    }

    // The same question asked by this translation unit, whose flags are what a
    // consumer of this build gets. Contracts() is answered where it is called.
    facts.push_back({"contract.this-tu.fp64",
                     boys::backend::ScalarFp64::Contracts() ? "1" : "0"});
    facts.push_back({"contract.this-tu.fp32",
                     boys::backend::ScalarFp32::Contracts() ? "1" : "0"});

    facts.push_back({"runtime.avx2", boys::BoysAvx2Available() ? "1" : "0"});

    AddFmaRoute(facts);
    return facts;
}

// --- the recorded table -----------------------------------------------------

/// The tag a recorded row opens with, and the version of the format: a fenced block that
/// does not open with it is prose, so the document can show the format without a row.
constexpr const char* kFormatTag = "boys.build-facts/1";

/// The keys that carry where and by whom a row was observed rather than what
/// the build does. They are reported when they move and never failed.
bool IsTag(const std::string& key) {
    static const char* const kTags[] = {
        "leg",           "recorded",       "cpu.model",     "cpu.model.source",
        "cpu.logical",   "compiler.version", "compiler.standard", "fma.route.tool",
        "fma.route.artifact",
    };
    for (const char* tag : kTags)
    {
        if (key == tag)
        {
            return true;
        }
    }
    return false;
}

/// Every row the table records, in document order: a row is a fenced block opening with the
/// format tag; anything else is prose and skipped, never mistaken for a fact.
std::vector<std::vector<Fact>> TableRows(const std::string& path, bool& tableFound) {
    std::vector<std::vector<Fact>> rows;
    std::ifstream table(path);
    tableFound = table.is_open();
    if (!tableFound)
    {
        return rows;
    }
    std::string line;
    bool inBlock = false;
    bool isRow = false;
    std::vector<Fact> candidate;
    while (ReadLine(table, line))
    {
        if (!inBlock)
        {
            if (StartsWith(line, "```"))
            {
                inBlock = true;
                isRow = false;
                candidate.clear();
            }
            continue;
        }
        if (StartsWith(line, "```"))
        {
            inBlock = false;
            if (isRow && !candidate.empty())
            {
                rows.push_back(candidate);
            }
            continue;
        }
        if (candidate.empty() && !isRow && line == kFormatTag)
        {
            isRow = true;
            continue;
        }
        const std::size_t at = line.find('=');
        if (at == std::string::npos)
        {
            continue;
        }
        candidate.push_back({line.substr(0, at), line.substr(at + 1)});
    }
    return rows;
}

/// The leg a row belongs to, or empty when it names none.
std::string LegOf(const std::vector<Fact>& row) {
    for (const Fact& fact : row)
    {
        if (fact.key == "leg")
        {
            return fact.value;
        }
    }
    return {};
}

int PrintRow(const std::vector<Fact>& facts) {
    std::printf("%s\n", kFormatTag);
    for (const Fact& fact : facts)
    {
        std::printf("%s=%s\n", fact.key.c_str(), fact.value.c_str());
    }
    return 0;
}

int CheckAgainst(const std::vector<Fact>& facts, const std::string& path) {
    const std::string leg = facts.front().value;
    bool tableFound = false;
    const std::vector<std::vector<Fact>> rows = TableRows(path, tableFound);
    if (!tableFound)
    {
        std::printf("\nbuild facts: no table at %s - cannot check this leg\n",
                    path.c_str());
        return 1;
    }
    std::vector<Fact> row;
    for (const std::vector<Fact>& candidate : rows)
    {
        if (LegOf(candidate) == leg)
        {
            row = candidate;
            break;
        }
    }
    if (row.empty())
    {
        // No row recorded yet is not a failure: the block above is that row. The legs the table
        // does carry are listed, so a leg name nothing will ever match is not silently unchecked.
        std::printf("\nbuild facts: the table records no row for `%s`. The block "
                    "above is that row:\nrecord it with\n"
                    "    python tools/gen_build_facts.py --record <file>\n",
                    leg.c_str());
        std::printf("build facts: rows recorded: ");
        for (const std::vector<Fact>& candidate : rows)
        {
            const std::string other = LegOf(candidate);
            if (!other.empty())
            {
                std::printf("`%s` ", other.c_str());
            }
        }
        std::printf("\n");
        return 0;
    }

    std::vector<std::string> notices;
    std::vector<std::string> drifts;
    std::size_t compared = 0;
    for (const Fact& fact : facts)
    {
        if (fact.key == "leg")
        {
            continue;
        }
        const Fact* other = nullptr;
        for (const Fact& candidate : row)
        {
            if (candidate.key == fact.key)
            {
                other = &candidate;
                break;
            }
        }
        if (other == nullptr)
        {
            drifts.push_back("unrecorded fact: " + fact.key + "=" + fact.value);
            continue;
        }
        ++compared;
        if (other->value != fact.value)
        {
            const std::string move =
                fact.key + ": recorded " + other->value + ", this build " + fact.value;
            (IsTag(fact.key) ? notices : drifts).push_back(move);
        }
    }
    for (const Fact& fact : row)
    {
        if (IsTag(fact.key))
        {
            continue;
        }
        bool present = false;
        for (const Fact& candidate : facts)
        {
            if (candidate.key == fact.key)
            {
                present = true;
                break;
            }
        }
        if (!present)
        {
            drifts.push_back("stale fact: " + fact.key + "=" + fact.value);
        }
    }

    for (const std::string& notice : notices)
    {
        std::printf("build facts: tag moved (reported, not gated): %s\n",
                    notice.c_str());
    }
    if (drifts.empty())
    {
        std::printf("build facts: `%s` matches its recorded row (%zu facts compared)\n",
                    leg.c_str(), compared);
        return 0;
    }
    std::printf("\nbuild facts: `%s` DIFFERS from its recorded row:\n", leg.c_str());
    for (const std::string& drift : drifts)
    {
        std::printf("  %s\n", drift.c_str());
    }
    std::printf("A build fact moved: this is a different build. Re-record it with\n"
                "    python tools/gen_build_facts.py --record <file>\n");
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    std::string leg;
    std::string checkPath;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--leg" && i + 1 < argc)
        {
            leg = argv[++i];
        } else if (argument == "--check" && i + 1 < argc)
        {
            checkPath = argv[++i];
        } else
        {
            std::printf("usage: boys-build-facts [--leg <name>] [--check <table>]\n");
            return 2;
        }
    }
    if (leg.empty())
    {
        leg = DerivedLegName();
    }

    const std::vector<Fact> facts = Gather(leg);
    PrintRow(facts);
    return checkPath.empty() ? 0 : CheckAgainst(facts, checkPath);
}
