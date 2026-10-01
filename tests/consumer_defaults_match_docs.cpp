// The default policy table of docs/lane-contract.md, read back against the four
// `DefaultPolicy*` constants it describes.
//
// A changed default is still a certified combination meeting its own bound, so
// no accuracy gate would notice it - this comparison keeps the table honest.

#include "boys/boys.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int gFailures = 0;

// One cell of a markdown table row, trimmed, with the code-span backticks it may
// carry removed: `boys::DefaultPolicyFp64` and the bare name are one name here.
std::string Trim(const std::string& s) {
    const std::size_t first = s.find_first_not_of(" \t\r\n");

    if (first == std::string::npos) {
        return {};
    }

    std::size_t last = s.find_last_not_of(" \t\r\n");
    std::string out = s.substr(first, last - first + 1);
    const std::size_t bfirst = out.find_first_not_of('`');

    if (bfirst == std::string::npos) {
        return {};
    }

    last = out.find_last_not_of('`');
    return out.substr(bfirst, last - bfirst + 1);
}

std::vector<std::string> SplitRow(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;

    for (char c : line) {
        if (c == '|') {
            cells.push_back(Trim(cell));
            cell.clear();
        } else {
            cell.push_back(c);
        }
    }

    cells.push_back(Trim(cell));
    return cells;
}

// Each axis is matched by the word the document uses for it, which is why these
// maps live here and not in the library: a reader meets "chebyshev" and "narrow".
bool RouteFrom(const std::string& word, boys::FitRoute& out) {
    if (word == "chebyshev") {
        out = boys::FitRoute::kChebyshev;
        return true;
    }

    if (word == "rational-minimax") {
        out = boys::FitRoute::kRationalMinimax;
        return true;
    }

    return false;
}

bool SchemeFrom(const std::string& word, boys::EvalScheme& out) {
    if (word == "Horner") {
        out = boys::EvalScheme::kHorner;
        return true;
    }

    if (word == "split-Clenshaw") {
        out = boys::EvalScheme::kSplitClenshaw;
        return true;
    }

    return false;
}

bool GranularityFrom(const std::string& word, boys::FitGranularity& out) {
    if (word == "narrow") {
        out = boys::FitGranularity::kNarrow;
        return true;
    }

    if (word == "shipped") {
        out = boys::FitGranularity::kShipped;
        return true;
    }

    return false;
}

bool PackFrom(const std::string& word, boys::PackAxis& out) {
    if (word == "arguments") {
        out = boys::PackAxis::kArguments;
        return true;
    }

    if (word == "orders") {
        out = boys::PackAxis::kOrders;
        return true;
    }

    return false;
}

// One row of the document's default table, against one policy.
template <typename Policy>
void CheckOne(const std::string& name,
              const std::vector<std::string>& cells,
              const char* expected_name) {
    if (cells.size() < 7) {
        std::printf("FAIL %s: the table row has %zu cells, expected at least 7\n",
                    expected_name,
                    cells.size());
        ++gFailures;
        return;
    }

    if (cells[2] != name) {
        std::printf("FAIL %s: the row names '%s' where '%s' was expected\n",
                    expected_name,
                    cells[2].c_str(),
                    name.c_str());
        ++gFailures;
        return;
    }

    boys::FitRoute route = boys::FitRoute::kChebyshev;
    boys::EvalScheme scheme = boys::EvalScheme::kHorner;
    boys::FitGranularity granularity = boys::FitGranularity::kNarrow;
    boys::PackAxis pack = boys::PackAxis::kArguments;

    const bool parsed = RouteFrom(cells[3], route) && SchemeFrom(cells[4], scheme) &&
                        GranularityFrom(cells[5], granularity) && PackFrom(cells[6], pack);

    if (!parsed) {
        std::printf("FAIL %s: a cell of the row is not a word this check knows: "
                    "'%s' / '%s' / '%s' / '%s'\n",
                    expected_name,
                    cells[3].c_str(),
                    cells[4].c_str(),
                    cells[5].c_str(),
                    cells[6].c_str());
        ++gFailures;
        return;
    }

    const bool same = route == Policy::kRoute && scheme == Policy::kScheme &&
                      granularity == Policy::kGranularity && pack == Policy::kPack;

    std::printf("%-28s document says %-15s %-15s %-8s %-10s -> %s\n",
                expected_name,
                cells[3].c_str(),
                cells[4].c_str(),
                cells[5].c_str(),
                cells[6].c_str(),
                same ? "agrees with the default this build uses" : "DISAGREES WITH THE CODE");

    if (!same) {
        ++gFailures;
    }
}

}  // namespace

int main() {
    const std::string path = std::string(BoysSourceDir) + "/docs/lane-contract.md";
    std::ifstream in(path);

    if (!in) {
        std::printf("FAIL: cannot read the document this check exists to trust: %s\n", path.c_str());
        return 1;
    }

    // The block is found by this header, not by a line number an edit would move.
    const std::string kHeader = "| Precision | Name | Fit route | Scheme | Granularity |";
    std::string line;
    bool in_table = false;
    int rows = 0;

    while (std::getline(in, line)) {
        if (!in_table) {
            in_table = line.rfind(kHeader, 0) == 0;
            continue;
        }

        // The separator row, then four rows of values.
        const std::vector<std::string> cells = SplitRow(line);

        if (cells.size() >= 3 && cells[2] == "boys::DefaultPolicyFp64") {
            CheckOne<boys::DefaultPolicyFp64>("boys::DefaultPolicyFp64", cells, "fp64");
            ++rows;
        } else if (cells.size() >= 3 && cells[2] == "boys::DefaultPolicyFp32") {
            CheckOne<boys::DefaultPolicyFp32>("boys::DefaultPolicyFp32", cells, "fp32");
            ++rows;
        } else if (cells.size() >= 3 && cells[2] == "boys::DefaultPolicyFp16") {
            CheckOne<boys::DefaultPolicyFp16>("boys::DefaultPolicyFp16", cells, "fp16");
            ++rows;
        } else if (cells.size() >= 3 && cells[2] == "boys::DefaultPolicyBf16") {
            CheckOne<boys::DefaultPolicyBf16>("boys::DefaultPolicyBf16", cells, "bf16");
            ++rows;
        } else if (rows > 0) {
            break;  // the table ended
        }

        if (line.empty() && rows > 0) {
            break;
        }
    }

    if (rows != 4) {
        std::printf("FAIL: found %d of the 4 default rows in %s\n", rows, path.c_str());
        return 1;
    }

    if (gFailures != 0) {
        std::printf("\n%d of the 4 rows disagree with the code. The document is what a reader "
                    "believes; the constants are what they call. One of them has to change.\n",
                    gFailures);
        return 1;
    }

    std::printf("\nPASS: all 4 documented defaults agree with the constants this build resolves "
                "them to\n");
    return 0;
}
