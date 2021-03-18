#include "apriori.h"
#include "utils.h"

#include <fstream>
#include <iostream>
#include <algorithm>

std::string ResultLineToString(const apriori::AprioriSolverResultLine & line) {
    return ToString(line.item_set) + '\t'
        + ToString(line.associative_item_set) + '\t'
        + ToStringPercentage(line.support) + '\t'
        + ToStringPercentage(line.confidence);
}

void SaveAprioriResults(
    const std::vector<apriori::AprioriSolverResultLine> & results,
    const std::string & output_filename
) {
    std::vector<std::string> lines;
    lines.reserve(results.size());
    std::transform(results.begin(), results.end(), std::back_inserter(lines), ResultLineToString);

    std::ofstream ofs(output_filename, std::ios::out);
    bool first_line = true;
    for (const auto line: lines) {
        if (!first_line) ofs << std::endl;
        ofs << line;
        first_line = false;
    }
    ofs.close();
}

auto main(int argc, const char** argv) -> int {
    std::cout << "[Apriori] 2016025241 YJ Kim - 2021 DataScience - Hanyang Univ. CSE" << '\n';

    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " minimum_support input_file_name output_file_name" << '\n';
        return -1;
    }

    int min_support = atoi(argv[1]);
    if (min_support <= 0) {
        std::cout << "[Apriori][Error] Invalid minimum_support" << '\n';
        return -1;
    }
    std::cout << "[Apriori] Minimum support: " << min_support << "%\n";

    const char* input_filename = argv[2];
    const char* output_filename = argv[3];

    apriori::AprioriSolver solver(input_filename);
    std::cout << "[Apriori] Loaded data from \"" << input_filename << "\"\n[Apriori] Generated Apriori solver class\n";
    auto results = solver.Solve(min_support);
    std::cout << "[Apriori] Solved Apriori on " << solver.Size() << " transactions\n";
    SaveAprioriResults(results, output_filename);
    std::cout << "[Apriori] Saved result on \"" << output_filename << "\"\n";
    
    return 0;
}
