#pragma once

#include "itemset.h"
#include "itemsetlist.h"

#include <vector>
#include <set>
#include <string>
#include <map>

namespace apriori {

    /*
     * Unit of result from AprioriSolver
     */
    struct AprioriSolverResultLine {
        std::vector<int> item_set;
        std::vector<int> associative_item_set;
        float support;
        float confidence;
    };

    /*
     * Solver class which uses Apriori algorithm
     */
    class AprioriSolver {
        std::vector<std::vector<int>> records;
        std::vector<int> GetItemIds() const;
        ItemSetList GetSingleItemSetList() const;
        std::vector<int> GetSupports(const ItemSetList & set_list) const;
        int LookupSupport(const ItemSet & set) const;
        std::vector<AprioriSolverResultLine> ExtractAssociationRules(
            const ItemSet & set,
            int support,
            const std::map<std::string, int> & support_map
        ) const;
    public:
        AprioriSolver(const std::string &path);
        std::vector<AprioriSolverResultLine> Solve(int min_support) const;
        uint Size() const;
    };

}
