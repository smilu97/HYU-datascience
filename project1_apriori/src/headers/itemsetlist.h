#pragma once

#include "itemset.h"

#include <vector>
#include <set>
#include <string>

namespace apriori {

    /*
     * List of ItemSet
     */
    class ItemSetList {
        std::vector<ItemSet> sets;
    public:
        ItemSetList();
        ItemSetList(std::vector<ItemSet> sets);
        ItemSetList SelfJoin() const;
        uint Size() const;

        std::vector<ItemSet>::const_iterator begin() const { return sets.begin(); }
        std::vector<ItemSet>::const_iterator end()   const { return sets.end();   }
    };

}
