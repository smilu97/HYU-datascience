#include "apriori.h"

using namespace apriori;

ItemSetList::ItemSetList() {}
ItemSetList::ItemSetList(std::vector<ItemSet> sets): sets(sets) {}

uint ItemSetList::Size() const { return sets.size(); }

/*
 * TODO: O(N^4) Too slow algorithm
 */
ItemSetList ItemSetList::SelfJoin() const {
    const int n = sets.size();
    std::vector<ItemSet> new_sets;
    for (auto i = sets.begin(); i != sets.end(); i++) {
        auto j = i;
        j++;
        for (; j != sets.end(); j++) {
            const auto new_set = (*i) + (*j);
            bool duplicated = false;
            for (const auto & k :new_sets) {
                if (k == new_set) {
                    duplicated = true;
                    break;
                }
            }
            if (!duplicated) {
                new_sets.push_back(new_set);
            }
        }
    }
    return new_sets;
}