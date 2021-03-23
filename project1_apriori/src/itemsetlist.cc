#include "apriori.h"

using namespace apriori;

ItemSetList::ItemSetList() {}
ItemSetList::ItemSetList(std::vector<ItemSet> sets): sets(sets) {}

unsigned int ItemSetList::Size() const { return sets.size(); }

/**
 * Get self-joined item set list
 * 
 * @return item set list
 */
ItemSetList ItemSetList::SelfJoin() const {
    const int n = sets.size();
    std::vector<ItemSet> new_sets;
    for (auto i = sets.begin(); i != sets.end(); i++) {
        const int tg_size = i->Size() + 1;
        auto j = i;
        j++;
        for (; j != sets.end(); j++) {
            if (!(i->Similar(*j))) continue;
            new_sets.push_back((*i) + (*j));
        }
    }
    return new_sets;
}