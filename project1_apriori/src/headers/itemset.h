#pragma once

#include <vector>
#include <set>
#include <string>

namespace apriori {

    /*
     * Set of item ids which occured at a single transaction
     */
    class ItemSet {
        std::set<int> s;
    public:
        const std::set<int> & GetPrimitive() const;
        ItemSet Merged(const ItemSet & tg) const;
        bool Equal(const ItemSet &tg) const;
        ItemSet();
        ItemSet(std::vector<int> item_ids);
        ItemSet(const std::string & s);
        uint Size() const;
        void Add(int item_id);
        void AddAll(const ItemSet & tg);
        bool Included(const std::vector<int> & v) const;
        std::vector<int> ToVector() const;
        ItemSet operator + (const ItemSet & tg) const;
        bool operator == (const ItemSet & tg) const;
        std::string ToString() const;
    };

}
