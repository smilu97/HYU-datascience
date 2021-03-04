#include "apriori.h"

using namespace apriori;


const std::set<int> & ItemSet::GetPrimitive() const {
    return s;
}

ItemSet ItemSet::Merged(const ItemSet & tg) const {
    ItemSet result;
    result.AddAll(*this);
    result.AddAll(tg);
    return result;
}

bool ItemSet::Equal(const ItemSet & tg) const {
    if (s.size() != tg.s.size()) return false;
    auto it1 = s.begin();
    auto it2 = tg.s.begin();
    while (it1 != s.end()) {
        if ((*it1) != (*it2)) return false;
        it1++;
        it2++;
    }
    return true;
}

std::string ItemSet::ToString() const {
    std::string result;
    for (auto id : s) {
        result += std::to_string(id);
        result += ',';
    }
    return result;
}

uint ItemSet::Size() const { return s.size(); }

ItemSet::ItemSet() {}
ItemSet::ItemSet(std::vector<int> item_ids) {
    for (auto id: item_ids) {
        Add(id);
    }
}
ItemSet::ItemSet(const std::string & s) {
    int begin = 0;
    for (int end = 0; end < s.size(); end++) {
        if (s[end] != ',') continue;
        const int id = atoi(s.substr(begin, end - begin).c_str());
        Add(id);
        begin = end + 1;
    }
}

void ItemSet::Add(int item_id) {
    s.insert(item_id);
}
void ItemSet::AddAll(const ItemSet & tg) {
    for (auto id: tg.s) {
        Add(id);
    }
}
bool ItemSet::Included(const std::vector<int> & v) const {
    int count = 0;
    for (int id: v) {
        if (s.find(id) != s.end()) {
            ++count;
        }
    }
    return s.size() == count;
}
std::vector<int> ItemSet::ToVector() const {
    return std::vector<int>(s.begin(), s.end());
}
ItemSet ItemSet::operator + (const ItemSet & tg) const {
    return Merged(tg);
}
bool ItemSet::operator == (const ItemSet & tg) const {
    return Equal(tg);
}