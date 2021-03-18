#include "apriori.h"

#include <algorithm>
#include <fstream>
#include <sstream>

using namespace apriori;

int AprioriSolver::LookupSupport(const ItemSet & set) const {
    int count = 0;
    for (const auto & rec: records) {
        if (set.Included(rec)) ++count;
    }
    return count;
}

std::vector<int> AprioriSolver::GetItemIds() const {
    std::vector<int> item_ids;
    for (auto & i: records) {
        for (auto id: i) {
            item_ids.push_back(id);
        }
    }
    std::sort(item_ids.begin(), item_ids.end());
    item_ids.erase(std::unique(item_ids.begin(), item_ids.end()), item_ids.end());

    return item_ids;
}

ItemSetList AprioriSolver::GetSingleItemSetList() const {
    const auto item_ids = GetItemIds();

    std::vector<ItemSet> vector_set;
    for (auto id : item_ids) {
        ItemSet set;
        set.Add(id);
        vector_set.push_back(set);
    }

    return ItemSetList(vector_set);
}

std::vector<int> AprioriSolver::GetSupports(const ItemSetList & set_list) const {
    std::vector<int> results;
    results.reserve(set_list.Size());
    for (const auto & set: set_list) {
        results.push_back(LookupSupport(set));
    }
    return results;
}

std::vector<AprioriSolverResultLine> AprioriSolver::ExtractAssociationRules(
    const ItemSet & set,
    int support,
    const std::map<std::string, int> & support_map
) const {
    const std::set<int> & s = set.GetPrimitive();
    std::vector<int> ids(s.begin(), s.end());
    std::vector<AprioriSolverResultLine> result;
    if (set.Size() <= 1) return result;

    const float f_support = ((float) support) / ((float) records.size());
    for (int a_size = 1; a_size < ids.size(); a_size++) {
        const int b_size = ids.size() - a_size;
        std::vector<bool> selector(ids.size(), false);
        for (int i = 0; i < a_size; i++) selector[ids.size() - (i+1)] = true;
        do {
            ItemSet a, b;
            for (int i = 0; i < ids.size(); i++) {
                if (selector[i]) a.Add(ids[i]);
                else b.Add(ids[i]);
            }
            const std::string a_key = a.ToString(), b_key = b.ToString();
            const int a_support = support_map.find(a_key)->second;
            const float confidence = ((float) support) / ((float) a_support);
            result.push_back({
                a.ToVector(),
                b.ToVector(),
                f_support,
                confidence,
            });
        } while (std::next_permutation(selector.begin(), selector.end()));
    }
    return result;
}

std::vector<AprioriSolverResultLine> AprioriSolver::Solve(int min_support) const {
    // Convert min_support value from floating percentage number into absolute count integer number
    min_support = (int) (((double) records.size() * min_support) / 100.0);

    ItemSetList set_list = GetSingleItemSetList();

    std::vector<ItemSet> frequent_sets;
    std::vector<int> frequent_supports;
    std::map<std::string, int> support_map;
    while (true) {
        std::vector<int> support_list = GetSupports(set_list);

        std::vector<ItemSet> new_sets;
        auto it = set_list.begin();
        for (int i = 0; i < support_list.size(); i++, it++) {
            if (support_list[i] > min_support) {
                new_sets.push_back(*it);
                frequent_sets.push_back(*it);
                frequent_supports.push_back(support_list[i]);
                support_map[it->ToString()] = support_list[i];
            }
        }

        if (new_sets.empty()) break;

        set_list = ItemSetList(new_sets).SelfJoin();
    }

    std::vector<AprioriSolverResultLine> results;
    for (int i = 0; i < frequent_sets.size(); i++) {
        const auto rules = ExtractAssociationRules(frequent_sets[i], frequent_supports[i], support_map);
        std::copy(rules.begin(), rules.end(), std::back_inserter(results));
    }
    return results;
}

uint AprioriSolver::Size() const { return records.size(); }

AprioriSolver::AprioriSolver(const std::string &path) {
    char * buf = new char[1024];

    // Create file input stream to read data
    std::ifstream ifs(path, std::ios::in);

    while (true) {
        // Read a line from input stream
        auto & is = ifs.getline(buf, 1024);

        std::stringstream stream;
        stream.str(buf);
        
        // Parse int list from a line into std::vector<int>
        std::vector<int> record;
        int tmp;
        while (stream >> tmp) {
            record.push_back(tmp);
        }

        this->records.push_back(record);

        if (is.eof()) break;
    }

    ifs.close();
    delete[] buf;
}
