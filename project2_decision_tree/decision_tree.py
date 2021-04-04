import numpy as np

class DecisionTreeNode:
  def __init__(self, indices, prev_atts, gate, result):
    self.indices = indices
    self.prev_atts = prev_atts
    self.children = []
    self.gate = gate
    self.result = result
  
  def query(self, record):
    for child in self.children:
      if child.gate is None:
        return child.query(record)
      gate = child.gate
      if record[gate[0]] == gate[1]:
        return child.query(record)
    return self

class DecisionTree:
  def __init__(self, db, col_tbs):
    self.db = db
    self.col_tbs = col_tbs
    self.col_sz = [len(t) for t in col_tbs]
    self.root = self.build_root()
  
  def predict(self, record):
    if len(record.shape) == 1:
      return self.root.query(record).result
    records = record.reshape((-1, record.shape[-1]))
    return np.array([self.root.query(r).result for r in records])
  
  def query_result(self, indices):
    sim_classes = self.db[indices, -1]
    return np.argmax(np.bincount(sim_classes))
  
  def build_root(self):
    indices = np.arange(self.db.shape[0])
    col_ans = len(self.col_sz) - 1
    root = DecisionTreeNode(indices, [col_ans], None, self.query_result(indices))
    self.build_children(root)
    return root
  
  def build_children(self, node):
    atts = [att for att in range(len(self.col_sz)) if att not in node.prev_atts]
    if len(atts) == 0: return
    gains = [self.measure(node.indices, att) for att in atts]
    t_att = atts[gains.index(max(gains))]
    sub_indices = self.split_by_att(node.indices, t_att)
    curr_atts = list(node.prev_atts)
    curr_atts.append(t_att)
    node.children = [\
      DecisionTreeNode(indices, curr_atts, (t_att, value), self.query_result(indices)) \
        for value, indices in enumerate(sub_indices) if len(indices) > 0]
    for child in node.children:
      self.build_children(child)

  def entropy(self, indices):
    classes = self.db[indices, -1]
    uniq, counts = np.unique(classes, return_counts=True)
    probs = counts / indices.shape[0]
    return -np.sum(probs * np.log2(probs))
  
  def measure(self, indices, att):
    return self.information_gain_ratio(indices, att)
  
  def information_gain(self, indices, att):
    prev = self.entropy(indices)
    sub_indices = self.split_by_att(indices, att)
    return prev - sum([self.entropy(s) * len(s) for s in sub_indices if len(s) > 0]) / indices.shape[0]
  
  def intrinsic_value(self, sub_indices):
    pbs = np.array([i.size for i in sub_indices if i.size > 0])
    pbs = pbs / np.sum(pbs)
    return -np.sum(pbs * np.log2(pbs))

  def information_gain_ratio(self, indices, att):
    prev = self.entropy(indices)
    sub_indices = self.split_by_att(indices, att)
    ig = prev - sum([self.entropy(s) * len(s) for s in sub_indices if len(s) > 0]) / indices.shape[0]
    return ig / (self.intrinsic_value(sub_indices) + 1e-8)
  
  def gini(self, indices, att):
    sub_indices = self.split_by_att(indices, att)
    pbs = np.array([i.size for i in sub_indices if i.size > 0])
    pbs = pbs / np.sum(pbs)
    return 1 - np.sum(np.square(pbs))

  def split_by_att(self, indices, att):
    sz_sub = self.col_sz[att]
    sub_indices = [list() for _ in range(sz_sub)]
    for index in indices:
      sub_indices[self.db[index, att]].append(index)
    return [np.array(l) for l in sub_indices]
