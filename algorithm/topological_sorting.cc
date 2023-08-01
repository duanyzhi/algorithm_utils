#include <iostream>
#include <queue>
#include <vector>

struct Node {
 public:
  Node(std::string name) : name_(name) {}
  void add(Node* node) {
    ++(node->in_degree);
    next_nodes.emplace_back(node);
  }
  int in_degree = 0;
  std::vector<Node*> next_nodes;

  std::string name_;
};

struct Graph {
 public:
  Graph() {}
  void add_node(Node* src, Node* dst) {};
  void add(Node* node) {
    all_node.emplace_back(node);
  };
  std::vector<Node*> all_node;
  void topological_sort();
};

void Graph::topological_sort() {
  std::queue<Node*> run_queue; 
  while (!all_node.empty()) {
    int run_id = -1;
    for (int i = 0; i < all_node.size(); i++) {
      Node* node = all_node[i];
      if (0 != node->in_degree) continue;
      for (Node* edge_node : node->next_nodes) {
        --(edge_node->in_degree);
      }
      run_id = i;
      run_queue.push(node);
      break;
    }
    if (-1 != run_id && !all_node.empty())
       all_node.erase(all_node.begin() + run_id);
  }
  while (!run_queue.empty()) {
    Node* n = run_queue.front();
    std::cout << n->name_ << " -> ";
    run_queue.pop();
  }
  std::cout << "\n";
}

int main() {
  Node* a = new Node("a");
  Node* b = new Node("b");
  Node* c = new Node("c");
  Node* d = new Node("d");
  Node* e = new Node("e");
  Node* f = new Node("f");

  a->add(b);
  a->add(c);

  e->add(c);
  b->add(d);
  c->add(d);
  f->add(e);

  Graph g;
  g.add(a);
  g.add(b);
  g.add(c);
  g.add(d);
  g.add(e);
  g.add(f);
  g.topological_sort();
}
