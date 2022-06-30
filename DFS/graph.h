#ifndef GRAPH_H_
#define GRAPH_H_

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>

struct Edge;
struct Node;
struct Graph;

using EdgeType = std::unordered_set<const Edge *>;

struct Node {
  Node(std::string name): name_(name) {}

 public:
  const std::string name() const { return name_; }
  void AddEdge(const Edge& edge) { edges_.insert(&edge); }
  const EdgeType edges() const { return edges_; }

 private:
  friend struct Graph;
  std::string name_;
  EdgeType edges_;
};

struct Edge {
 public:
  const int id() { return id_; }
  const Node* src_node() const { return src_node_; }
  const Node* dst_node() const { return dst_node_; }

 private:
  Edge(Node* src, Node* dst): src_node_(src), dst_node_(dst) {}
  Edge() {}

  friend struct Graph; // Graph can access private data;
  int id_;
  Node* src_node_;
  Node* dst_node_;
};

struct Graph {
 public:
  Graph() {}
  void AddNode(Node& node);
  void AddEdge(Node& src, Node& dst);
  const std::vector<Node*> nodes() const { return nodes_; }
  const std::vector<Edge*> edges() const { return edges_; }
  void DFS();
  void update(); // ??? update node and edge

 private:
  std::vector<Node*> nodes_;
  std::vector<Edge*> edges_;
};

std::ostream& operator<<(std::ostream &out, Node& node);

std::ostream& operator<<(std::ostream &out, Edge& edge);

std::ostream& operator<<(std::ostream &out, Graph& graph);

#endif // GRAPH_H_
