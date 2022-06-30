#include "graph.h"

void Graph::AddNode(Node& node) {
  nodes_.push_back(&node);
}


void Graph::AddEdge(Node& src, Node& dst) {
 Edge* e = nullptr;
 e->id_ = edges_.size();
 e->src_node_ = &src;
 e->dst_node_ = &dst;
 edges_.push_back(e);
}

std::ostream& operator<<(std::ostream &out, Node& node) {
  out << node.name() << "\n";
  return out;
}

std::ostream& operator<<(std::ostream &out, Edge& edge) {
  out << "Edge id: " << edge.id() << ". src node " << edge.src_node()->name() <<
      ". dst node: " << edge.dst_node()->name() << "\n";
  return out;
}

std::ostream& operator<<(std::ostream &out, Graph& graph) {
  std::stringstream ss;
  ss << "------------------------------------------------\n";
  ss << "GRAPH INFOS:\n";
  for (auto node : graph.nodes()) {
    ss << "node name: " << node->name() << "\n";
  }
  for (auto edge : graph.edges()) {
    ss << "edge id: " << edge->id() << " node is " << edge->src_node()->name()
       << " to " << edge->dst_node()->name() << "\n";
  }
  out << ss.str();
  return out;
}
