#include "graph.h"

int main() {
  Graph g;
  Node n0("0"), n1("1"), n2("2"), n3("3"), n4("4");
  // Edge e1(n0, n1), e2(n0, n2), e3(n0, n3), e4(n1, n2), e5(n2, n4);
  g.AddNode(n0);
  g.AddNode(n1);
  g.AddNode(n2);
  g.AddNode(n3);
  g.AddNode(n4);
  g.AddEdge(n0, n1);
  g.AddEdge(n0, n2);
  g.AddEdge(n0, n3);
  g.AddEdge(n1, n2);
  g.AddEdge(n2, n4);
  std::cout << g;
}
