#include <pstream.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <numeric>
#include <filesystem>
#include <lemon/list_graph.h>
#include <lemon/smart_graph.h>
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <lemon/cost_scaling.h>
#include <lemon/capacity_scaling.h>
#include <lemon/cycle_canceling.h>
#include <lemon/dimacs.h>
namespace fs = std::filesystem;
using namespace lemon;
using namespace std;

template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}


int main(int argc, char** argv)
{
  SmartDigraph g;
  SmartDigraph::ArcMap<int> lowerMap(g);
  SmartDigraph::ArcMap<int> capacityMap(g);
  SmartDigraph::ArcMap<int> costMap(g);
  SmartDigraph::NodeMap<int> supplyMap(g);
  readDimacsMin(std::cin, g, lowerMap, capacityMap, costMap, supplyMap);

  int acc = 0;
  for (SmartDigraph::NodeIt n(g); n != INVALID; ++n)
    acc += supplyMap[n];
  if(acc != 0){
    cout << "There is no feasible solution! Jumping to next file." << endl;
    return 0;
  }

  int algo = stoi(argv[1]);
  int time = -1;
  long long cost = -1;
  if(algo == 0){
    //NetworkSimplex
    NetworkSimplex<SmartDigraph> ns(g);
    ns.lowerMap(lowerMap);
    ns.upperMap(capacityMap);
    ns.costMap(costMap);
    ns.supplyMap(supplyMap);
    ns.supplyType(NetworkSimplex<SmartDigraph>::GEQ);
    auto start = std::chrono::steady_clock::now();
    ns.run();
    time = since<std::chrono::microseconds>(start).count();
    cost = ns.totalCost<long long>();
  }else if(algo == 2){
    //SSP
    CapacityScaling<SmartDigraph> cas(g);
    cas.lowerMap(lowerMap);
    cas.upperMap(capacityMap);
    cas.costMap(costMap);
    cas.supplyMap(supplyMap);
    auto start = std::chrono::steady_clock::now();
    cas.run(1);
    time = since<std::chrono::microseconds>(start).count();
    cost = cas.totalCost<long long>();
  }else if(algo == 3){
    //CAS
    CapacityScaling<SmartDigraph> cas(g);
    cas.lowerMap(lowerMap);
    cas.upperMap(capacityMap);
    cas.costMap(costMap);
    cas.supplyMap(supplyMap);
    auto start = std::chrono::steady_clock::now();
    cas.run();
    time = since<std::chrono::microseconds>(start).count();
    cost = cas.totalCost<long long>();
  }else if(algo == 4){
    //Cycle Canceling
    CycleCanceling<SmartDigraph> cc(g);
    cc.lowerMap(lowerMap);
    cc.upperMap(capacityMap);
    cc.costMap(costMap);
    cc.supplyMap(supplyMap);
    auto start = std::chrono::steady_clock::now();
    cc.run(CycleCanceling<SmartDigraph>::SIMPLE_CYCLE_CANCELING);
    time = since<std::chrono::microseconds>(start).count(); 
    cost = cc.totalCost<long long>();
  }else if(algo == 5){
    CycleCanceling<SmartDigraph> cc(g);
    cc.lowerMap(lowerMap);
    cc.upperMap(capacityMap);
    cc.costMap(costMap);
    cc.supplyMap(supplyMap);
    auto start = std::chrono::steady_clock::now();
    cc.run(CycleCanceling<SmartDigraph>::MINIMUM_MEAN_CYCLE_CANCELING);
    time = since<std::chrono::microseconds>(start).count();
    cost = cc.totalCost<long long>(); 
  }else if(algo == 6){
    CycleCanceling<SmartDigraph> cc(g);
    cc.lowerMap(lowerMap);
    cc.upperMap(capacityMap);
    cc.costMap(costMap);
    cc.supplyMap(supplyMap);
    auto start = std::chrono::steady_clock::now();
    cc.run(CycleCanceling<SmartDigraph>::CANCEL_AND_TIGHTEN);
    time = since<std::chrono::microseconds>(start).count();
    cost = cc.totalCost<long long>(); 
  }
  cout << time << " " << cost << endl;
  return 0;

}    


