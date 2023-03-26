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
#include <stdio.h>
#include <processthreadsapi.h>

#ifdef _WIN32
#include <Windows.h>
double get_cpu_time(){
    FILETIME a,b,c,d;
    if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
            (double)(d.dwLowDateTime |
            ((unsigned long long)d.dwHighDateTime << 32));
    }else{
        //  Handle error
        return 0.0;
    }
}
#else
#include <time.h>
#include <sys/time.h>
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}
#endif

namespace fs = std::filesystem;
using namespace lemon;
using namespace std;


int main(int argc, char** argv)
{
  SmartDigraph g;
  SmartDigraph::ArcMap<int> lowerMap(g);
  SmartDigraph::ArcMap<int> capacityMap(g);
  SmartDigraph::ArcMap<int> costMap(g);
  SmartDigraph::NodeMap<int> supplyMap(g);
  double start_readin = get_cpu_time();
  readDimacsMin(std::cin, g, lowerMap, capacityMap, costMap, supplyMap);
  double end_readin = get_cpu_time();
  printf("%.0f ",(end_readin - start_readin) * 1000000.0);
  
  int acc = 0;
  for (SmartDigraph::NodeIt n(g); n != INVALID; ++n)
    acc += supplyMap[n];
  if(acc != 0){
    cout << "There is no feasible solution! Jumping to next file." << endl;
    return 0;
  }
  double c_start;
  double c_end;

  int algo = stoi(argv[1]);
  long long cost = -1;

  if(algo == 0){
    //NetworkSimplex
    NetworkSimplex<SmartDigraph> ns(g);
    ns.lowerMap(lowerMap);
    ns.upperMap(capacityMap);
    ns.costMap(costMap);
    ns.supplyMap(supplyMap);
    ns.supplyType(NetworkSimplex<SmartDigraph>::GEQ);
    c_start = get_cpu_time();
    ns.run();
    c_end = get_cpu_time();
    cost = ns.totalCost<long long>();
  }else if(algo == 2){
    //SSP
    CapacityScaling<SmartDigraph> cas(g);
    cas.lowerMap(lowerMap);
    cas.upperMap(capacityMap);
    cas.costMap(costMap);
    cas.supplyMap(supplyMap);
    c_start = get_cpu_time();
    cas.run(1);
    c_end = get_cpu_time();
    cost = cas.totalCost<long long>();
  }else if(algo == 3){
    //CAS
    CapacityScaling<SmartDigraph> cas(g);
    cas.lowerMap(lowerMap);
    cas.upperMap(capacityMap);
    cas.costMap(costMap);
    cas.supplyMap(supplyMap);
    c_start = get_cpu_time();
    cas.run();
    c_end = get_cpu_time();
    cost = cas.totalCost<long long>();
  }else if(algo == 4){
    //Cycle Canceling
    CycleCanceling<SmartDigraph> cc(g);
    cc.lowerMap(lowerMap);
    cc.upperMap(capacityMap);
    cc.costMap(costMap);
    cc.supplyMap(supplyMap);
    c_start = get_cpu_time();
    cc.run(CycleCanceling<SmartDigraph>::SIMPLE_CYCLE_CANCELING);
    c_end = get_cpu_time();
    cost = cc.totalCost<long long>();
  }else if(algo == 5){
    CycleCanceling<SmartDigraph> cc(g);
    cc.lowerMap(lowerMap);
    cc.upperMap(capacityMap);
    cc.costMap(costMap);
    cc.supplyMap(supplyMap);
    c_start = get_cpu_time();
    cc.run(CycleCanceling<SmartDigraph>::MINIMUM_MEAN_CYCLE_CANCELING);
    c_end = get_cpu_time();
    cost = cc.totalCost<long long>(); 
  }else if(algo == 6){
    CycleCanceling<SmartDigraph> cc(g);
    cc.lowerMap(lowerMap);
    cc.upperMap(capacityMap);
    cc.costMap(costMap);
    cc.supplyMap(supplyMap);
    c_start = get_cpu_time();
    cc.run(CycleCanceling<SmartDigraph>::CANCEL_AND_TIGHTEN);
    c_end = get_cpu_time();
    cost = cc.totalCost<long long>(); 
  }
  double elapsed = (c_end - c_start) * 1000000.0;
  printf("%.0f ",elapsed);
  printf("%lu\n",(unsigned long)cost);
  return 0;
}