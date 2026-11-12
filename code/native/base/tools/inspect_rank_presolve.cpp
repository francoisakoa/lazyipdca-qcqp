#include "lazyipdca_qcqp/rank_presolve.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char** argv){
  try {
    if(argc<2||argc>4){std::cerr<<"Usage: inspect_rank_presolve input.qcq [report.json] [remove-implied-bounds:0|1]\n";return 2;}
    if(argc==4&&std::string(argv[3])!="0"&&std::string(argv[3])!="1")throw std::invalid_argument("remove-implied-bounds must be 0 or 1");
    auto p=lazyipdca::qcqp::read_qcq_compact(argv[1]);
    lazyipdca::qcqp::ConstraintPresolve reduced(p,argc<4||std::string(argv[3])=="1");
    auto s=reduced.report_json();
    if(argc>=3){std::ofstream out(argv[2]);if(!out)throw std::runtime_error("report open failed");out<<s<<'\n';}
    else std::cout<<s<<'\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
