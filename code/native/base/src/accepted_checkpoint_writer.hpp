#pragma once
#include "lazyipdca_qcqp/result.hpp"
#include <cmath>
#include <cstdio>
#include <cerrno>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

namespace lazyipdca::qcqp::detail {
// Diagnostic state only: this is never the final endpoint or a restart image.
inline std::size_t write_accepted_checkpoint(const Result& r,const std::string& path,
    int iteration,double elapsed) {
  if(iteration<0||!std::isfinite(elapsed)||elapsed<0)
    throw std::runtime_error("invalid accepted checkpoint identity");
  std::ostringstream o;o<<std::setprecision(17)
    <<"{\n  \"schema\":\"lazyipdca-accepted-checkpoint-v1\",\n"
    <<"  \"kind\":\"accepted_checkpoint\",\n  \"status\":\"checkpoint_only\",\n"
    <<"  \"success\":false,\n  \"audit_performed\":false,\n  \"termination_audit_valid\":false,\n"
    <<"  \"coordinate_scope\":\"full_original_problem\",\n"
    <<"  \"multiplier_scope\":\"original_function_units\",\n"
    <<"  \"outer_iteration\":"<<iteration<<",\n  \"engine_elapsed_seconds\":"<<elapsed;
  auto vector=[&](const char* name,const Vec& values) {
    o<<",\n  \""<<name<<"\":[";
    for(std::size_t j=0;j<values.size();++j) {
      if(!std::isfinite(values[j]))throw std::runtime_error("nonfinite accepted checkpoint array");
      if(j)o<<',';
      o<<values[j];
    }
    o<<']';
  };
  vector("x",r.x);vector("certificate_equality_multipliers",r.certificate_equality_multipliers);
  vector("certificate_inequality_multipliers",r.certificate_inequality_multipliers);
  vector("certificate_lower_multipliers",r.certificate_lower_multipliers);
  vector("certificate_upper_multipliers",r.certificate_upper_multipliers);o<<"\n}\n";
  const std::string bytes=o.str();struct stat info{};
  if(lstat(path.c_str(),&info)==0) {
    if(!S_ISREG(info.st_mode)||info.st_uid!=geteuid()||(info.st_mode&0022))
      throw std::runtime_error("checkpoint destination is not an owner-controlled regular file");
  } else if(errno!=ENOENT)throw std::runtime_error("cannot inspect checkpoint destination");
  std::string temporary=path+".tmp.XXXXXX";std::vector<char> name(temporary.begin(),temporary.end());name.push_back(0);
  const int fd=mkstemp(name.data());if(fd<0)throw std::runtime_error("cannot create checkpoint temporary");
  FILE* f=fdopen(fd,"wb");if(!f){close(fd);unlink(name.data());throw std::runtime_error("cannot open checkpoint temporary");}
  bool okay=fchmod(fd,0600)==0&&fwrite(bytes.data(),1,bytes.size(),f)==bytes.size()&&fflush(f)==0&&fsync(fd)==0;
  if(fclose(f)!=0)okay=false;
  if(!okay||rename(name.data(),path.c_str())!=0){unlink(name.data());throw std::runtime_error("atomic checkpoint write failed");}
  return bytes.size();
}
}
