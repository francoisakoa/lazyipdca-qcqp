#include "lazyipdca_qcqp/io.hpp"
#include "qcq_bound_sentinels.hpp"
#include <charconv>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string_view>
#include <stdexcept>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
namespace lazyipdca::qcqp {
namespace {
class MappedInput {
public:
  const char* data=nullptr;std::size_t size=0;
#if defined(__unix__) || defined(__APPLE__)
  int fd=-1;
  explicit MappedInput(const std::string& path) {
    fd=open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("cannot open QCQ file: "+path);
    struct stat s;if(fstat(fd,&s)!=0||s.st_size<=0){close(fd);throw std::runtime_error("empty or unreadable QCQ file: "+path);}
    size=(std::size_t)s.st_size;void* p=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);
    if(p==MAP_FAILED){close(fd);throw std::runtime_error("cannot map QCQ file: "+path);}data=(const char*)p;
  }
  ~MappedInput(){if(data)munmap((void*)data,size);if(fd>=0)close(fd);}
#else
  std::string storage;
  explicit MappedInput(const std::string& path){std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open QCQ file: "+path);storage.assign(std::istreambuf_iterator<char>(in),{});data=storage.data();size=storage.size();}
#endif
  MappedInput(const MappedInput&)=delete;
};
class Scanner {
  const char* p;const char* end;
public:
  explicit Scanner(const MappedInput& input):p(input.data),end(p+input.size){}
  void space(){while(p<end&&(*p==' '||*p=='\n'||*p=='\r'||*p=='\t'||*p=='\v'||*p=='\f'))++p;}
  std::string_view token(){space();const char* b=p;while(p<end&&*p!=' '&&*p!='\n'&&*p!='\r'&&*p!='\t'&&*p!='\v'&&*p!='\f')++p;if(b==p)throw std::runtime_error("QCQ parse error: unexpected EOF");return {b,(std::size_t)(p-b)};}
  void expect(std::string_view want){if(token()!=want)throw std::runtime_error("QCQ parse error: expected "+std::string(want));}
  int integer(){auto t=token();int x;auto b=t.data();if(!t.empty()&&t[0]=='+')++b;auto r=std::from_chars(b,t.data()+t.size(),x);if(r.ec!=std::errc()||r.ptr!=t.data()+t.size())throw std::runtime_error("QCQ parse error: invalid integer");return x;}
  static double number(std::string_view t){
    if(t=="0"||t=="-0"||t=="+0")return t=="-0"?-0.0:0.0;
    if(t=="INF"||t=="+INF"||t=="inf"||t=="+inf")return INFINITY;
    if(t=="-INF"||t=="-inf")return -INFINITY;
    // Match the legacy std::stod rounding and accepted numeric syntax.
    std::size_t n=0;double x=std::stod(std::string(t),&n);if(n!=t.size())throw std::runtime_error("QCQ parse error: invalid floating value");return x;
  }
  double real(){return number(token());}
  Vec vector(int n){Vec v(n);for(double& x:v)x=real();return v;}
  void sparse_linear(QuadraticFunction& q,int n){
    q.compact_linear_dimension=n;int j=0;space();
    static constexpr char zeros[]="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    while(j<n){
      // Skip exact canonical zero tokens in blocks; never skip a coefficient
      // based on magnitude, tolerance or position in the model.
      if(n-j>=16&&end-p>=32&&std::memcmp(p,zeros,32)==0){j+=16;p+=32;continue;}
      const double x=real();if(!std::isfinite(x))throw std::runtime_error("QCQ parse error: nonfinite linear coefficient");
      if(x!=0.0)q.compact_linear.push_back({j,x});
      ++j;space();
    }
  }
  QuadraticFunction quadratic(int n,bool sparse_format){
    expect("quadratic");QuadraticFunction q;expect("name");q.name=token();expect("constant");q.constant=real();
    if(sparse_format){
      expect("linear_nnz");const int count=integer();
      if(count<0||count>n)throw std::runtime_error("QCQ parse error: invalid linear count");
      q.compact_linear_dimension=n;int previous=-1;
      for(int k=0;k<count;++k){const int j=integer();const double x=real();
        if(j<=previous||j>=n||!std::isfinite(x))throw std::runtime_error("QCQ parse error: invalid sparse linear entry");
        previous=j;if(x!=0.0)q.compact_linear.push_back({j,x});
      }
    }else{expect("linear");sparse_linear(q,n);}
    expect("hessian_nnz");int nnz=integer();if(nnz<0)throw std::runtime_error("QCQ parse error: negative Hessian nnz");
    q.hessian.n=n;q.hessian.entries.reserve(nnz);
    for(int k=0;k<nnz;++k){int i=integer(),j=integer();double x=real();if(i>j)std::swap(i,j);q.hessian.entries.push_back({i,j,x});}
    expect("end_quadratic");q.validate(n);return q;
  }
};
}
QCQPProblem read_qcq_compact(const std::string& path){
  MappedInput input(path);Scanner in(input);QCQPProblem p;
  const auto magic=in.token();const bool sparse_format=magic=="LAZYIPDCA_QCQP_V2";
  if(!sparse_format&&magic!="LAZYIPDCA_QCQP_V1")throw std::runtime_error("unsupported QCQ format");
  in.expect("name");p.name=in.token();in.expect("n");p.n=in.integer();
  if(p.n<=0)throw std::runtime_error("QCQ parse error: n must be positive");
  in.expect("x0");p.x0=in.vector(p.n);in.expect("lb");p.lb=in.vector(p.n);in.expect("ub");p.ub=in.vector(p.n);
  detail::canonicalize_qcq_absent_bound_sentinels(p);
  in.expect("objective");p.objective=in.quadratic(p.n,sparse_format);
  in.expect("equalities");int me=in.integer();if(me<0)throw std::runtime_error("QCQ parse error: negative equality count");p.equalities.reserve(me);for(int i=0;i<me;++i)p.equalities.push_back(in.quadratic(p.n,sparse_format));
  in.expect("inequalities");int mi=in.integer();if(mi<0)throw std::runtime_error("QCQ parse error: negative inequality count");p.inequalities.reserve(mi);for(int i=0;i<mi;++i)p.inequalities.push_back(in.quadratic(p.n,sparse_format));
  in.expect("end_problem");p.validate();return p;
}
}
