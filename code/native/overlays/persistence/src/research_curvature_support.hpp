#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace lazyipdca::qcqp::research {
// SHA-256 of a canonical little-endian, length-delimited binary64 stream.
// Used only in qualification traces, never in the timed cache comparison.
class Sha256 {
  std::array<std::uint32_t,8> h{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
  std::array<unsigned char,64> block{};std::size_t used=0;std::uint64_t length=0;
  static std::uint32_t rr(std::uint32_t x,int n){return(x>>n)|(x<<(32-n));}
  void compress(){
    static constexpr std::uint32_t k[64]={
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::uint32_t w[64];for(int j=0;j<16;++j)w[j]=(std::uint32_t(block[4*j])<<24)|(std::uint32_t(block[4*j+1])<<16)|(std::uint32_t(block[4*j+2])<<8)|block[4*j+3];
    for(int j=16;j<64;++j){auto x=w[j-15],y=w[j-2];w[j]=w[j-16]+(rr(x,7)^rr(x,18)^(x>>3))+w[j-7]+(rr(y,17)^rr(y,19)^(y>>10));}
    auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];
    for(int j=0;j<64;++j){auto t=v+(rr(e,6)^rr(e,11)^rr(e,25))+((e&f)^(~e&g))+k[j]+w[j];auto u=(rr(a,2)^rr(a,13)^rr(a,22))+((a&b)^(a&c)^(b&c));v=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;}
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;
  }
public:
  void byte(unsigned char x){block[used++]=x;++length;if(used==64){compress();used=0;}}
  void raw(const std::string& s){for(unsigned char x:s)byte(x);}
  void integer(std::uint64_t v){for(int i=0;i<8;++i){byte(static_cast<unsigned char>(v&255));v>>=8;}}
  void real(double v){static_assert(sizeof(double)==8,"binary64 required");std::uint64_t bits;std::memcpy(&bits,&v,8);integer(bits);}
  template<class V>void vector_real(const V& a){integer(a.size());for(double v:a)real(v);}
  template<class V>void vector_integer(const V& a){integer(a.size());for(auto v:a)integer(static_cast<std::uint64_t>(v));}
  std::string finish(){auto bits=length*8;byte(0x80);while(used!=56)byte(0);for(int i=7;i>=0;--i)byte(static_cast<unsigned char>(bits>>(8*i)));std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto v:h)out<<std::setw(8)<<v;return out.str();}
};
inline std::size_t& cache_instance_count(){static std::size_t n=0;return n;}
inline std::size_t next_cache_instance(){return ++cache_instance_count();}
inline void cache_session_completed(const std::string& mode,const std::string& path){
  if(path.empty())return;
  std::ofstream out(path,std::ios::app);
  if(!out)throw std::runtime_error("cannot complete curvature research evidence: "+path);
  out<<"{\"schema\":\"research-curvature-cache-session-v1\",\"mode\":\""<<mode
     <<"\",\"model_instances\":"<<cache_instance_count()<<",\"solver_returned\":true}\n";
  out.flush();if(!out)throw std::runtime_error("curvature session evidence write failed");
}
inline void validate_cache_mode(const std::string& mode,bool active,const std::string& engine,
    const std::string& study,bool objective_lazy){
  if(mode!="reuse"&&mode!="rebuild")throw std::invalid_argument("research-curvature-cache expects reuse|rebuild");
  if(active&&((engine!="majorants"&&engine!="auto")||study!="off"||objective_lazy))
    throw std::invalid_argument("curvature-cache ablation supports fixed R3 majorants only: study off and objective lazy convexification off");
}
}
