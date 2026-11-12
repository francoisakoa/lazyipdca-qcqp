#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
namespace lazyipdca::qcqp::detail {
// Strict JSON reader for a public frozen primal vector. Other JSON fields are
// parsed and discarded; no dual, slack, barrier or solver history is imported.
class FrozenStartJson {
  const std::string& text;std::size_t pos=0,expected;Vec values;bool found=false;
  [[noreturn]] void fail(const char* why)const{throw std::runtime_error(std::string("frozen-start JSON: ")+why+" at byte "+std::to_string(pos));}
  void space(){while(pos<text.size()&&(text[pos]==' '||text[pos]=='\n'||text[pos]=='\r'||text[pos]=='\t'))++pos;}
  char peek(){space();return pos<text.size()?text[pos]:'\0';}
  bool take(char c){if(peek()!=c)return false;++pos;return true;}
  void need(char c){if(!take(c))fail("unexpected delimiter");}
  static bool digit(char c){return c>='0'&&c<='9';}
  static int hex(char c){return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1;}
  std::string string(){
    need('"');std::string out;
    while(pos<text.size()){
      unsigned char c=text[pos++];if(c=='"')return out;
      if(c<0x20)fail("unescaped string control character");
      if(c!='\\'){out.push_back(char(c));continue;}
      if(pos==text.size())fail("truncated string escape");
      char e=text[pos++];
      if(e=='"'||e=='\\'||e=='/')out.push_back(e);
      else if(e=='b')out.push_back('\b');else if(e=='f')out.push_back('\f');
      else if(e=='n')out.push_back('\n');else if(e=='r')out.push_back('\r');else if(e=='t')out.push_back('\t');
      else if(e=='u'){
        unsigned code=0;for(int j=0;j<4;++j){if(pos==text.size()||hex(text[pos])<0)fail("invalid unicode escape");code=16*code+hex(text[pos++]);}
        // Only ASCII x0 is significant. A non-ASCII key cannot alias it.
        out.push_back(code<128?char(code):char(0x80));
      }else fail("invalid string escape");
    }
    fail("unterminated string");
  }
  std::string number(){
    space();const auto start=pos;if(pos<text.size()&&text[pos]=='-')++pos;
    if(pos==text.size()||!digit(text[pos]))fail("number required");
    if(text[pos]=='0')++pos;else while(pos<text.size()&&digit(text[pos]))++pos;
    if(pos<text.size()&&text[pos]=='.'){++pos;if(pos==text.size()||!digit(text[pos]))fail("invalid fraction");while(pos<text.size()&&digit(text[pos]))++pos;}
    if(pos<text.size()&&(text[pos]=='e'||text[pos]=='E')){++pos;if(pos<text.size()&&(text[pos]=='+'||text[pos]=='-'))++pos;if(pos==text.size()||!digit(text[pos]))fail("invalid exponent");while(pos<text.size()&&digit(text[pos]))++pos;}
    return text.substr(start,pos-start);
  }
  void value(int depth){
    if(depth>64)fail("nesting limit exceeded");const char c=peek();
    if(c=='"'){string();return;}
    if(c=='{'){
      need('{');if(take('}'))return;
      do{string();need(':');value(depth+1);}while(take(','));need('}');return;
    }
    if(c=='['){need('[');if(take(']'))return;do{value(depth+1);}while(take(','));need(']');return;}
    for(const char* word:{"true","false","null"})if(text.compare(pos,std::char_traits<char>::length(word),word)==0){pos+=std::char_traits<char>::length(word);return;}
    number();
  }
  void vector(){
    need('[');if(take(']'))return;
    do{
      const auto token=number();char* end=nullptr;const double v=std::strtod(token.c_str(),&end);
      if(end!=token.c_str()+token.size()||!std::isfinite(v))fail("x0 must contain finite numbers");
      if(values.size()>=expected)fail("x0 dimension mismatch");values.push_back(v);
    }while(take(','));need(']');
  }
 public:
  FrozenStartJson(const std::string& input,std::size_t n):text(input),expected(n){}
  Vec read(){
    need('{');if(!take('}')){
      do{const auto key=string();need(':');if(key=="x0"){if(found)fail("duplicate x0");found=true;vector();}else value(0);}while(take(','));need('}');
    }
    space();if(pos!=text.size())fail("trailing content");
    if(!found||values.size()!=expected)fail("missing x0 or dimension mismatch");return values;
  }
};
inline Vec read_frozen_start(const std::string& path,std::size_t n){
  std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("cannot open frozen-start JSON: "+path);
  const std::string text((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
  if(file.bad())throw std::runtime_error("cannot read frozen-start JSON: "+path);
  return FrozenStartJson(text,n).read();
}
} // namespace lazyipdca::qcqp::detail
