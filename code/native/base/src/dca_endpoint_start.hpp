// Primal-dual warm start from an independently validated endpoint vector file.
// This is NOT a trajectory checkpoint: filter/bundle history is reset and
// strictly positive slacks are reconstructed from the original inequalities.
// Workspace scaling is frozen at the ORIGINAL QCQ x0 before this import.
void dca_endpoint_start(Workspace& w,const std::string& path) {
  std::ifstream file(path,std::ios::binary);
  if(!file)throw std::runtime_error("cannot open DCA endpoint vectors: "+path);
  const std::uint16_t endian=1;
  if(*reinterpret_cast<const unsigned char*>(&endian)!=1)
    throw std::runtime_error("DCA endpoint vector format requires a little-endian host");
  char magic[8];file.read(magic,8);
  if(!file||std::string(magic,8)!="DCAEP001")throw std::runtime_error("invalid DCA endpoint vector header");
  auto read=[&](std::size_t expected){
    std::uint64_t count=0;file.read(reinterpret_cast<char*>(&count),8);
    if(!file||count!=expected)throw std::runtime_error("DCA endpoint vector dimension mismatch");
    Vec values(count);file.read(reinterpret_cast<char*>(values.data()),std::streamsize(count*sizeof(double)));
    if(!file||!finite(values))throw std::runtime_error("truncated/nonfinite DCA endpoint vector");
    return values;
  };
  const auto x=read(w.original.n),eq=read(w.me),iq=read(w.original.inequalities.size()),
             lo=read(w.original.n),up=read(w.original.n);
  if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing DCA endpoint data");
  for(int j=0;j<w.original.n;++j)if(w.index[j]<0 && x[j]!=w.fixed[j])
    throw std::runtime_error("DCA endpoint violates an exactly fixed coordinate");
  for(const auto* v:{&iq,&lo,&up})for(double z:*v)if(z<0)
    throw std::runtime_error("negative endpoint inequality multiplier");
  for(int j=0;j<w.n;++j)w.x[j]=x[w.free[j]];
  for(int e=0;e<w.me;++e)w.lambda[e]=eq[e]*w.objective.scale/w.equalities[e].scale;
  for(int i=0;i<w.mi;++i){
    const int origin=w.inequality_origin[i],kind=w.inequality_kind[i];
    const double raw=kind==0?iq[origin]:kind<0?lo[origin]:up[origin];
    w.z[i]=raw*w.objective.scale/w.inequalities[i].scale;
    w.active[i]=raw>0;
    w.s[i]=std::max(1e-14,-w.inequalities[i].original_at(w.x));
  }
  if(!finite(w.lambda)||!finite(w.z))throw std::runtime_error("nonfinite scaled endpoint multipliers");
  w.evaluate();
  if(w.st.live_display_level>0)std::cout<<"[DCA-ENDPOINT-START] slacks=reconstructed history=reset scaling=original-qcq-x0 source="<<path<<"\n";
}
