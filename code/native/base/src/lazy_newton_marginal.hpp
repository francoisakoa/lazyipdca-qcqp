// Included inside Workspace. Reuse a complete rejected factor; a probe never
// serves as a step. All retained equality RHS are zero, hence each returned
// primal vector is tangent up to the independently checked linear residual.
std::vector<AtomWitness> tangent_witnesses() {
  std::vector<AtomWitness> witnesses;
  if(!current_factor||current_factor->curvature_defect()==0)return witnesses;
  const auto started=Clock::now();++stats.atom_oracle_calls;
  auto finish=[&](){stats.atom_oracle_seconds+=seconds(started);};
  auto inverse=[&](const Vec& v,Vec& x) {
    Vec rhs(dim,0.0);std::copy(v.begin(),v.end(),rhs.begin());std::string reason;
    ++stats.atom_oracle_solves;
    if(!current_factor->solve_curvature_probe(rhs,reason)||!finite(rhs))return false;
    x.assign(rhs.begin(),rhs.begin()+nr);return true;
  };
  const int limit=std::min(nr,std::min(20,std::max(8,4*current_factor->curvature_defect())));
  Vec q(nr);for(int j=0;j<nr;++j)q[j]=std::sin((j+1)*1.713+stats.accepted_steps*.137);
  double norm=std::sqrt(dot(q,q));for(double& v:q)v/=norm;
  std::vector<Vec> basis,images;
  for(int j=0;j<limit;++j) {
    Vec v;if(!inverse(q,v)){++stats.atom_oracle_failures;finish();return {};}
    basis.push_back(q);images.push_back(v);
    for(int pass=0;pass<2;++pass)for(const auto& b:basis)axpy(-dot(b,v),b,v);
    norm=std::sqrt(dot(v,v));
    if(!(norm>1e-13*std::max(1.0,std::sqrt(dot(images.back(),images.back())))))break;
    for(double& a:v)a/=norm;
    q.swap(v);
  }
  const int m=(int)basis.size();if(m==0){finish();return {};}
  Vec small(m*m),eval;
  for(int i=0;i<m;++i)for(int j=i;j<m;++j)
    small[i+j*m]=small[j+i*m]=.5*(dot(basis[i],images[j])+dot(basis[j],images[i]));
  NewtonAtoms::eigensystem(small,m,eval);
  const double cutoff=64*std::numeric_limits<double>::epsilon()*std::max(1.0,inf_norm(eval));
  for(int j=0;j<m&&eval[j]<-cutoff&&witnesses.size()<8;++j) {
    Vec v(nr,0.0),tangent;
    for(int i=0;i<m;++i)axpy(small[i+j*m],basis[i],v);
    // One additional inverse maps away all normal components of the initial
    // seed and limits loss of tangency from Krylov orthogonalization.
    if(!inverse(v,tangent)){++stats.atom_oracle_failures;continue;}
    norm=std::sqrt(dot(tangent,tangent));if(!(norm>0.0)||!std::isfinite(norm))continue;
    for(double& a:tangent)a/=norm;
    double error=0.0;
    for(const auto& row:ar) {
      double residual=0.0,scale=0.0;
      for(std::size_t k=0;k<row.support.size();++k){residual+=row.values[k]*tangent[row.support[k]];scale+=std::abs(row.values[k]);}
      error=std::max(error,std::abs(residual)/std::max(1.0,scale));
    }
    if(error>1e-8)continue;
    NewtonAtom dense;dense.support.resize(nr);std::iota(dense.support.begin(),dense.support.end(),0);dense.values=tangent;
    const auto ray=atom_rayleigh(dense);
    if(!std::isfinite(ray.first)||ray.first>=-ray.second)continue;
    stats.atom_maximum_tangent_error=std::max(stats.atom_maximum_tangent_error,error);
    witnesses.push_back({std::move(tangent),ray.first,ray.first});
  }
  stats.atom_tangent_witnesses+=(int)witnesses.size();
  if(st.live_curvature)std::cout<<"[ATOM-ORACLE] k="<<stats.accepted_steps<<" defect="<<current_factor->curvature_defect()
     <<" krylov="<<m<<" witnesses="<<witnesses.size()<<" seconds="<<seconds(started)<<std::endl;
  finish();return witnesses;
}
