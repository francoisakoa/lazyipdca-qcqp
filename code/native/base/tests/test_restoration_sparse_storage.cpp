#include "../src/restoration.cpp"
#include <cstdlib>
#include <iomanip>
using namespace lazyipdca::qcqp;

static void require(bool c,const char* message) {
  if(!c) {std::cerr<<"FAIL "<<message<<"\n"; std::exit(1);}
}

class RecordingFactor final : public SymmetricFactorization {
 public:
  explicit RecordingFactor(std::unique_ptr<SymmetricFactorization> inner)
      : inner_(std::move(inner)) {}
  bool factor(int n,int primal,const std::vector<SymmetricTriplet>& entries,
              std::string& message) override {
    dimension=n; positive=primal; triplets=entries.size();
    return inner_->factor(n,primal,entries,message);
  }
  bool solve(Vec& rhs,std::string& message) const override {
    return inner_->solve(rhs,message);
  }
  const char* name() const override { return inner_->name(); }
  int dimension=0,positive=0;
  std::size_t triplets=0;
 private:
  std::unique_ptr<SymmetricFactorization> inner_;
};

static void check_dense_columns() {
  // Independent dense-Gram reference on a full-row-rank Jacobian with three
  // shared columns, unrelated to any benchmark's equations or variable names.
  constexpr int m=96,n=133;
  LegacyResidualJacobian rj;rj.sparse=true;
  rj.sparse_rows.resize(m);rj.residual.resize(m);
  for(int a=0;a<m;++a) {
    rj.residual[a]=0.03*(a%13-6);
    rj.sparse_rows[a]={{0,0.2+0.01*a},{1,-0.5+0.003*a},
                       {2,0.7+0.02*(a%9)},{a+3,1.0+0.01*(a%7)}};
  }
  std::vector<int> free;
  for(int j=0;j<n;++j)free.push_back(j);
  for(double damping:{1e-2,1e-12}) {
    Settings ref;ref.linear_backend="native";std::string message;
    auto fref=make_symmetric_factorization(ref,message);Vec expected;
    require(sparse_row_space_lm_direction(rj,free,damping,*fref,expected,n,
                                         message,false,false),"dense Gram reference");
    for(const std::string backend:{"native","cholesk","ldl","mumps"}) {
      if(!linear_backend_compiled(backend))continue;
      Settings st;st.linear_backend=backend;st.linear_metis=true;
      auto inner=make_symmetric_factorization(st,message);
      require(bool(inner),"border backend exists");
      RecordingFactor f(std::move(inner));Vec actual;
      const bool ok=sparse_row_space_lm_direction(rj,free,damping,f,actual,n,message);
      if(!ok) std::cerr<<backend<<": "<<message<<"\n";
      require(ok,"bordered solve");
      require(f.dimension==m+3 && f.positive==m,"border inertia dimensions");
      require(f.triplets==5*m+3,"linear storage of shared columns");
      double error=0.0;
      for(int j=0;j<n;++j)error=std::max(error,std::abs(expected[j]-actual[j]));
      require(error<1e-8*std::max(1.0,inf_norm(expected)),"dense reference direction parity");
      // Check the LM first-order equation using only the original Jacobian,
      // not the assembled bordered matrix or the factorization's diagnostics.
      const double shift=std::max(1e-15,damping*static_cast<double>(n)/m);
      Vec stationarity(n,0.0);
      for(int j=0;j<n;++j)stationarity[j]=shift*actual[j];
      for(int a=0;a<m;++a) {
        double norm2=0.0,jd=0.0;
        for(auto [j,v]:rj.sparse_rows[a]) {norm2+=v*v;jd+=v*actual[j];}
        const double residual=(jd+rj.residual[a])/std::max(1.0,norm2);
        for(auto [j,v]:rj.sparse_rows[a])stationarity[j]+=v*residual;
      }
      require(inf_norm(stationarity)<1e-8,"independent original LM stationarity");
    }
  }
}
int main() {
  check_dense_columns();
  QCQPProblem p;
  p.n=67;
  p.x0.resize(p.n);
  p.lb.assign(p.n,-INFINITY);p.ub.assign(p.n,INFINITY);
  for(int j=0;j<p.n;++j) p.x0[j]=(j%7-3)*0.25;
  p.objective.linear.assign(p.n,0.0);p.objective.hessian.n=p.n;
  for(int k=0;k<19;++k) {
    QuadraticFunction q;q.hessian.n=p.n;q.linear.assign(p.n,0.0);
    q.constant=0.1*(k+1);
    const int i=k*3,j=i+1,t=i+2;
    q.hessian.entries={{i,i,0.25},{i,j,1e16},{i,j,-1e16},
                       {i,j,-0.625},{j,j,-1.25},{j,t,0.3}};
    q.linear[i]=0.31;q.linear[j]=-0.9;
    q.signed_square={{i,0.5},{j,-0.3}};
    p.equalities.push_back(q);
  }
  auto g=p.equalities.front();g.constant=-100.0;p.inequalities.push_back(g);
  g.constant=100.0;p.inequalities.push_back(g);
  const auto dense=legacy_residual_jacobian(p,p.x0);
  const auto sparse=legacy_residual_jacobian(p,p.x0,true);
  require(sparse.rows.empty(),"sparse Jacobian must not allocate dense rows");
  require(binary_equal(dense.residual,sparse.residual),"residual parity");
  Vec direction=p.x0;
  for(int j=0;j<p.n;++j) direction[j]=(j%3?1.0:-2.0);
  for(std::size_t i=0;i<dense.rows.size();++i) {
    Vec expanded(p.n,0.0);
    for(auto e:sparse.sparse_rows[i]) expanded[e.first]=e.second;
    require(binary_equal(expanded,dense.rows[i]),"gradient coordinate bytewise parity");
    const double a=legacy_row_dot(dense,i,direction),b=legacy_row_dot(sparse,i,direction);
    require(std::memcmp(&a,&b,sizeof(double))==0,"row-dot bytewise parity");
  }
  const double a=legacy_squared_merit(dense),b=legacy_squared_merit_values(p,p.x0,{});
  require(std::memcmp(&a,&b,sizeof(double))==0,"residual-only merit bytewise parity");
  std::vector<int> free;
  for(int j=0;j<p.n;++j) if(j%11!=0)free.push_back(j);
  Settings st;st.linear_backend="native";std::string message;
  auto fa=make_symmetric_factorization(st,message),fb=make_symmetric_factorization(st,message);
  Vec da,db;
  require(sparse_row_space_lm_direction(dense,free,0.0125,*fa,da,p.n,message,true),"dense LM solve");
  require(sparse_row_space_lm_direction(sparse,free,0.0125,*fb,db,p.n,message,true),"sparse LM solve");
  require(binary_equal(da,db),"active-bound LM direction bytewise parity");
  for(auto* group:{&p.equalities,&p.inequalities}) for(auto& q:*group) {
    q.compact_linear_dimension=p.n;
    for(int j=0;j<p.n;++j)if(q.linear[j]!=0.0) q.compact_linear.push_back({j,q.linear[j]});
    q.linear.clear();
  }
  const auto compact=legacy_residual_jacobian(p,p.x0,true);
  require(binary_equal(compact.residual,sparse.residual),"compact residual parity");
  require(compact.sparse_rows==sparse.sparse_rows,"compact sparse gradient parity");
  std::cout<<"PASS gradient, signed-square, Hessian duplicates, compact/dense linear, inactive inequality, free-variable LM and residual-only merit exact parity\n";
}
