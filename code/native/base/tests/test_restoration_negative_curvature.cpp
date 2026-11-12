#include "../src/restoration.cpp"
#include <cstdlib>

using namespace lazyipdca::qcqp;

static void require(bool condition,const char* message) {
  if(!condition){std::cerr<<"FAIL "<<message<<"\n";std::exit(1);}
}

// Independent unit spheres have zero first-order residual gradients at their
// centres, while the squared-residual merit has strictly negative curvature.
// The construction exercises a general degenerate feasibility problem rather
// than any named benchmark or a geometrically supplied feasible point.
static QCQPProblem spheres(int number=200) {
  QCQPProblem p;p.name="degenerate_quadratic_feasibility";p.n=3*number;
  p.x0.assign(p.n,0);p.lb.assign(p.n,-INFINITY);p.ub.assign(p.n,INFINITY);
  p.objective.compact_linear_dimension=p.n;p.objective.hessian.n=p.n;
  for(int i=0;i<number;++i) {
    QuadraticFunction h;h.name="sphere_"+std::to_string(i);h.constant=-1;
    h.compact_linear_dimension=p.n;h.hessian.n=p.n;
    for(int d=0;d<3;++d)h.hessian.entries.push_back({3*i+d,3*i+d,2});
    p.equalities.push_back(std::move(h));
  }
  p.validate();return p;
}

int main() {
  Settings st;st.verbose=false;st.live_display_level=0;st.linear_backend="native";
  st.feasibility_tolerance=1e-8;st.restoration_max_iterations=50;
  st.restoration_max_restarts=0;st.restoration_minimax_enabled=false;
  auto p=spheres();const Vec frozen=p.x0;Vec x=frozen;
  auto step=restoration_negative_curvature_step(p,x,st);
  require(step.attempted&&step.accepted,"negative-curvature escape not accepted");
  require(step.rayleigh<0&&step.matvecs<=25,"uncertified or unbounded curvature probe");
  require(x!=frozen&&p.x0==frozen,"escape did not preserve the supplied initial point");
  long double before=0,after=0;
  for(const auto& h:p.equalities){const double a=h.value(frozen),b=h.value(x);before+=0.5L*a*a;after+=0.5L*b*b;}
  require(after<0.99L*before,"actual original merit does not decrease");

  // A Phase-I stage may request a tenfold reduction instead of terminal
  // precision. Its zero-Jacobian anchor at theta=1 is NOT near target=.1.
  // The old 10*target threshold disabled this exact saddle escape.
  auto coarse_problem=spheres(2);coarse_problem.objective.compact_linear={{0,1}};
  Settings coarse=st;coarse.feasibility_tolerance=.1;
  coarse.restoration_failed_linearization_context=true;
  const Vec coarse_start=coarse_problem.x0;Vec coarse_x=coarse_start;
  const auto coarse_escape=restoration_negative_curvature_step(coarse_problem,coarse_x,coarse);
  require(coarse_escape.attempted&&coarse_escape.accepted,
      "coarse phase-I target blocked a violated zero-Jacobian escape");
  require(coarse_escape.merit_after<coarse_escape.merit_before&&coarse_problem.x0==coarse_start,
      "coarse escape changed frozen start or lacked actual merit decrease");
  // Nonzero J'r already supplies a first-order descent direction, even if
  // the squared merit also has negative curvature. A coarse target must not
  // eagerly replace that useful linearized model with a curvature escape.
  QCQPProblem first_order;first_order.name="nonblind_coarse_feasibility";first_order.n=1;
  first_order.x0={.1};first_order.lb={-INFINITY};first_order.ub={INFINITY};
  first_order.objective.linear={0};first_order.objective.hessian.n=1;
  QuadraticFunction nonblind;nonblind.constant=-1;nonblind.linear={0};nonblind.hessian.n=1;
  nonblind.hessian.entries={{0,0,2}};first_order.equalities={nonblind};first_order.validate();
  Vec first_order_x=first_order.x0;const auto first_order_step=restoration_negative_curvature_step(first_order,first_order_x,coarse);
  require(!first_order_step.attempted&&!first_order_step.accepted&&first_order_x==first_order.x0,
      "coarse target eagerly explored curvature despite a useful first-order model");
  auto far_target=coarse;far_target.feasibility_tolerance=.001;first_order_x=first_order.x0;
  require(restoration_negative_curvature_step(first_order,first_order_x,far_target).accepted,
      "historical curvature path far above its target was changed");
  // A global nonzero J'r does not make every residual row solvable. A
  // violated quadratic row at its centre is blind even while another affine
  // row provides ordinary progress. The curvature escape must remain eligible.
  QCQPProblem mixed; mixed.name="partially_blind_coarse_feasibility";mixed.n=2;
  mixed.x0={0,.5};mixed.lb={-INFINITY,-INFINITY};mixed.ub={INFINITY,INFINITY};
  mixed.objective.linear={0,0};mixed.objective.hessian.n=2;
  QuadraticFunction blind_row;blind_row.constant=-1;blind_row.linear={0,0};blind_row.hessian.n=2;blind_row.hessian.entries={{0,0,2}};
  QuadraticFunction nonblind_row;nonblind_row.constant=-2;nonblind_row.linear={0,1};nonblind_row.hessian.n=2;
  mixed.equalities={blind_row,nonblind_row};mixed.validate();Vec mixed_x=mixed.x0;
  auto mixed_coarse=coarse;mixed_coarse.feasibility_tolerance=.15;
  const auto mixed_escape=restoration_negative_curvature_step(mixed,mixed_x,mixed_coarse);
  require(mixed_escape.attempted&&mixed_escape.accepted&&mixed_escape.merit_after<mixed_escape.merit_before,
      "nonzero global gradient masked a violated blind quadratic row");
  // Infinity is an absent bound on either side, including historical +INF
  // lower sentinels. This is a search ray, never a surrogate finite box.
  auto sentinel=coarse_problem;sentinel.lb.assign(sentinel.n,INFINITY);
  Vec sentinel_x=sentinel.x0;
  const auto sentinel_escape=restoration_negative_curvature_step(sentinel,sentinel_x,coarse);
  require(sentinel_escape.accepted&&sentinel_x==coarse_x,
      "absent bound sentinels altered the unbounded curvature search");
  auto near=coarse;near.feasibility_tolerance=1;
  Vec near_x=coarse_start;
  require(!restoration_negative_curvature_step(coarse_problem,near_x,near).attempted&&near_x==coarse_start,
      "point already meeting coarse phase-I target was perturbed");
  auto optimization=p;optimization.objective.compact_linear={{0,1.0}};
  Vec control=frozen;auto no_objective=restoration_negative_curvature_step(optimization,control,st);
  require(!no_objective.attempted&&control==frozen,"optimization trajectory changed");
  auto feasible=p;for(int i=0;i<200;++i)feasible.x0[3*i]=1;
  control=feasible.x0;auto no_feasible=restoration_negative_curvature_step(feasible,control,st);
  require(!no_feasible.attempted&&control==feasible.x0,"feasible point changed");
  auto affine=p;
  for(int i=0;i<200;++i){auto& h=affine.equalities[i];h.hessian.entries.clear();h.compact_linear={{3*i,1}};}
  control=frozen;auto no_negative=restoration_negative_curvature_step(affine,control,st);
  require(no_negative.attempted&&!no_negative.accepted&&control==frozen,"nonnegative merit curvature changed iterate");
  auto expired=st;expired.restoration_deadline_seconds=0;
  control=frozen;auto no_budget=restoration_negative_curvature_step(p,control,expired);
  require(!no_budget.attempted&&control==frozen,"expired deadline ignored");

  const auto restored=FeasibilityRestoration(st).solve(p,frozen);
  require(restored.success&&restored.final_feasibility<=st.feasibility_tolerance,"integrated degenerate restoration did not converge");
  require(restored.row_space_solves>0&&restored.iterations>=2,"restoration accounting omitted curvature/LM work");
  require(p.x0==frozen,"integrated restoration changed frozen x0");
  require(feasibility(p,restored.x).total_inf<=st.feasibility_tolerance,"independent feasibility check failed");
  auto small=spheres(10);const Vec small_frozen=small.x0;
  require(!large_sparse_equality_chain_candidate(small),"small dense LM route changed");
  control=small_frozen;const auto small_step=restoration_negative_curvature_step(small,control,st);
  require(small_step.attempted&&small_step.accepted,"small degenerate feasibility excluded by dimension");
  const auto small_restored=FeasibilityRestoration(st).solve(small,small_frozen);
  require(small_restored.success&&small_restored.row_space_solves==0,"small dense LM continuation failed");
  require(small.x0==small_frozen&&feasibility(small,small_restored.x).total_inf<=st.feasibility_tolerance,"small original feasibility/start audit failed");
  small.objective.compact_linear={{0,1}};control=small_frozen;
  require(!restoration_negative_curvature_step(small,control,st).attempted&&control==small_frozen,"small nonconstant objective guard changed");
  auto medium=spheres(100);const Vec medium_frozen=medium.x0;
  const auto medium_restored=FeasibilityRestoration(st).solve(medium,medium_frozen);
  require(medium_restored.success&&medium_restored.exact_hessian_solves>0&&medium_restored.row_space_solves==0,"medium exact restoration bypassed curvature escape");
  require(medium.x0==medium_frozen&&feasibility(medium,medium_restored.x).total_inf<=st.feasibility_tolerance,"medium original feasibility/start audit failed");
  // The same Newton displacement for an anonymous affine residual with
  // three different physical units. Only the descent eligibility changes:
  // historical absolute-slope gating rejects all three tiny exact steps.
  for(double units:{.1,1.,2.}) {
    QCQPProblem q;q.n=1;q.x0={1+1e-8};q.lb={-INFINITY};q.ub={INFINITY};
    q.objective.linear={1};q.objective.hessian.n=1;
    QuadraticFunction h;h.constant=-units;h.linear={units};h.hessian.n=1;q.equalities={h};
    const auto residual=residual_model(q,q.x0,{});
    const auto gradient=merit_gradient(residual,1);Vec diagonal,radius;
    transformed_curvature_gershgorin(residual,q.x0,{},diagonal,radius);
    ExactSparseRowDotRuntimeState runtime;
    auto context=st;context.restoration_failed_linearization_context=true;
    context.restoration_private_precision_context=true;
    auto direction=exact_newton_direction(q,q.x0,residual,{},std::vector<char>(1,0),
        gradient,diagonal,radius,context,0,nullptr,runtime);
    require(direction.success&&std::abs(q.x0[0]+direction.d[0]-1)<1e-14,
        "scaled private Newton fixture lost the physical root");
    require(std::abs(h.value({q.x0[0]+direction.d[0]}))<std::abs(h.value(q.x0)),
        "scaled private Newton fixture lacked actual original merit decrease");
    context.restoration_private_precision_context=false;
    const auto coarse_historical=exact_newton_direction(q,q.x0,residual,{},std::vector<char>(1,0),
        gradient,diagonal,radius,context,0,nullptr,runtime);
    require(!coarse_historical.success,"coarse failed-linearization phase acquired relative descent");
    context.restoration_private_precision_context=true;
    context.restoration_failed_linearization_context=false;
    const auto historical=exact_newton_direction(q,q.x0,residual,{},std::vector<char>(1,0),
        gradient,diagonal,radius,context,0,nullptr,runtime);
    require(!historical.success,"historical absolute-slope eligibility changed");
  }
  std::cout<<"PASS degenerate equality feasibility, actual merit descent, fixed x0, optimization/feasible/positive-curvature/deadline guards and integrated convergence\n";
}
