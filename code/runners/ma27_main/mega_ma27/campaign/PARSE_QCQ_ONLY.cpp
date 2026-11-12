#include "lazyipdca_qcqp/io.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

// Compatibility probe only: neither solver nor linear-algebra backend is called.
// The public reader is the entry point used by the direct Ipopt interface;
// the compact reader is the entry point used by the native Lazy CLI.
int main(int argc, char** argv) {
  int failed = 0;
  if (argc < 2) return 2;
  for (int k = 1; k < argc; ++k) {
    try {
      const auto compact = lazyipdca::qcqp::read_qcq_compact(argv[k]);
      const auto public_model = lazyipdca::qcqp::read_qcq(argv[k]);
      if (compact.n != public_model.n || compact.x0 != public_model.x0 ||
          compact.lb != public_model.lb || compact.ub != public_model.ub ||
          compact.equalities.size() != public_model.equalities.size() ||
          compact.inequalities.size() != public_model.inequalities.size()) {
        throw std::runtime_error("public/compact parser shape or start mismatch");
      }
      std::cout << "PASS\t" << argv[k] << '\t' << compact.n << '\t'
                << compact.equalities.size() << '\t' << compact.inequalities.size()
                << "\tpublic_and_compact\n";
    } catch (const std::exception& e) {
      ++failed;
      std::cout << "FAIL\t" << argv[k] << '\t' << e.what() << '\n';
    }
  }
  return failed ? 1 : 0;
}
