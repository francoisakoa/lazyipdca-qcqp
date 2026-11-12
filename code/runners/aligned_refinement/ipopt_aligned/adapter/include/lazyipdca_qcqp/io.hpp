#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include <string>

namespace lazyipdca::qcqp {
QCQPProblem read_qcq_compact(const std::string& path);
QCQPProblem read_qcq(const std::string& path);
void write_result_json(const Result& result, const std::string& problem_name, const std::string& path);
std::string result_json(const Result& result, const std::string& problem_name);
}
