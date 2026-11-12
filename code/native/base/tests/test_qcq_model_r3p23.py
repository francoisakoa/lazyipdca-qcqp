#!/usr/bin/env python3
"""Dependency-free tests for the exact-QCQ parser and evaluator."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from qcq_model_r3p23 import read_qcq  # noqa: E402


MINI_QCQ = textwrap.dedent(
    """\
    LAZYIPDCA_QCQP_V1
    name mini_exact_qcqp
    n 2
    x0 1 2
    lb 0 -INF
    ub 3 INF
    objective
    quadratic
    name objective
    constant 1
    linear 2 -3
    hessian_nnz 3
    0 0 2
    0 1 4
    1 1 -2
    end_quadratic
    equalities 1
    quadratic
    name sum
    constant -3
    linear 1 1
    hessian_nnz 0
    end_quadratic
    inequalities 1
    quadratic
    name disk
    constant -5
    linear 0 0
    hessian_nnz 2
    0 0 2
    1 1 2
    end_quadratic
    end_problem
    """
)


class ExactQCQModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "mini.qcq"
        self.path.write_text(MINI_QCQ, encoding="utf-8")
        self.model = read_qcq(self.path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dimensions_bounds_and_sparse_structure(self) -> None:
        self.assertEqual(self.model.name, "mini_exact_qcqp")
        self.assertEqual(self.model.n, 2)
        self.assertEqual(len(self.model.equalities), 1)
        self.assertEqual(len(self.model.inequalities), 1)
        self.assertEqual(self.model.jacobian_structural_nnz(), 4)
        self.assertEqual(self.model.lagrangian_hessian_structural_nnz(), 3)
        self.assertEqual(self.model.lb[1], float("-inf"))
        self.assertEqual(self.model.ub[1], float("inf"))

    def test_exact_objective_gradient_and_constraints(self) -> None:
        point = [1.0, 2.0]
        self.assertAlmostEqual(self.model.objective_value(point), 2.0)
        self.assertEqual(self.model.objective.gradient(point, 2), [12.0, -3.0])
        self.assertEqual(self.model.constraint_values(point), [0.0, 0.0])
        self.assertEqual(
            self.model.residuals(point),
            {
                "equality_inf": 0.0,
                "inequality_inf": 0.0,
                "bound_inf": 0.0,
                "primal_inf": 0.0,
            },
        )

    def test_kkt_stationarity_uses_constraint_and_bound_signs(self) -> None:
        point = [1.0, 2.0]
        # Equality lambda=-3 and disk upper multiplier=1 give
        # grad L before bounds = [11, -2].  lam_x=[-11, 2] cancels it.
        residuals = self.model.kkt_residuals(
            point,
            constraint_multipliers=[-3.0, 1.0],
            bound_multipliers=[-11.0, 2.0],
        )
        self.assertEqual(residuals["dual_inf"], 0.0)
        self.assertEqual(residuals["complementarity_inf"], 11.0)
        self.assertEqual(residuals["dual_sign_inf"], 2.0)

    def test_rejects_wrong_dense_vector_dimension(self) -> None:
        malformed = self.path.with_name("malformed.qcq")
        malformed.write_text(MINI_QCQ.replace("x0 1 2", "x0 1"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "x0 has 1 entries; expected 2"):
            read_qcq(malformed)


if __name__ == "__main__":
    unittest.main()
