#!/usr/bin/env python3
"""Build and separately qualify native Ipopt+MA27 and Ipopt+MUMPS comparators.

The new campaign freezes its newly compiled comparator. Old campaign binaries,
software version labels and numerical outcomes are never reused as evidence.
CasADi, when selected, supplies native headers/libraries only: no symbolic model
or CasADi optimizer API is used. An installed HSL library is mandatory.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def selected_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def discover_hsl_library(explicit=None, ipopt_prefix=None):
    """Resolve existing HSL; no download, solver fallback, or fake qualification."""
    choices = [str(explicit)] if explicit not in (None, "auto") else []
    choices.extend(os.environ[key] for key in ("IPOPT_HSL_LIBRARY", "HSL_LIBRARY")
                   if os.environ.get(key))
    if choices:
        resolved = set()
        for item in choices:
            path = Path(item).expanduser().resolve(strict=True)
            if not path.is_file():
                raise RuntimeError("HSL is not a regular library file: " + str(path))
            resolved.add(str(path))
        if len(resolved) != 1:
            raise RuntimeError("Conflicting explicit HSL paths: " + ", ".join(sorted(resolved)))
        return next(iter(resolved))

    # This exact user installation was already recorded in the handover. It
    # is only a discovery candidate; the real MA27 solve still must qualify it.
    known = Path("/data/projects/IPDCA_QCQP/deps/hsl_ma27/setup_hHIqA2RD/build/.libs/libcoinhsl.so.2.2.6")
    if known.is_file():
        return str(known.resolve())
    candidates = []
    for root in (ipopt_prefix, os.environ.get("CONDA_PREFIX"),
                 os.environ.get("IPOPT_PREFIX"), sys.prefix, "/usr/local", "/usr"):
        if not root:
            continue
        for leaf in ("lib", "lib64", "lib/x86_64-linux-gnu"):
            directory = Path(root) / leaf
            for pattern in ("libcoinhsl.so*", "libhsl.so*"):
                candidates.extend(directory.glob(pattern))
    deps = Path("/data/projects/IPDCA_QCQP/deps")
    if deps.is_dir():
        for subtree in deps.glob("hsl*"):
            if subtree.is_dir():
                candidates.extend(subtree.glob("**/libcoinhsl.so*"))
                candidates.extend(subtree.glob("**/libhsl.so*"))
    resolved = sorted({str(path.resolve()) for path in candidates if path.is_file()})
    if len(resolved) == 1:
        return resolved[0]
    if len(resolved) > 1:
        raise RuntimeError("Several HSL libraries found; pass --hsl-library explicitly: " + ", ".join(resolved))
    raise RuntimeError("No installed MA27/HSL library found. Pass --hsl-library /absolute/path/libcoinhsl.so. "
                       "The campaign will not replace MA27 with another linear solver.")


def load_build_binding(root, manifest_path=None):
    """Verify exactly the bundled C++ source, selected library, and built CLI."""
    root = Path(root).expanduser().resolve(strict=True)
    ipopt = root / "ipopt"
    manifest_path = Path(manifest_path or ipopt / "build/BUILD_IPOPT.json").resolve(strict=True)
    build = json.loads(manifest_path.read_text())
    if build.get("schema") != "native-ipopt-build-v1" or not build.get("sources"):
        raise RuntimeError("Missing native Ipopt source/build binding")
    for name in ("test_direct_ipopt_structure", "test_direct_ipopt_value_accuracy", "test_sparse_scale"):
        if not isinstance(build.get("tests", {}).get(name), dict) or build["tests"][name].get("passed") is not True:
            raise RuntimeError("Required native comparator check is absent or failed: " + name)
    for name, expected in build["sources"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unconfined Ipopt source path: " + name)
        path = ipopt / relative
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise RuntimeError("Bundled comparator source changed after build: " + name)
    binary = Path(build["binary"]).resolve(strict=True)
    if digest(binary) != build["binary_sha256"]:
        raise RuntimeError("Built comparator binary changed")
    library = Path(build["ipopt_library"]).resolve(strict=True)
    if digest(library) != build["ipopt_library_sha256"]:
        raise RuntimeError("Selected native Ipopt library changed")
    header = Path(build["ipopt_header"]).resolve(strict=True)
    if digest(header) != build["ipopt_header_sha256"]:
        raise RuntimeError("Selected native Ipopt configuration header changed")
    version = re.search(r"\bipopt=(\d+\.\d+\.\d+)(?:;|\])", build["version"])
    if not version:
        raise RuntimeError("Compiled native Ipopt version was not recorded")
    qualification = load_module(ipopt / "ma27_qualification/ma27_qualification.py",
                                "_rebuilt_campaign_ma27_qualification")
    # The inherited real-solve protocol remains intact. These two identities
    # bind THIS build, rather than pretending it is the historical binary.
    qualification.EXPECTED_BINARY_SHA256 = build["binary_sha256"]
    qualification.EXPECTED_IPOPT_VERSION = version.group(1)
    return build, qualification, manifest_path


def runtime_library_roots(build, hsl_library, ipopt_prefix=None):
    paths = [str(Path(build["ipopt_library"]).resolve().parent),
             str(Path(hsl_library).resolve().parent)]
    for prefix in (ipopt_prefix, os.environ.get("CONDA_PREFIX"), os.environ.get("IPOPT_PREFIX")):
        if prefix and (Path(prefix) / "lib").is_dir():
            paths.append(str((Path(prefix) / "lib").resolve()))
    return list(dict.fromkeys(paths))


def qualify_mumps(root, *, manifest_path=None, output=None):
    """Real MUMPS solve on the bound native adapter; no HSL or fallback."""
    root = Path(root).expanduser().resolve(strict=True)
    build, common, manifest = load_build_binding(root, manifest_path)
    fresh = Path(output or root / "validation/ipopt_mumps" /
                 (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + str(time.time_ns())))
    fresh = fresh.resolve()
    fresh.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "ipopt-mumps-runtime-qualification-v1", "qualified": False,
               "requested_backend": "mumps", "state": "NOT_QUALIFIED",
               "expected_binary_sha256": build["binary_sha256"],
               "expected_ipopt_version": common.EXPECTED_IPOPT_VERSION,
               "auditor_sha256": common.EXPECTED_AUDITOR_SHA256,
               "mumps_pivot_order": 5, "fallback_solver_used": False,
               "real_campaign_launched": False,
               "binary": common.file_identity(build["binary"]),
               "build_manifest": common.file_identity(manifest)}
    try:
        toy = root / "ipopt/ma27_qualification/toy.qcq"
        endpoint, log = fresh / "smoke_endpoint.json", fresh / "smoke.log"
        command = [build["binary"], "--problem", str(toy), "--json", str(endpoint),
                   "--linear-solver", "mumps", "--time-limit", "20", "--max-iter", "100",
                   "--tol", "1e-8", "--constr-viol-tol", "1e-6", "--dual-inf-tol", "1e-6",
                   "--compl-inf-tol", "1e-6", "--acceptable-iter", "0", "--print-level", "5"]
        environment = common.single_thread_environment()
        libraries = [str(Path(build["ipopt_library"]).resolve().parent)]
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
        environment["LD_DEBUG"] = "libs"
        environment["LD_DEBUG_OUTPUT"] = str(fresh / "loader")
        workdir = fresh / "empty_working_directory"; workdir.mkdir()
        receipt.update(smoke_command=command, toy_input=common.file_identity(toy))
        start = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.run(command, cwd=workdir, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=45, check=False)
        receipt.update(smoke_returncode=process.returncode,
                       smoke_process_wall_seconds=time.monotonic() - start,
                       runtime=common._runtime_dependencies(fresh))
        data = json.loads(endpoint.read_text())
        evidence = backend_evidence(data, log.read_text(errors="replace"),
                                    {"ipopt_version": common.EXPECTED_IPOPT_VERSION},
                                    require_banner=True, method="ipopt_mumps")
        auditor = common._load_auditor()
        audit = auditor.audit(auditor.read_model(toy), data, 1e-6)
        iterations, core = data.get("iterations"), data.get("core_s")
        budget_pass = (type(iterations) is int and 0 <= iterations <= 100 and
                       type(core) in (float, int) and 0 <= core <= 20 and
                       receipt["smoke_process_wall_seconds"] <= 20 and
                       data.get("optimization_wall_limit_seconds") == 20 and
                       data.get("max_iterations") == 100)
        precise_path = root / "validation_tools/precise_original_auditor.py"
        if not precise_path.is_file():
            raise RuntimeError("Independent Decimal150 original-QCQP auditor absent")
        precise = load_module(precise_path, "_mumps_smoke_precise_original_auditor")
        interval = precise.audit_decimal(auditor.read_model(toy), data)
        loaded = receipt["runtime"]["loaded_libraries"]
        mumps_loaded = any("mumps" in Path(item["path"]).name.lower() for item in loaded)
        strict = (audit.get("strict") is True and evidence["pass"] and budget_pass and
                  mumps_loaded and interval.get("strict") is True)
        receipt.update(smoke_result={"strict": strict, "audit": audit,
            "interval150_audit": interval, "backend_evidence": evidence,
            "budget_pass": budget_pass, "actual_mumps_library_loaded": mumps_loaded},
            smoke_endpoint=common.file_identity(endpoint), smoke_log=common.file_identity(log))
        if digest(toy) != receipt["toy_input"]["sha256"] or digest(build["binary"]) != build["binary_sha256"]:
            raise RuntimeError("MUMPS qualification input or binary changed")
        if not strict:
            raise RuntimeError("Real MUMPS smoke solve failed backend, original KKT or budget checks")
        receipt.update(qualified=True, state="QUALIFIED_MUMPS")
        write_json(fresh / "QUALIFICATION.json", receipt)
        return receipt, fresh / "QUALIFICATION.json"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        receipt["reason"] = str(exc)
        write_json(fresh / "QUALIFICATION.json", receipt)
        raise


def validate_mumps_qualification(path, binary, version):
    """Recheck independent MUMPS evidence, without accepting an MA27 receipt."""
    path = Path(path); receipt = json.loads(path.read_text())
    if (receipt.get("schema") != "ipopt-mumps-runtime-qualification-v1" or
        receipt.get("qualified") is not True or receipt.get("state") != "QUALIFIED_MUMPS" or
        receipt.get("requested_backend") != "mumps" or receipt.get("mumps_pivot_order") != 5 or
        receipt.get("expected_ipopt_version") != version or
        receipt.get("smoke_result", {}).get("strict") is not True):
        raise RuntimeError("No successful real MUMPS qualification")
    if digest(binary) != receipt["binary"]["sha256"] or digest(binary) != receipt["expected_binary_sha256"]:
        raise RuntimeError("MUMPS qualified binary changed")
    auditor_path = Path(receipt["toy_input"]["path"]).parent / "original_auditor.py"
    expected_auditor = "d500ff62371d7409180245e6f946c011f5b973537c600789d1c3b4671be3df8a"
    if receipt.get("auditor_sha256") != expected_auditor or digest(auditor_path) != expected_auditor:
        raise RuntimeError("MUMPS qualification original auditor differs")
    for key in ("binary", "build_manifest", "toy_input", "smoke_endpoint", "smoke_log"):
        identity = receipt[key]
        if digest(identity["path"]) != identity["sha256"]:
            raise RuntimeError("MUMPS qualification evidence changed: " + key)
    for identity in receipt["runtime"]["loaded_libraries"]:
        if digest(identity["path"]) != identity["sha256"]:
            raise RuntimeError("MUMPS numerical dependency changed")
    data = json.loads(Path(receipt["smoke_endpoint"]["path"]).read_text())
    log = Path(receipt["smoke_log"]["path"]).read_text(errors="replace")
    if not backend_evidence(data, log, {"ipopt_version": version},
                            require_banner=True, method="ipopt_mumps")["pass"]:
        raise RuntimeError("MUMPS qualification backend differs")
    auditor = load_module(auditor_path, "_validate_mumps_original_auditor")
    if auditor.audit(auditor.read_model(receipt["toy_input"]["path"]), data, 1e-6).get("strict") is not True:
        raise RuntimeError("MUMPS stored endpoint no longer passes the full original audit")
    if receipt["smoke_result"].get("interval150_audit", {}).get("strict") is not True:
        raise RuntimeError("MUMPS qualification lacks the successful Decimal150 audit")
    return receipt


def build_and_qualify(root, *, ipopt_prefix=None, hsl_library=None, output=None):
    root = Path(root).expanduser().resolve(strict=True)
    ipopt = root / "ipopt"
    output = Path(output or ipopt / "build").expanduser().resolve()
    evidence = root / "validation/ipopt_ma27"
    evidence.mkdir(parents=True, exist_ok=True)
    stage = "ma27"
    try:
        hsl = discover_hsl_library(hsl_library, ipopt_prefix)
        builder = load_module(ipopt / "build_adapter.py", "_campaign_native_ipopt_builder")
        selected_prefix = Path(ipopt_prefix).expanduser().resolve(strict=True) if ipopt_prefix is not None else builder.pinned_casadi_prefix()
        with selected_environment({"IPOPT_PREFIX": selected_prefix}):
            builder.build(output, ipopt_prefix=selected_prefix)
        build, qualification, manifest = load_build_binding(root, output / "BUILD_IPOPT.json")
        libraries = runtime_library_roots(build, hsl, ipopt_prefix)
        loader = os.pathsep.join(libraries)
        fresh = evidence / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        if fresh.exists():
            fresh = fresh.with_name(fresh.name + "_" + str(time.time_ns()))
        with selected_environment({"LD_LIBRARY_PATH": loader}):
            receipt = qualification.qualify_ma27(build["binary"], hsl, fresh, ipopt_prefix)
        receipt.update(software_origin="newly compiled native adapter from bundled source",
                       numerical_reuse_from_previous_campaign=False,
                       build_manifest=str(manifest), build_manifest_sha256=digest(manifest))
        write_json(fresh / "QUALIFICATION.json", receipt)
        stage = "mumps"
        mumps_receipt, mumps_path = qualify_mumps(root, manifest_path=manifest)
        config = {
            "ipopt_binary": build["binary"], "ipopt_binary_sha256": build["binary_sha256"],
            "ipopt_version": qualification.EXPECTED_IPOPT_VERSION,
            "hsl_library": hsl, "hsl_library_sha256": digest(hsl),
            "ma27_qualification": str(fresh / "QUALIFICATION.json"),
            "ma27_qualification_sha256": digest(fresh / "QUALIFICATION.json"),
            "mumps_qualification": str(mumps_path),
            "mumps_qualification_sha256": digest(mumps_path),
            "ipopt_build_manifest": str(manifest), "ipopt_build_manifest_sha256": digest(manifest),
            "ipopt_source_root": str(root), "library_roots": libraries,
            "ipopt_linear_solver": "ma27", "historical_numeric_reuse_allowed": False,
            "ipopt_mumps_linear_solver": "mumps", "ipopt_mumps_pivot_order": 5,
            "ipopt_comparators": {
                "ipopt_qcqp": {"linear_solver": "ma27", "qualification": str(fresh / "QUALIFICATION.json")},
                "ipopt_mumps": {"linear_solver": "mumps", "mumps_pivot_order": 5,
                                "qualification": str(mumps_path)}},
            "runtime_dependency_sha256": {
                item["path"]: item["sha256"] for item in
                receipt["runtime"]["loaded_libraries"] + mumps_receipt["runtime"]["loaded_libraries"]},
        }
        write_json(evidence / "RUNTIME.json", config)
        return config
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        write_json(evidence / "NOT_QUALIFIED.json", {
            "qualified": False, "reason": str(exc), "requested_backends": ["ma27", "mumps"],
            "failed_stage": stage,
            "fallback_solver_used": False, "campaign_launched": False})
        raise


# Numerical library defaults are part of this new comparator protocol.  The
# original six-profile campaign scheduler and task/audit interfaces are kept.
DEFAULT_IPOPT_VERSION = '3.14.11'
DEFAULT_FIXTURES = ('active_bound', 'quadratic_inequality', 'quadratic_equality', 'large_finite_bound')
DEFAULT_OPTIONS = dict(mu_strategy='monotone', tol=1e-8, constr_viol_tol=1e-4,
    dual_inf_tol=1.0, compl_inf_tol=1e-4, acceptable_iter=15,
    acceptable_tol=1e-6, acceptable_constr_viol_tol=1e-2,
    acceptable_dual_inf_tol=1e10, acceptable_compl_inf_tol=1e-2,
    acceptable_obj_change_tol=1e20, s_max=100.0,
    nlp_scaling_method='gradient-based', hessian_approximation='exact',
    fixed_variable_treatment='make_parameter', bound_relax_factor=1e-8)
DEFAULT_EXPLICIT_OVERRIDES = {'max_iter', 'max_cpu_time', 'max_wall_time',
    'linear_solver', 'hsllib_if_MA27', 'option_file_name_empty',
    'print_level', 'print_user_options'}


def _checked_file(path, expected, label):
    path = Path(path).expanduser().resolve(strict=True)
    if not expected or digest(path) != expected:
        raise RuntimeError(label + ' changed or lacks its SHA256')
    return path


def default_installation(configuration, *, complete=False):
    """Read the actual qualified default adapter receipt; no legacy fallback."""
    path = _checked_file(configuration['ipopt_defaults_installation'],
        configuration['ipopt_defaults_installation_sha256'], 'Default installation')
    defaults = json.loads(path.read_text())
    if defaults.get('schema') != 'ipopt-defaults-full-corpus-installation-v1':
        raise RuntimeError('A qualified full-corpus Ipopt-default installation is required')
    if (defaults.get('cases'), defaults.get('tasks'), defaults.get('datasets')) != (4078, 8156, 28):
        raise RuntimeError('Both default Ipopt backends must cover all 4078 cases and 28 datasets')
    if set(defaults.get('qualified_backends', [])) != {'ma27', 'mumps'}:
        raise RuntimeError('Both MA27 and MUMPS must be qualified with library defaults')
    if defaults.get('budgets') != {'seconds': 7200, 'max_iterations': 5000}:
        raise RuntimeError('Default Ipopt task budgets differ from the common campaign')
    if (configuration.get('ipopt_version') != DEFAULT_IPOPT_VERSION or
        configuration.get('ipopt_linear_solver') != 'ma27' or
        configuration.get('ipopt_mumps_linear_solver') != 'mumps' or
        configuration.get('ipopt_mumps_pivot_order') != 7 or
        configuration.get('ipopt_mumps_ordering_policy') != 'library-default'):
        raise RuntimeError('Two library-default comparator profiles are required; MUMPS ordering is observed, never forced')
    for new, old in (('default_binary', 'ipopt_binary'), ('hsl_library', 'hsl_library')):
        actual = _checked_file(defaults[new], defaults.get(new + '_sha256'), new)
        if actual != Path(configuration[old]).resolve() or defaults[new + '_sha256'] != configuration.get(old + '_sha256'):
            raise RuntimeError('Campaign and default-adapter identities differ: ' + old)
    _checked_file(defaults['ipopt_library'], defaults.get('ipopt_library_sha256'), 'Loaded Ipopt library')
    qpath = _checked_file(defaults['qualification'], defaults.get('qualification_sha256'), 'Default backend qualification')
    for key in ('ma27_qualification', 'mumps_qualification'):
        if Path(configuration[key]).resolve() != qpath or configuration.get(key + '_sha256') != defaults['qualification_sha256']:
            raise RuntimeError('Campaign backend receipt differs from default qualification: ' + key)
    bpath = _checked_file(defaults['build'], defaults.get('build_sha256'), 'Default comparator build')
    build = json.loads(bpath.read_text())
    if build.get('schema') != 'isolated-ipopt-library-defaults-build-v1' or build.get('original_installation_modified') is not False:
        raise RuntimeError('Isolated default comparator build proof unavailable')
    for key in ('binary_sha256', 'ipopt_library_sha256'):
        expected = defaults['default_binary_sha256'] if key == 'binary_sha256' else defaults[key]
        if build.get(key) != expected:
            raise RuntimeError('Default comparator build and installation differ: ' + key)
    if build.get('ipopt_version') != DEFAULT_IPOPT_VERSION:
        raise RuntimeError('Unexpected default comparator build version')
    if complete:
        contract = _checked_file(defaults['contract'], defaults.get('contract_sha256'), 'Complete frozen corpus contract')
        import gzip
        corpus = json.loads(gzip.decompress(contract.read_bytes()) if contract.suffix == '.gz' else contract.read_bytes())
        cases = corpus.get('cases', [])
        if len(cases) != 4078 or len({c['case_id'] for c in cases}) != 4078:
            raise RuntimeError('Default comparator corpus is incomplete or duplicated')
        # Bind the delivered adapter's source files, rather than relying only
        # on the version compiled into its header.
        cpp = [Path(x) for x in build.get('compile_command', []) if isinstance(x, str) and x.endswith('.cpp')]
        roots = {x.parent.parent.resolve() for x in cpp}
        if len(cpp) != 4 or len(roots) != 1 or not build.get('source_sha256'):
            raise RuntimeError('Default comparator source compilation binding unavailable')
        source = roots.pop()
        for name, expected in build['source_sha256'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('Unconfined default comparator source identity')
            _checked_file(source / relative, expected, 'Default adapter source ' + name)
        _checked_file(source.parent / 'PATCH_CONTRACT.json', build.get('patch_contract_sha256'), 'Default adapter patch contract')
    defaults['_installation_path'] = str(path)
    defaults['_installation_sha256'] = configuration['ipopt_defaults_installation_sha256']
    defaults['_qualification_path'] = str(qpath)
    return defaults


def _default_option_errors(endpoint, backend, seconds, max_outer):
    errors = []
    if endpoint.get('adapter_version') != 'ipopt-library-defaults-original-qcqp-v1':
        errors.append('default_adapter_identity_absent')
    options = endpoint.get('effective_options')
    if not isinstance(options, dict):
        return errors + ['effective_default_options_absent']
    expected = dict(DEFAULT_OPTIONS, linear_solver=backend, max_iter=max_outer,
        max_cpu_time=seconds, max_wall_time=seconds)
    if backend == 'mumps':expected['mumps_pivot_order'] = 7
    for key, value in expected.items():
        if options.get(key) != value:
            errors.append('effective_option_differs:' + key)
    if backend == 'mumps' and (type(endpoint.get('requested_mumps_ordering')) is not int or endpoint['requested_mumps_ordering'] != 0):
        errors.append('mumps_ordering_override_not_zero')
    return errors


def backend_evidence(endpoint, log, configuration, require_banner=False, method='ipopt_qcqp'):
    """Observe backend/defaults separately from the original-KKT audit."""
    if method not in ('ipopt_qcqp', 'ipopt_mumps'):
        raise RuntimeError('Unknown separately qualified Ipopt comparator method')
    backend = 'ma27' if method == 'ipopt_qcqp' else 'mumps'
    expected = configuration.get('ipopt_version')
    if expected != DEFAULT_IPOPT_VERSION:
        raise RuntimeError('Default comparator requires the qualified Ipopt 3.14.11 runtime')
    budget = configuration.get('task_budgets', {})
    seconds = budget.get('seconds', 7200);max_outer = budget.get('max_outer', 5000)
    errors = _default_option_errors(endpoint, backend, seconds, max_outer)
    if endpoint.get('solver') != 'Ipopt' or endpoint.get('ipopt_version') != expected or endpoint.get('linear_solver') != backend:
        errors.append('endpoint_backend_or_version_differs')
    banners = re.findall(r'running with linear solver\s+([A-Za-z][A-Za-z0-9_]*)', log, flags=re.I)
    versions = re.findall(r'This is Ipopt version\s+(\d+\.\d+\.\d+)', log, flags=re.I)
    if any(x.casefold() != backend for x in banners):errors.append('runtime_backend_banner_differs')
    if any(x != expected for x in versions):errors.append('runtime_version_banner_differs')
    if require_banner and (not banners or not versions):errors.append('required_runtime_banner_absent')
    markers = []
    for line in log.splitlines():
        if 'IPOPT_DEFAULTS_EFFECTIVE ' not in line:continue
        try:markers.append(json.loads(line.split('IPOPT_DEFAULTS_EFFECTIVE ', 1)[1]))
        except (ValueError, TypeError):errors.append('default_marker_not_valid_json')
    if not markers:errors.append('pre_solve_default_marker_absent')
    for marker in markers:
        if marker.get('ipopt_version') != expected or marker.get('effective_options') != endpoint.get('effective_options'):
            errors.append('pre_solve_and_endpoint_options_differ')
        if marker.get('input_representation_overrides') != endpoint.get('input_representation_overrides'):
            errors.append('bound_encoding_declaration_differs')
        overrides = marker.get('explicit_nonrepresentation_overrides')
        if not isinstance(overrides, list) or len(overrides) != len(DEFAULT_EXPLICIT_OVERRIDES) or set(overrides) != DEFAULT_EXPLICIT_OVERRIDES:
            errors.append('explicit_numerical_option_override_declared')
    effective = endpoint.get('effective_options') if isinstance(endpoint.get('effective_options'),dict) else {}
    return dict(method=method, requested=backend, endpoint_requested_backend=endpoint.get('linear_solver'),
        requested_mumps_ordering=endpoint.get('requested_mumps_ordering'),
        effective_mumps_pivot_order=effective.get('mumps_pivot_order'),
        expected_compiled_ipopt_version=expected, observed_banners=banners,
        observed_ipopt_versions=versions, banner_required=bool(require_banner),
        backend_observed=bool(banners), defaults_verified=not errors,
        pass_=not errors, **{'pass':not errors}, violations=sorted(set(errors)), KKT_certificate=False,
        note=None if banners else 'Backend banner absent; task certification still requires the separate original audit.')


def validate_ipopt_runtime(configuration):
    """Verify all eight original-audited default fixture receipts without solving."""
    defaults = default_installation(configuration, complete=True)
    qualification = json.loads(Path(defaults['_qualification_path']).read_text())
    if (qualification.get('schema') != 'ipopt-defaults-target-qualification-v1' or
        qualification.get('all_original_strict') is not True or
        set(qualification.get('qualified_backends', [])) != {'ma27', 'mumps'}):
        raise RuntimeError('Complete independent original-audited default qualification unavailable')
    records = qualification.get('records', [])
    wanted = {(fixture, backend) for fixture in DEFAULT_FIXTURES for backend in ('ma27', 'mumps')}
    if len(records) != 8 or {(r.get('fixture'), r.get('backend')) for r in records} != wanted:
        raise RuntimeError('Exactly four independently audited default fixtures per backend are required')
    folder = Path(defaults['_qualification_path']).parent / 'qualification'
    evidence = {'ma27':[], 'mumps':[]}
    for record in records:
        fixture, backend = record['fixture'], record['backend']
        task = folder / (fixture + '__' + backend)
        inp = _checked_file(folder / (fixture + '.qcq'), record.get('input_sha256'), 'Default qualification input')
        endpoint_path = _checked_file(task / 'endpoint.json', record.get('endpoint_sha256'), 'Default qualification endpoint')
        audit_path = _checked_file(task / 'AUDIT.json', record.get('audit_sha256'), 'Default original qualification audit')
        endpoint = json.loads(endpoint_path.read_text());audit = json.loads(audit_path.read_text())
        cfg = dict(configuration, task_budgets={'seconds':60, 'max_outer':300})
        checked = backend_evidence(endpoint, (task / 'solver.log').read_text(errors='replace'), cfg,
            require_banner=True, method='ipopt_qcqp' if backend == 'ma27' else 'ipopt_mumps')
        if not checked['pass'] or record.get('proof', {}).get('runtime_version') != DEFAULT_IPOPT_VERSION:
            raise RuntimeError('Default runtime fixture proof failed: ' + fixture + '/' + backend)
        if record.get('proof', {}).get('effective_options') != endpoint.get('effective_options'):
            raise RuntimeError('Default fixture option receipts disagree')
        import math
        if (audit.get('strict_qcqp') is not True or audit.get('original_bounds_used') is not True or
            audit.get('input_sha256') != digest(inp) or audit.get('endpoint_sha256') != digest(endpoint_path) or
            any(type(audit.get(k)) not in (int,float) or not math.isfinite(audit[k]) or not 0 <= audit[k] <= 1e-6
                for k in ('primal', 'dual', 'complementarity'))):
            raise RuntimeError('Independent original-KKT fixture audit failed: ' + fixture + '/' + backend)
        evidence[backend].append(dict(fixture=fixture, input_sha256=digest(inp),
            endpoint_sha256=digest(endpoint_path), audit_sha256=digest(audit_path), proof=checked))
    return dict(schema='previous-mega-ipopt-default-runtime-validation-v1',
        ipopt_version=DEFAULT_IPOPT_VERSION, qualified_backends=['ma27', 'mumps'],
        default_installation_sha256=defaults['_installation_sha256'], backend_evidence=evidence,
        optimizer_invocations=0, original_KKT_fixture_count=8, library_defaults=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--ipopt-prefix", type=Path)
    parser.add_argument("--hsl-library", default="auto")
    args = parser.parse_args()
    try:
        config = build_and_qualify(args.root, ipopt_prefix=args.ipopt_prefix,
                                  hsl_library=args.hsl_library)
        print(json.dumps(config, indent=2))
        return 0
    except Exception as exc:
        print("IPOPT_COMPARATORS_NOT_QUALIFIED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
