# Copyright (C) 2026 M.R. Siavash Katebzadeh <mr@katebzadeh.xyz>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Mandatory local profiles, selected by model and actual vLLM release only.

Precedence: profile defaults < ambient argv/env < experiment serving intent.
Qualifiers and Ascend versions are provenance, not recipe-match gates. The
whole registry is validated before selection; missing/invalid profiles fail.
Intent transport never applies defaults or changes the process environment.
The returned metadata belongs under ``serving_profile`` at integration.
"""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from importlib import metadata as distribution_metadata
from pathlib import Path
from urllib.parse import urlsplit


_MODELS = {"qwen3-" + size.lower(): "Qwen/Qwen3-" + size for size in ("4B", "8B", "14B")}
_RELEASES = {"0.11.0", "0.23.0"}
_JSON_OPTIONS = {"--compilation-config", "--additional-config"}
_BOOLEAN_OPTIONS = {"--enforce-eager", "--enable-prefix-caching", "--disable-log-requests", "--disable-log-stats"}
_REVIEWED_ENV = ("PYTORCH_NPU_ALLOC_CONF", "TASK_QUEUE_ENABLE")
_PROTECTED = {
    "model": "experiment.model (MODEL)",
    "port": "lanes[].backend_url / lane port configuration",
    "device": "lanes[].ascend_devices / device_map",
    "devices": "lanes[].ascend_devices / device_map",
    "tensor-parallel-size": "experiment.tp (TP_SIZE)",
    "data-parallel-size": "authoritative experiment parallelism configuration (DP_SIZE)",
    "pipeline-parallel-size": "authoritative experiment parallelism configuration (PP_SIZE)",
    "decode-context-parallel-size": "authoritative experiment parallelism configuration",
    "prefill-context-parallel-size": "authoritative experiment parallelism configuration",
    "max-num-batched-tokens": "lanes[].max_num_batched_tokens / sweep.max_num_batched_tokens",
    "max-num-seqs": "lanes[].max_num_seqs / sweep.max_num_seqs",
    "scheduling-policy": "runner-owned priority scheduling policy",
    "enable-chunked-prefill": "runner-owned chunked prefill policy",
    "trust-remote-code": "runner-owned model loading policy",
    "config": "authoritative experiment configuration / sweep",
}
_PROTECTED_ALIASES = {"-tp": "tensor-parallel-size", "-dp": "data-parallel-size", "-pp": "pipeline-parallel-size", "-c": "config", "-m": "model"}
_PROTECTED_ENV = {
    "MODEL", "TP_SIZE", "DP_SIZE", "PP_SIZE", "PRESETS_JSON", "RESULT_JSON", "PROFILE_JSON",
    "ASCEND_DEVICES", "ASCEND_P_DEVICES", "ASCEND_D_DEVICES", "ASCEND_RT_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "LANE_DEVICES_JSON",
    "ONLINE_SERVING", "VLLM_ROOT", "PATH", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
    "DEMO_STATE_DIR", "KILL_AFTER_CLIENT", "WAIT_SECS", "WAIT_BAR_WIDTH", "RESCUE_PRESET_ID",
    "ROUTER_PORT", "LANE_P_PORT", "LANE_D_PORT", "ROUTER_EXTRA_ARGS", "ROUTING_POLICY", "ROUTING_KWARGS",
    "CLIENT_MAX_ARRIVALS", "CLIENT_EXTRA_ARGS", "CLIENT_BASE_URL", "SLO_MAX_LATENCY_S", "EXPERIMENT_METADATA",
    "POWER_MONITOR_ENABLED", "POWER_INTERVAL_S", "KV_MONITOR_ENABLED", "KV_MONITOR_INTERVAL_S", "KV_MONITOR_TIMEOUT_S",
    "VLLM_SERVE_EXTRA_ARGS", "VLLM_SERVING_OVERRIDES", "VLLM_EXPECTED_SERVING_CONDITION",
}
_TUNING_ENV_PREFIXES = ("VLLM_", "HCCL_", "OMP_", "MKL_", "PYTORCH_", "TORCH_", "ASCEND_", "NPU_", "CANN_", "ATB_", "ACL_", "NCCL_", "CUDA_", "AIV_", "FLASH_", "RAY_", "GLOO_", "GOMP_", "KMP_", "ENABLE_FLASHCOMM")
_TUNING_ENV_NAMES = {"TOKENIZERS_PARALLELISM", "MALLOC_ARENA_MAX", "INF_NAN_MODE_ENABLE", "MULTI_STREAM_MEMORY_REUSE"}
_SWEEP_KEYS = {"port", "freq", "frequency", "frequency-mhz", "maxseq", "max-seqs", "max-num-seqs"}


def runtime_versions():
    """Read installed distribution metadata; VLLM_VERSION is not authoritative."""
    versions = {}
    for key, distribution in (("vllm", "vllm"), ("vllm_ascend", "vllm-ascend")):
        try:
            versions[key] = distribution_metadata.version(distribution)
        except distribution_metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def runtime_package_context():
    """Capture parent distribution versions and top-level package paths, without imports."""
    origins = {}
    for name in ("vllm", "vllm_ascend"):
        try:
            spec = importlib.util.find_spec(name)
            roots = list(spec.submodule_search_locations or []) if spec else []
            if not spec or not spec.origin or spec.origin in {"built-in", "frozen"} or len(roots) != 1:
                origins[name] = None
            else:
                origins[name] = {"origin": str(Path(spec.origin).resolve()), "root": str(Path(roots[0]).resolve())}
        except (ImportError, AttributeError, TypeError, ValueError, OSError) as exc:
            raise ValueError("Invalid runtime package discovery for " + name) from exc
    return {"versions": runtime_versions(), "origins": origins}


_RUNTIME_PACKAGE_PROBE = """
import importlib.util
import importlib.metadata
import json
from pathlib import Path

versions, origins = {}, {}
for name, distribution in (("vllm", "vllm"), ("vllm_ascend", "vllm-ascend")):
    spec = importlib.util.find_spec(name)
    roots = list(spec.submodule_search_locations or []) if spec else []
    if not spec or not spec.origin or spec.origin in {"built-in", "frozen"} or len(roots) != 1:
        origins[name] = None
    else:
        origins[name] = {"origin": str(Path(spec.origin).resolve()), "root": str(Path(roots[0]).resolve())}
    try:
        versions[name] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"versions": versions, "origins": origins}))
"""


def _validate_runtime_package_context(context):
    names = {"vllm", "vllm_ascend"}
    if not isinstance(context, dict) or set(context) != {"versions", "origins"}:
        raise ValueError("Invalid runtime package context")
    for field in ("versions", "origins"):
        if not isinstance(context[field], dict) or set(context[field]) != names:
            raise ValueError("Invalid runtime package " + field)
    for name in names:
        version, origin = context["versions"][name], context["origins"][name]
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Missing runtime package distribution version for " + name)
        if not isinstance(origin, dict) or set(origin) != {"origin", "root"} or any(
            not isinstance(value, str) or not value or not Path(value).is_absolute() for value in origin.values()
        ) or Path(origin["origin"]).parent != Path(origin["root"]):
            raise ValueError("Missing or invalid runtime package origin/root for " + name)


def verify_runtime_package_context(anchor, cwd, environment):
    """Metadata-only, bounded discovery in the exact server cwd/env; never import the packages."""
    _validate_runtime_package_context(anchor)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _RUNTIME_PACKAGE_PROBE], cwd=str(cwd), env=environment,
            check=True, capture_output=True, text=True, timeout=10,
        )
        discovered = json.loads(completed.stdout)
        _validate_runtime_package_context(discovered)
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise ValueError("Server runtime package preflight failed") from exc
    if discovered != anchor:
        raise ValueError("Server runtime package origins/versions mismatch with parent anchor: check VLLM_ROOT and PYTHONPATH")
    return discovered


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _json_value(value, option):
    try:
        parsed = json.loads(value, object_pairs_hook=_unique_object) if isinstance(value, str) else value
        _json(parsed)  # Reject nonfinite numbers, including inside objects.
        return deepcopy(parsed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Malformed JSON for " + option) from exc


def _credential_name(name):
    name = re.sub(r"([a-z])([A-Z])", r"\1-\2", name).lower().replace("_", "-")
    return bool(re.search(r"(?:^|-)(?:api-?key|access-?key|secret|token|password|passwd|credential[s]?|authorization|private-?key)(?:$|-)", name))


def _reject_credentials(value):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Serving JSON object keys must be strings")
            if _credential_name(key):
                raise ValueError("Explicit credentials are not serving tuning; inherit credentials outside serving configuration")
            _reject_credentials(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_credentials(item)
    elif isinstance(value, str):
        if "\x00" in value:
            raise ValueError("Serving values must not contain NUL")
        if value.lstrip().startswith(("{", "[")):
            _reject_credentials(_json_value(value, "serving intent"))
        elif "@" in value:
            text = value.strip()
            try:
                url = urlsplit(text if "://" in text or text.startswith("//") else "//" + text)
                userinfo = url.username is not None or url.password is not None
            except ValueError:
                raise ValueError("Invalid serving URL value") from None
            if userinfo:
                raise ValueError("Explicit credentials are not serving tuning; inherit credentials outside serving configuration")


def _merge_objects(default, explicit):
    if not isinstance(default, dict) or not isinstance(explicit, dict):
        return deepcopy(explicit)
    result = deepcopy(default)
    for key, value in explicit.items():
        result[key] = _merge_objects(result.get(key), value)
    return result


def _flag(token):
    name = token.split("=", 1)[0].replace("_", "-")
    return "--compilation-config" if name == "-O" else name


def _protected_owner(token):
    name = _flag(token)
    if name.startswith("--"):
        name = name[2:]
        # Reject even ambiguous long abbreviations that could reach a fixed knob.
        for protected, owner in _PROTECTED.items():
            if name and (protected.startswith(name) or ("no-" + protected).startswith(name)):
                return owner
        positive = name[3:] if name.startswith("no-") else name
        for family, option in (("tensor-parallel-", "tensor-parallel-size"), ("data-parallel-", "data-parallel-size"), ("pipeline-parallel-", "pipeline-parallel-size")):
            if positive.startswith(family):
                return _PROTECTED[option]
    if name.startswith("-") and name != "-":
        # argparse also accepts attached short-option values, e.g. -cfile.yaml.
        for alias, option in _PROTECTED_ALIASES.items():
            if name.startswith(alias) or alias.startswith(name):
                return _PROTECTED[option]
    return None


def _check_option(token):
    # Never echo an inline value: it could contain credentials.
    name = _flag(token)
    if _credential_name(name):
        raise ValueError("Explicit credentials are not serving tuning; inherit credentials outside serving configuration")
    owner = _protected_owner(token)
    if owner:
        raise ValueError("Protected serving option " + name + "; use " + owner)
    if token == "--":
        raise ValueError("Serving extra argv cannot contain a positional separator; use experiment.model")
    if name.startswith("-O.") or any(name.startswith(option + ".") for option in _JSON_OPTIONS):
        raise ValueError("Use whole compilation-config/additional-config JSON objects in serving.args, not dotted CLI options")


def _is_option(token):
    return token.startswith("-") and token != "-" and not re.match(r"^-\d", token)


def _boolean(value, option):
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)) and str(value).lower() in {"true", "false", "1", "0"}:
        return str(value).lower() in {"true", "1"}
    raise ValueError("Invalid boolean for " + option)


def _normalize_args(defaults):
    if not isinstance(defaults, Mapping):
        raise ValueError("Serving args must be an option map")
    result = {}
    for name, value in defaults.items():
        if not isinstance(name, str):
            raise ValueError("Serving option names must be strings")
        option = name if name.startswith("-") else "--" + name
        _check_option(option)
        option = _flag(option)
        if "=" in name or not re.fullmatch(r"--[a-z][a-z0-9]*(?:-[a-z0-9]+)*", option):
            raise ValueError("Serving options must be long option names without values")
        negative = option.startswith("--no-")
        positive = "--" + option[5:] if negative else option
        if option not in _JSON_OPTIONS and (negative or isinstance(value, bool) or positive in _BOOLEAN_OPTIONS):
            value = _boolean(value, option)
            option, value = positive, not value if negative else value
        if option in result:
            raise ValueError("Duplicate serving option: " + option)
        if value is None and option not in _JSON_OPTIONS:
            raise ValueError("Missing value for " + option)
        value = _json_value(value, option) if option in _JSON_OPTIONS else _json_value(_json_checked(value), option)
        _reject_credentials(value)
        result[option] = value
    return result


def _json_checked(value):
    try:
        _reject_credentials(value)
        return _json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid serving JSON value; credentials must inherit outside serving configuration") from exc


def _environment(environment):
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
        or not isinstance(value, str) or "\x00" in value
        for key, value in environment.items()
    ):
        raise ValueError("Serving environment must be a string map with valid environment names/values")
    return dict(environment)


def _protected_env(key):
    name = key.upper()
    return name in _PROTECTED_ENV or name.startswith(("SKIP_", "VLLM_SERVING_PROFILE", "SERVING_PROFILE")) or name.endswith("_VISIBLE_DEVICES")


def normalize_serving_config(serving):
    """Normalize optional {'args': option-map, 'env': string-map}, independently of runtime."""
    if serving is None:
        serving = {}
    if not isinstance(serving, Mapping) or set(serving) - {"args", "env"}:
        raise ValueError("serving must be an object containing only args and env")
    args = _normalize_args(serving.get("args", {}))
    env = _environment(serving.get("env", {}))
    _reject_credentials(env)
    for key in env:
        if _protected_env(key):
            raise ValueError("Protected serving environment " + key + "; use authoritative experiment configuration / sweep / runner lifecycle fields")
    return {"args": {key[2:]: value for key, value in sorted(args.items())}, "env": env}


def merge_serving_configs(base, override):
    """Layer root/workload/experiment intent; only the two config JSON options merge objects."""
    result, override = normalize_serving_config(base), normalize_serving_config(override)
    for key, value in override["args"].items():
        result["args"][key] = _merge_objects(result["args"].get(key), value) if "--" + key in _JSON_OPTIONS else deepcopy(value)
    result["env"].update(override["env"])
    result["args"] = dict(sorted(result["args"].items()))
    return result


def _environment_serving_config(environment):
    if "VLLM_SERVING_OVERRIDES" not in environment:
        return normalize_serving_config(None)
    try:
        intent = _json_value(environment["VLLM_SERVING_OVERRIDES"], "VLLM_SERVING_OVERRIDES")
        if not isinstance(intent, dict):
            raise ValueError("intent must be an object")
        return normalize_serving_config(intent)
    except ValueError as exc:
        raise ValueError("Invalid VLLM_SERVING_OVERRIDES: " + str(exc)) from exc


def apply_serving_config(environment, serving):
    """Return a copied environment transporting merged intent, without applying any defaults."""
    result = _environment(environment)
    result["VLLM_SERVING_OVERRIDES"] = _json(merge_serving_configs(_environment_serving_config(result), serving))
    return result


def _explicit_argv(explicit_args):
    if isinstance(explicit_args, (str, bytes)):
        raise ValueError("explicit_args must be argv, not a shell string")
    try:
        explicit = list(explicit_args)
    except TypeError as exc:
        raise ValueError("explicit_args must be argv") from exc
    for token in explicit:
        if not isinstance(token, str) or "\x00" in token:
            raise ValueError("Explicit argv tokens must be strings without NUL")
        if _is_option(token):
            _check_option(token)
            value = token[2:].removeprefix("=") if token.startswith("-O") else token.split("=", 1)[-1]
            _reject_credentials(value)
        else:
            _reject_credentials(token)
    return explicit


def _value_text(value):
    return value if isinstance(value, str) else _json(value)


def _emit(option, value):
    if option in _JSON_OPTIONS or isinstance(value, dict):
        return [option, _json(value)]
    if isinstance(value, bool):
        return [option if value else "--no-" + option[2:]]
    if isinstance(value, list):
        return [option, *(_value_text(item) for item in value)]
    return [option, _value_text(value)]


def merge_serving_args(defaults, explicit_args, overrides=None):
    """Merge profile < ambient argv < experiment option map; preserve unrelated tokens.

    Names may omit ``--`` and use underscores. Lists become multi-value argv;
    dicts become JSON. Only the two config JSON options merge objects. Repeated
    explicit occurrences use the last value, and boolean opposites share a key.
    Protected flags/aliases/abbreviations always raise, even with no defaults.
    """
    explicit = _explicit_argv(explicit_args)
    settings = _normalize_args(defaults)
    experiment = _normalize_args({} if overrides is None else overrides)
    shapes = dict(settings)
    for option, value in experiment.items():
        shapes.setdefault(option, value)
    ambient, unrelated = {}, []
    index = 0
    while index < len(explicit):
        token = explicit[index]
        if token.startswith("-O") and token != "-O" and not token.startswith(("-O=", "-O.")):
            token = "-O=" + token[2:]
        option = _flag(token)
        negative = option.startswith("--no-")
        positive = "--" + option[5:] if negative else option
        if positive in shapes and isinstance(shapes[positive], bool) and positive not in _JSON_OPTIONS:
            option = positive
        if option in _JSON_OPTIONS:
            shapes.setdefault(option, None)
        if option not in shapes:
            unrelated.append(token)
            index += 1
            continue
        inline = token.split("=", 1)[1] if "=" in token else None
        index += 1
        if isinstance(shapes[option], bool) and option not in _JSON_OPTIONS:
            if inline is None and index < len(explicit) and explicit[index].lower() in {"true", "false", "1", "0"}:
                inline = explicit[index]
                index += 1
            value = True if inline is None else _boolean(inline, option)
            ambient[option] = not value if negative else value
            continue
        values = [] if inline is None else [inline]
        multiple = (isinstance(shapes[option], list) or option == "--served-model-name") and option not in _JSON_OPTIONS
        if multiple:
            while index < len(explicit) and not _is_option(explicit[index]):
                values.append(explicit[index])
                index += 1
        elif inline is None and index < len(explicit) and not _is_option(explicit[index]):
            values.append(explicit[index])
            index += 1
        if not values:
            raise ValueError("Missing value for " + option)
        if option in _JSON_OPTIONS:
            ambient[option] = _json_value(values[0], option)
            _reject_credentials(ambient[option])
        else:
            ambient[option] = values if multiple else values[0]
    result = []
    for option in shapes:
        value = settings.get(option)
        for layer in (ambient, experiment):
            if option in layer:
                value = _merge_objects(value, layer[option]) if option in _JSON_OPTIONS else layer[option]
        result.extend(_emit(option, value))
    return result + unrelated


def _validate_profile(entry):
    for field in ("profile_id", "rationale"):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            raise ValueError("Invalid profile " + field)
    if entry.get("model") not in _MODELS.values():
        raise ValueError("Invalid profile model")
    versions = entry.get("versions")
    if not isinstance(versions, dict) or set(versions) != {"vllm", "vllm_ascend"} or (
        not isinstance(versions["vllm"], str) or versions["vllm"] not in _RELEASES
        or not isinstance(versions["vllm_ascend"], str) or not re.fullmatch(r"\d+\.\d+\.\d+", versions["vllm_ascend"])
    ):
        raise ValueError("Invalid profile versions")
    if type(entry.get("revision")) is not int or entry["revision"] < 1:
        raise ValueError("Invalid profile revision")
    if entry.get("status") != "ready":
        raise ValueError("Invalid profile status")
    defaults = entry.get("defaults")
    if not isinstance(defaults, dict) or set(defaults) != {"args", "env"} or not all(
        isinstance(defaults[key], dict) for key in ("args", "env")
    ):
        raise ValueError("Invalid profile defaults")
    merge_serving_args(defaults["args"], [])
    _environment(defaults["env"])
    _reject_credentials(defaults["env"])
    for key, value in defaults["env"].items():
        if key not in _REVIEWED_ENV:
            raise ValueError("Unreviewed environment: " + key)
        if not isinstance(value, str):
            raise ValueError("Invalid environment: " + key)
    documentation = entry.get("documentation_version")
    validated = entry.get("validated_version")
    if not isinstance(documentation, str) or documentation != versions["vllm"]:
        raise ValueError("Invalid profile documentation_version")
    if "validated_version" not in entry or (validated is not None and (
        not isinstance(validated, str) or not re.fullmatch(r"\d+\.\d+\.\d+", validated)
    )):
        raise ValueError("Invalid profile validated_version")
    provenance, evidence = entry.get("source_provenance"), entry.get("source_evidence")
    if not isinstance(provenance, dict) or not isinstance(evidence, list) or not evidence:
        raise ValueError("Invalid profile source provenance/evidence")
    for source in [provenance, *evidence]:
        fields = ("section", "official_url") if source is provenance else ("section", "official_url", "url", "commit", "context")
        if not isinstance(source, dict) or any(
            not isinstance(source.get(field), str) or not source[field].strip() for field in fields
        ):
            raise ValueError("Invalid profile source provenance/evidence")
        if "/v" + documentation + "/" not in source["official_url"]:
            raise ValueError("Source official_url must use documentation_version")
        if source is not provenance and (
            source.get("documentation_version") != documentation or "validated_version" not in source
            or source["validated_version"] != validated or source["section"] != provenance["section"]
            or source["official_url"] != provenance["official_url"]
        ):
            raise ValueError("Inconsistent source provenance/evidence versions or section")
    guidance = entry.get("guidance")
    if not isinstance(guidance, list) or any(
        not isinstance(item, dict) or item.get("active") is not False or any(
            not isinstance(item.get(field), str) or not item[field].strip() for field in ("candidate", "scope")
        ) for item in guidance
    ):
        raise ValueError("Invalid nonactive profile guidance")


def _load_profiles():
    path = Path(__file__).with_suffix(".json")
    try:
        registry = _json_value(path.read_text(encoding="utf-8"), "serving registry")
    except (OSError, UnicodeError) as exc:
        raise ValueError("Cannot read mandatory serving registry " + str(path) + "; restore src/serving_profiles.json before launch") from exc
    if not isinstance(registry, dict) or type(registry.get("schema_version")) is not int or registry["schema_version"] != 1:
        raise ValueError("Unsupported serving registry schema")
    entries = registry.get("profiles")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("Serving registry profiles must be objects in a list")
    seen_ids, seen_matches = set(), set()
    for entry in entries:
        _validate_profile(entry)
        profile_id = entry["profile_id"]
        identity = (entry["model"], entry["versions"]["vllm"])
        if profile_id in seen_ids:
            raise ValueError("Duplicate profile_id: " + profile_id)
        if identity in seen_matches:
            raise ValueError("Duplicate model/versions: " + str(identity))
        seen_ids.add(profile_id)
        seen_matches.add(identity)
    return entries


def _model_identity(model):
    if model in _MODELS.values():
        return model, "canonical"
    if "/" not in model or model.startswith(("/", "./", "../", "~/")):
        canonical = _MODELS.get(Path(model).name.lower())
        if canonical:
            return canonical, "local-basename"
    return None, None


def _release(version):
    if not isinstance(version, str):
        return None
    match = re.fullmatch(r"v?(\d+\.\d+\.\d+)(?:\+[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*)?", version)
    return match[1] if match and match[1] in _RELEASES else None


def _relevant_environment(environment, extra_keys=()):
    result = {}
    for key, value in environment.items():
        if _protected_env(key) or _credential_name(key):
            continue
        if key in _REVIEWED_ENV or key in extra_keys or key in _TUNING_ENV_NAMES or key.startswith(_TUNING_ENV_PREFIXES):
            # Inherited credentials still reach the child, but never provenance.
            try:
                _reject_credentials(value)
            except ValueError:
                continue
            result[key] = value
    return result


def resolve_serving_profile(model, explicit_args, environment, versions=None, qualifiers=None, overrides=None):
    """Return ``(extra_argv, env_updates, metadata)`` without mutating inputs.

    Every successful result is applied. Only model and running vLLM release
    select a profile; Ascend versions and qualifiers are recorded context.
    Read transported intent and layer direct overrides above it. Profile env
    defaults are reviewed; experiment tuning env is validated independently.
    Apply returned env_updates only to the final copied child-lane environment.
    """
    explicit = _explicit_argv(explicit_args)
    environment = _environment(environment)
    intent = merge_serving_configs(_environment_serving_config(environment), overrides)
    if qualifiers is not None and not isinstance(qualifiers, Mapping):
        raise ValueError("Serving qualifiers must be a provenance object")
    context = dict(qualifiers or {})
    _json_checked(context)
    requested = str(model) if model is not None else ""
    canonical, match = _model_identity(requested)
    if canonical is None:
        raise ValueError("Unknown serving model; set experiment.model to Qwen/Qwen3-4B, Qwen/Qwen3-8B or Qwen/Qwen3-14B, or an exact local basename")
    entries = _load_profiles()
    if versions is None:
        versions = runtime_versions()
    if not isinstance(versions, Mapping):
        raise ValueError("Actual runtime versions must be a mapping")
    actual = {key: versions.get(key) for key in ("vllm", "vllm_ascend")}
    release = _release(actual["vllm"])
    if release is None:
        raise ValueError("Missing or unsupported actual vLLM release; use running vLLM 0.11.0 or 0.23.0 (local +suffix allowed, rc/dev unsupported)")
    if actual["vllm_ascend"] is not None and not isinstance(actual["vllm_ascend"], str):
        raise ValueError("Actual vllm_ascend version must be a string or unknown (None)")
    entry = next((item for item in entries if item["model"] == canonical and item["versions"]["vllm"] == release), None)
    if entry is None:
        raise ValueError("Missing mandatory serving profile for " + canonical + " / vLLM " + release + "; restore the matching src/serving_profiles.json entry before launch")
    defaults = entry["defaults"]
    args = merge_serving_args(defaults["args"], explicit, intent["args"])
    updates = {key: environment.get(key, value) for key, value in defaults["env"].items()}
    updates.update(intent["env"])
    inherited = _relevant_environment(environment, intent["env"])
    metadata = {
        "status": "applied",
        "reason": "compatible-profile",
        "profile_id": entry["profile_id"],
        "profile_revision": entry["revision"],
        "profile_hash": hashlib.sha256(_json(entry).encode()).hexdigest(),
        "source_provenance": deepcopy(entry["source_provenance"]),
        "documentation_version": entry["documentation_version"],
        "validated_version": entry["validated_version"],
        "source_evidence": deepcopy(entry["source_evidence"]),
        "rationale": entry["rationale"],
        "guidance": deepcopy(entry.get("guidance", [])),
        "model": {"requested": requested, "canonical": canonical, "match": match},
        "versions": actual,
        "profile_versions": dict(entry["versions"]),
        "qualifiers": deepcopy(context),
        "effective_extra_args": args,
        "relevant_env": _relevant_environment({**environment, **updates}, intent["env"]),
        "explicit_overrides": {"args": explicit, "env": inherited, "serving": intent},
    }
    return args, updates, metadata


def _condition_args(args):
    result, index = [], 0
    while index < len(args):
        token = args[index]
        option = _flag(token)
        if option.startswith("--") and option[2:] in _SWEEP_KEYS:
            index += 1
            if "=" not in token and index < len(args) and not _is_option(args[index]):
                index += 1
        else:
            result.append(token)
            index += 1
    return result


def serving_condition_key(metadata):
    """Compact deterministic grouping key, excluding timing and sweep knobs."""
    condition = {key: metadata.get(key) for key in ("versions", "profile_id", "profile_revision", "profile_hash", "status", "model")}
    if "package_origins" in metadata:
        condition["package_origins"] = metadata["package_origins"]
    condition["effective_extra_args"] = _condition_args(metadata.get("effective_extra_args", []))
    condition["relevant_env"] = dict(metadata.get("relevant_env", {}))
    condition["qualifiers"] = {
        key: value for key, value in metadata.get("qualifiers", {}).items()
        if key.replace("_", "-") not in _SWEEP_KEYS
    }
    return "sp-" + hashlib.sha256(_json(condition).encode()).hexdigest()[:24]
