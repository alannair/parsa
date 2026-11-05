#!/usr/bin/env python
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

import argparse
import getpass
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import tempfile
from pathlib import Path
from types import MappingProxyType


def parse_device_list(devices):
    return [int(x) for x in devices.split(",")]


MODEL = "qwen3-4b"
DEVICES = parse_device_list(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0"))
PROXY_ENV = dict.fromkeys(
    ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
     "http_proxy", "https_proxy", "all_proxy", "ftp_proxy"), ""
)
PROXY_ENV.update({"NO_PROXY": "*", "no_proxy": "*"})

MODEL_DIR = "/data1/models"
DATASET_DIR = os.environ.get("DATASET_DIR", "/data1/datasets")

VLLM_IMAGE = "quay.io/ascend/vllm-ascend:v0.11.0"
VLLM_ASCEND_REPO = "https://github.com/vllm-project/vllm-ascend.git"
VLLM_ASCEND_TAG = "v0.11.0"
# Published image's org.opencontainers.image.revision label; also the v0.11.0 tag.
VLLM_ASCEND_COMMIT = "2f1aed98ccdb0fcbe1ff4fd0abab225bfd8d0367"
BEAM_IMAGE = "vllm-ascend-beam"
PARSA_IMAGE = "vllm-ascend-parsa"
V023_PYARROW_VERSION = "24.0.0"

PARSA_PROFILES = MappingProxyType({
    "0.11": MappingProxyType({
        "version": "0.11.0",
        "dockerfile": "Dockerfile.0.11",
        "vllm_tag": "v0.11.0",
        # Official tag metadata; retain legacy tag-based clone/worktree commands.
        "vllm_commit": "b8b302cde434df8c9289a2b465406b47ebab1c2d",
        "ascend_tag": VLLM_ASCEND_TAG,
        "ascend_commit": VLLM_ASCEND_COMMIT,
        "base_image": VLLM_IMAGE,
        "parsa_dir": "vllm-parsa",
        "upstream_dir": "vllm-upstream",
        "ascend_dir": "vllm-ascend",
        "patch": "parsa-vllm.patch",
        "image": PARSA_IMAGE,
    }),
    "0.23": MappingProxyType({
        "version": "0.23.0",
        "dockerfile": "Dockerfile.0.23",
        "vllm_tag": "v0.23.0",
        # User-supplied release pins; verified against the v0.23.0 source tags.
        "vllm_commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "ascend_tag": "v0.23.0",
        "ascend_commit": "5cb98caaadeff42b5b62b996e34bb2aaa29d20fd",
        "base_image": "quay.io/ascend/vllm-ascend:v0.23.0",
        "parsa_dir": "vllm-parsa-0.23.0",
        "upstream_dir": "vllm-0.23.0",
        "ascend_dir": "vllm-ascend-0.23.0",
        "patch": "parsa-vllm-0.23.0.patch",
        "image": "vllm-ascend-parsa:0.23.0",
    }),
})

BEAM_NPU_PROFILE = MappingProxyType({
    **PARSA_PROFILES["0.23"],
    "beam_dir": "vllm-beam-npu-0.23.0",
    "beam_commit": "38d461ca6a4694d944b8496b36091ff6c42ef6ad",
    "patches": ("beam-vllm-0.23.0.patch", "beam-npu-vllm-0.23.0.patch"),
    "image": "vllm-ascend-beam:0.23.0",
})


def parse_version(value):
    if value not in PARSA_PROFILES:
        raise argparse.ArgumentTypeError("--version must be one of: %s" % ", ".join(PARSA_PROFILES))
    return value


def resolve_parsa_profile(version="0.11"):
    profile = PARSA_PROFILES[parse_version(version)]
    dockerfile = "Dockerfile.%s" % version
    if profile.get("dockerfile") != dockerfile:
        raise ValueError("PARSA profile %s Dockerfile must be %s" % (version, dockerfile))
    if version != "0.11":
        invalid = [
            key for key in ("vllm_commit", "ascend_commit")
            if not isinstance(profile.get(key), str)
            or re.fullmatch(r"[0-9a-f]{40}", profile[key]) is None
            or profile[key] == "0" * 40
        ]
        if not isinstance(profile.get("base_image"), str) or re.fullmatch(r"\S+", profile["base_image"]) is None:
            invalid.append("base_image")
        if invalid:
            raise ValueError("PARSA profile %s is not ready; verified immutable values required: %s" % (version, ", ".join(invalid)))
    return profile


def _resolve_backend_profile(backend, version, *source_paths):
    parse_version(version)
    if backend not in ("parsa", "beam") and version != "0.11":
        raise ValueError("--version %s is only supported for the parsa and beam backends" % version)
    profile = resolve_parsa_profile(version)
    if backend == "beam" and version == "0.23":
        profile = BEAM_NPU_PROFILE
    if version != "0.11":
        legacy_names = {PARSA_PROFILES["0.11"][key] for key in ("upstream_dir", "parsa_dir", "ascend_dir")}
        legacy_names.update(("vllm-beam", "vllm-ascend-beam"))
        for path in source_paths:
            if path is None:
                continue
            normalized = os.path.abspath(path)
            resolved = os.path.realpath(normalized)
            if any(legacy_names.intersection(candidate.split(os.sep)) for candidate in (normalized, resolved, os.path.realpath(path))):
                raise ValueError("%s profile %s refuses legacy v0.11 source path: %s (resolves to %s)" % (backend.upper(), version, path, resolved))
    return profile


backend_to_image = {
    "beam": BEAM_IMAGE,
    "parsa": PARSA_IMAGE,
    "vllm": VLLM_IMAGE,
}

default_dirs = {
    "beam": {"vllm": "vllm-beam", "vllm-ascend": "vllm-ascend"},
    "parsa": {"vllm": "vllm-parsa", "vllm-ascend": "vllm-ascend"},
}


def find_free_port(preference=0):
    """
    Find a free port, optionally at a preferred number.

    Be mindful that this will not reserve the port, so another process can
    "find" the same port and race you to it
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while True:
            try:
                s.bind(("localhost", preference))
                _, port = s.getsockname()
                return port
            except OSError as ex:
                if preference == 0:
                    raise RuntimeError("No free ports available") from ex
                else:
                    preference += 1


def pretty_print_command(cmd):
    lines = []
    for p in cmd:
        if p.startswith("-"):
            lines.append("\\\n\t")
        lines.append(p)
        lines.append(" ")
    print("".join(lines), flush=True)


def _beam_npu_vllm_args(vllm_args):
    """Normalize explicit booleans to EngineArgs flags and reject NPU conflicts."""
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "0":
        raise ValueError("BEAM v0.23 NPU requires VLLM_USE_V2_MODEL_RUNNER=0")
    required = {"enforce-eager": True, "async-scheduling": False}
    flags = set(required) | {
        "custom-profiler", "use-s1", "use-s1-dvfs-only", "use-s2",
        "use-window-based-dvfs-only", "measure-fidelity", "disagg-mode",
    }
    args = list(vllm_args)
    normalized = []
    seen = set()
    i = 0
    while i < len(args):
        argument = args[i]
        key, separator, value = argument.partition("=")
        key = key.replace("_", "-")
        negative = key.startswith("--no-")
        name = key[5:] if negative else key[2:]
        if not key.startswith("--") or name not in flags:
            normalized.append(argument)
            i += 1
            continue
        if not separator and i + 1 < len(args) and args[i + 1].lower() in ("true", "false", "1", "0"):
            i += 1
            separator, value = "=", args[i]
        enabled = not negative
        if separator:
            if value.lower() not in ("true", "false", "1", "0"):
                raise ValueError("%s requires a boolean value (true/false or 1/0)" % key)
            enabled = (value.lower() in ("true", "1")) != negative
        if name in required and enabled != required[name]:
            raise ValueError("BEAM v0.23 NPU requires --enforce-eager and --no-async-scheduling; conflicting option: %s" % argument)
        seen.add(name)
        normalized.append("--" + ("" if enabled else "no-") + name)
        i += 1
    return [
        "--" + ("" if enabled else "no-") + name
        for name, enabled in required.items() if name not in seen
    ] + normalized


def run(
    backend,
    devices,
    port=None,
    model=MODEL,
    shell=False,
    mount_vllm=None,
    mount_ascend=None,
    vllm_args=(),
    version="0.11",
):
    profile = _resolve_backend_profile(backend, version, mount_vllm, mount_ascend)
    beam_npu = backend == "beam" and version == "0.23"
    if beam_npu:
        vllm_args = _beam_npu_vllm_args(vllm_args)
    if port is None:
        port = find_free_port()
        print(f"Binding to port {port}")

    image = profile["image"] if backend == "parsa" or beam_npu else backend_to_image.get(backend)
    name = "-".join(
        (
            getpass.getuser(),
            "experiment",
            backend,
            *map(str, devices),
        )
    )

    env = {
        **PROXY_ENV,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, devices)),
        "ASCEND_DEVICES": ",".join(map(str, devices)),
        "VLLM_VERSION": profile["version"],  # used to pacify vllm-ascend with mounted vllm versions
        "MODEL": model,  # passed to inside the container
        "TASK_QUEUE_ENABLE": "1",
    }
    if beam_npu:
        env.update({"BEAM_NPU_MODE": "1", "VLLM_USE_V2_MODEL_RUNNER": "0"})

    volumes = {
        "/usr/local/dcmi": "/usr/local/dcmi",
        "/usr/local/bin/npu-smi": "/usr/local/bin/npu-smi",
        "/usr/local/Ascend/driver": "/usr/local/Ascend/driver",
        "/etc/ascend_install.info": "/etc/ascend_install.info",
        MODEL_DIR: "/models",
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))): "/workspace",
        "/mnt": "/mnt",
        DATASET_DIR: "/datasets",
    }

    if mount_vllm is not None:
        volumes[os.path.abspath(mount_vllm)] = "/vllm-workspace/vllm"
    if mount_ascend is not None:
        volumes[os.path.abspath(mount_ascend)] = "/vllm-workspace/vllm-ascend"

    ports = {str(port): "8000"}
    cmd_flags = [
        "--rm",
        "--privileged",
        f"--name={name}",
        "--shm-size=18gb",
    ]

    for pair in env.items():
        cmd_flags.append("-e")
        cmd_flags.append("=".join(pair))
    for pair in volumes.items():
        cmd_flags.append("-v")
        cmd_flags.append(":".join(pair))
    for pair in ports.items():
        cmd_flags.append("-p")
        cmd_flags.append(":".join(pair))

    if vllm_args:
        vllm_args = ["--", *vllm_args]
    container_command = ["python", "/workspace/setup/deploy.py", "container-init", *vllm_args]

    if shell:
        cmd_flags.append("-it")
        container_command = ["bash"]

    command = [
        "docker",
        "run",
        *cmd_flags,
        image,
        *container_command,
    ]
    pretty_print_command(command)
    subprocess.check_call(command, env={**os.environ, **PROXY_ENV})


def container_init(vllm_args=()):
    """
    this is used as the entrypoint inside the container
    it does the "last-mile" setup:
    - import installing the mounted vllm module
    - configuring everything
    - launching vllm
    """
    if os.environ.get("BEAM_NPU_MODE") == "1":
        vllm_args = _beam_npu_vllm_args(vllm_args)
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    model = os.environ.get("MODEL", MODEL)
    # yada yada
    # launch vllm with flags
    # environment variables are alreadys set globally in docker
    #
    subprocess.check_call(
        [
            "vllm",
            "serve",
            os.path.join("/models", model),
            "-tp",
            str(len(devices)),
            "--port",
            "8000",
            *vllm_args,
        ],
        cwd="/vllm-workspace/vllm",
    )


def _docker_build_command():
    """Choose Buildx when available, including installations on old Docker CLIs."""
    result = subprocess.run(
        ["docker", "buildx", "version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        return ["docker", "buildx", "build"], True

    standalone = os.path.expanduser("~/.docker/cli-plugins/docker-buildx")
    if os.path.isfile(standalone) and os.access(standalone, os.X_OK):
        return [standalone, "build"], True
    return ["docker", "build"], False


def _python_tag_for_image(image):
    """Read the CPython ABI used by the image that will receive the wheel."""
    tag = subprocess.check_output(
        [
            "docker", "run", "--rm", "--network=none",
            "--entrypoint", "python3", image,
            "-c", "import sys; print('cp%d%d' % sys.version_info[:2])",
        ],
        text=True,
    ).strip()
    if re.fullmatch(r"cp3\d\d", tag) is None:
        raise ValueError("could not determine a supported CPython wheel tag from %s: %r" % (image, tag))
    return tag


def _find_offline_pyarrow_wheel(location, python_tag):
    """Find the locally supplied PyArrow 24 wheel matching the image's CPython ABI."""
    path = Path(location or "wheelhouse").expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob(f"pyarrow-{V023_PYARROW_VERSION}-*.whl"))
    elif path.is_file():
        candidates = [path]
    else:
        raise ValueError(
            "PyArrow wheel not found at %s. Supply --wheelhouse with a directory "
            "containing the pyarrow==24.0.0 CPython 3.11 aarch64 wheel, "
            "or the path to that wheel." % path
        )

    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one PyArrow 24.0.0 wheel at %s, found %d" %
            (path, len(candidates))
        )
    wheel = candidates[0]
    pattern = (
        rf"pyarrow-{re.escape(V023_PYARROW_VERSION)}-{re.escape(python_tag)}-"
        rf"{re.escape(python_tag)}-manylinux_[0-9_]+_aarch64\.whl"
    )
    if not re.fullmatch(pattern, wheel.name):
        python_version = "%s.%s" % (python_tag[2], int(python_tag[3:]))
        raise ValueError(
            "Expected a pyarrow==24.0.0 CPython %s aarch64 manylinux wheel; got %s" %
            (python_version, wheel.name)
        )
    return wheel


def _build_v023_offline_pyarrow(base_image, final_image, wheel):
    """Repair PyArrow without connecting the image build to any network."""
    # A temporary build context lets --wheelhouse refer to an arbitrary host path.
    # The repaired image inherits every layer and source checkout from base_image.
    with tempfile.TemporaryDirectory(prefix="vllm-pyarrow-offline-") as temp_dir:
        context = Path(temp_dir)
        wheels = context / "wheelhouse"
        wheels.mkdir()
        shutil.copy2(wheel, wheels / wheel.name)
        dockerfile = context / "Dockerfile"
        dockerfile.write_text(
            f"FROM {base_image}\n"
            "COPY wheelhouse/ /opt/wheelhouse/\n"
            "RUN python3 -m pip uninstall -y pyarrow && "
            "python3 -m pip install --no-index --no-deps --no-cache-dir "
            f"/opt/wheelhouse/pyarrow-{V023_PYARROW_VERSION}-*.whl\n"
            "RUN python3 -c \"import importlib.metadata as md, numpy, pyarrow; "
            "print('NumPy:', numpy.__version__); "
            "print('PyArrow:', pyarrow.__version__); "
            f"assert pyarrow.__version__ == {V023_PYARROW_VERSION!r}; "
            f"assert md.version('pyarrow') == {V023_PYARROW_VERSION!r}\"\n",
            encoding="utf-8",
        )
        cmd = [
            "docker", "build", "--network=none", "--pull=false",
            "--file", str(dockerfile), "--tag", final_image, str(context),
        ]
        print("Applying PyArrow 24.0.0 offline repair to v0.23 image:", flush=True)
        pretty_print_command(cmd)
        subprocess.check_call(cmd, env={**os.environ, **PROXY_ENV})

    # Test the exported runtime image as well as the Docker build layer.
    verify = [
        "docker", "run", "--rm", "--network", "none",
        "--entrypoint", "python3", final_image,
        "-c", "import numpy, pyarrow; "
        f"assert pyarrow.__version__ == {V023_PYARROW_VERSION!r}; "
        "print('NumPy:', numpy.__version__, 'PyArrow:', pyarrow.__version__)",
    ]
    pretty_print_command(verify)
    subprocess.check_call(verify, env={**os.environ, **PROXY_ENV})


def build_image(
    backend,
    vllm_dir,
    vllm_ascend_dir,
    use_host=True,
    version="0.11",
    wheelhouse=None,
    offline_from=None,
):
    profile = _resolve_backend_profile(backend, version, vllm_dir, vllm_ascend_dir)
    is_v023 = backend in ("parsa", "beam") and version == "0.23"
    if not is_v023 and (wheelhouse is not None or offline_from is not None):
        raise ValueError(
            "--wheelhouse and --offline-from are only supported for parsa or beam --version 0.23"
        )
    # Resolve the wheel before running any expensive Docker build.
    python_tag = _python_tag_for_image(profile["base_image"]) if is_v023 else None
    wheel = _find_offline_pyarrow_wheel(wheelhouse, python_tag) if is_v023 else None
    if backend == "parsa" or is_v023:
        directories = {
            "vllm": profile["beam_dir" if backend == "beam" else "parsa_dir"],
            "vllm-ascend": profile["ascend_dir"],
        }
    else:
        directories = default_dirs[backend]
    vllm_dir = vllm_dir if vllm_dir is not None else directories["vllm"]
    vllm_ascend_dir = (
        vllm_ascend_dir
        if vllm_ascend_dir is not None
        else directories["vllm-ascend"]
    )
    _resolve_backend_profile(backend, version, vllm_dir, vllm_ascend_dir)
    build_args = {
        **PROXY_ENV,
        "VLLM_DIR": vllm_dir,
        "VLLM_ASCEND_DIR": vllm_ascend_dir,
    }
    if version != "0.11":
        build_args.update({
            "VLLM_BASE_IMAGE": profile["base_image"], "VLLM_VERSION": profile["version"],
        })

    # 0.23 first builds the original image under a separate local tag; the
    # offline wheel stage then produces the final tag used by `run`.
    base_tag = profile["image"] + "-pre-pyarrow" if is_v023 else None
    builder, using_buildx = _docker_build_command()
    cmd = [
        *builder,
        "-t",
        (
            base_tag if is_v023
            else (profile["image"] if backend == "parsa" else backend_to_image.get(backend))
        ),
    ]
    if is_v023 and using_buildx:
        # The second offline build must be able to resolve the first image locally
        # even when buildx is configured with a non-Docker builder driver.
        cmd.append("--load")
    cmd.extend(["--file", profile["dockerfile"]])

    if use_host:
        print("Using host network...")
        cmd.append("--network=host")

    for pair in build_args.items():
        cmd.append("--build-arg")
        cmd.append("=".join(pair))

    labels = {}
    if backend == "parsa":
        labels = {
            "org.opencontainers.image.version": profile["version"],
            "org.parsa.vllm.revision": profile["vllm_commit"],
            "org.parsa.vllm-ascend.revision": profile["ascend_commit"],
            "org.parsa.patch": profile["patch"],
        }
    elif backend == "beam" and is_v023:
        labels = {
            "org.opencontainers.image.version": profile["version"],
            "org.beam.vllm.revision": profile["vllm_commit"],
            "org.beam.core.revision": profile["beam_commit"],
            "org.beam.vllm-ascend.revision": profile["ascend_commit"],
            "org.beam.patches": " ".join(profile["patches"]),
        }
    for pair in labels.items():
        cmd.append("--label")
        cmd.append("=".join(pair))

    cmd.append(".")

    if is_v023 and offline_from:
        # Reuse a prebuilt local 0.23 image and skip rebuilding native vLLM.
        # This is useful when the host is air-gapped as well as the container.
        print("Using existing local v0.23 image:", offline_from, flush=True)
        if backend == "beam":
            # Provenance labels are required here; they are not runtime proof.
            try:
                actual_labels = json.loads(subprocess.check_output(
                    ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", offline_from],
                    text=True,
                ))
            except (json.JSONDecodeError, UnicodeError):
                raise ValueError("BEAM v0.23 --offline-from image has malformed label metadata; expected a JSON object") from None
            if not isinstance(actual_labels, dict):
                raise ValueError("BEAM v0.23 --offline-from image label metadata must be a JSON object")
            invalid = [key for key, value in labels.items() if actual_labels.get(key) != value]
            if invalid:
                raise ValueError("BEAM v0.23 --offline-from image has missing or mismatched provenance labels: %s" % ", ".join(invalid))
        else:
            subprocess.check_call(
                ["docker", "image", "inspect", offline_from],
                stdout=subprocess.DEVNULL,
            )
        subprocess.check_call(["docker", "tag", offline_from, base_tag])
    else:
        print("docker build command is")
        pretty_print_command(cmd)
        subprocess.check_call(cmd, env={**os.environ, **PROXY_ENV})

    if is_v023:
        _build_v023_offline_pyarrow(base_tag, profile["image"], wheel)


def _patch_already_applied(checkout, message):
    """Skip only verified postimages, never a partial or merely conflicting patch."""
    try:
        with open(message, encoding="utf-8") as handle:
            sections = re.split(r"^diff --git ", handle.read(), flags=re.MULTILINE)[1:]
        if not sections:
            return False
        postimages = {}
        for section in sections:
            lines = section.splitlines()
            paths = lines[0].split() if lines else []
            if len(paths) != 2 or not paths[0].startswith("a/") or paths[1] != "b/" + paths[0][2:]:
                return False
            relative = paths[0][2:]
            if any(ch in relative for ch in ('"', "\\", "\0")) or any(part in ("", ".", "..", ".git") for part in relative.split("/")) or relative in postimages:
                return False
            if any(line.startswith(("old mode ", "new mode ", "deleted file mode ", "rename ", "copy ", "similarity index ", "dissimilarity index ")) for line in lines):
                return False
            indexes = [line for line in lines if line.startswith("index ")]
            if len(indexes) != 1:
                return False
            index = re.fullmatch(r"index ([0-9a-f]+)\.\.([0-9a-f]{9,64})(?: (100644|100755))?", indexes[0])
            if index is None or not any(ch != "0" for ch in index[2]):
                return False
            modes = [line.removeprefix("new file mode ") for line in lines if line.startswith("new file mode ")]
            if len(modes) > 1 or modes and (any(ch != "0" for ch in index[1]) or index[3] not in (None, modes[0])):
                return False
            if not modes and not any(ch != "0" for ch in index[1]):
                return False
            mode = modes[0] if modes else index[3]
            if mode not in ("100644", "100755"):
                return False
            postimages[relative] = (index[2], mode)

        root = os.path.realpath(checkout)
        for relative, (digest, mode) in postimages.items():
            path = os.path.join(root, relative)
            if os.path.realpath(path) != path:
                return False
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or bool(info.st_mode & stat.S_IXUSR) != (mode == "100755"):
                return False
            actual = subprocess.check_output(
                ["git", "-C", checkout, "hash-object", "--no-filters", "--", relative],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if not actual.startswith(digest):
                return False
        return subprocess.call(
            ["git", "-C", checkout, "apply", "--reverse", "--check", message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) == 0
    except (OSError, UnicodeError, subprocess.CalledProcessError):
        return False


def apply_vllm_patch(checkout, patch, start_message=1):
    checkout = os.path.abspath(checkout)
    patch = os.path.abspath(patch)
    with open(patch, "rb") as handle:
        is_raw = handle.read(11) == b"diff --git "
    # format-patch mailboxes contain dependent changes; apply one message at a time.
    with tempfile.TemporaryDirectory() as directory:
        if is_raw:
            messages = ["1"]
        else:
            subprocess.check_call(["git", "mailsplit", "--keep-cr", "-o" + directory, patch])
            messages = sorted(os.listdir(directory))
        if not messages:
            raise ValueError("no patch messages found in %s" % patch)
        if type(start_message) is not int or not 1 <= start_message <= len(messages):
            raise ValueError("start_message must be an integer within 1..%d" % len(messages))
        for name in messages[start_message - 1:]:
            message = patch if is_raw else os.path.join(directory, name)
            print("Applying %s message %s" % (os.path.basename(patch), name), flush=True)
            try:
                subprocess.check_call(["git", "-C", checkout, "apply", "--check", message])
            except subprocess.CalledProcessError:
                if not _patch_already_applied(checkout, message):
                    raise
                print("Already applied: %s message %s (postimages and reverse check verified)" % (os.path.basename(patch), name), flush=True)
                continue
            subprocess.check_call(["git", "-C", checkout, "apply", message])


def ensure_vllm_ascend(root, version="0.11"):
    profile = resolve_parsa_profile(version)
    checkout = os.path.join(os.path.abspath(root), profile["ascend_dir"])
    if not os.path.lexists(checkout):
        subprocess.check_call([
            "git", "clone", "--depth=1", "--branch", profile["ascend_tag"], "--single-branch",
            VLLM_ASCEND_REPO, checkout,
        ])
    if os.path.islink(checkout) or not os.path.isdir(checkout) or not os.path.exists(os.path.join(checkout, ".git")):
        raise FileExistsError("vLLM-Ascend path is not a Git checkout; refusing to overwrite: %s" % checkout)
    top = subprocess.check_output(["git", "-C", checkout, "rev-parse", "--show-toplevel"], text=True).strip()
    if os.path.realpath(top) != os.path.realpath(checkout):
        raise FileExistsError("vLLM-Ascend path is not its own Git checkout root: %s" % checkout)
    head = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
    if head != profile["ascend_commit"]:
        raise ValueError("vLLM-Ascend revision mismatch: expected %s, found %s; checkout left untouched" % (profile["ascend_commit"], head))
    status = subprocess.check_output(["git", "--no-optional-locks", "-C", checkout, "status", "--porcelain", "--untracked-files=all"], text=True)
    if status.strip():
        raise ValueError("vLLM-Ascend checkout has local changes; left untouched: %s" % checkout)
    print("vLLM-Ascend ready: %s (%s, %s)" % (checkout, profile["ascend_tag"], head), flush=True)
    return checkout


def prepare_vllm(root=None, ascend_only=False, version="0.11"):
    profile = resolve_parsa_profile(version)
    root = os.path.abspath(root or os.path.dirname(__file__))
    ascend_options = {} if version == "0.11" else {"version": version}
    if ascend_only:
        ensure_vllm_ascend(root, **ascend_options)
        return
    upstream = os.path.join(root, profile["upstream_dir"])
    checkouts = (
        ("parsa" if version == "0.11" else "parsa-" + profile["version"],
         os.path.join(root, profile["parsa_dir"]), (os.path.join(root, profile["patch"]),)),
    )
    if version == "0.11":
        checkouts += (("beam", os.path.join(root, "vllm-beam"), (os.path.join(root, "beam-vllm.patch"),)),)
    else:
        checkouts += ((
            "beam-port-v0.23", os.path.join(root, BEAM_NPU_PROFILE["beam_dir"]),
            tuple(os.path.join(root, patch) for patch in BEAM_NPU_PROFILE["patches"]),
        ),)
    paths = (upstream, *(checkout for _, checkout, _ in checkouts))
    existing = [path for path in paths if os.path.lexists(path)]
    if existing:
        raise FileExistsError("refusing to overwrite existing vLLM paths: %s" % ", ".join(existing))
    missing = [patch for _, _, patches in checkouts for patch in patches if not os.path.isfile(patch)]
    if missing:
        raise FileNotFoundError("missing vLLM patch files: %s" % ", ".join(missing))

    ensure_vllm_ascend(root, **ascend_options)
    subprocess.check_call([
        "git", "clone", "--depth=1", "--branch", profile["vllm_tag"], "--single-branch",
        "https://github.com/vllm-project/vllm.git", upstream,
    ])
    baseline = profile["vllm_tag"]
    if version != "0.11":
        head = subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip()
        if head != profile["vllm_commit"]:
            raise ValueError("vLLM revision mismatch: expected %s, found %s; refusing to create patched worktrees" % (profile["vllm_commit"], head))
        baseline = profile["vllm_commit"]
    for branch, checkout, patches in checkouts:
        subprocess.check_call(["git", "-C", upstream, "worktree", "add", "-b", branch, checkout, baseline])
        for patch in patches:
            apply_vllm_patch(checkout, patch)


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="action")

    run_parser = subparsers.add_parser("run")
    run_parser.set_defaults(action="run")
    run_parser.add_argument(
        "--shell",
        help="give me a bash shell instead of launching vllm",
        action="store_true",
    )
    run_parser.add_argument(
        "--port",
        default=None,
        type=int,
        help="port to run vllm on, the runner will pick a free one if not specified",
    )
    run_parser.add_argument(
        "--devices",
        type=parse_device_list,
        default=DEVICES,
        help="device list to use for deployment",
    )
    run_parser.add_argument(
        "--vllm", help="mount a vllm directory inside the container", default=None
    )
    run_parser.add_argument(
        "--ascend",
        help="mount a vllm-ascend directory inside the container",
        default=None,
    )
    run_parser.add_argument(
        "--model",
        help="model to use, must be located in /data1/models",
        default=MODEL,
    )
    run_parser.add_argument(
        "backend",
        help="Backend to run: parsa, beam, or vllm",
        choices=["parsa", "beam", "vllm"],
    )
    run_parser.add_argument("vllm_args", help="flags to pass to vllm serve", nargs="*")

    cs_parser = subparsers.add_parser("container-init")
    cs_parser.set_defaults(action="init")
    cs_parser.add_argument("vllm_args", help="flags to pass to vllm serve", nargs="*")

    image = subparsers.add_parser(
        "build-image", help="Build a docker image for the beam backend"
    )
    image.set_defaults(action="build-image")
    image.add_argument("backend", choices=["beam", "parsa"])
    image.add_argument("--vllm", help="vllm package source to use", default=None)
    image.add_argument("--ascend", help="vllm-ascend source to use", default=None)
    image.add_argument(
        "--host-network",
        action="store_true",
        help="use host network during build process",
    )
    image.add_argument(
        "--wheelhouse",
        default=None,
        help="v0.23 only: directory containing the offline PyArrow 24.0.0 ARM64 wheel (default: ./wheelhouse), or wheel path",
    )
    image.add_argument(
        "--offline-from",
        metavar="LOCAL_IMAGE",
        default=None,
        help="v0.23 only: reuse an existing local v0.23 image as the base; skip the normal vLLM build",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-vllm", help="Prepare pinned vLLM-Ascend and patched Parsa/BEAM vLLM worktrees"
    )
    prepare_parser.set_defaults(action="prepare-vllm")
    prepare_parser.add_argument(
        "--ascend-only", action="store_true", help="Only check/clone the image-matched vLLM-Ascend source; leave vLLM worktrees untouched"
    )

    for command_parser in (run_parser, image, prepare_parser):
        command_parser.add_argument("--version", type=parse_version, default="0.11", help="vLLM profile: 0.11 (default) or 0.23 (PARSA/BEAM NPU, pinned sources)")

    args = parser.parse_args(argv)
    if args.action in ("run", "build-image", "prepare-vllm"):
        try:
            if args.action == "prepare-vllm":
                resolve_parsa_profile(args.version)
            else:
                _resolve_backend_profile(args.backend, args.version, args.vllm, args.ascend)
        except ValueError as error:
            parser.error(str(error))
    if args.action == "run":
        try:
            run(
                args.backend,
                args.devices,
                port=args.port,
                shell=args.shell,
                vllm_args=args.vllm_args,
                mount_vllm=args.vllm,
                mount_ascend=args.ascend,
                model=args.model,
                version=args.version,
            )
        except ValueError as error:
            parser.error(str(error))
    elif args.action == "init":
        container_init(args.vllm_args)
    elif args.action == "build-image":
        try:
            build_image(
                args.backend,
                args.vllm,
                args.ascend,
                args.host_network,
                version=args.version,
                wheelhouse=args.wheelhouse,
                offline_from=args.offline_from,
            )
        except ValueError as error:
            parser.error(str(error))
    elif args.action == "prepare-vllm":
        prepare_vllm(ascend_only=args.ascend_only, version=args.version)


if __name__ == "__main__":
    main()
