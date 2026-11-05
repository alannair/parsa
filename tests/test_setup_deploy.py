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
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "setup"))
import deploy


class PrepareVllmTests(unittest.TestCase):
    def v023_profiles(self, **overrides):
        # Synthetic pins exercise selection only; they are not official revisions/images.
        profile = {
            **deploy.PARSA_PROFILES["0.23"],
            "vllm_commit": "1" * 40,
            "ascend_commit": "2" * 40,
            "base_image": "example.invalid/vllm-ascend:synthetic-0.23.0",
            **overrides,
        }
        return {**deploy.PARSA_PROFILES, "0.23": MappingProxyType(profile)}

    def legacy_source_paths(self, root):
        paths = []
        for name in ("vllm-upstream", "vllm-parsa", "vllm-beam", "vllm-ascend", "vllm-ascend-beam"):
            checkout = root / name
            checkout.mkdir()
            (checkout / "nested").mkdir()
            alias = root / ("alias-" + name)
            alias.symlink_to(checkout, target_is_directory=True)
            paths.extend((name, "./" + name + "/", str(checkout),
                          str(root) + "/unused/../" + name + "/.",
                          str(alias), str(alias / "nested"), str(checkout / "nested")))
            work = root / ("dotdot-" + name)
            work.mkdir()
            branch = root / ("external-" + name) / "branch"
            branch.mkdir(parents=True)
            (branch.parent / "current").mkdir()
            (work / "alias").symlink_to(branch, target_is_directory=True)
            (work / "current").symlink_to(checkout, target_is_directory=True)
            # abspath first hides the external branch; realpath first hides current.
            paths.append(str(work / "alias") + "/../current")
            (work / "nested-alias").symlink_to(checkout / "nested", target_is_directory=True)
            paths.append(str(work / "nested-alias") + "/..")
        return paths

    def test_resolve_parsa_profile_defaults_to_v011_and_keeps_legacy_artifacts(self):
        profile = deploy.resolve_parsa_profile()
        self.assertIs(profile, deploy.resolve_parsa_profile("0.11"))
        self.assertEqual(profile["version"], "0.11.0")
        self.assertEqual(profile["vllm_tag"], "v0.11.0")
        self.assertEqual(profile["vllm_commit"], "b8b302cde434df8c9289a2b465406b47ebab1c2d")
        self.assertEqual(profile["parsa_dir"], "vllm-parsa")
        self.assertEqual(profile["upstream_dir"], "vllm-upstream")
        self.assertEqual(profile["ascend_dir"], "vllm-ascend")
        self.assertEqual(profile["patch"], "parsa-vllm.patch")
        self.assertEqual(profile["image"], "vllm-ascend-parsa")
        self.assertEqual(profile["base_image"], "quay.io/ascend/vllm-ascend:v0.11.0")
        self.assertEqual(profile["dockerfile"], "Dockerfile.0.11")
        self.assertEqual(profile["ascend_commit"], "2f1aed98ccdb0fcbe1ff4fd0abab225bfd8d0367")
        with self.assertRaises(TypeError):
            profile["parsa_dir"] = "other"
        with self.assertRaises(TypeError):
            deploy.PARSA_PROFILES["0.11"] = {}

    def test_resolve_parsa_profile_selects_isolated_v023_artifacts(self):
        profile = deploy.resolve_parsa_profile("0.23")
        self.assertEqual(profile["version"], "0.23.0")
        self.assertEqual(profile["vllm_tag"], "v0.23.0")
        self.assertEqual(profile["vllm_commit"], "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665")
        self.assertEqual(profile["ascend_tag"], "v0.23.0")
        self.assertEqual(profile["ascend_commit"], "5cb98caaadeff42b5b62b996e34bb2aaa29d20fd")
        self.assertEqual(profile["base_image"], "quay.io/ascend/vllm-ascend:v0.23.0")
        self.assertEqual(profile["dockerfile"], "Dockerfile.0.23")
        self.assertEqual(profile["parsa_dir"], "vllm-parsa-0.23.0")
        self.assertEqual(profile["upstream_dir"], "vllm-0.23.0")
        self.assertEqual(profile["ascend_dir"], "vllm-ascend-0.23.0")
        self.assertEqual(profile["patch"], "parsa-vllm-0.23.0.patch")
        self.assertEqual(profile["image"], "vllm-ascend-parsa:0.23.0")
        with self.assertRaises(TypeError):
            profile["ascend_commit"] = "3" * 40

    def test_parse_version_rejects_unknown_versions(self):
        for version in ("0.11", "0.23"):
            self.assertEqual(deploy.parse_version(version), version)
        for version in ("0.22", "0.11.0", "v0.23.0", "", " 0.23"):
            with self.subTest(version=version):
                with self.assertRaises(argparse.ArgumentTypeError):
                    deploy.parse_version(version)

    def test_build_command_supports_standard_standalone_and_legacy_builders(self):
        with patch.object(
            deploy.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ):
            self.assertEqual(
                deploy._docker_build_command(), (["docker", "buildx", "build"], True)
            )

        with patch.object(
            deploy.subprocess, "run", return_value=SimpleNamespace(returncode=1)
        ), patch.object(deploy.os.path, "isfile", return_value=True), patch.object(
            deploy.os, "access", return_value=True
        ):
            standalone = deploy._docker_build_command()
        self.assertEqual(
            standalone,
            ([str(Path.home() / ".docker/cli-plugins/docker-buildx"), "build"], True),
        )

        with patch.object(
            deploy.subprocess, "run", return_value=SimpleNamespace(returncode=1)
        ), patch.object(deploy.os.path, "isfile", return_value=False):
            self.assertEqual(deploy._docker_build_command(), (["docker", "build"], False))

    def test_pyarrow_wheel_selection_matches_image_cpython_tag(self):
        with patch.object(
            deploy.subprocess, "check_output", return_value="cp312\n"
        ) as check_output:
            self.assertEqual(deploy._python_tag_for_image("example:v0.23"), "cp312")
        self.assertEqual(check_output.call_args.args[0][0:4], [
            "docker", "run", "--rm", "--network=none",
        ])

        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / (
                "pyarrow-24.0.0-cp312-cp312-manylinux_2_28_aarch64.whl"
            )
            wheel.touch()
            self.assertEqual(deploy._find_offline_pyarrow_wheel(directory, "cp312"), wheel)
            with self.assertRaisesRegex(ValueError, "CPython 3.11"):
                deploy._find_offline_pyarrow_wheel(directory, "cp311")

    def test_offline_pyarrow_repair_uses_legacy_dockerfile_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / (
                "pyarrow-24.0.0-cp312-cp312-manylinux_2_28_aarch64.whl"
            )
            wheel.touch()
            commands = []

            def record_command(command, **_kwargs):
                commands.append(command)
                if command[1] == "build":
                    dockerfile = Path(command[command.index("--file") + 1]).read_text()
                    self.assertIn("RUN python3 -c", dockerfile)
                    self.assertNotIn("\nimport numpy", dockerfile)

            with patch.object(
                deploy.subprocess, "check_call", side_effect=record_command
            ):
                deploy._build_v023_offline_pyarrow(
                    "vllm-ascend-beam:pre-pyarrow",
                    "vllm-ascend-beam:0.23.0",
                    wheel,
                )
            self.assertEqual(len(commands), 2)

    def test_unverified_v023_is_rejected_before_any_command_or_patch(self):
        profiles = self.v023_profiles(vllm_commit=None, ascend_commit=None, base_image=None)
        with patch.object(deploy, "PARSA_PROFILES", profiles):
            with self.assertRaisesRegex(ValueError, "0.23.*vllm_commit.*ascend_commit.*base_image"):
                deploy.resolve_parsa_profile("0.23")
        operations = (
            lambda: deploy.run("parsa", [0], version="0.23"),
            lambda: deploy.build_image("parsa", None, None, version="0.23"),
            lambda: deploy.ensure_vllm_ascend("/unused", version="0.23"),
            lambda: deploy.prepare_vllm("/unused", version="0.23"),
            lambda: deploy.prepare_vllm("/unused", ascend_only=True, version="0.23"),
        )
        for operation in operations:
            with patch.object(deploy, "PARSA_PROFILES", profiles), \
                    patch.object(deploy.subprocess, "check_call") as check_call, \
                    patch.object(deploy.subprocess, "check_output") as check_output, \
                    patch.object(deploy, "find_free_port") as find_port, \
                    patch.object(deploy, "apply_vllm_patch") as apply_patch:
                with self.assertRaisesRegex(ValueError, "0.23.*not ready"):
                    operation()
            check_call.assert_not_called()
            check_output.assert_not_called()
            find_port.assert_not_called()
            apply_patch.assert_not_called()

    def test_v023_requires_complete_commit_pins_and_base_image(self):
        for field in ("vllm_commit", "ascend_commit", "base_image"):
            invalid_values = (None, "", " ") if field == "base_image" else (None, "v0.23.0", "a" * 39, "g" * 40, "0" * 40)
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles(**{field: value})):
                        with self.assertRaisesRegex(ValueError, field):
                            deploy.resolve_parsa_profile("0.23")

    def test_other_backends_reject_v023_before_docker_even_with_complete_pins(self):
        with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                patch.object(deploy.subprocess, "check_call") as check_call, \
                patch.object(deploy, "find_free_port") as find_port:
            for backend in ("beam", "vllm"):
                with self.assertRaisesRegex(ValueError, "only supported for the parsa backend"):
                    deploy.run(backend, [0], version="0.23")
            with self.assertRaisesRegex(ValueError, "only supported for the parsa backend"):
                deploy.build_image("beam", None, None, version="0.23")
        check_call.assert_not_called()
        find_port.assert_not_called()

    def test_cli_defaults_all_profile_commands_to_v011(self):
        with patch.object(deploy, "run") as run, \
                patch.object(deploy, "build_image") as build, \
                patch.object(deploy, "prepare_vllm") as prepare:
            deploy.main(["run", "parsa", "--port", "8000"])
            self.assertEqual(run.call_args.kwargs["version"], "0.11")
            deploy.main(["build-image", "parsa"])
            self.assertEqual(build.call_args.kwargs["version"], "0.11")
            deploy.main(["prepare-vllm"])
            prepare.assert_called_once_with(ascend_only=False, version="0.11")

    def test_cli_forwards_v023_when_profile_pins_are_supplied(self):
        with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                patch.object(deploy, "run") as run, \
                patch.object(deploy, "build_image") as build, \
                patch.object(deploy, "prepare_vllm") as prepare:
            deploy.main(["run", "parsa", "--version", "0.23", "--port", "8000"])
            self.assertEqual(run.call_args.kwargs["version"], "0.23")
            deploy.main(["build-image", "parsa", "--version", "0.23"])
            self.assertEqual(build.call_args.kwargs["version"], "0.23")
            deploy.main(["prepare-vllm", "--version", "0.23", "--ascend-only"])
            prepare.assert_called_once_with(ascend_only=True, version="0.23")

    def test_documented_v023_cli_uses_production_profile(self):
        with patch.object(deploy, "prepare_vllm") as prepare, \
                patch.object(deploy.subprocess, "check_call") as check_call, \
                patch.object(deploy, "find_free_port", return_value=8000), \
                patch.object(sys, "stderr") as stderr:
            deploy.main(["prepare-vllm", "--version", "0.23"])
            prepare.assert_called_once_with(ascend_only=False, version="0.23")

            deploy.main(["build-image", "parsa", "--version", "0.23"])
            command = check_call.call_args.args[0]
            builder, _ = deploy._docker_build_command()
            self.assertEqual(command[:len(builder) + 2], [*builder, "-t", "vllm-ascend-parsa:0.23.0"])
            self.assertIn("VLLM_BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.23.0", command)
            self.assertIn("VLLM_VERSION=0.23.0", command)

            deploy.main(["run", "parsa", "--version", "0.23", "--devices", "0,1"])
            command = check_call.call_args.args[0]
            self.assertEqual(command[0:2], ["docker", "run"])
            self.assertIn("vllm-ascend-parsa:0.23.0", command)
            self.assertIn("VLLM_VERSION=0.23.0", command)
            self.assertIn("ASCEND_RT_VISIBLE_DEVICES=0,1", command)
            self.assertEqual(command[-3:], ["python", "/workspace/setup/deploy.py", "container-init"])

            deploy.main(["run", "--version", "0.23", "--devices", "0,1", "parsa", "--", "--help"])
            command = check_call.call_args.args[0]
            self.assertEqual(command[-5:], ["python", "/workspace/setup/deploy.py", "container-init", "--", "--help"])
        self.assertEqual(check_call.call_count, 3)

    def test_production_v023_build_reaches_docker_with_exact_vendor_base(self):
        with patch.object(deploy.subprocess, "check_call") as check_call:
            deploy.build_image("parsa", None, None, use_host=False, version="0.23")
        command = check_call.call_args.args[0]
        builder, _ = deploy._docker_build_command()
        self.assertEqual(command[:len(builder) + 2], [*builder, "-t", "vllm-ascend-parsa:0.23.0"])
        self.assertIn("VLLM_BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.23.0", command)

    def test_v011_default_build_matches_explicit_version(self):
        for backend in ("parsa", "beam"):
            with self.subTest(backend=backend), patch.object(deploy.subprocess, "check_call") as check_call:
                deploy.build_image(backend, None, None, use_host=False)
                default_command = check_call.call_args.args[0]
                deploy.build_image(backend, None, None, use_host=False, version="0.11")
                self.assertEqual(check_call.call_args.args[0], default_command)
            self.assertEqual(check_call.call_count, 2)

    def test_cli_rejects_unsupported_or_unready_profiles_before_dispatch(self):
        for arguments, message in (
            (["run", "beam", "--version", "0.23"], "only supported for the parsa backend"),
            (["build-image", "beam", "--version", "0.23"], "only supported for the parsa backend"),
            (["run", "parsa", "--version", "0.22"], "--version must be one of: 0.11, 0.23"),
            (["build-image", "parsa", "--version", "0.22"], "--version must be one of: 0.11, 0.23"),
            (["prepare-vllm", "--version", "0.22"], "--version must be one of: 0.11, 0.23"),
            (["run", "parsa", "--version", "0.23"], "0.23 is not ready"),
            (["build-image", "parsa", "--version", "0.23"], "0.23 is not ready"),
            (["prepare-vllm", "--version", "0.23"], "0.23 is not ready"),
        ):
            with self.subTest(arguments=arguments):
                with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles(
                        vllm_commit=None, ascend_commit=None, base_image=None)), \
                        patch.object(deploy, "run") as run, \
                        patch.object(deploy, "build_image") as build, \
                        patch.object(deploy, "prepare_vllm") as prepare, \
                        patch.object(sys, "stderr") as stderr:
                    with self.assertRaises(SystemExit) as error:
                        deploy.main(arguments)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, "".join(item.args[0] for item in stderr.write.call_args_list))
                run.assert_not_called()
                build.assert_not_called()
                prepare.assert_not_called()

    def test_v023_run_uses_matching_image_and_runtime_version(self):
        with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                patch.object(deploy.subprocess, "check_call") as check_call:
            deploy.run("parsa", [0], port=8000, version="0.23")
        command = check_call.call_args.args[0]
        self.assertIn("vllm-ascend-parsa:0.23.0", command)
        self.assertIn("VLLM_VERSION=0.23.0", command)
        self.assertNotIn("VLLM_VERSION=0.11.0", command)

    def test_build_defaults_preserve_legacy_commands_for_both_backends(self):
        for backend, image, directory in (
            ("parsa", "vllm-ascend-parsa", "vllm-parsa"),
            ("beam", "vllm-ascend-beam", "vllm-beam"),
        ):
            with patch.object(deploy.subprocess, "check_call") as check_call:
                deploy.build_image(backend, None, None, use_host=False)
            command = check_call.call_args.args[0]
            builder, _ = deploy._docker_build_command()
            self.assertEqual(command[:len(builder) + 2], [*builder, "-t", image])
            self.assertEqual(command[command.index("--file") + 1], "Dockerfile.0.11")
            self.assertIn("VLLM_DIR=" + directory, command)
            self.assertIn("VLLM_ASCEND_DIR=vllm-ascend", command)
            self.assertFalse(any(value.startswith(("VLLM_BASE_IMAGE=", "VLLM_VERSION=")) for value in command))
            self.assertFalse(any(value.startswith(("VLLM_CONSTRAINTS_FILE=", "VLLM_SOURCE_REVISION=", "VLLM_ASCEND_SOURCE_REVISION=")) for value in command))

    def test_build_arguments_route_each_profile_to_its_image_and_sources(self):
        for backend, version, expected in (
            ("parsa", "0.11", {
                "VLLM_DIR": "vllm-parsa", "VLLM_ASCEND_DIR": "vllm-ascend",
            }),
            ("beam", "0.11", {
                "VLLM_DIR": "vllm-beam", "VLLM_ASCEND_DIR": "vllm-ascend",
            }),
            ("parsa", "0.23", {
                "VLLM_DIR": "vllm-parsa-0.23.0", "VLLM_ASCEND_DIR": "vllm-ascend-0.23.0",
                "VLLM_BASE_IMAGE": "quay.io/ascend/vllm-ascend:v0.23.0", "VLLM_VERSION": "0.23.0",
            }),
        ):
            with self.subTest(backend=backend, version=version):
                with patch.object(deploy.subprocess, "check_call") as check_call:
                    deploy.build_image(backend, None, None, use_host=False, version=version)
                command = check_call.call_args.args[0]
                arguments = dict(command[index + 1].split("=", 1)
                                 for index, value in enumerate(command) if value == "--build-arg")
                self.assertEqual({key: arguments.get(key) for key in expected}, expected)
                self.assertFalse(any(key in arguments for key in (
                    "VLLM_CONSTRAINTS_FILE", "VLLM_SOURCE_REVISION", "VLLM_ASCEND_SOURCE_REVISION",
                )))

    def test_build_rejects_mismatched_profile_dockerfile_before_docker(self):
        for version, wrong_dockerfile in (
            ("0.11", "Dockerfile.0.23"),
            ("0.23", "Dockerfile.0.11"),
            ("0.11", None),
            ("0.23", ""),
        ):
            profiles = {
                **deploy.PARSA_PROFILES,
                version: MappingProxyType({**deploy.PARSA_PROFILES[version],
                                           "dockerfile": wrong_dockerfile}),
            }
            with self.subTest(version=version, dockerfile=wrong_dockerfile):
                with patch.object(deploy, "PARSA_PROFILES", profiles), \
                        patch.object(deploy.subprocess, "check_call") as check_call, \
                        patch.object(deploy.subprocess, "check_output") as check_output:
                    with self.assertRaisesRegex(ValueError, "Dockerfile"):
                        deploy.build_image("parsa", None, None, version=version)
                check_call.assert_not_called()
                check_output.assert_not_called()

    def test_v023_build_selects_isolated_sources_and_honors_overrides(self):
        for vllm, ascend, expected_vllm, expected_ascend in (
            (None, None, "vllm-parsa-0.23.0", "vllm-ascend-0.23.0"),
            ("custom-vllm", "custom-ascend", "custom-vllm", "custom-ascend"),
            ("vllm-parsa-custom", "vllm-ascend-custom", "vllm-parsa-custom", "vllm-ascend-custom"),
        ):
            with patch.object(deploy.subprocess, "check_call") as check_call, \
                    patch.object(deploy.subprocess, "check_output") as check_output:
                deploy.build_image("parsa", vllm, ascend, version="0.23")
            command = check_call.call_args.args[0]
            self.assertIn("vllm-ascend-parsa:0.23.0", command)
            self.assertIn("VLLM_DIR=" + expected_vllm, command)
            self.assertIn("VLLM_ASCEND_DIR=" + expected_ascend, command)
            self.assertIn("VLLM_BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.23.0", command)
            self.assertEqual(command[command.index("--file") + 1], "Dockerfile.0.23")
            self.assertIn("VLLM_VERSION=0.23.0", command)
            check_output.assert_not_called()

    def test_v023_build_labels_selected_profile_revisions_and_patch(self):
        with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles(
                vllm_commit="3" * 40, ascend_commit="4" * 40,
                patch="custom-parsa-vllm-0.23.0.patch")), \
                patch.object(deploy.subprocess, "check_call") as check_call:
            deploy.build_image("parsa", None, None, version="0.23")
        command = check_call.call_args.args[0]
        labels = [command[index + 1] for index, value in enumerate(command) if value == "--label"]
        self.assertCountEqual(labels, [
            "org.opencontainers.image.version=0.23.0",
            "org.parsa.vllm.revision=" + "3" * 40,
            "org.parsa.vllm-ascend.revision=" + "4" * 40,
            "org.parsa.patch=custom-parsa-vllm-0.23.0.patch",
        ])
        self.assertEqual(command[-1], ".")

    def test_v011_parsa_build_labels_provenance_for_default_and_explicit_version(self):
        for options in ({}, {"version": "0.11"}):
            with self.subTest(options=options):
                with patch.object(deploy.subprocess, "check_call") as check_call:
                    deploy.build_image("parsa", "custom-vllm", "custom-ascend", use_host=False, **options)
                command = check_call.call_args.args[0]
                labels = [command[index + 1] for index, value in enumerate(command) if value == "--label"]
                self.assertCountEqual(labels, [
                    "org.opencontainers.image.version=0.11.0",
                    "org.parsa.vllm.revision=b8b302cde434df8c9289a2b465406b47ebab1c2d",
                    "org.parsa.vllm-ascend.revision=2f1aed98ccdb0fcbe1ff4fd0abab225bfd8d0367",
                    "org.parsa.patch=parsa-vllm.patch",
                ])
                builder, _ = deploy._docker_build_command()
                self.assertEqual(command[:len(builder) + 2], [*builder, "-t", "vllm-ascend-parsa"])
                self.assertIn("VLLM_DIR=custom-vllm", command)
                self.assertIn("VLLM_ASCEND_DIR=custom-ascend", command)
                self.assertFalse(any(value.startswith(("VLLM_BASE_IMAGE=", "VLLM_VERSION=")) for value in command))

    def test_v011_explicit_build_and_run_match_defaults_without_label_requirements(self):
        for backend in ("parsa", "beam"):
            with self.subTest(backend=backend):
                with patch.object(deploy.subprocess, "check_call") as check_call, \
                        patch.object(deploy.subprocess, "check_output") as check_output:
                    deploy.build_image(backend, None, None, use_host=False)
                    default_build = check_call.call_args.args[0]
                    deploy.build_image(backend, None, None, use_host=False, version="0.11")
                    self.assertEqual(check_call.call_args.args[0], default_build)
                    self.assertEqual("--label" in default_build, backend == "parsa")
                    deploy.run(backend, [0], port=8000)
                    default_run = check_call.call_args.args[0]
                    deploy.run(backend, [0], port=8000, version="0.11")
                    self.assertEqual(check_call.call_args.args[0], default_run)
                    self.assertIn("VLLM_VERSION=0.11.0", default_run)
                check_output.assert_not_called()
                self.assertEqual(check_call.call_count, 4)

    def test_v023_shell_uses_matching_image_and_runtime_version(self):
        with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                patch.object(deploy.subprocess, "check_call") as check_call:
            deploy.run("parsa", [0], port=8000, shell=True, version="0.23")
        command = check_call.call_args.args[0]
        self.assertEqual(command[-2:], ["vllm-ascend-parsa:0.23.0", "bash"])
        self.assertIn("-it", command)
        self.assertIn("VLLM_VERSION=0.23.0", command)
        self.assertNotIn("VLLM_VERSION=0.11.0", command)

    def test_v023_build_rejects_legacy_source_overrides_before_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.legacy_source_paths(Path(directory))
            for source in paths:
                for index in (0, 1):
                    sources = [None, None]
                    sources[index] = source
                    with self.subTest(source=source, index=index):
                        with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                                patch.object(deploy.subprocess, "check_call") as check_call:
                            with self.assertRaisesRegex(ValueError, "legacy v0.11 source"):
                                deploy.build_image("parsa", *sources, version="0.23")
                        check_call.assert_not_called()

    def test_v023_build_validates_effective_sources_with_legacy_default_symlinks(self):
        for default_name, legacy_name, mixed_sources in (
            ("vllm-parsa-0.23.0", "vllm-parsa", (None, "custom-ascend")),
            ("vllm-ascend-0.23.0", "vllm-ascend", ("custom-vllm", None)),
        ):
            with self.subTest(default=default_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                legacy = root / legacy_name
                legacy.mkdir()
                (root / default_name).symlink_to(legacy, target_is_directory=True)
                with patch.object(deploy.os, "getcwd", return_value=directory):
                    for sources in ((None, None), mixed_sources):
                        with self.subTest(sources=sources), \
                                patch.object(deploy.subprocess, "check_call") as check_call:
                            with self.assertRaisesRegex(ValueError, "legacy v0.11 source"):
                                deploy.build_image("parsa", *sources, version="0.23")
                            check_call.assert_not_called()
                with patch.object(deploy.subprocess, "check_call") as check_call:
                    deploy.build_image("parsa", "custom-vllm", "custom-ascend", version="0.23")
                    check_call.assert_called_once()
                    command = check_call.call_args.args[0]
                    self.assertIn("VLLM_DIR=custom-vllm", command)
                    self.assertIn("VLLM_ASCEND_DIR=custom-ascend", command)

    def test_v023_run_rejects_legacy_mounts_before_port_selection_or_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.legacy_source_paths(Path(directory))
            for source in paths:
                for mount in ("mount_vllm", "mount_ascend"):
                    for shell in (False, True):
                        with self.subTest(source=source, mount=mount, shell=shell):
                            with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                                    patch.object(deploy.subprocess, "check_call") as check_call, \
                                    patch.object(deploy, "find_free_port") as find_port:
                                with self.assertRaisesRegex(ValueError, "legacy v0.11 source"):
                                    deploy.run("parsa", [0], shell=shell, version="0.23", **{mount: source})
                            check_call.assert_not_called()
                            find_port.assert_not_called()

    def test_v023_run_rejects_normalized_symlink_dotdot_legacy_mount_before_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            branch = root / "external" / "branch"
            branch.mkdir(parents=True)
            (branch.parent / "current").mkdir()
            legacy = work / "vllm-parsa"
            legacy.mkdir()
            (work / "alias").symlink_to(branch, target_is_directory=True)
            (work / "current").symlink_to(legacy, target_is_directory=True)
            source = str(work / "alias") + "/../current"
            self.assertEqual(deploy.os.path.realpath(source), str(branch.parent / "current"))
            self.assertEqual(deploy.os.path.realpath(deploy.os.path.abspath(source)), str(legacy))
            with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                    patch.object(deploy.subprocess, "check_call") as check_call, \
                    patch.object(deploy, "find_free_port") as find_port:
                with self.assertRaisesRegex(ValueError, "legacy v0.11 source"):
                    deploy.run("parsa", [0], mount_vllm=source, version="0.23")
            find_port.assert_not_called()
            check_call.assert_not_called()

    def test_v023_run_accepts_nonlegacy_custom_mounts_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            aliases = []
            for name in ("vllm-parsa-0.23.0", "vllm-ascend-0.23.0"):
                source = root / name
                source.mkdir()
                alias = root / ("custom-" + name)
                alias.symlink_to(source, target_is_directory=True)
                sources.append(source)
                aliases.append(alias)
            for vllm, ascend in (sources, aliases):
                with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                        patch.object(deploy.subprocess, "check_call") as check_call:
                    deploy.run("parsa", [0], port=8000, mount_vllm=vllm, mount_ascend=ascend, version="0.23")
                command = check_call.call_args.args[0]
                self.assertIn(str(vllm) + ":/vllm-workspace/vllm", command)
                self.assertIn(str(ascend) + ":/vllm-workspace/vllm-ascend", command)

    def test_cli_rejects_v023_legacy_source_overrides_before_dispatch(self):
        for command in ("run", "build-image"):
            for flag in ("--vllm", "--ascend"):
                for source in ("vllm-upstream", "vllm-parsa", "vllm-beam", "vllm-ascend", "vllm-ascend-beam"):
                    with self.subTest(command=command, flag=flag, source=source):
                        with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                                patch.object(deploy, "run") as run, \
                                patch.object(deploy, "build_image") as build, \
                                patch.object(sys, "stderr") as stderr:
                            with self.assertRaises(SystemExit) as error:
                                deploy.main([command, "parsa", "--version", "0.23", flag, source])
                        self.assertEqual(error.exception.code, 2)
                        self.assertIn("legacy v0.11 source", "".join(item.args[0] for item in stderr.write.call_args_list))
                        run.assert_not_called()
                        build.assert_not_called()

    def test_v011_preserves_legacy_build_overrides_and_run_mounts(self):
        with patch.object(deploy.subprocess, "check_call") as check_call:
            deploy.build_image("parsa", "vllm-parsa", "vllm-ascend")
            command = check_call.call_args.args[0]
            self.assertIn("VLLM_DIR=vllm-parsa", command)
            self.assertIn("VLLM_ASCEND_DIR=vllm-ascend", command)
            deploy.run("parsa", [0], port=8000, mount_vllm="vllm-parsa", mount_ascend="vllm-ascend")
            command = check_call.call_args.args[0]
            self.assertIn(deploy.os.path.abspath("vllm-parsa") + ":/vllm-workspace/vllm", command)
            self.assertIn(deploy.os.path.abspath("vllm-ascend") + ":/vllm-workspace/vllm-ascend", command)

    def test_prepare_v023_is_isolated_and_checks_exact_upstream_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("vllm-upstream", "vllm-parsa", "vllm-beam", "vllm-ascend"):
                (root / name).mkdir()
            (root / "parsa-vllm-0.23.0.patch").touch()
            upstream = str(root / "vllm-0.23.0")
            parsa = str(root / "vllm-parsa-0.23.0")
            with patch.object(deploy, "ensure_vllm_ascend") as ensure_ascend, \
                    patch.object(deploy, "apply_vllm_patch") as apply_patch, \
                    patch.object(deploy.subprocess, "check_call") as check_call, \
                    patch.object(deploy.subprocess, "check_output", return_value="0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665\n") as check_output:
                deploy.prepare_vllm(root, version="0.23")
            ensure_ascend.assert_called_once_with(str(root), version="0.23")
            self.assertEqual(check_call.call_args_list, [
                call(["git", "clone", "--depth=1", "--branch", "v0.23.0", "--single-branch",
                      "https://github.com/vllm-project/vllm.git", upstream]),
                call(["git", "-C", upstream, "worktree", "add", "-b", "parsa-0.23.0", parsa, "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"]),
            ])
            check_output.assert_called_once_with(["git", "-C", upstream, "rev-parse", "HEAD"], text=True)
            apply_patch.assert_called_once_with(parsa, str(root / "parsa-vllm-0.23.0.patch"))
            self.assertTrue(all((root / name).is_dir() for name in ("vllm-upstream", "vllm-parsa", "vllm-beam", "vllm-ascend")))

    def test_prepare_v023_rejects_wrong_upstream_revision_before_worktree_or_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parsa-vllm-0.23.0.patch").touch()
            with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                    patch.object(deploy, "ensure_vllm_ascend"), \
                    patch.object(deploy, "apply_vllm_patch") as apply_patch, \
                    patch.object(deploy.subprocess, "check_call") as check_call, \
                    patch.object(deploy.subprocess, "check_output", return_value="3" * 40):
                with self.assertRaisesRegex(ValueError, "vLLM revision mismatch"):
                    deploy.prepare_vllm(root, version="0.23")
            self.assertEqual(check_call.call_count, 1)
            self.assertEqual(check_call.call_args.args[0][1], "clone")
            apply_patch.assert_not_called()

    def test_prepare_v023_ascend_only_preserves_both_versions_of_vllm_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("vllm-upstream", "vllm-parsa", "vllm-beam", "vllm-0.23.0", "vllm-parsa-0.23.0"):
                (root / name).mkdir()
            with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                    patch.object(deploy, "ensure_vllm_ascend") as ensure_ascend, \
                    patch.object(deploy.subprocess, "check_call") as check_call, \
                    patch.object(deploy, "apply_vllm_patch") as apply_patch:
                deploy.prepare_vllm(root, ascend_only=True, version="0.23")
            ensure_ascend.assert_called_once_with(str(root), version="0.23")
            check_call.assert_not_called()
            apply_patch.assert_not_called()

    def test_v023_ascend_checkout_uses_selected_pin_and_preserves_dirty_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "vllm-ascend-0.23.0"

            def clone(command):
                checkout.mkdir()
                (checkout / ".git").mkdir()

            with patch.object(deploy.subprocess, "check_call", side_effect=clone) as check_call, \
                    patch.object(deploy.subprocess, "check_output", side_effect=[str(checkout), "5cb98caaadeff42b5b62b996e34bb2aaa29d20fd", ""]):
                self.assertEqual(deploy.ensure_vllm_ascend(root, version="0.23"), str(checkout))
            check_call.assert_called_once_with([
                "git", "clone", "--depth=1", "--branch", "v0.23.0", "--single-branch",
                "https://github.com/vllm-project/vllm-ascend.git", str(checkout),
            ])
            for head, status in (("3" * 40, ""), ("2" * 40, " M local.py\n")):
                with patch.object(deploy, "PARSA_PROFILES", self.v023_profiles()), \
                        patch.object(deploy.subprocess, "check_call") as check_call, \
                        patch.object(deploy.subprocess, "check_output", side_effect=[str(checkout), head, status]):
                    with self.assertRaises(ValueError):
                        deploy.ensure_vllm_ascend(root, version="0.23")
                check_call.assert_not_called()

    def test_run_mounts_code_root_and_server_data_for_shell_and_serving(self):
        code_root = str(Path(deploy.__file__).resolve().parents[1])
        self.assertEqual(deploy.MODEL_DIR, "/data1/models")
        self.assertEqual(deploy.DATASET_DIR, deploy.os.environ.get("DATASET_DIR", "/data1/datasets"))
        for shell in (False, True):
            with patch.object(deploy.os, "getcwd", return_value="/unrelated"), \
                    patch.object(deploy.subprocess, "check_call") as check_call:
                deploy.run("parsa", [0, 1, 2, 3], port=8000, shell=shell)
            command = check_call.call_args.args[0]
            self.assertIn("vllm-ascend-parsa", command)
            self.assertIn("VLLM_VERSION=0.11.0", command)
            self.assertIn(code_root + ":/workspace", command)
            self.assertNotIn("/unrelated:/workspace", command)
            self.assertIn("/data1/models:/models", command)
            self.assertIn(deploy.DATASET_DIR + ":/datasets", command)
            if shell:
                self.assertEqual(command[-1], "bash")
                self.assertIn("-it", command)
            else:
                self.assertEqual(command[-3:], ["python", "/workspace/setup/deploy.py", "container-init"])

    def test_clones_exact_tag_and_patches_two_worktree_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for patch_name in ("parsa-vllm.patch", "beam-vllm.patch"):
                (root / patch_name).touch()

            with patch.object(deploy.subprocess, "check_call") as check_call, \
                    patch.object(deploy, "apply_vllm_patch") as apply_patch, \
                    patch.object(deploy, "ensure_vllm_ascend") as ensure_ascend, \
                    patch.object(deploy.subprocess, "check_output") as check_output:
                deploy.prepare_vllm(root)
            ensure_ascend.assert_called_once_with(str(root))
            check_output.assert_not_called()

            upstream = str(root / "vllm-upstream")
            parsa = str(root / "vllm-parsa")
            beam = str(root / "vllm-beam")
            self.assertEqual(check_call.call_args_list, [
                call(["git", "clone", "--depth=1", "--branch", "v0.11.0", "--single-branch",
                      "https://github.com/vllm-project/vllm.git", upstream]),
                call(["git", "-C", upstream, "worktree", "add", "-b", "parsa", parsa, "v0.11.0"]),
                call(["git", "-C", upstream, "worktree", "add", "-b", "beam", beam, "v0.11.0"]),
            ])
            self.assertEqual(apply_patch.call_args_list, [
                call(parsa, str(root / "parsa-vllm.patch")),
                call(beam, str(root / "beam-vllm.patch")),
            ])

    def test_applies_mailbox_messages_in_order(self):
        def split_mailbox(command):
            if command[1] == "mailsplit":
                directory = Path(command[3][2:])
                (directory / "0002").write_bytes(b"second message")
                (directory / "0001").write_bytes(b"first message")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = str(root / "vllm-beam")
            mailbox = str(root / "beam-vllm.patch")
            with patch.object(deploy.subprocess, "check_call", side_effect=split_mailbox) as check_call:
                deploy.apply_vllm_patch(checkout, mailbox)
            split_command = check_call.call_args_list[0].args[0]
            message_dir = Path(split_command[3][2:])
            self.assertEqual(split_command, ["git", "mailsplit", "--keep-cr", "-o" + str(message_dir), mailbox])
            self.assertEqual(check_call.call_args_list[1:], [
                call(["git", "-C", checkout, "apply", "--check", str(message_dir / "0001")]),
                call(["git", "-C", checkout, "apply", str(message_dir / "0001")]),
                call(["git", "-C", checkout, "apply", "--check", str(message_dir / "0002")]),
                call(["git", "-C", checkout, "apply", str(message_dir / "0002")]),
            ])

    def test_stops_before_applying_a_failed_message(self):
        def fail_second_check(command):
            if command[1] == "mailsplit":
                directory = Path(command[3][2:])
                for name in ("0001", "0002", "0003"):
                    (directory / name).write_bytes(b"message")
            elif "--check" in command and command[-1].endswith("0002"):
                raise deploy.subprocess.CalledProcessError(1, command)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(deploy.subprocess, "check_call", side_effect=fail_second_check) as check_call:
                with self.assertRaises(deploy.subprocess.CalledProcessError):
                    deploy.apply_vllm_patch(root / "vllm-beam", root / "beam-vllm.patch")
            self.assertEqual(len(check_call.call_args_list), 4)
            self.assertIn("--check", check_call.call_args_list[-1].args[0])
            self.assertTrue(check_call.call_args_list[-1].args[0][-1].endswith("0002"))

    def test_rejects_an_empty_mailbox(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(deploy.subprocess, "check_call") as check_call:
                with self.assertRaises(ValueError):
                    deploy.apply_vllm_patch(root / "vllm-beam", root / "beam-vllm.patch")
            self.assertEqual(len(check_call.call_args_list), 1)

    def test_explicit_resume_starts_at_the_requested_message(self):
        def split_mailbox(command):
            if command[1] == "mailsplit":
                directory = Path(command[3][2:])
                for name in ("0001", "0002"):
                    (directory / name).write_bytes(b"message")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(deploy.subprocess, "check_call", side_effect=split_mailbox) as check_call:
                deploy.apply_vllm_patch(root / "vllm-beam", root / "beam-vllm.patch", start_message=2)
            self.assertEqual(len(check_call.call_args_list), 3)
            self.assertTrue(all(item.args[0][-1].endswith("0002") for item in check_call.call_args_list[1:]))
            for start in (0, 3, True):
                with patch.object(deploy.subprocess, "check_call", side_effect=split_mailbox):
                    with self.assertRaises(ValueError):
                        deploy.apply_vllm_patch(root / "vllm-beam", root / "beam-vllm.patch", start_message=start)

    def test_duplicate_gate_requires_postimage_hash_mode_and_reverse_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "vllm-beam"
            checkout.mkdir()
            target = checkout / "file.py"
            content = b"new\n"
            target.write_bytes(content)
            digest = hashlib.sha1(b"blob 4\0" + content).hexdigest()
            message = root / "0018"
            patch_text = (
                "diff --git a/file.py b/file.py\nnew file mode 100644\n"
                "index 000000000..%s\n--- /dev/null\n+++ b/file.py\n@@ -0,0 +1 @@\n+new\n"
                % digest[:9]
            )
            message.write_text(patch_text, encoding="utf-8")
            with patch.object(deploy.subprocess, "check_output", return_value=digest + "\n") as hash_object, \
                    patch.object(deploy.subprocess, "call", return_value=0) as reverse_check:
                self.assertTrue(deploy._patch_already_applied(str(checkout), str(message)))
                hash_object.return_value = "f" * 40 + "\n"
                self.assertFalse(deploy._patch_already_applied(str(checkout), str(message)))
                hash_object.return_value = digest + "\n"
                target.chmod(0o755)
                self.assertFalse(deploy._patch_already_applied(str(checkout), str(message)))
                target.chmod(0o644)
                reverse_check.return_value = 1
                self.assertFalse(deploy._patch_already_applied(str(checkout), str(message)))
                reverse_check.return_value = 0
                for malformed in (
                    patch_text.replace("file.py", "../file.py"),
                    patch_text.replace("b/file.py", "b/other.py", 1),
                    patch_text + patch_text,
                    patch_text + "diff --git ",
                    patch_text + "rename from file.py\n",
                    patch_text.replace(digest[:9], "000000000"),
                    patch_text.replace(digest[:9], digest[:8]),
                    patch_text + "index 000000000..%s\n" % digest[:9],
                ):
                    message.write_text(malformed, encoding="utf-8")
                    self.assertFalse(deploy._patch_already_applied(str(checkout), str(message)))
                message.write_text(patch_text, encoding="utf-8")
                outside = root / "outside.py"
                outside.write_bytes(content)
                target.unlink()
                target.symlink_to(outside)
                self.assertFalse(deploy._patch_already_applied(str(checkout), str(message)))

    def test_verified_duplicate_is_skipped_but_actual_apply_failure_is_not(self):
        def git_command(command):
            if command[1] == "mailsplit":
                Path(command[3][2:], "0018").write_bytes(b"message")
            elif "--check" in command:
                raise deploy.subprocess.CalledProcessError(1, command)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(deploy.subprocess, "check_call", side_effect=git_command) as check_call, \
                    patch.object(deploy, "_patch_already_applied", return_value=True) as duplicate:
                deploy.apply_vllm_patch(root / "vllm-beam", root / "beam-vllm.patch")
                self.assertEqual(len(check_call.call_args_list), 2)
                duplicate.assert_called_once()

            def failed_apply(command):
                if command[1] == "mailsplit":
                    Path(command[3][2:], "0001").write_bytes(b"message")
                elif "--check" not in command:
                    raise deploy.subprocess.CalledProcessError(1, command)

            with patch.object(deploy.subprocess, "check_call", side_effect=failed_apply), \
                    patch.object(deploy, "_patch_already_applied", return_value=True) as duplicate:
                with self.assertRaises(deploy.subprocess.CalledProcessError):
                    deploy.apply_vllm_patch(root / "vllm-beam", root / "beam-vllm.patch")
                duplicate.assert_not_called()

    def test_refuses_existing_checkout_paths_before_cloning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vllm-beam").mkdir()
            with patch.object(deploy.subprocess, "check_call") as check_call:
                with self.assertRaises(FileExistsError):
                    deploy.prepare_vllm(root)
            check_call.assert_not_called()

    def test_ascend_only_preserves_existing_vllm_worktrees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("vllm-upstream", "vllm-parsa", "vllm-beam"):
                (root / name).mkdir()
            with patch.object(deploy, "ensure_vllm_ascend") as ensure_ascend, \
                    patch.object(deploy.subprocess, "check_call") as check_call, \
                    patch.object(deploy, "apply_vllm_patch") as apply_patch:
                deploy.prepare_vllm(root, ascend_only=True)
            ensure_ascend.assert_called_once_with(str(root))
            check_call.assert_not_called()
            apply_patch.assert_not_called()

    def test_ascend_clone_is_pinned_to_the_image_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "vllm-ascend"

            def clone(command):
                checkout.mkdir()
                (checkout / ".git").mkdir()

            with patch.object(deploy.subprocess, "check_call", side_effect=clone) as check_call, \
                    patch.object(deploy.subprocess, "check_output", side_effect=[str(checkout) + "\n", deploy.VLLM_ASCEND_COMMIT + "\n", ""]):
                self.assertEqual(deploy.ensure_vllm_ascend(root), str(checkout))
            check_call.assert_called_once_with([
                "git", "clone", "--depth=1", "--branch", "v0.11.0", "--single-branch",
                "https://github.com/vllm-project/vllm-ascend.git", str(checkout),
            ])
            self.assertEqual(deploy.VLLM_ASCEND_COMMIT, "2f1aed98ccdb0fcbe1ff4fd0abab225bfd8d0367")

    def test_existing_ascend_checkout_is_reused_or_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "vllm-ascend"
            (checkout / ".git").mkdir(parents=True)
            sentinel = checkout / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            for head, status, valid in (
                (deploy.VLLM_ASCEND_COMMIT, "", True),
                ("0" * 40, "", False),
                (deploy.VLLM_ASCEND_COMMIT, " M file.py\n", False),
                (deploy.VLLM_ASCEND_COMMIT, "?? extra.py\n", False),
            ):
                with patch.object(deploy.subprocess, "check_call") as check_call, \
                        patch.object(deploy.subprocess, "check_output", side_effect=[str(checkout), head, status]):
                    if valid:
                        self.assertEqual(deploy.ensure_vllm_ascend(root), str(checkout))
                    else:
                        with self.assertRaises(ValueError):
                            deploy.ensure_vllm_ascend(root)
                check_call.assert_not_called()
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            with patch.object(deploy.subprocess, "check_call") as check_call, \
                    patch.object(deploy.subprocess, "check_output", return_value=str(root)):
                with self.assertRaises(FileExistsError):
                    deploy.ensure_vllm_ascend(root)
            check_call.assert_not_called()
            self.assertEqual(deploy.default_dirs["beam"]["vllm-ascend"], "vllm-ascend")
            self.assertEqual(deploy.default_dirs["parsa"]["vllm-ascend"], "vllm-ascend")


if __name__ == "__main__":
    unittest.main()
