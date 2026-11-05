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

class BackendAdapter:

    __slots__ = ("backend_url", "name", "supports_priority", "capabilities")

    def __init__(self, backend_url, name="backend", supports_priority=True, capabilities=None):
        self.backend_url = str(backend_url).rstrip("/")
        self.name = str(name)
        self.supports_priority = bool(supports_priority)
        self.capabilities = dict(capabilities or {})

    def request_url(self, path):
        return self.backend_url.rstrip("/") + str(path)

    def metric_url(self):
        return self.backend_url.rstrip("/") + "/metrics"

    def admin_kv_projection_url(self):
        return self.backend_url.rstrip("/") + "/admin/parsa/kv-projection"

    def health_url(self):
        return self.backend_url.rstrip("/") + "/health"

    def supports_kv_projection(self):
        return bool(self.capabilities.get("kv_projection", False))

    def prepare_body(self, body, priority=0):
        payload = dict(body or {})
        if priority != 0 and self.supports_priority:
            payload["priority"] = int(priority)
        return payload


class VLLMBackendAdapter(BackendAdapter):

    __slots__ = ()

    def __init__(self, backend_url, capabilities=None, supports_priority=True):
        merged = dict(capabilities or {})
        merged.setdefault("streaming", True)
        merged.setdefault("priority", bool(supports_priority))
        merged.setdefault("kv_projection", False)
        super().__init__(
            backend_url,
            name="vllm",
            supports_priority=supports_priority,
            capabilities=merged,
        )


def build_backend_adapter(kind, backend_url, capabilities=None):
    adapter_kind = str(kind or "vllm").strip().lower()
    if adapter_kind in ("vllm", "openai"):
        supports_priority = bool((capabilities or {}).get("priority", True))
        return VLLMBackendAdapter(
            backend_url,
            capabilities=capabilities,
            supports_priority=supports_priority,
        )
    raise ValueError("unsupported backend adapter: %s" % adapter_kind)
