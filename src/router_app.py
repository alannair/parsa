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

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from .controller import OverloadedError
from .kv_projection import fetch_kv_projection_snapshot


logger = logging.getLogger(__name__)
AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60


def _http_exception(status_code, detail):
    from fastapi import HTTPException

    raise HTTPException(status_code=status_code, detail=detail)


def validate_json_request(request):
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        _http_exception(415, "content-type must be application/json")


class EnergyAwareRouter:

    __slots__ = ("controller",)

    def __init__(self, controller):
        self.controller = controller

    def _forward_headers(self, request):
        headers = {}
        authorization = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")
        if authorization:
            headers["Authorization"] = authorization
        if api_key:
            headers["X-API-Key"] = api_key
        return headers

    def _backend_metadata_headers(self, choice):
        headers = {}
        if not getattr(choice, "request_id", ""):
            return headers
        headers["X-Energy-Aware-Request-Id"] = str(choice.request_id)
        if self.controller.specialization_gap is not None:
            return headers
        headers["X-Energy-Aware-L-In"] = str(choice.routing_l_in)
        headers["X-Energy-Aware-L-Out-Predicted-Raw"] = str(choice.routing_l_out_predicted_raw)
        headers["X-Energy-Aware-L-Out-Effective"] = str(choice.routing_l_out_effective)
        headers["X-Energy-Aware-SLO-Class"] = str(getattr(choice, "slo_class", "default"))
        headers["X-Energy-Aware-Request-Value"] = str(getattr(choice, "request_value", 1))
        if self.controller.soft_admission_enabled or self.controller.queue_enabled:
            headers["X-Energy-Aware-Admission-Tier"] = str(getattr(choice, "admission_tier", ""))
            headers["X-Energy-Aware-Admission-Reason"] = str(getattr(choice, "admission_reason", ""))
            headers["X-Energy-Aware-Route-Attempts"] = str(getattr(choice, "route_attempts", 0))
            headers["X-Energy-Aware-Best-Effort"] = "1" if getattr(choice, "admission_tier", "") == "best_effort" else "0"
        if self.controller.feedback_enabled or self.controller.kv_project_enabled:
            if getattr(choice, "requested_max_latency_s", None) is not None:
                headers["X-Energy-Aware-Max-Latency-S"] = str(choice.requested_max_latency_s)
            if getattr(choice, "requested_ttft_s", None) is not None:
                headers["X-Energy-Aware-Target-TTFT-S"] = str(choice.requested_ttft_s)
            if getattr(choice, "requested_tpot_s", None) is not None:
                headers["X-Energy-Aware-Target-TPOT-S"] = str(choice.requested_tpot_s)
        if self.controller.kv_project_enabled:
            quantiles = dict(getattr(choice, "predicted_output_quantiles", {}) or {})
            if "p50" in quantiles:
                headers["X-Energy-Aware-Predicted-Output-Tokens-P50"] = str(quantiles["p50"])
            if "p90" in quantiles:
                headers["X-Energy-Aware-Predicted-Output-Tokens-P90"] = str(quantiles["p90"])
            if "p95" in quantiles:
                headers["X-Energy-Aware-Predicted-Output-Tokens-P95"] = str(quantiles["p95"])
        if self.controller.depon_enabled:
            if getattr(choice, "workflow_id", ""):
                headers["X-Energy-Aware-Workflow-Id"] = str(choice.workflow_id)
            if getattr(choice, "stage_id", None) is not None:
                headers["X-Energy-Aware-Stage-Id"] = str(choice.stage_id)
            headers["X-Energy-Aware-Critical-Path"] = "1" if getattr(choice, "critical_path", False) else "0"
            headers["X-Energy-Aware-Request-Kind"] = str(getattr(choice, "request_kind", "default"))
        return headers

    def _routing_headers(self, choice):
        headers = {
            "X-Energy-Aware-Preset-Id": choice.preset.preset_id,
            "X-Energy-Aware-Rescue": "1" if choice.used_rescue else "0",
            "X-Energy-Aware-Routing-Policy": self.controller.routing_policy,
            "X-Energy-Aware-Random-LB": "1" if self.controller.random_lb else "0",
            "X-Energy-Aware-L-In": str(choice.routing_l_in),
            "X-Energy-Aware-L-Out-Predicted-Raw": str(choice.routing_l_out_predicted_raw),
            "X-Energy-Aware-L-Out-Effective": str(choice.routing_l_out_effective),
        }
        if self.controller.specialization_gap is not None:
            # No predictor ran; do not advertise the Choice's neutral placeholders as predictions.
            for key in ("X-Energy-Aware-L-In", "X-Energy-Aware-L-Out-Predicted-Raw", "X-Energy-Aware-L-Out-Effective"):
                headers.pop(key, None)
        if self.controller.soft_admission_enabled or self.controller.queue_enabled:
            headers.update(
                {
                    "X-Energy-Aware-Admission-Tier": str(getattr(choice, "admission_tier", "")),
                    "X-Energy-Aware-Admission-Reason": str(getattr(choice, "admission_reason", "")),
                    "X-Energy-Aware-Route-Attempts": str(getattr(choice, "route_attempts", 0)),
                    "X-Energy-Aware-SLO-Class": str(getattr(choice, "slo_class", "default")),
                    "X-Energy-Aware-Best-Effort": "1" if getattr(choice, "admission_tier", "") == "best_effort" else "0",
                    "X-Energy-Aware-Request-Id": str(getattr(choice, "request_id", "")),
                }
            )
        if self.controller.feedback_enabled:
            headers.update(
                {
                    "X-Energy-Aware-Target-TTFT-S": "" if getattr(choice, "requested_ttft_s", None) is None else str(choice.requested_ttft_s),
                    "X-Energy-Aware-Target-TPOT-S": "" if getattr(choice, "requested_tpot_s", None) is None else str(choice.requested_tpot_s),
                    "X-Energy-Aware-Request-Value": str(getattr(choice, "request_value", 1)),
                }
            )
        if self.controller.depon_enabled:
            headers.update(
                {
                    "X-Energy-Aware-Workflow-Id": str(getattr(choice, "workflow_id", "")),
                    "X-Energy-Aware-Stage-Id": "" if getattr(choice, "stage_id", None) is None else str(choice.stage_id),
                    "X-Energy-Aware-Critical-Path": "1" if getattr(choice, "critical_path", False) else "0",
                }
            )
        return headers

    def _response_headers(self, routing_headers, content_type=None):
        headers = dict(routing_headers)
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    async def _apply_routing(self, body):
        request_id = uuid.uuid4().hex
        context = self.controller._begin_observation(request_id)
        try:
            return await self._prepare_routing(body, request_id, context)
        except asyncio.CancelledError:
            self.controller._finish_observation(context, "cancelled")
            raise
        except Exception as exc:
            self.controller._finish_observation(context, "rejected" if isinstance(exc, OverloadedError) else "failed")
            raise

    async def _prepare_routing(self, body, request_id, context):
        if self.controller.dispatch_loop_enabled:
            choice = await self.controller.choose_or_wait(body, request_id=request_id, _observation_context=context)
        else:
            choice = self.controller.choose(body, request_id=request_id, _observation_context=context)
        priority = self.controller.suggested_priority(choice, body)
        if self.controller.specialization_gap is not None:
            body = dict(body)
            metadata = dict(body.get("energy_aware") or {})
            metadata.pop("specialization_gap", None)
            if metadata:
                body["energy_aware"] = metadata
            else:
                body.pop("energy_aware", None)
        payload = choice.preset.adapter.prepare_body(body, priority=priority)
        return request_id, choice, payload

    def _finish_proxy(self, request_id, state, choice, started_s, status_code, *, outcome=None):
        if outcome is not None:
            self.controller._finish_observation(choice._observation_context, outcome)
        elapsed_s = max(0.0, time.time() - started_s)
        state.on_finish(elapsed_s)
        self.controller.metrics.record_completion(choice, elapsed_s, status_code=status_code)
        self.controller.on_request_finish(request_id, choice, elapsed_s, status_code=status_code)

    async def _proxy_non_stream(self, session, url, payload, forward_headers, request_id, choice, state):
        from fastapi.responses import Response

        started_s = time.time()
        status_code = None
        outcome = "failed"
        try:
            async with session.post(url, json=payload, headers=forward_headers) as upstream:
                status_code = upstream.status
                body = await upstream.read()
                if 200 <= status_code < 300:
                    outcome = "completed"
                if status_code == 422:
                    logger.warning("upstream 422: %s", body[:800].decode("utf-8", "replace"))
                if 200 <= status_code < 300 or 400 <= status_code < 500:
                    content_type = upstream.headers.get("Content-Type")
                    return Response(
                        content=body,
                        status_code=status_code,
                        headers=self._response_headers(self._routing_headers(choice), content_type),
                    )
                _http_exception(status_code, body.decode("utf-8", "replace"))
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception:
            outcome = "failed"
            raise
        finally:
            self._finish_proxy(request_id, state, choice, started_s, status_code, outcome=outcome)

    async def _proxy_stream(self, session, url, payload, forward_headers, request_id, choice, state):
        from fastapi.responses import Response, StreamingResponse

        started_s = time.time()
        status_code = None
        finished = False
        upstream = None
        outcome = "failed"

        def finish_once():
            nonlocal finished
            if finished:
                return
            finished = True
            self._finish_proxy(request_id, state, choice, started_s, status_code, outcome=outcome)

        try:
            upstream = await session.post(url, json=payload, headers=forward_headers)
            status_code = upstream.status
            if 400 <= status_code < 500:
                body = await upstream.read()
                await upstream.release()
                finish_once()
                return Response(
                    content=body,
                    status_code=status_code,
                    headers=self._response_headers(self._routing_headers(choice), upstream.headers.get("Content-Type")),
                )
            if not 200 <= status_code < 300:
                body = await upstream.read()
                await upstream.release()
                finish_once()
                _http_exception(status_code, body.decode("utf-8", "replace"))

            async def iterator():
                nonlocal outcome
                try:
                    async for chunk in upstream.content.iter_chunked(1024):
                        if self.controller.feedback_enabled and chunk:
                            self.controller.feedback_model.observe_stream_chunk(request_id)
                        yield chunk
                    outcome = "completed"
                except (asyncio.CancelledError, GeneratorExit):
                    outcome = "cancelled"
                    raise
                finally:
                    try:
                        await upstream.release()
                    except Exception:
                        pass
                    finish_once()

            response_type = StreamingResponse
            if self.controller.shadow is not None:
                controller = self.controller
                observation_context = choice._observation_context

                class ShadowStreamingResponse(StreamingResponse):
                    async def __call__(self, scope, receive, send):
                        # A disconnect can return before the iterator ever starts.
                        terminal_status = "cancelled"
                        try:
                            await super().__call__(scope, receive, send)
                            if outcome == "completed":
                                terminal_status = "completed"
                        except Exception:
                            terminal_status = "failed"
                            raise
                        finally:
                            # Observation only; iterator-owned proxy cleanup stays there.
                            controller._finish_observation(observation_context, terminal_status)

                response_type = ShadowStreamingResponse
            return response_type(
                iterator(),
                status_code=status_code,
                headers=self._routing_headers(choice),
                media_type="text/event-stream",
            )
        except asyncio.CancelledError:
            self.controller._finish_observation(choice._observation_context, "cancelled")
            raise
        except Exception:
            if upstream is not None:
                try:
                    await upstream.release()
                except Exception:
                    pass
            finish_once()
            raise

    async def handle_completion(self, request, path):
        validate_json_request(request)
        body = await request.json()
        request_id, choice, payload = await self._apply_routing(body)
        try:
            return await self._handle_routed_completion(request, path, body, request_id, choice, payload)
        except asyncio.CancelledError:
            self.controller._finish_observation(choice._observation_context, "cancelled")
            raise
        except Exception:
            self.controller._finish_observation(choice._observation_context, "failed")
            raise

    async def _handle_routed_completion(self, request, path, body, request_id, choice, payload):
        state = self.controller.queue_manager.get(choice.preset.preset_id)
        self.controller.on_request_dispatch(request_id, choice)
        state.on_start()
        session = request.app.state.upstream_session
        url = choice.preset.adapter.request_url(path)
        forward_headers = self._forward_headers(request)
        forward_headers.update(self._backend_metadata_headers(choice))
        if bool(body.get("stream")):
            return await self._proxy_stream(session, url, payload, forward_headers, request_id, choice, state)
        return await self._proxy_non_stream(session, url, payload, forward_headers, request_id, choice, state)


@asynccontextmanager
async def shadow_lifespan(controller, *, headroom_producer=None, sleep=asyncio.sleep):
    """The exact shadow task lifecycle used by create_app; no web imports needed."""
    if controller.shadow is None:
        yield None
        return
    producer = controller.collect_shadow_headroom if headroom_producer is None else headroom_producer
    task = asyncio.create_task(controller.shadow.run(producer, sleep=sleep))
    try:
        yield task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def create_app(controller, metrics_calibration=None):
    import aiohttp
    from aiohttp import ClientConnectorError
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from .metrics_collector import calibration_loop

    router = EnergyAwareRouter(controller)

    async def probe_kv_projection(session):
        if not controller.kv_project_enabled:
            return
        timeout_s = max(0.01, float(controller.config.kv_project.get("startup_probe_timeout_ms", 500) or 500) / 1000.0)
        strict = bool(controller.config.kv_project.get("strict", True))
        for preset in controller.registry.snapshot():
            if not preset.adapter.supports_kv_projection():
                message = "kv projection required but backend capability disabled for %s" % preset.preset_id
                if strict:
                    raise RuntimeError(message)
                logger.warning(message)
                continue
            try:
                snapshot = await fetch_kv_projection_snapshot(
                    session,
                    preset.adapter.admin_kv_projection_url(),
                    timeout_s=timeout_s,
                    expected_engine_id=preset.preset_id,
                )
                controller.kv_projection_store.update(preset.preset_id, snapshot)
            except Exception as exc:
                if strict:
                    raise
                controller.kv_projection_store.record_error(preset.preset_id, str(exc))
                logger.warning("kv projection probe failed for %s: %s", preset.preset_id, exc)

    async def kv_projection_loop(session):
        if not controller.kv_project_enabled:
            return
        interval_s = max(0.01, float(controller.config.kv_project.get("poll_interval_ms", 50) or 50) / 1000.0)
        timeout_s = max(0.01, float(controller.config.kv_project.get("request_timeout_ms", 50) or 50) / 1000.0)
        while True:
            for preset in controller.registry.snapshot():
                if not preset.adapter.supports_kv_projection():
                    controller.kv_projection_store.record_error(preset.preset_id, "kv_projection_capability_disabled")
                    continue
                try:
                    snapshot = await fetch_kv_projection_snapshot(
                        session,
                        preset.adapter.admin_kv_projection_url(),
                        timeout_s=timeout_s,
                        expected_engine_id=preset.preset_id,
                    )
                    controller.kv_projection_store.update(preset.preset_id, snapshot)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    controller.kv_projection_store.record_error(preset.preset_id, str(exc))
            await asyncio.sleep(interval_s)

    async def lifespan(app):
        timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT_SECONDS)
        task = None
        planner_task = None
        kv_project_task = None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            app.state.upstream_session = session
            if metrics_calibration:
                metrics_urls, scales, cfg, interval_s = metrics_calibration
                task = asyncio.create_task(calibration_loop(metrics_urls, scales, cfg, interval_s=interval_s))
                app.state.calibration_task = task
            if controller.kv_project_enabled:
                await probe_kv_projection(session)
                kv_project_task = asyncio.create_task(kv_projection_loop(session))
                app.state.kv_projection_task = kv_project_task
            if controller.dispatch_loop_enabled:
                async def planner_loop():
                    interval_s = max(0.01, float(controller.config.queue.get("tick_ms", 50) or 50) / 1000.0)
                    while True:
                        await controller.plan_pending_dispatches()
                        await asyncio.sleep(interval_s)

                planner_task = asyncio.create_task(planner_loop())
                app.state.planner_task = planner_task
            async with shadow_lifespan(controller) as shadow_task:
                if shadow_task is not None:
                    app.state.shadow_task = shadow_task
                try:
                    yield
                finally:
                    if kv_project_task is not None:
                        kv_project_task.cancel()
                        await asyncio.gather(kv_project_task, return_exceptions=True)
                    if planner_task is not None:
                        planner_task.cancel()
                        await asyncio.gather(planner_task, return_exceptions=True)
                    if task is not None:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

    app = FastAPI(title="Energy-aware vLLM router", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ClientConnectorError)
    async def upstream_connect_error(request: Request, exc):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(OverloadedError)
    async def overload_error(request: Request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await router.handle_completion(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await router.handle_completion(request, "/v1/completions")

    return app
