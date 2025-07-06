import asyncio, contextlib, random
from collections import deque
from dhanhq import MarketFeed as BaseMarketFeed
from utils import log_text
from helper import is_within_capture_window

class AsyncMarketFeed(BaseMarketFeed):
    def __init__(self, dhan_context, instruments, version="v2", max_buffer=100):
        super().__init__(dhan_context, instruments, version)
        self.queue   = asyncio.Queue()
        self.buffer  = deque(maxlen=max_buffer)
        self.running = False
        self._callback = None
        self._task     = None

    def start(self):
        self._task = asyncio.create_task(self._run_async())

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            log_text("Market feed cancelled.")

    async def get_data_async(self, timeout=0.5):
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def _keepalive(self):
        while self.running and self.ws:
            try:
                await self.ws.ping()
            except Exception:
                break
            await asyncio.sleep(8)

    async def _run_async(self):
        self.running = True
        reconnect_attempts = 0
        keep_task = None

        while self.running:

            try:
                await self.connect()
                log_text("Connected to Dhan market feed.")
                if not self.ws:
                    log_text("🔴 WebSocket not alive after connect — skipping retry loop.")
                    await asyncio.sleep(10)
                    continue

                reconnect_attempts = 0
                keep_task = asyncio.create_task(self._keepalive())

                while self.running:
                    try:
                        raw = await self.get_instrument_data()
                        parsed = raw
                        if parsed:
                            await self.queue.put(parsed)
                            self.buffer.append(parsed)
                            if self._callback:
                                self._callback(parsed)
                    except asyncio.CancelledError:
                        self.running = False
                        break
                    except Exception as inner:
                        log_text(f"Stream error: {inner} reconnecting …")
                        if keep_task and not keep_task.done():
                            keep_task.cancel()
                        try:
                            await self.ws.close()
                        except Exception:
                            pass
                        break

            except Exception as e:
                log_text(f"Connection failed: {e}")

            if keep_task:
                keep_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await keep_task
                keep_task = None

            reconnect_attempts += 1
            if reconnect_attempts > 50:
                log_text("Too many reconnects — shutting down feed.")
                break

            backoff = min(2 ** reconnect_attempts + random.uniform(0, 2), 30)
            await asyncio.sleep(backoff)
