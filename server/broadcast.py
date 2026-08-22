from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING

from aiohttp.web_ws import WebSocketResponse

from json_utils import json_dumps
from websocket_utils import ws_send_str_many

if TYPE_CHECKING:
    from bug.game_bug import GameBug
    from game import Game
    from pychess_global_app_state import PychessGlobalAppState
import logging

log = logging.getLogger(__name__)


async def broadcast_streams(app_state: PychessGlobalAppState) -> None:
    """Send live_streams to lobby"""
    live_streams = app_state.twitch.live_streams + app_state.youtube.live_streams
    response = {"type": "streams", "items": live_streams}
    log.debug("broadcasting streams to lobby: %s", response)
    await app_state.lobby.lobby_broadcast(response)


async def round_broadcast(
    game: Game | GameBug,
    response: Mapping[str, object],
    full: bool = False,
    channels: set[asyncio.Queue[str]] | None = None,
) -> None:
    # Snapshot live collections because awaits below let other tasks add/remove
    # spectators/channels while we're broadcasting.
    spectators = tuple(game.spectators)
    players = tuple(game.non_bot_players) if full else ()
    ch = tuple(channels) if channels is not None else ()

    if not spectators and not players and not ch:
        return

    # Encode the response ONCE and fan it out as a raw string. Previously each
    # recipient (potentially hundreds of spectators on a popular game) caused
    # its own independent msgspec encode of the identical payload, which is
    # pure wasted CPU on the single-threaded event loop. One encode here turns
    # an O(spectators) cost into O(1).
    # Guard against bad payloads: preserve the non-raising behaviour that
    # ws_send_json_many() had (it catches serialization errors internally).
    try:
        payload = json_dumps(response)
    except Exception:
        log.exception(
            "round_broadcast: failed to serialize response for game %s: %r", game.id, response
        )
        return

    # Collect all WebSocket objects from spectators and players into one flat
    # list and call ws_send_str_many once.  The previous code awaited each
    # user's send_game_message_str() in a sequential loop, adding a context
    # switch per user even though the payload was identical.  A single
    # asyncio.gather call inside ws_send_str_many is strictly faster.
    game_id = game.id
    all_ws: list[WebSocketResponse] = []
    for user in (*spectators, *players):
        ws_set = user.game_sockets.get(game_id)
        if ws_set:
            all_ws.extend(ws_set)

    if all_ws:
        await ws_send_str_many(all_ws, payload)

    for queue in ch:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            log.warning(
                "Dropping slow /api/ongoing subscriber with %d queued messages",
                queue.qsize(),
            )
            queue.shutdown(immediate=True)
            if channels is not None:
                channels.discard(queue)
        except asyncio.QueueShutDown:
            if channels is not None:
                channels.discard(queue)
