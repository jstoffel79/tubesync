'''
    Detects sustained YouTube throttling and applies a temporary,
    instance-wide cooldown (meeb/tubesync#1529).

    Throttling tends to show up as a burst of the same handful of errors
    (429s, "Too Many Requests", the "Sign in to confirm you're not a bot"
    wall, etc) across many different videos/channels at once, rather than
    a single video failing. So: count throttle-signature errors in a
    rolling window; once enough land, verify against a known-good, cheap
    request before trusting it (a single flaky video shouldn't trigger a
    multi-day cooldown); if confirmed, stop hitting YouTube for a while
    and let things cool off.
'''


import json
import time
from pathlib import Path

from django.conf import settings

from common.logger import log


STATE_FILE = Path(settings.CONFIG_BASE_DIR) / 'throttle_state.json'

# how many throttle-signature errors...
ERROR_THRESHOLD = getattr(settings, 'THROTTLE_ERROR_THRESHOLD', 5)
# ...within this many seconds...
WINDOW_SECONDS = getattr(settings, 'THROTTLE_WINDOW_SECONDS', 30 * 60)
# ...before verifying and (if confirmed) cooling down for this many seconds.
# Default is the middle of the 48-72h range requested in the issue.
COOLDOWN_SECONDS = getattr(settings, 'THROTTLE_COOLDOWN_SECONDS', 60 * 60 * 60)
# A short, always-available public video used as the "did this actually
# get better" check. Overridable in settings for self-hosters who would
# rather point this at their own known-good source.
VERIFY_URL = getattr(
    settings, 'THROTTLE_VERIFY_URL',
    'https://www.youtube.com/watch?v=jNQXAC9IVRw',  # "Me at the zoo"
)

THROTTLE_SIGNATURES = (
    'HTTP Error 429',
    'Too Many Requests',
    'giving up after',
    "Sign in to confirm you're not a bot",
)


def _load():
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save(state):
    try:
        STATE_FILE.write_text(json.dumps(state))
    except OSError as e:
        log.warning(f'throttle: failed to persist state to {STATE_FILE}: {e}')


def is_throttle_error(msg):
    msg = str(msg)
    return any(sig in msg for sig in THROTTLE_SIGNATURES)


def in_cooldown():
    '''
        Returns (True, seconds_remaining) if a confirmed cooldown is
        active, otherwise (False, 0).
    '''
    state = _load()
    until = state.get('cooldown_until')
    if not until:
        return False, 0
    remaining = until - time.time()
    if remaining <= 0:
        return False, 0
    return True, remaining


def clear_cooldown():
    state = _load()
    state.pop('cooldown_until', None)
    state['hits'] = []
    _save(state)


def _verify_throttled():
    # local import: avoids a hard import-time dependency loop with youtube.py
    import yt_dlp
    from .youtube import get_yt_opts

    opts = get_yt_opts()
    opts.update({
        'skip_download': True,
        'simulate': True,
        'extract_flat': True,
        'check_formats': False,
    })
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            y.extract_info(VERIFY_URL, download=False)
    except yt_dlp.utils.DownloadError as e:
        return is_throttle_error(str(e))
    return False


def record_error(msg):
    '''
        Record a throttle-signature error. Once ERROR_THRESHOLD land inside
        WINDOW_SECONDS, verify with a single known-good request before
        starting a cooldown, so a single bad video doesn't pause the whole
        instance for days.
    '''
    if not is_throttle_error(msg):
        return
    now = time.time()
    state = _load()
    hits = [t for t in state.get('hits', []) if now - t < WINDOW_SECONDS]
    hits.append(now)
    state['hits'] = hits
    _save(state)
    if len(hits) < ERROR_THRESHOLD:
        return

    log.warning(
        f'throttle: {len(hits)} throttle-signature errors in the last '
        f'{WINDOW_SECONDS}s, verifying before starting a cooldown'
    )
    if _verify_throttled():
        state['cooldown_until'] = now + COOLDOWN_SECONDS
        state['hits'] = []
        _save(state)
        log.warning(
            f'throttle: confirmed, pausing downloads/indexing for '
            f'{COOLDOWN_SECONDS}s (until {time.ctime(now + COOLDOWN_SECONDS)})'
        )
    else:
        # false alarm: don't let the same burst re-trigger verification
        # on every subsequent error until the window rolls over
        state['hits'] = []
        _save(state)
        log.info('throttle: verification request succeeded, not starting a cooldown')
