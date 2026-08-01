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


import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from django.conf import settings

from common.logger import log


STATE_FILE = Path(settings.CONFIG_BASE_DIR) / 'throttle_state.json'
LOCK_FILE = STATE_FILE.with_suffix('.lock')

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
    # Write to a temp file and rename over the real path: os.replace() is
    # atomic on the same filesystem, so a reader never sees a truncated
    # or partially-written file, even if this process is killed mid-write.
    tmp_path = STATE_FILE.with_suffix('.json.tmp')
    try:
        tmp_path.write_text(json.dumps(state))
        os.replace(tmp_path, STATE_FILE)
    except OSError as e:
        log.warning(f'throttle: failed to persist state to {STATE_FILE}: {e}')


@contextmanager
def _locked_state():
    '''
        Read-modify-write the state file under an exclusive file lock, so
        concurrent Huey workers calling record_error()/clear_cooldown() at
        the same time can't lose one another's updates.
    '''
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, 'a+') as lock_fp:
        fcntl.flock(lock_fp, fcntl.LOCK_EX)
        try:
            state = _load()
            yield state
            _save(state)
        finally:
            fcntl.flock(lock_fp, fcntl.LOCK_UN)


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
    with _locked_state() as state:
        state.pop('cooldown_until', None)
        state['hits'] = []


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
    # Increment the hit counter under lock (fast), then release the lock
    # before doing the slow network verification below -- otherwise every
    # other worker's record_error()/clear_cooldown() call would block for
    # the duration of a yt-dlp request.
    with _locked_state() as state:
        hits = [t for t in state.get('hits', []) if now - t < WINDOW_SECONDS]
        hits.append(now)
        state['hits'] = hits
        hit_count = len(hits)
    if hit_count < ERROR_THRESHOLD:
        return

    log.warning(
        f'throttle: {hit_count} throttle-signature errors in the last '
        f'{WINDOW_SECONDS}s, verifying before starting a cooldown'
    )
    confirmed = _verify_throttled()
    with _locked_state() as state:
        # always clear hits here: either we're starting a cooldown, or this
        # was a false alarm and shouldn't re-trigger verification on every
        # subsequent error until the window rolls over
        state['hits'] = []
        if confirmed:
            state['cooldown_until'] = now + COOLDOWN_SECONDS
    if confirmed:
        log.warning(
            f'throttle: confirmed, pausing downloads/indexing for '
            f'{COOLDOWN_SECONDS}s (until {time.ctime(now + COOLDOWN_SECONDS)})'
        )
    else:
        log.info('throttle: verification request succeeded, not starting a cooldown')
