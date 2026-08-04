'''
    Start, stop and manage scheduled tasks. These are generally triggered by Django
    signals (see signals.py).
'''


import os
import random
import requests
import time
import uuid
from collections import deque as queue
from io import BytesIO
from pathlib import Path
from datetime import timedelta
from shutil import copyfile, rmtree
from django import db
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django_huey import lock_task as huey_lock_task, task as huey_task # noqa
from django_huey import db_periodic_task, db_task, signal as huey_signal
from huey import crontab as huey_crontab, signals as huey_signals
from huey.exceptions import TaskLockedException
from common.huey import CancelExecution, dynamic_retry, register_huey_signals, LockPool
from common.logger import log
from common.models import TaskHistory
from common.errors import (
    DownloadFailedException,
    NoFormatException, NoMediaException, NoThumbnailException,
    QuerySetEmptyError,
)
from common.utils import (  django_queryset_generator as qs_gen,
                            remove_enclosed, seconds_to_timestr, )
from .choices import Val, IndexSchedule, TaskQueue
from .models import Source, Media, MediaServer, Metadata
from .throttle import in_cooldown as throttle_in_cooldown, record_error as throttle_record_error
from .utils import get_remote_image, resize_image_to_height, filter_response
from .youtube import YouTubeError

atomic = db.transaction.atomic
db_vendor = db.connection.vendor
# register_huey_signals() is called at the bottom of this file instead of
# here, passing clear_stale_media_locks as the worker on_startup hook --
# it needs to be defined first, and this needs to run in worker processes
# specifically (see register_huey_signals()'s own docstring).


def get_task_map():
    TASK_MAP = {
        'index_source': Source,
        'download_media_image': Media,
        'download_media_file': Media,
        'download_media_metadata': Media,
        'save_all_media_for_source': Source,
        'rename_all_media_for_source': Source,
        'refresh_formats': Media,
    }
    return { f"sync.tasks.{k}": v for k,v in TASK_MAP.items() }

def _task_model_and_uuid(task, /, *, task_map=None, model_url_map=None):
    '''
        Resolves a task down to (model, url, instance_uuid) without hitting
        the database. Shared by map_task_to_instance() and the batched
        map_tasks_to_instances().
    '''
    TASK_MAP = task_map if task_map is not None else get_task_map()
    MODEL_URL_MAP = model_url_map if model_url_map is not None else {
        Source: 'sync:source',
        Media: 'sync:media-item',
    }
    model = TASK_MAP.get(task.name, None)
    if not model:
        return None, None, None
    url = MODEL_URL_MAP.get(model, None)
    if not url:
        return None, None, None
    task_args = task.task_params
    if len(task_args) != 2:
        return None, None, None
    args, kwargs = task_args
    if len(args) == 0:
        return None, None, None
    try:
        instance_uuid = uuid.UUID(args[0])
    except (TypeError, ValueError, AttributeError):
        return None, None, None
    return model, url, instance_uuid


def map_task_to_instance(task):
    '''
        Reverse-maps a scheduled backgrond task to an instance. Requires the task name
        to be a known task function and the first argument to be a UUID. This is used
        because UUID's are incompatible with background_task's "creator" feature.

        For rendering many tasks at once (e.g. the tasks list view), prefer
        map_tasks_to_instances() instead: this does one .get() query per
        call, which turns into an N+1 query storm over a large task list.
    '''
    model, url, instance_uuid = _task_model_and_uuid(task)
    if not model:
        return None, None
    try:
        instance = model.objects.get(pk=instance_uuid)
        return instance, url
    except model.DoesNotExist:
        return None, None


def map_tasks_to_instances(tasks):
    '''
        Batched equivalent of calling map_task_to_instance() once per task.
        Looks up each referenced model with a single "pk__in=" query instead
        of one query per task, then returns a dict of
        {task.id: (instance_or_None, url_or_None)}.
    '''
    TASK_MAP = get_task_map()
    MODEL_URL_MAP = {
        Source: 'sync:source',
        Media: 'sync:media-item',
    }
    # task.id -> (model, url, instance_uuid)
    resolved = {
        task.id: _task_model_and_uuid(task, task_map=TASK_MAP, model_url_map=MODEL_URL_MAP)
        for task in tasks
    }
    # collect the set of pks to fetch per model
    pks_by_model = {}
    for model, _url, instance_uuid in resolved.values():
        if model is None:
            continue
        pks_by_model.setdefault(model, set()).add(instance_uuid)
    # one query per model instead of one query per task
    instances_by_model = {
        model: {obj.pk: obj for obj in model.objects.filter(pk__in=pks)}
        for model, pks in pks_by_model.items()
    }
    result = {}
    for task_id, (model, url, instance_uuid) in resolved.items():
        if model is None:
            result[task_id] = (None, None)
            continue
        instance = instances_by_model.get(model, {}).get(instance_uuid)
        result[task_id] = (instance, url) if instance else (None, None)
    return result


def get_error_message(task):
    '''
        Extract an error message from a failed task. This is the last line of the
        last_error field with the method name removed.
    '''
    if not task.has_error():
        return ''
    stacktrace_lines = task.last_error.strip().split('\n')
    if len(stacktrace_lines) == 0:
        return ''
    error_message = stacktrace_lines[-1].strip()
    if ':' not in error_message:
        return ''
    return error_message.split(':', 1)[1].strip()


def update_task_status(task, status):
    if not task:
        return False
    if not hasattr(task, '_verbose_name'):
        task._verbose_name = remove_enclosed(
            task.verbose_name, '[', ']', ' ',
        )
    if status is None:
        task.verbose_name = task._verbose_name
    else:
        task.verbose_name = f'[{status}] {task._verbose_name}'
    try:
        task.save(update_fields={'verbose_name'})
    except db.DatabaseError as e:
        if 'Save with update_fields did not affect any rows.' == str(e):
            pass
        else:
            raise
    return True


def get_source_completed_tasks(source_id, only_errors=False):
    '''
        Returns a queryset of TaskHistory objects for a source by source ID.
    '''
    qs = get_model_tasks(source_id)
    if only_errors:
        qs = qs.filter(failed_at__isnull=False)
    return qs.order_by('-failed_at')


def get_model_tasks(model_pk, /, name=None, qs=None):
    if qs is None:
        qs = TaskHistory.objects.all()
    if name is not None:
        qs = qs.filter(name__endswith=name)
    #return qs.filter(task_params__0__0=model_pk)
    return qs.filter(task_params__istartswith=f'[["{model_pk}"')

def get_running_tasks(arg_dt=None, /):
    max_run_time = getattr(settings, 'MAX_RUN_TIME', 3600)
    return TaskHistory.objects.running(
        now=arg_dt,
        within=timezone.timedelta(seconds=max_run_time),
    )


def get_queue_status():
    '''
        Live depth of each Huey queue: how many tasks are ready to run right
        now (pending) vs scheduled for a future eta. Used by the Tasks page
        to show what's actually queued, distinct from TaskHistory (which is
        historical bookkeeping, not the live queue state).
    '''
    from django_huey import DJANGO_HUEY, get_queue
    queues = []
    for qn in DJANGO_HUEY.get('queues', dict()):
        try:
            q = get_queue(qn)
            queues.append(dict(
                name=qn,
                pending=q.pending_count(),
                scheduled=q.scheduled_count(),
            ))
        except Exception as e:
            log.warning(f'get_queue_status: could not read queue {qn}: {e}')
    return queues


def get_task_breakdown(limit=8):
    '''
        Per-task-name breakdown of currently pending (enqueued, not yet
        started) work, for the Tasks page. get_queue_status() alone only
        shows aggregate depth per huey queue, which hides *what* is
        actually queued -- this matters because several task types share
        the same 'limited' single-worker queue, and one type (e.g.
        refresh_formats, proactively scheduled for every not-yet-
        downloaded video after indexing) can vastly outnumber and crowd
        out another (e.g. download_media_file, the actual video
        downloads) for a very long time without this being visible
        anywhere else on the page.
    '''
    from django.db.models import Count
    rows = (
        TaskHistory.objects
        .filter(start_at__isnull=True, failed_at__isnull=True)
        .values('name')
        .annotate(count=Count('id'))
        .order_by('-count')[:limit]
    )
    return [
        dict(name=row['name'].rsplit('.', 1)[-1], count=row['count'])
        for row in rows
    ]


def get_cgroup_status():
    '''
        Reads this container's own cgroup v2 accounting -- the same limits
        Kubernetes actually enforces -- rather than generic host stats, so
        the Tasks page reflects the real ceiling this pod is running under.
        Returns an empty dict (rather than raising) on cgroup v1 hosts or
        anywhere these files aren't present, so this is safe to call
        unconditionally.
    '''
    status = {}
    try:
        stat_text = Path('/sys/fs/cgroup/cpu.stat').read_text()
        stats = dict(
            line.split(' ', 1) for line in stat_text.splitlines() if ' ' in line
        )
        nr_periods = int(stats.get('nr_periods', 0) or 0)
        nr_throttled = int(stats.get('nr_throttled', 0) or 0)
        if nr_periods:
            status['cpu_throttled_pct'] = round(100 * nr_throttled / nr_periods, 1)
    except (OSError, ValueError):
        pass
    try:
        mem_current = int(Path('/sys/fs/cgroup/memory.current').read_text().strip())
        mem_max_raw = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        status['memory_current_mb'] = round(mem_current / (1024 * 1024), 1)
        if mem_max_raw != 'max':
            mem_max = int(mem_max_raw)
            status['memory_max_mb'] = round(mem_max / (1024 * 1024), 1)
            status['memory_pct'] = round(100 * mem_current / mem_max, 1)
    except (OSError, ValueError):
        pass
    return status


def get_ffmpeg_status():
    '''
        Live view of any ffmpeg/ffprobe processes actually running in this
        container right now, with accumulated CPU time. The Running task
        list shows which video is being worked on and its last reported
        progress label, but not whether ffmpeg itself is genuinely making
        progress or has hung -- this is exactly how the earlier stuck
        keyframe/remux postprocessing incident was diagnosed: by hand,
        checking /proc for a live ffmpeg process and how much CPU time
        it had accumulated. Surfacing it here means that check no longer
        requires a shell into the pod.
    '''
    processes = []
    try:
        clock_ticks = os.sysconf('SC_CLK_TCK')
    except (ValueError, OSError):
        clock_ticks = 100
    try:
        proc_entries = list(Path('/proc').iterdir())
    except OSError:
        return processes
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / 'comm').read_text().strip()
        except OSError:
            continue
        if comm not in ('ffmpeg', 'ffprobe'):
            continue
        cpu_seconds = None
        try:
            stat_fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            cpu_seconds = round((int(stat_fields[11]) + int(stat_fields[12])) / clock_ticks, 1)
        except (OSError, IndexError, ValueError):
            pass
        target = None
        try:
            cmdline = [
                c for c in (entry / 'cmdline').read_bytes().decode(errors='replace').split('\x00')
                if c
            ]
            if cmdline:
                # last arg is almost always the output path/filename for
                # how tubesync invokes ffmpeg -- good enough for an
                # at-a-glance "what is this working on"
                target = cmdline[-1]
        except OSError:
            pass
        processes.append(dict(
            pid=entry.name,
            comm=comm,
            cpu_seconds=cpu_seconds,
            target=target,
        ))
    return processes


def get_download_progress():
    '''
        Whether a download_media_file task is genuinely in progress right
        now, regardless of which phase it's in. get_ffmpeg_status() alone
        only catches the brief remux/keyframe-cut postprocessing window --
        most of a download's wall-clock time is spent in yt-dlp's own
        network fetch, with no ffmpeg process running at all, so "no
        ffmpeg process" on its own reads as "nothing is happening" even
        during a perfectly normal download. Reuses
        get_genuinely_running_uuids() so an orphaned/stale lock (see
        clear_stale_media_locks()) doesn't get reported as an active
        download.
    '''
    lock_stale_after = getattr(settings, 'LOCK_STALE_AFTER_SECONDS', 3 * 60 * 60)
    running_uuids = get_genuinely_running_uuids(lock_stale_after)
    qs = TaskHistory.objects.running(within=lock_stale_after).filter(
        name='sync.tasks.download_media_file',
    ).order_by('-start_at')
    for task in qs:
        try:
            media_id = str(task.task_params[0][0])
        except (IndexError, TypeError, KeyError):
            continue
        if media_id not in running_uuids:
            continue
        try:
            media = Media.objects.get(pk=media_id)
        except Media.DoesNotExist:
            continue
        elapsed = None
        if task.start_at:
            elapsed = int((timezone.now() - task.start_at).total_seconds())
        result = dict(
            active=True,
            media_key=media.key,
            media_name=media.name,
            elapsed_seconds=elapsed,
        )
        result.update(_get_download_size_and_rate(media))
        return result
    return dict(active=False)


_download_progress_sample_cache = {}


def _get_download_size_and_rate(media):
    '''
        Current bytes-on-disk and estimated total size for the in-progress
        download, plus a rate computed from the previous sample -- there's
        no separate "progress" tracking mechanism (yt-dlp runs quiet, and
        the video download itself may be delegated to aria2c as a
        subprocess, see _aria2c_opts()), so this reads the actual
        partially-written file(s) in YOUTUBE_DL_TEMPDIR directly rather
        than trying to hook into either downloader's internals.

        No exact file size is cached in Media.formats (yt-dlp doesn't
        always report one for YouTube's adaptive formats), so total size
        is *estimated* from cached bitrate * media.duration -- shown as
        an approximation, same as yt-dlp's own CLI does in this situation.

        Uses the *largest* matching file, not a sum -- verified live that
        multiple files legitimately coexist for one download at once
        (separate video/audio streams while downloading, then the
        original(s) *and* a growing merged .temp.mkv once ffmpeg starts
        muxing them), and summing them double-counts data that exists in
        both the source stream(s) and the merge output, wildly
        overstating progress (measured live: reported ~29GB for a video
        whose real total was ~6.5GB during the merge phase).

        Rate is computed from a single prior sample stored in a
        module-level cache keyed by media UUID (cleared once the media
        stops appearing as actively downloading), rather than blocking to
        take two samples -- this fits the existing ~30s Tasks-page
        refresh cadence instead of adding latency to page loads.
    '''
    tempdir = getattr(settings, 'YOUTUBE_DL_TEMPDIR', None)
    if not tempdir:
        return {}
    marker = f'[{media.key}]'
    try:
        candidates = [
            entry for entry in Path(tempdir).iterdir()
            if marker in entry.name
        ]
        # download_media() gives each download its own scratch dir named
        # '.yt_dlp-{key}-*' (see temp_dir_prefix in youtube.py) -- partial
        # fragment/merge files often live in there instead of directly in
        # tempdir, so a plain iterdir() alone misses an in-progress
        # download entirely. Only descend into the matching one, not a
        # full recursive scan of every download's scratch dir.
        for subdir in Path(tempdir).glob(f'.yt_dlp-{media.key}-*'):
            if subdir.is_dir():
                try:
                    candidates.extend(subdir.iterdir())
                except OSError:
                    continue
    except OSError:
        return {}
    if not candidates:
        return {}
    current_bytes = 0
    for path in candidates:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        current_bytes = max(current_bytes, size)

    total_bytes_estimate = None
    try:
        format_str = media.get_format_str()
    except Exception:
        format_str = None
    if format_str and media.duration:
        wanted_ids = set(str(format_str).split('+'))
        kbps_total = 0
        for fmt in media.iter_formats():
            if fmt.get('id') in wanted_ids:
                kbps_total += (fmt.get('vbr') or 0) + (fmt.get('abr') or 0)
        if kbps_total:
            total_bytes_estimate = int(kbps_total * 1000 / 8 * media.duration)

    rate_bytes_per_sec = None
    now = timezone.now()
    cache_key = str(media.pk)
    previous = _download_progress_sample_cache.get(cache_key)
    if previous is not None:
        prev_bytes, prev_time = previous
        elapsed = (now - prev_time).total_seconds()
        if elapsed > 0 and current_bytes >= prev_bytes:
            rate_bytes_per_sec = (current_bytes - prev_bytes) / elapsed
    _download_progress_sample_cache[cache_key] = (current_bytes, now)
    # Bound growth: only one download can genuinely be "active" per the
    # caller's own query, but stale entries from finished/failed
    # downloads would otherwise accumulate here forever.
    if len(_download_progress_sample_cache) > 20:
        oldest_key = min(
            _download_progress_sample_cache,
            key=lambda k: _download_progress_sample_cache[k][1],
        )
        if oldest_key != cache_key:
            _download_progress_sample_cache.pop(oldest_key, None)

    return dict(
        current_mb=round(current_bytes / (1024 * 1024), 1),
        total_mb_estimate=(
            round(total_bytes_estimate / (1024 * 1024), 1)
            if total_bytes_estimate else None
        ),
        rate_mb_per_sec=(
            round(rate_bytes_per_sec / (1024 * 1024), 2)
            if rate_bytes_per_sec is not None else None
        ),
    )


def get_running_tasks_by_name(arg_str, instance_id, /):
    name = arg_str
    if '.' not in name:
        name = f'sync.tasks.{name}'
    tqs = get_model_tasks(instance_id, qs=get_running_tasks())
    return tqs.filter(name=name)

def get_media_download_task(media_id):
    tqs = get_running_tasks_by_name('download_media_file', media_id)
    return tqs.first() or False
    
def get_media_thumbnail_task(media_id):
    tqs = get_running_tasks_by_name('download_media_image', media_id)
    return tqs.first() or False

def get_source_index_task(source_id):
    tqs = get_running_tasks_by_name('index_source', source_id)
    return tqs.first() or False


def get_tasks(task_name, id=None, /, instance=None):
    assert not (id is None and instance is None)
    arg = str(id or instance.pk)
    return get_running_tasks_by_name(str(task_name), arg)

def get_first_task(task_name, id=None, /, *, instance=None):
    tqs = get_tasks(task_name, id, instance).order_by('scheduled_at')
    return tqs.first() or False

def get_media_metadata_task(media_id):
    return get_first_task('sync.tasks.download_media_metadata', media_id)


def cleanup_completed_tasks():
    days_to_keep = getattr(settings, 'COMPLETED_TASKS_DAYS_TO_KEEP', 30)
    delta = timezone.now() - timedelta(days=days_to_keep)
    log.info(f'Deleting completed tasks older than {days_to_keep} days '
             f'(end_at before {delta})')
    TaskHistory.objects.filter(end_at__lt=delta).delete()


def save_model(instance):
    with atomic(durable=False):
        instance.save()
    if 'sqlite' != db_vendor:
        return

    # work around for SQLite and its many
    # "database is locked" errors
    arg = getattr(settings, 'SQLITE_DELAY_FLOAT', 1.5)
    time.sleep(random.expovariate(arg))


def update_model(instance, **kwargs):
    qs = instance.__class__.objects.all()
    return qs.filter(
        pk=instance.pk,
    ).update(**kwargs)


_container_start_time_cache = None


def get_container_start_time():
    '''
        Wall-clock time this container's PID 1 started -- a hard,
        unambiguous signal distinct from any elapsed-time heuristic.

        Why this matters: on 2026-08-01, tubesync was evicted and
        rescheduled to a different node mid-download (a stuck Longhorn
        volume attachment). The old pod was killed non-gracefully, so
        several media:{uuid} locks and their TaskHistory "running" rows
        were left behind in Postgres (which survives across pods) with
        no process anywhere left to ever complete or release them --
        but they still looked "running" by any purely elapsed-time
        check, since their start_at was within the trust window.
        Comparing a lock's start_at against *this container's own start
        time* catches that case unconditionally: nothing from a previous
        pod generation ever ran in the current container. Cached because
        PID 1 cannot be replaced without the whole container restarting.
    '''
    global _container_start_time_cache
    if _container_start_time_cache is not None:
        return _container_start_time_cache
    try:
        clock_ticks = os.sysconf('SC_CLK_TCK')
        with open('/proc/uptime') as f:
            uptime = float(f.read().split()[0])
        boot_time = time.time() - uptime
        with open('/proc/1/stat') as f:
            fields = f.read().rsplit(')', 1)[1].split()
        starttime_ticks = int(fields[19])
        _container_start_time_cache = boot_time + (starttime_ticks / clock_ticks)
    except (OSError, ValueError, IndexError):
        _container_start_time_cache = None
    return _container_start_time_cache


def get_genuinely_running_uuids(lock_stale_after):
    '''
        Media/Source UUIDs backed by either a genuinely-running task, or a
        legitimately still-*pending* one, within `lock_stale_after`
        seconds. Shared by clear_stale_media_locks() (acts on this) and
        get_lock_status() (just reports it, for the Tasks page).

        The pending half matters because index_source()'s indexing loop
        acquires an `index_media:{uuid}` lock and enqueues
        migrate_to_metadata() for every newly-indexed media item
        synchronously, well before that task actually gets dequeued and
        run (the `filesystem` queue only has 2 workers) -- during a large
        indexing burst (e.g. enabling several sources at once), hundreds
        of locks can be acquired for tasks that haven't started yet.
        Counting only "genuinely running" (start_at set) missed all of
        these, misclassifying a completely normal backlog as orphaned
        locks -- which clear_stale_media_locks() would then have cleared
        and *duplicate*-rescheduled on its next 10-minute sweep, on top of
        the original enqueued task that was always going to run anyway.
    '''
    container_start = get_container_start_time()
    running_uuids = set()

    def _collect(qs):
        for params, start_at in qs:
            if container_start is not None and start_at is not None:
                if start_at.timestamp() < container_start:
                    # orphaned by a pod/container generation change -- not
                    # really running, no matter how "fresh" it looks.
                    continue
            try:
                args = params[0]
                if args:
                    running_uuids.add(str(args[0]))
            except (IndexError, TypeError, KeyError):
                continue

    _collect(TaskHistory.objects.running(within=lock_stale_after).values_list(
        'task_params', 'start_at',
    ))

    time_limit = timezone.now() - timedelta(seconds=lock_stale_after)
    _collect(TaskHistory.objects.filter(
        start_at__isnull=True,
        failed_at__isnull=True,
        end_at__gt=time_limit,
    ).values_list('task_params', 'start_at'))

    return running_uuids


@db_periodic_task(
    huey_crontab(minute=29, strict=True,),
    priority=10,
    expires=10*60,
    queue=Val(TaskQueue.DB),
)
def delete_deleted_sources():
    now = timezone.now()
    end_time = now + timezone.timedelta(minutes=5)
    qs = Source.objects.filter(
        key__endswith='/deleted',
    )
    for source in qs_gen(qs):
        if timezone.now() > end_time:
            log.info('delete_deleted_sources: beyond end_time')
            return
        log.info(f'Deleting: {source.pk}')
        result = source.delete()
        log.debug(f'Result: {result!r}')


@db_periodic_task(
    huey_crontab(minute=45, strict=True,),
    priority=10,
    expires=10*60,
    queue=Val(TaskQueue.DB),
)
def prune_huey_task_history():
    '''
        Huey's `task_history:*` KV storage was previously only pruned once,
        at worker startup (see register_huey_signals() in common/huey.py).
        Tasks that are interrupted/killed before firing a SIGNAL_COMPLETE
        never get pruned by that signal handler, so this storage could
        grow unbounded between restarts. Run it daily instead.
    '''
    from django_huey import DJANGO_HUEY, get_queue
    from common.huey import prune_task_history
    for qn in DJANGO_HUEY.get('queues', dict()):
        q = get_queue(qn)
        pruned = prune_task_history(q)
        if pruned:
            log.info(f'prune_huey_task_history: removed {pruned} entries from queue: {qn}')


@db_periodic_task(
    huey_crontab(minute='*/10', strict=True,),
    priority=10,
    expires=5*60,
    queue=Val(TaskQueue.DB),
)
def clear_stale_media_locks():
    '''
        Recovers automatically from the exact scenario hit in production:
        a `media:{uuid}` or `index_media:{uuid}` lock left held by a task
        that died or hung without releasing it (e.g. a stuck ffmpeg
        subprocess, or a pod restart mid-task) -- this required manually
        diagnosing and clearing the lock by hand.

        Huey's TaskLock is a bare KV flag with no timestamp of its own
        (see huey.api.TaskLock), so staleness can't be judged from the
        lock alone. Instead, cross-reference every currently-held
        media/index_media lock against TaskHistory (which IS timestamped)
        for a task that's actually still running against that UUID.

        Uses LOCK_STALE_AFTER_SECONDS (default 3h), not MAX_RUN_TIME
        (12h) -- MAX_RUN_TIME is tuned for the Tasks page's "Running"
        display, where it's fine to keep showing a legitimately
        long-running download as running for most of a day. But that
        same generous window made this sweeper *itself* useless: seen
        in production, huey worker processes restarted while a handful
        of media:{uuid} locks were held; the owning tasks were dead, but
        their TaskHistory rows still satisfied the 12h "running" check,
        so this function saw them as running_uuids and refused to touch
        their locks, requiring the exact manual diagnosis this function
        exists to avoid. A single video's download+transcode essentially
        never legitimately takes 3h, so this is a much more meaningful
        trust window for "is this lock's owner still plausibly alive" --
        and on top of that, get_genuinely_running_uuids() also discards
        anything that started before this container itself did (see
        get_container_start_time()), which is what actually would have
        caught the 2026-08-01 pod-eviction incident regardless of window
        size.

        Clearing the lock alone isn't enough to actually recover, though:
        huey has no mechanism to notice "the worker executing this task
        died" and requeue it -- the lock's orphaned owner had already
        been dequeued, so it's simply gone, and the media is left
        permanently un-downloaded until something else happens to notice.
        So for each orphaned lock, also explicitly reschedule the
        underlying work if it still needs doing.
    '''
    from django_huey import DJANGO_HUEY, get_queue, lock_task

    lock_stale_after = getattr(settings, 'LOCK_STALE_AFTER_SECONDS', 3 * 60 * 60)
    running_uuids = get_genuinely_running_uuids(lock_stale_after)

    lock_prefixes = ('media:', 'index_media:')
    cleared = 0
    rescheduled = 0
    for qn in DJANGO_HUEY.get('queues', dict()):
        q = get_queue(qn)
        try:
            rows = q.storage.sql('select key from kv', results=True)
        except Exception as e:
            log.warning(f'clear_stale_media_locks: could not read lock storage for queue {qn}: {e}')
            continue
        for (key,) in rows:
            if not isinstance(key, str) or '.lock.' not in key:
                continue
            _, _, lock_name = key.partition('.lock.')
            for prefix in lock_prefixes:
                if not lock_name.startswith(prefix):
                    continue
                lock_uuid = lock_name[len(prefix):]
                if lock_uuid in running_uuids:
                    break
                lock_task(lock_name, queue=qn).clear()
                cleared += 1
                log.warning(f'clear_stale_media_locks: cleared orphaned lock: '
                            f'{lock_name} (queue={qn})')
                if prefix == 'media:':
                    try:
                        media = Media.objects.get(pk=lock_uuid)
                    except (Media.DoesNotExist, ValueError):
                        break
                    if not media.downloaded and not media.manual_skip:
                        TaskHistory.schedule(
                            download_media_file,
                            str(media.pk),
                            remove_duplicates=True,
                            vn_fmt=_('Downloading media for "{}" (recovered after an orphaned lock)'),
                            vn_args=(media.name,),
                        )
                        rescheduled += 1
                        log.warning(f'clear_stale_media_locks: rescheduled download for '
                                    f'orphaned media: {media.key}')
                elif prefix == 'index_media:':
                    try:
                        media = Media.objects.get(pk=lock_uuid)
                    except (Media.DoesNotExist, ValueError):
                        break
                    still_pending = Metadata.objects.filter(
                        media__isnull=True, source=media.source, key=media.key,
                    ).exists()
                    if still_pending:
                        TaskHistory.schedule(
                            migrate_to_metadata,
                            str(media.pk),
                            remove_duplicates=True,
                            vn_fmt=_('Migrating metadata for "{}" (recovered after an orphaned lock)'),
                            vn_args=(media.name,),
                        )
                        rescheduled += 1
                        log.warning(f'clear_stale_media_locks: rescheduled metadata migration for '
                                    f'orphaned media: {media.key}')
                break
    if cleared:
        log.warning(f'clear_stale_media_locks: cleared {cleared} orphaned lock(s), '
                    f'rescheduled {rescheduled} task(s)')
    close_out_orphaned_task_history()


def close_out_orphaned_task_history():
    '''
        Closes out any TaskHistory row stuck satisfying the "running"
        convention (start_at == end_at) from a *previous* container
        generation -- i.e. its owning worker process was killed
        mid-execution (a pod restart, a manual worker SIGTERM during a
        deploy, the node eviction incidents from tonight) and no
        terminal huey signal (SIGNAL_COMPLETE/SIGNAL_ERROR) ever fired
        for it, so nothing else ever revisits or closes the row.

        This is the exact same container-generation check
        get_genuinely_running_uuids() already uses for locks, but
        applied directly to TaskHistory rows themselves, for every task
        type -- not just media/index_media lock holders. Caught live:
        a clear_stale_media_locks() run itself got interrupted this way
        and stayed shown as "Running" on the Tasks page for over an
        hour, since clear_stale_media_locks isn't a lock-holder and so
        was invisible to the lock-specific sweep above.
    '''
    container_start = get_container_start_time()
    if container_start is None:
        return
    running_qs = TaskHistory.objects.running()
    closed = 0
    for t in running_qs.only('pk', 'start_at', 'name'):
        if t.start_at is None or t.start_at.timestamp() >= container_start:
            continue
        t.failed_at = timezone.now()
        t.end_at = t.failed_at
        t.last_error = (
            'orphaned: owning worker process was killed mid-execution '
            '(container restarted before this task finished)'
        )
        t.save(update_fields=['failed_at', 'end_at', 'last_error'])
        closed += 1
    if closed:
        log.warning(f'close_out_orphaned_task_history: closed {closed} '
                    f'task(s) stuck "running" from a previous container generation')


def get_lock_status():
    '''
        Read-only version of the same check clear_stale_media_locks() acts
        on, for the Tasks page: how many media/index_media locks are
        currently held, and how many of those don't have a genuinely
        running task backing them right now (the same orphan signal,
        including the container-generation check). A non-zero
        `orphaned` here means something is stuck -- either waiting for
        the next sweep (it runs every 10 minutes) or, if this stays
        nonzero across repeated checks, something to look into directly.
    '''
    from django_huey import DJANGO_HUEY, get_queue

    lock_stale_after = getattr(settings, 'LOCK_STALE_AFTER_SECONDS', 3 * 60 * 60)
    running_uuids = get_genuinely_running_uuids(lock_stale_after)

    held = 0
    orphaned = 0
    for qn in DJANGO_HUEY.get('queues', dict()):
        q = get_queue(qn)
        try:
            rows = q.storage.sql('select key from kv', results=True)
        except Exception as e:
            log.warning(f'get_lock_status: could not read lock storage for queue {qn}: {e}')
            continue
        for (key,) in rows:
            if not isinstance(key, str) or '.lock.' not in key:
                continue
            _, _, lock_name = key.partition('.lock.')
            if not (lock_name.startswith('media:') or lock_name.startswith('index_media:')):
                continue
            held += 1
            lock_uuid = lock_name.split(':', 1)[1]
            if lock_uuid not in running_uuids:
                orphaned += 1
    return dict(held=held, orphaned=orphaned)


def get_throttle_status():
    '''
        Surfaces sync.throttle's cooldown state (see meeb/tubesync#1529)
        on the Tasks page directly -- so "why has nothing downloaded in
        a while" can be answered by looking at the page instead of
        having to know this module exists and query it by hand.
    '''
    from .throttle import in_cooldown
    cooling, remaining = in_cooldown()
    return dict(cooling=cooling, remaining_seconds=int(remaining))


def get_queue_activity():
    '''
        Per-queue "is a yt-dlp/YouTube call actually in flight right now"
        status, for the Tasks page. There is no literal per-queue rate
        limit in this app -- sync.throttle's cooldown (get_throttle_status)
        is global, since YouTube doesn't care which internal queue made a
        request. What IS meaningfully per-queue is concurrency: 'limited'
        (index_source/download_media_file/upgrade_media) enforces one
        yt-dlp call at a time via its single worker, and the separate
        'yt_dlp_aux_call' lock pool (refresh_formats/download_media_metadata,
        moved off 'limited' onto 'network' to stop a huge backlog of one
        from crowding out the other) allows up to 2 concurrent -- so at
        most 3 yt-dlp calls can be in flight cluster-wide. Showing whether
        each is currently busy makes that concurrency model visible
        instead of implicit.
    '''
    limited_running = TaskHistory.objects.running(within=3*60*60).filter(
        name__in=(
            'sync.tasks.index_source',
            'sync.tasks.download_media_file',
            'sync.tasks.upgrade_media',
        ),
    ).exists()
    aux_pool = LockPool('sync.tasks.yt_dlp_aux_call.slot', 3, queue=Val(TaskQueue.NET))
    return dict(
        limited_busy=limited_running,
        yt_dlp_aux_busy=aux_pool.is_locked(),
    )


def get_pod_info():
    '''
        Surfaces which pod/node is actually serving this page, via
        kubernetes' Downward API (env vars populated from fieldRef in the
        Deployment spec). Outside kubernetes -- e.g. local dev -- these env
        vars are simply unset and the fields render blank.
    '''
    import os
    return dict(
        pod_name=os.environ.get('POD_NAME', ''),
        pod_ip=os.environ.get('POD_IP', ''),
        node_name=os.environ.get('NODE_NAME', ''),
    )


@db_periodic_task(
    huey_crontab(minute=40, strict=True,),
    priority=100,
    expires=15*60,
    queue=Val(TaskQueue.DB),
)
def upcoming_media():
    qs = Media.objects.filter(
        manual_skip=True,
        published__isnull=False,
        published__gte=(
            # previous hour
            timezone.now() - timezone.timedelta(hours=1, minutes=1)
        ),
    )
    for media in qs_gen(qs):
        valid, hours = media.wait_for_premiere()
        if valid:
            save_model(media)
        log.debug(f'upcoming_media: wait_for_premiere: {media.key}: {valid=} {hours=}')


@db_periodic_task(
    huey_crontab(minute=59, strict=True,),
    priority=100,
    expires=30*60,
    queue=Val(TaskQueue.DB),
)
def schedule_indexing():
    now = timezone.now()
    next_hour = now + timezone.timedelta(hours=1, minutes=1)
    qs = Source.objects.filter(
        index_schedule__gt=Val(IndexSchedule.NEVER),
    )
    for source in qs_gen(qs):
        previous_run = next_hour - timezone.timedelta(
            seconds=source.index_schedule
        )
        skip_source = (
            not source.is_active or
            source.target_schedule >= next_hour or
            (source.last_crawl and source.last_crawl >= previous_run)
        )
        if skip_source:
            continue
        try:
            # clear all existing media locks
            media_qs = Media.objects.filter(source=source).only('uuid')
            for media in qs_gen(media_qs):
                huey_lock_task(
                    f'media:{media.uuid}',
                    queue=Val(TaskQueue.DB),
                ).clear()
        except QuerySetEmptyError as e:
            msg = f'missing media from "{source.name}": {source.pk}: {e.key}'
            log.exception(msg, exc_info=e)
            pass
        # schedule a new indexing task
        log.info(f'Scheduling an indexing task for source "{source.name}": {source.pk}')
        TaskHistory.schedule(
            index_source,
            str(source.pk),
            delay=300,
            expires=40*60,
            remove_duplicates=True,
            vn_fmt=_('Index media from source "{}"'),
            vn_args=(source.name,),
        )


def schedule_media_servers_update():
    # Schedule a task to update media servers
    log.info('Scheduling media server updates')
    for mediaserver in MediaServer.objects.all():
        rescan_media_server(str(mediaserver.pk))


def contains_http429(q, task_id, /):
    from huey.exceptions import TaskException
    try:
        q.result(preserve=True, id=task_id)
    except TaskException as e:
        return True if 'HTTPError 429: Too Many Requests' in str(e) else False
    return False


def wait_for_errors(model, /, *, queue_name=None, task_name=None):
    '''
        If other tasks in `queue_name` have recently hit a 429, reschedule
        this task via CancelExecution(retry=True) instead of running now.

        This used to block the calling worker thread in a `time.sleep(5)`
        loop for up to `10 * count` seconds. With several tasks hitting
        429s at once, most/all threads in a queue could end up parked in
        that sleep simultaneously, starving the queue's throughput for as
        long as the delay lasted. Raising CancelExecution(retry=True)
        instead hands the wait back to Huey's own retry/backoff scheduling,
        so the worker thread is freed immediately to pick up other tasks.
    '''
    if task_name is None:
        task_name=tuple((
            'sync.tasks.download_media_file',
            'sync.tasks.download_media_metadata',
        ))
    elif isinstance(task_name, str):
        task_name = tuple((task_name,))
    tasks = list()
    for tn in task_name:
        ft = get_first_task(tn, instance=model)
        if ft:
            tasks.append(ft)

    total_count = int()
    if queue_name:
        from django_huey import get_queue
        q = get_queue(queue_name)
        total_count += sum([ 1 if contains_http429(q, k) else 0 for k in q.all_results() ])
    if total_count <= 0:
        return

    delay = 10 * total_count
    time_str = seconds_to_timestr(delay)
    log.info(f'waiting for errors: 429 (approx. {time_str}): {model}, rescheduling')
    for task in tasks:
        update_task_status(task, 'paused (429)')
    try:
        raise CancelExecution(_('waiting for 429 errors to clear'), retry=True)
    finally:
        for task in tasks:
            update_task_status(task, None)


@db_task(priority=90, queue=Val(TaskQueue.FS))
def cleanup_old_media(durable=True):
    with atomic(durable=durable):
        for source in qs_gen(Source.objects.filter(delete_old_media=True, days_to_keep__gt=0)):
            delta = timezone.now() - timedelta(days=source.days_to_keep)
            mqs = source.media_source.defer(
                'metadata',
            ).filter(
                downloaded=True,
                download_date__lt=delta,
            )
            for media in qs_gen(mqs):
                log.info(f'Deleting expired media: {source} / {media} '
                         f'(now older than {source.days_to_keep} days / '
                         f'download_date before {delta})')
                with atomic(durable=False):
                    # .delete() also triggers a pre_delete/post_delete signals that remove files
                    media.delete()
    schedule_media_servers_update()


@db_task(priority=90, queue=Val(TaskQueue.FS))
def cleanup_removed_media(source_id, video_keys):
    try:
        source = Source.objects.get(pk=source_id)
    except Source.DoesNotExist as e:
        # Task triggered but the Source has been deleted, delete the task
        raise CancelExecution(_('no such source'), retry=False) from e
    if not source.delete_removed_media:
        return
    log.info(f'Cleaning up media no longer in source: {source}')
    mqs = Media.objects.defer(
        'metadata',
    ).filter(
        source=source,
    )
    with atomic(durable=True):
        for media in qs_gen(mqs):
            if media.key not in video_keys:
                log.info(f'{media.name} is no longer in source, removing')
                with atomic(durable=False):
                    media.delete()
    schedule_media_servers_update()


def save_db_batch(qs, objs, fields, /):
    assert hasattr(qs, 'bulk_update')
    assert callable(qs.bulk_update)
    assert hasattr(objs, '__len__')
    assert callable(objs.__len__)
    assert isinstance(fields, (tuple, list, set, frozenset))

    num_updated = 0
    num_objs = len(objs)
    with atomic(durable=False):
        num_updated = qs.bulk_update(objs=objs, fields=fields)
    if num_objs == num_updated:
        # this covers at least: list, set, deque
        if hasattr(objs, 'clear') and callable(objs.clear):
            objs.clear()
    return num_updated


@db_task(delay=60, priority=80, retries=10, retry_delay=60, queue=Val(TaskQueue.DB))
def migrate_to_metadata(media_id):
    try:
        media = Media.objects.get(pk=media_id)
    except Media.DoesNotExist as e:
        # Task triggered but the media no longer exists, do nothing
        log.error(f'Task migrate_to_metadata(pk={media_id}) called but no '
                  f'media exists with ID: {media_id}')
        raise CancelExecution(_('no such media'), retry=False) from e

    migrating_lock = huey_lock_task(
        f'index_media:{media.uuid}',
        queue=Val(TaskQueue.FS),
    )
    try:
        data = Metadata.objects.get(
            media__isnull=True,
            source=media.source,
            key=media.key,
        )
    except Metadata.DoesNotExist as e:
        migrating_lock.acquired = False
        raise CancelExecution(_('no indexed data to migrate to metadata'), retry=False) from e

    with huey_lock_task(
        f'media:{media.uuid}',
        queue=Val(TaskQueue.DB),
    ):
        video = data.value
        fields = lambda f, m: m.get_metadata_field(f)
        timestamp = video.get(fields('timestamp', media), None)
        for key in ('epoch', 'availability', 'extractor_key',):
            field = fields(key, media)
            value = video.get(field)
            existing_value = media.get_metadata_first_value(key)
            if value is None:
                if 'epoch' == key:
                    value = timestamp
                elif 'extractor_key' == key:
                    value = data.site
            if value is not None:
                if existing_value and ('epoch' == key or value == existing_value):
                    continue
                media.save_to_metadata(field, value)
    migrating_lock.acquired = False


@db_task(delay=30, priority=80, queue=Val(TaskQueue.LIMIT))
def index_source(source_id):
    '''
        Indexes media available from a Source object.
    '''
    db.reset_queries()
    cleanup_completed_tasks()
    # deleting expired media should happen any time an index task is requested.
    # cleanup_old_media is a @db_task -- calling it bare enqueues a *separate*
    # async task on the filesystem queue rather than running it here, so
    # indexing N sources back-to-back enqueued N independent, full-table (not
    # scoped to the current source) cleanup passes. With 2 filesystem workers,
    # two of those could run concurrently and race on deleting the same
    # expired row: one deletes it while the other's pre_delete signal handler
    # (media_pre_delete) tries to .save() its now-gone in-memory copy, which
    # Django resolves as an INSERT fallback -- colliding with the
    # (source_id, key) unique constraint. Reproduced live in production
    # (psycopg.errors.UniqueViolation on sync_media_source_id_key) while
    # enabling several sources at once. call_local() runs it synchronously,
    # inline, on this already-serialized 'limited' queue worker instead,
    # which is what the comment above always intended.
    cleanup_old_media.call_local()
    try:
        source = Source.objects.get(pk=source_id)
    except Source.DoesNotExist as e:
        # Task triggered but the Source has been deleted, delete the task
        raise CancelExecution(_('no such source'), retry=False) from e
    # An inactive Source would return an empty list for videos anyway
    if not source.is_active:
        return False
    indexing_lock = huey_lock_task(
        f'source:{source.uuid}',
        queue=Val(TaskQueue.FS),
    )
    # be sure that this is locked
    if not indexing_lock.acquired:
        indexing_lock.acquired = True
    # update the target schedule column
    source.task_run_at_dt
    update_model(source, target_schedule=source.target_schedule)
    # Reset any errors
    source.has_failed = False
    # Index the source in an isolated child process so yt-dlp's peak
    # memory usage for very large channels doesn't accumulate in this
    # long-lived worker process. A real OS subprocess, not
    # multiprocessing.Pool/Process: huey's own process-pool workers are
    # themselves daemonic (huey/consumer.py sets p.daemon = True), and
    # daemonic processes are forbidden from having multiprocessing children
    # (AssertionError, confirmed in production via the equivalent bug in
    # download_media_file). subprocess.Popen has no such restriction.
    import json
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, 'manage.py', 'index_media_worker', str(source_id)],
        cwd='/app',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not proc.stdout.strip():
        raise NoMediaException(
            f'Source "{source}" (ID: {source_id}) indexing subprocess exited '
            f'without a result (exit code {proc.returncode}): {proc.stderr}'
        )
    payload = json.loads(proc.stdout)
    if payload['status'] == 'error':
        raise NoMediaException(
            f'Source "{source}" (ID: {source_id}) indexing subprocess failed: '
            f'{payload.get("message")}'
        )
    videos = queue(payload['videos'])
    if not videos:
        source.has_failed = True
        update_model(source, has_failed=source.has_failed)
        indexing_lock.acquired = False
        raise NoMediaException(f'Source "{source}" (ID: {source_id}) returned no '
                               f'media to index, is the source key valid? Check the '
                               f'source configuration is correct and that the source '
                               f'is reachable')
    # Got some media, update the last crawl timestamp
    source.last_crawl = timezone.now()
    update_model(
        source,
        has_failed=source.has_failed,
        last_crawl=source.last_crawl,
    )
    num_videos = len(videos)
    log.info(f'Found {num_videos} media items for source: {source}')
    tvn_format = '{:,}' + f'/{num_videos:,}'
    db_batch_data = queue(list(), maxlen=50)
    db_fields_data = frozenset((
        'retrieved',
        'site',
        'value',
    ))
    db_batch_media = queue(list(), maxlen=10)
    db_fields_media = frozenset((
        'duration',
        'published',
        'title',
    ))
    fields = lambda f, m: m.get_metadata_field(f)
    task = get_source_index_task(source_id)
    if task:
        task._verbose_name = remove_enclosed(
            task.verbose_name, '[', ']', ' ',
            valid='0123456789/,',
            end=task.verbose_name.find('Index'),
        )
    vn = 0
    video_keys = set()
    try:
        while len(videos) > 0:
            vn += 1
            video = videos.popleft()
            # Create or update each video as a Media object
            key = video.get(source.key_field, None)
            if not key:
                # Video has no unique key (ID), it can't be indexed
                continue
            video_keys.add(key)
            if len(db_batch_data) == db_batch_data.maxlen:
                save_db_batch(Metadata.objects, db_batch_data, db_fields_data)
            if len(db_batch_media) == db_batch_media.maxlen:
                save_db_batch(Media.objects, db_batch_media, db_fields_media)
            update_task_status(task, tvn_format.format(vn))
            media_defaults = dict()
            # create a dummy instance to use its functions
            media = Media(source=source, key=key)
            media_defaults['duration'] = float(video.get(fields('duration', media), None) or 0) or None
            media_defaults['title'] = str(video.get(fields('title', media), ''))[:200]
            site = video.get(fields('ie_key', media), None)
            timestamp = video.get(fields('timestamp', media), None)
            try:
                published_dt = media.ts_to_dt(timestamp)
            except AssertionError:
                pass
            else:
                if published_dt:
                    media_defaults['published'] = published_dt
            # Retrieve or create the actual media instance
            media, new_media = source.media_source.only(
                'uuid',
                'source',
                'key',
                *db_fields_media,
            ).get_or_create(defaults=media_defaults, source=source, key=key)
            db_batch_media.append(media)
            data, new_data = source.videos.defer('value').filter(
                media__isnull=True,
            ).get_or_create(source=source, key=key)
            if site:
                data.site = site
            data.retrieved = source.last_crawl
            data.value = { k: v for k,v in video.items() if v is not None }
            db_batch_data.append(data)
            migrating_lock = huey_lock_task(
                f'index_media:{media.uuid}',
                queue=Val(TaskQueue.FS),
            )
            # `migrating_lock.acquired = True` runs through a property setter,
            # whose return value an assignment statement always discards -- so
            # the previous `if not migrating_lock.acquired: migrating_lock.acquired
            # = True` both raced (check, then separate set) and never actually
            # knew whether the acquire succeeded. migrate_to_metadata() was
            # then scheduled unconditionally, so a migration already in flight
            # for this media (lock held) could get a second, overlapping one.
            # Call __enter__() directly to get a real success/failure result.
            try:
                migrating_lock.__enter__()
            except Exception:
                already_migrating = True
            else:
                already_migrating = False
            if not already_migrating:
                migrate_to_metadata(str(media.pk))
            if not new_media:
                # update the existing media
                for key, value in media_defaults.items():
                    setattr(media, key, value)
                log.debug(f'Indexed media: {vn}: {source} / {media}')
            else:
                # log the new media instances
                log.info(f'Indexed new media: {source} / {media}')
                log.info(f'Scheduling tasks to download thumbnail for: {media.key}')
                thumbnail_fmt = 'https://i.ytimg.com/vi/{}/{}default.jpg'
                for num, prefix in enumerate(reversed(('hq', 'sd', 'maxres',))):
                    thumbnail_url = thumbnail_fmt.format(
                        media.key,
                        prefix,
                    )
                    download_media_image.schedule(
                        (str(media.pk), thumbnail_url,),
                        priority=10+(5*num),
                        delay=65-(30*num),
                    )
                priority = download_media_metadata.settings.get('default_priority', 50)
                if source.download_media:
                    priority += 5
                else:
                    priority -= 5
                log.info(f'Scheduling task to download metadata for: {media.url}')
                TaskHistory.schedule(
                    download_media_metadata,
                    str(media.pk),
                    priority=priority,
                    remove_duplicates=True,
                    vn_fmt=_('Downloading metadata for: "{}": {}'),
                    vn_args=(media.key, media.name,),
                )
    finally:
        # Always flush whatever Metadata/Media updates have been
        # accumulated so far, even if this task is interrupted or raises
        # mid-run -- otherwise up to `maxlen` buffered updates per batch
        # only ever existed in these in-memory deques and are lost.
        save_db_batch(Metadata.objects, db_batch_data, db_fields_data)
        save_db_batch(Media.objects, db_batch_media, db_fields_media)
    # Reset task.verbose_name to the saved value
    update_task_status(task, None)
    # Cleanup of media no longer available from the source
    cleanup_removed_media(str(source.pk), video_keys)
    # Clear references to indexed data
    videos = video = None
    db_batch_data.clear()
    db_batch_media.clear()
    # Let the checking task run
    indexing_lock.acquired = False
    # Create the checking task
    TaskHistory.schedule(
        save_all_media_for_source,
        str(source.pk),
        remove_duplicates=True,
        vn_fmt = _('Checking all media for "{}"'),
        vn_args=(
            source.name,
        ),
    )
    return True


@dynamic_retry(db_task, priority=100, retries=15, queue=Val(TaskQueue.FS))
def check_source_directory_exists(source_id):
    '''
        Checks the output directory for a source exists and is writable, if it does
        not attempt to create it. This is a task so if there are permission errors
        they are logged as failed tasks.
    '''
    try:
        source = Source.objects.get(pk=source_id)
    except Source.DoesNotExist as e:
        # Task triggered but the Source has been deleted, delete the task
        raise CancelExecution(_('no such source'), retry=False) from e
    # Check the source output directory exists
    if not source.directory_exists():
        # Try to create it
        log.info(f'Creating directory: {source.directory_path}')
        source.make_directory()


@dynamic_retry(db_task, delay=10, priority=90, retries=15, queue=Val(TaskQueue.NET))
def download_source_images(source_id):
    '''
        Downloads an image and save it as a local thumbnail attached to a
        Source instance.
    '''
    try:
        source = Source.objects.get(pk=source_id)
    except Source.DoesNotExist as e:
        # Task triggered but the source no longer exists, do nothing
        log.error(f'Task download_source_images(pk={source_id}) called but no '
                  f'source exists with ID: {source_id}')
        raise CancelExecution(_('no such source'), retry=False) from e
    avatar, banner, thumbnail = source.get_image_url
    log.info(f'Thumbnail URL for source with ID: {source_id} / {source} '
        f'Avatar: {avatar} '
        f'Banner: {banner} '
        f'Thumbnail: {thumbnail}')
    images = (
        (thumbnail, ('thumbnail.jpg',)),
        (banner,    ('banner.jpg', 'background.jpg')),
        (avatar,    ('poster.jpg', 'season-poster.jpg')),
    )
    for url, file_names in images:
        if url is None:
            continue
        i = get_remote_image(url)
        image_file = BytesIO()
        i.save(image_file, 'JPEG', quality=85, optimize=True, progressive=True)
        for file_name in file_names:
            image_file.seek(0)
            file_path = source.directory_path / file_name
            with open(file_path, 'wb') as f:
                f.write(image_file.read())
        i = image_file = None

    log.info(f'Thumbnail downloaded for source with ID: {source_id} / {source}')


@db_task(delay=60, priority=90, retries=5, retry_delay=60, queue=Val(TaskQueue.FS))
@atomic(durable=True)
def delete_media(media_id):
    try:
        media = Media.objects.get(pk=media_id)
    except Media.DoesNotExist as e:
        raise CancelExecution(_('no such media'), retry=False) from e
    else:
        migrating_lock = huey_lock_task(
            f'index_media:{media.uuid}',
            queue=Val(TaskQueue.FS),
        )
        # Check migrating_lock *inside* the media:{uuid} lock, not before
        # acquiring it: migrate_to_metadata() also does its actual field
        # mutations inside a media:{uuid} lock, so checking-then-acting
        # outside it left a window where a migration could start between
        # the check here and the mutation below.
        with huey_lock_task(
            f'media:{media.uuid}',
            queue=Val(TaskQueue.DB),
        ):
            if migrating_lock.acquired:
                raise CancelExecution(_('media indexing in progress'), retry=True)
            media.delete()


@db_task(delay=60, priority=70, retries=5, retry_delay=60, queue=Val(TaskQueue.FS))
@atomic(durable=True)
def rename_media(media_id):
    try:
        media = Media.objects.get(pk=media_id)
    except Media.DoesNotExist as e:
        raise CancelExecution(_('no such media'), retry=False) from e
    else:
        migrating_lock = huey_lock_task(
            f'index_media:{media.uuid}',
            queue=Val(TaskQueue.FS),
        )
        # See delete_media(): check inside the media:{uuid} lock to close
        # the TOCTOU window against migrate_to_metadata()'s own mutations.
        with huey_lock_task(
            f'media:{media.uuid}',
            queue=Val(TaskQueue.DB),
        ):
            if migrating_lock.acquired:
                raise CancelExecution(_('media indexing in progress'), retry=True)
            media.rename_files()


@db_task(delay=60, priority=80, retries=5, retry_delay=60, queue=Val(TaskQueue.FS))
@atomic(durable=True)
def save_media(media_id):
    try:
        media = Media.objects.get(pk=media_id)
    except Media.DoesNotExist as e:
        raise CancelExecution(_('no such media'), retry=False) from e
    else:
        migrating_lock = huey_lock_task(
            f'index_media:{media.uuid}',
            queue=Val(TaskQueue.FS),
        )
        # See delete_media(): check inside the media:{uuid} lock to close
        # the TOCTOU window against migrate_to_metadata()'s own mutations.
        with huey_lock_task(
            f'media:{media.uuid}',
            queue=Val(TaskQueue.DB),
        ):
            if migrating_lock.acquired:
                raise CancelExecution(_('media indexing in progress'), retry=True)
            save_model(media)


@db_task(delay=60, priority=50, retries=6, retry_delay=3600, queue=Val(TaskQueue.LIMIT))
def upgrade_media(media_id):
    try:
        media = Media.objects.get(pk=media_id)
    except Media.DoesNotExist as e:
        # Task triggered but the media no longer exists, do nothing
        raise CancelExecution(_('no such media'), retry=False) from e
    else:
        if not media.downloaded:
            raise CancelExecution(_('media not downloaded'))
        format_str = media.get_format_str()
        downloaded_dict = media.get_display_format(format_str)
        downloaded_height = downloaded_dict.get('height')
        if not downloaded_height:
            if media.source.is_audio:
                raise CancelExecution(_('upgrading audio is unsupported'), retry=False)
            raise CancelExecution(_('media height not available'))
        media.downloaded = False
        new_dict = media.get_display_format(format_str)
        media.downloaded = True
        format_height = new_dict.get('height')
        if not format_height or format_height <= downloaded_height:
            raise CancelExecution(_('downloaded media is better'))
        download_media_file.call_local(str(media.pk), override=True)

@dynamic_retry(db_task, backoff_func=lambda n: min(30*n, 300), priority=60, retries=30, queue=Val(TaskQueue.NET))
@LockPool('sync.tasks.yt_dlp_aux_call.slot', 3, queue=Val(TaskQueue.NET))
def download_media_metadata(media_id):
    '''
        Downloads the metadata for a media item.

        Runs on 'network' (shared 'yt_dlp_aux_call' serialization lock
        with refresh_formats) rather than 'limited' (download_media_file's
        single-worker queue) for the same reason refresh_formats was
        moved: 28,799 pending metadata fetches were crowding out actual
        video downloads for the same single worker. Still capped to one
        at a time (via the shared lock) so this doesn't add a second
        concurrent yt-dlp call against YouTube alongside whatever
        'limited' is doing.

        retries=3/retry_delay=600 (flat) was a bug: TaskLockedException
        from losing the race for the shared lock counts as a normal
        failure against this task's own retry budget, same as any other
        exception -- with only 3 tries and a 10-minute flat delay,
        contention against refresh_formats (which gets retries=15 and a
        much more patient backoff) could exhaust this task's retries
        before it ever actually ran, since both compete for the exact
        same lock. Verified in production: 23,262 of these had never run,
        oldest since 2026-07-26 -- i.e. this task type was being starved
        out entirely. priority=60 (vs refresh_formats' 50) means this
        should usually win the lock once both are eligible to try again;
        raising retries and shortening the backoff (30s, capped at 5min --
        vs refresh_formats' n*3600+600) makes sure it actually gets that
        many more, much sooner chances instead of dying quietly.
    '''
    try:
        media = Media.objects.get(pk=media_id)
    except Media.DoesNotExist as e:
        # Task triggered but the media no longer exists, do nothing
        log.error(f'Task download_media_metadata(pk={media_id}) called but no '
                  f'media exists with ID: {media_id}')
        raise CancelExecution(_('no such media'), retry=False) from e
    if media.manual_skip:
        log.info(f'Task for ID: {media_id} / {media} skipped, due to task being manually skipped.')
        return
    source = media.source
    cooling, remaining = throttle_in_cooldown()
    if cooling:
        log.info(f'download_media_metadata: in throttle cooldown ({int(remaining)}s left), '
                 f'rescheduling: {media.key}')
        raise CancelExecution(_('in throttle cooldown'), retry=True)
    wait_for_errors(
        media,
        queue_name=Val(TaskQueue.LIMIT),
        task_name='sync.tasks.download_media_metadata',
    )
    metadata_lock = huey_lock_task(
        f'index_media:{media.uuid}',
        queue=Val(TaskQueue.FS),
    )
    keep_metadata_lock = False
    try:
        metadata_lock.__enter__()
    except TaskLockedException as e:
        raise CancelExecution(_('media indexing in progress'), retry=True) from e
    try:
        metadata = media.index_metadata()
    except YouTubeError as e:
        e_str = str(e)
        raise_exception = True
        if ': Premieres in ' in e_str:
            now = timezone.now()
            published_datetime = None

            parts = e_str.split(': ', 1)[1].rsplit(' ', 2)
            unit = lambda p: str(p[-1]).lower()
            number = lambda p: int(str(p[-2]), base=10)
            log.debug(parts)
            try:
                if 'days' == unit(parts):
                    published_datetime = now + timedelta(days=number(parts))
                if 'hours' == unit(parts):
                    published_datetime = now + timedelta(hours=number(parts))
                if 'minutes' == unit(parts):
                    published_datetime = now + timedelta(minutes=number(parts))
                log.debug(unit(parts))
                log.debug(number(parts))
            except Exception as ee:
                log.exception(ee)
                pass

            if published_datetime:
                media.published = published_datetime
                media.manual_skip = True
                media.save()
                raise_exception = False
        if raise_exception:
            throttle_record_error(e_str)
            raise
        log.debug(str(e))
    else:
        keep_metadata_lock = True
    finally:
        metadata_lock.acquired = keep_metadata_lock
    response = metadata
    if getattr(settings, 'SHRINK_NEW_MEDIA_METADATA', False):
        response = filter_response(metadata, True)
    media.ingest_metadata(response)
    pointer_dict = {'_using_table': True}
    media.metadata = media.metadata_dumps(arg_dict=pointer_dict)
    upload_date = media.upload_date
    # Media must have a valid upload date
    if upload_date:
        media.published = timezone.make_aware(upload_date)
    timestamp = media.get_metadata_first_value(
        ('release_timestamp', 'timestamp',),
        arg_dict=response,
    )
    try:
        published_dt = media.ts_to_dt(timestamp)
    except AssertionError:
        pass
    else:
        if published_dt:
            media.published = published_dt

    # Store title in DB so it's fast to access
    if media.metadata_title:
        media.title = media.metadata_title[:200]

    # Store duration in DB so it's fast to access
    if media.metadata_duration:
        media.duration = media.metadata_duration

    # Don't filter media here, the post_save signal will handle that
    try:
        media.save()
    except Exception:
        raise
    else:
        log.info(f'Saved {len(media.metadata_dumps())} bytes of metadata for: '
                 f'{source} / {media}: {media_id}')
    finally:
        metadata_lock.acquired = False


@dynamic_retry(db_task, delay=10, priority=90, retries=15, queue=Val(TaskQueue.NET))
def download_media_image(media_id, url):
    '''
        Downloads an image from a URL and save it as a local thumbnail attached to a
        Media instance.
    '''
    try:
        media = Media.objects.get(pk=media_id)
    except Media.DoesNotExist as e:
        # Task triggered but the media no longer exists, do nothing
        raise CancelExecution(_('no such media'), retry=False) from e
    if media.skip or media.manual_skip:
        # Media was toggled to be skipped after the task was scheduled
        log.warn(f'Download task triggered for media: {media} (UUID: {media.pk}) but '
                 f'it is now marked to be skipped, not downloading thumbnail')
        return False
    width = getattr(settings, 'MEDIA_THUMBNAIL_WIDTH', 430)
    height = getattr(settings, 'MEDIA_THUMBNAIL_HEIGHT', 240)
    try:
        try:
            i = get_remote_image(url)
        except requests.HTTPError as re:
            if 404 != re.response.status_code:
                raise
            raise NoThumbnailException(re.response.reason) from re
    except NoThumbnailException as e:
        log.exception(str(e.__cause__))
        return False
    if (i.width > width) and (i.height > height):
        log.info(f'Resizing {i.width}x{i.height} thumbnail to '
                 f'{width}x{height}: {url}')
        i = resize_image_to_height(i, width, height)
    image_file = BytesIO()
    i.save(image_file, 'JPEG', quality=85, optimize=True, progressive=True)
    image_file.seek(0)
    media.thumb.save(
        'thumb',
        SimpleUploadedFile(
            'thumb',
            image_file.read(),
            'image/jpeg',
        ),
        save=True
    )
    i = image_file = None
    log.info(f'Saved thumbnail for: {media} from: {url}')
    # After media is downloaded, copy the updated thumbnail.
    copy_thumbnail = (
        media.downloaded and
        media.source.copy_thumbnails and
        media.thumb_file_exists
    )
    if copy_thumbnail:
        log.info(f'Copying media thumbnail from: {media.thumb.path} '
                 f'to: {media.thumbpath}')
        copyfile(media.thumb.path, media.thumbpath)        
    return True

@huey_signal(huey_signals.SIGNAL_COMPLETE, queue=Val(TaskQueue.NET))
def on_complete_download_media_image(signal_name, task_obj, exception_obj=None, /, *, huey=None):
    assert huey_signals.SIGNAL_COMPLETE == signal_name
    assert huey is not None
    if 'download_media_image' != task_obj.name:
        return
    result = huey.result(preserve=True, id=task_obj.id)
    # clear False/True from the results storage
    if result is False or result is True:
        huey.result(preserve=False, id=task_obj.id)

@db_task(
    delay=60, priority=70, queue=Val(TaskQueue.LIMIT),
    retries=3, retry_delay=300,
    # This used to also set `timeout=3*60*60` to guard against a stuck
    # ffmpeg/NFS hang here wedging the single-process 'limited' worker
    # forever. That guard does not actually work (see
    # `_download_media_in_subprocess`'s docstring) -- huey's SIGALRM-based
    # timeout cannot interrupt the TASK_KILLABLE NFS wait it exists to
    # guard against. The real enforcement now happens via the subprocess
    # deadline below, so huey's own `timeout` is left unset here: it would
    # only add a second, non-functional layer for this specific hang, and
    # for genuinely slow-but-progressing large 4K downloads/remuxes there
    # is no good universal huey-level timeout value anyway.
)
def download_media_file(media_id, override=False):
    '''
        Downloads the media to disk and attaches it to the Media instance.
    '''
    try:
        media = Media.objects.get(pk=media_id)
    except Media.DoesNotExist as e:
        # Task triggered but the media no longer exists, do nothing
        raise CancelExecution(_('no such media'), retry=False) from e
    else:
        if not media.download_checklist(override):
            # any condition that needs to reschedule the task
            # should raise an exception to avoid this
            return

    cooling, remaining = throttle_in_cooldown()
    if cooling:
        log.info(f'download_media_file: in throttle cooldown ({int(remaining)}s left), '
                 f'rescheduling: {media.key}')
        raise CancelExecution(_('in throttle cooldown'), retry=True)

    wait_for_errors(
        media,
        queue_name=Val(TaskQueue.LIMIT),
    )
    with huey_lock_task(
        f'media:{media.uuid}',
        queue=Val(TaskQueue.DB),
    ):
        filepath = media.filepath
        container = format_str = None
        log.info(f'Downloading media: {media} (UUID: {media.pk}) to: "{filepath}"')
        try:
            import json
            import subprocess
            import sys
            # A real OS subprocess, not multiprocessing.Process: huey's own
            # process-pool workers are themselves daemonic (huey/consumer.py
            # sets p.daemon = True), and daemonic processes are forbidden
            # from having multiprocessing children (asserted in
            # multiprocessing/process.py) -- confirmed by reproducing
            # `AssertionError: daemonic processes are not allowed to have
            # children` in production. subprocess.Popen has no such
            # restriction, and still lets us enforce a real wall-clock
            # deadline with Popen.kill() (SIGKILL) from outside the process
            # doing the actual I/O -- which is what's needed here, since
            # /downloads is a `hard` NFS4 mount and ffmpeg/yt-dlp file I/O
            # against it can block in the kernel's TASK_KILLABLE state,
            # which SIGALRM (huey's own `timeout=`) cannot interrupt but
            # SIGKILL can.
            proc = subprocess.Popen(
                [sys.executable, 'manage.py', 'download_media_worker', str(media_id)],
                cwd='/app',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=3*60*60)
            except subprocess.TimeoutExpired:
                log.error(f'download_media_file: subprocess for {media.key} '
                          f'exceeded deadline, killing it (pid {proc.pid})')
                proc.kill()
                stdout, stderr = proc.communicate(timeout=30)
                raise DownloadFailedException(
                    f'Downloading media {media} (UUID: {media.pk}) timed out '
                    f'and the download subprocess had to be killed'
                )
            if not stdout.strip():
                raise DownloadFailedException(
                    f'Download subprocess for {media} (UUID: {media.pk}) '
                    f'exited without a result (exit code {proc.returncode}): {stderr}'
                )
            payload = json.loads(stdout)
            if payload['status'] == 'error':
                category = payload.get('category')
                if category == 'format_unavailable':
                    media.failed_format(payload.get('format'))
                    raise CancelExecution(_('format did not work'), retry=True)
                elif category == 'youtube_error':
                    throttle_record_error(payload.get('message', ''))
                    raise YouTubeError(payload.get('message', ''))
                elif category == 'no_format':
                    raise NoFormatException(payload.get('message', ''))
                else:
                    raise DownloadFailedException(payload.get('message', 'unknown subprocess error'))
            format_str, container = payload['format_str'], payload['container']
        except NoFormatException:
            # Try refreshing formats
            if media.has_metadata:
                log.debug(f'Scheduling a task to refresh metadata for: {media.key}: "{media.name}"')
                TaskHistory.schedule(
                    refresh_formats,
                    str(media.pk),
                    remove_duplicates=True,
                    vn_fmt = _('Refreshing formats for "{}"'),
                    vn_args=(media.key,),
                )
            raise
        else:
            if not os.path.exists(filepath):
                # Try refreshing formats
                if media.has_metadata:
                    log.debug(f'Scheduling a task to refresh metadata for: {media.key}: "{media.name}"')
                    TaskHistory.schedule(
                        refresh_formats,
                        str(media.pk),
                        remove_duplicates=True,
                        vn_fmt = _('Refreshing formats for "{}"'),
                        vn_args=(media.key,),
                    )
                # Expected file doesn't exist on disk
                err = (
                    f'Failed to download media: {media} (UUID: {media.pk}) to disk, '
                    f'expected outfile does not exist: {filepath}'
                )
                log.error(err)
                # Raising an error here triggers the task to be re-attempted (or fail)
                raise DownloadFailedException(err)

            # Media has been downloaded successfully
            media.download_finished(format_str, container, filepath)
            media.save()
            media.rename_files()
            media.copy_thumbnail()
            media.write_nfo_file()
            # Try to download a better format later, if the settings allow this
            if getattr(settings, 'VIDEO_HEIGHT_UPGRADE', False):
                upgrade_media(str(media.pk))
            # Schedule a task to update media servers
            schedule_media_servers_update()


@db_task(delay=30, expires=210, priority=100, queue=Val(TaskQueue.NET))
def rescan_media_server(mediaserver_id):
    '''
        Attempts to request a media rescan on a remote media server.
    '''
    try:
        mediaserver = MediaServer.objects.get(pk=mediaserver_id)
    except MediaServer.DoesNotExist as e:
        # Task triggered but the media server no longer exists, do nothing
        raise CancelExecution(_('no such server'), retry=False) from e
    # Request an rescan / update
    log.info(f'Updating media server: {mediaserver}')
    mediaserver.update()


@dynamic_retry(db_task, backoff_func=lambda n: (n*3600)+600, priority=50, retries=15, queue=Val(TaskQueue.NET))
@LockPool('sync.tasks.yt_dlp_aux_call.slot', 3, queue=Val(TaskQueue.NET))
def refresh_formats(media_id):
    '''
        Runs on the 'network' queue (which has multiple workers) rather
        than 'limited' (download_media_file's single-worker queue) so a
        large backlog of format-refreshes -- e.g. proactively scheduled
        for every can_download=False video by save_all_media_for_source
        after indexing several large sources at once -- doesn't crowd
        out actual video downloads for the same single worker. Reproduced
        in production: 177k pending refresh_formats vs 17 pending
        download_media_file, both competing for one worker, at ~48s per
        refresh_formats call (~98 days to drain before downloads would
        get a fair share).

        Still serialized to exactly one at a time via huey_lock_task
        (not truly concurrent even though 'network' has several worker
        threads) so this doesn't start hammering YouTube with parallel
        format-list requests -- preserving the same "one yt-dlp call
        against YouTube at a time" property 'limited' existed for, just
        without sharing a worker with downloads specifically.
    '''
    try:
        media = Media.objects.get(pk=media_id)
    except Media.DoesNotExist as e:
        raise CancelExecution(_('no such media'), retry=False) from e
    else:
        wait_for_errors(
            media,
            queue_name=Val(TaskQueue.LIMIT),
        )
        save, retry, msg = media.refresh_formats()
        if save is not True:
            log.warning(f'Refreshing formats for "{media.key}" failed: {msg}')
            exc = CancelExecution(
                _('failed to refresh formats for:'),
                f'{media.key} / {media.uuid}:',
                msg,
                retry=retry,
            )
            # combine the strings
            exc.args = (' '.join(map(str, exc.args)),)
            # store instance details
            exc.instance = dict(
                key=media.key,
                model='Media',
                uuid=str(media.pk),
            )
            # store the function results
            exc.reason = msg
            exc.save = save
            raise exc
        # the metadata has already been saved, trigger the post_save signal
        log.info(f'Saving refreshed formats for "{media.key}": {msg}')
        media.save()


@db_task(delay=300, priority=80, retries=5, retry_delay=600, queue=Val(TaskQueue.FS))
@atomic(durable=True)
def rename_all_media_for_source(source_id):
    try:
        source = Source.objects.get(pk=source_id)
    except Source.DoesNotExist as e:
        # Task triggered but the source no longer exists, do nothing
        log.error(f'Task rename_all_media_for_source(pk={source_id}) called but no '
                  f'source exists with ID: {source_id}')
        raise CancelExecution(_('no such source'), retry=False) from e
    # Check that the settings allow renaming
    rename_sources_setting = getattr(settings, 'RENAME_SOURCES') or list()
    create_rename_tasks = (
        (
            source.directory and
            source.directory in rename_sources_setting
        ) or
        getattr(settings, 'RENAME_ALL_SOURCES', False)
    )
    if not create_rename_tasks:
        return
    mqs = Media.objects.filter(
        source=source,
        downloaded=True,
    )
    for media in qs_gen(mqs):
        migrating_lock = huey_lock_task(
            f'index_media:{media.uuid}',
            queue=Val(TaskQueue.FS),
        )
        try:
            with huey_lock_task(
                f'media:{media.uuid}',
                queue=Val(TaskQueue.DB),
            ):
                # Checked inside the media:{uuid} lock (not before
                # acquiring it) so a migration can't start in the gap
                # between the check and rename_files() below -- see
                # delete_media()/rename_media()/save_media().
                if migrating_lock.acquired:
                    # good luck to you in the queue!
                    rename_media(str(media.pk))
                    continue
                with atomic(durable=False):
                    media.rename_files()
        except TaskLockedException:
            rename_media(str(media.pk))


@dynamic_retry(db_task, delay=600, priority=70, retries=15, queue=Val(TaskQueue.FS))
@huey_lock_task('sync.tasks.save_all_media_for_source', queue=Val(TaskQueue.FS))
def save_all_media_for_source(source_id):
    '''
        Iterates all media items linked to a source and saves them to
        trigger the post_save signal for every media item. Used when a
        source has its parameters changed and all media needs to be
        checked to see if its download status has changed.
    '''
    db.reset_queries()
    try:
        source = Source.objects.get(pk=source_id)
    except Source.DoesNotExist as e:
        # Task triggered but the source no longer exists, do nothing
        log.error(f'Task save_all_media_for_source(pk={source_id}) called but no '
                  f'source exists with ID: {source_id}')
        raise CancelExecution(_('no such source'), retry=False) from e

    # Keep out of the way of the index task!
    # SQLite will be locked for a while if we start
    # a large source, which reschedules a more costly task.
    indexing_lock = huey_lock_task(
        f'source:{source.uuid}',
        queue=Val(TaskQueue.FS),
    )
    if indexing_lock.acquired:
        raise CancelExecution(_('source indexing in progress'))

    refresh_qs = Media.objects.all().only(
        'pk',
        'uuid',
        'key',
        'title', # for name property
    ).filter(
        source=source,
        can_download=False,
        skip=False,
        manual_skip=False,
        downloaded=False,
        metadata__isnull=False,
    )
    save_qs = Media.objects.all().only(
        'pk',
        'uuid',
    ).filter(
        source=source,
    )
    saved_later = set()
    for media in qs_gen(refresh_qs):
        if media.has_metadata:
            saved_later.add(str(media.pk))
            TaskHistory.schedule(
                refresh_formats,
                str(media.pk),
                remove_duplicates=True,
                vn_fmt = _('Refreshing formats for "{}"'),
                vn_args=(media.key,),
            )

    # Trigger the post_save signal for each media item linked to this source as various
    # flags may need to be recalculated
    saved_now = {
        str(media.pk)
        for media in qs_gen(save_qs)
        if str(media.pk) not in saved_later
    }
    save_media.map(saved_now)

    TaskHistory.schedule(
        rename_all_media_for_source,
        str(source.pk),
        remove_duplicates=True,
        vn_fmt = _('Renaming downloaded media from source "{}"'),
        vn_args=(source.name,),
    )


@dynamic_retry(db_task, delay=90, priority=99, queue=Val(TaskQueue.FS))
def delete_all_media_for_source(source_id, source_name, source_directory):
    source = None
    assert source_id
    assert source_name
    assert source_directory
    try:
        source = Source.objects.get(pk=source_id)
    except Source.DoesNotExist:
        # Task triggered but the source no longer exists, do nothing
        log.warning(f'Task delete_all_media_for_source(pk={source_id}) called but no '
                  f'source exists with ID: {source_id}')
        #raise CancelExecution(_('no such source'), retry=False) from e
        pass # this task can run after a source was deleted
    mqs = Media.objects.all().defer(
        'metadata',
    ).filter(
        source=source or source_id,
    )
    delete_media.map({
        str(media.pk)
        for media in qs_gen(mqs)
    })
    with atomic(durable=True):
        mqs.update(manual_skip=True, skip=True)
        log.info(f'Deleting media for source: {source_name}')
        mqs.delete()
    # Remove the directory, if the user requested that
    directory_path = Path(source_directory)
    remove = (
        (source and source.delete_removed_media) or
        (directory_path / '.to_be_removed').is_file()
    )
    if source:
        with atomic(durable=True):
            source.delete()
    if remove:
        log.info(f'Deleting directory for: {source_name}: {directory_path}')
        rmtree(directory_path, True)


# Run once per worker process at startup (not in gunicorn web workers --
# see register_huey_signals()'s docstring), so recovery from an
# ungracefully-killed previous pod happens immediately on the new pod
# coming up, instead of waiting for clear_stale_media_locks' own 10-minute
# periodic schedule to happen to fire.
register_huey_signals(on_worker_startup=clear_stale_media_locks)


