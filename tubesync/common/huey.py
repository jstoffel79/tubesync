import datetime
import subprocess
import threading
import time
import uuid
from functools import partial, wraps
from huey import (
    CancelExecution, Huey as huey_Huey,
    signals, utils,
)
from huey.api import TaskLock
from huey.exceptions import TaskLockedException
from huey.storage import SqliteStorage as huey_SqliteStorage
from pathlib import Path
from .timestamp import datetime_to_timestamp, timestamp_to_datetime
from .utils import get_usable_cpu_count


def _set_acquired(self, value=True):
    if value is not True:
        return self.clear()
    try:
        self.__enter__()
    except Exception:
        return False
    else:
        return True
#TaskLock._set_acquired = _set_acquired
TaskLock.acquired = property(
    TaskLock.is_locked,
    _set_acquired,
    TaskLock.clear,
)


class LockPool:
    '''
        N independent named TaskLocks, first-available-wins -- huey's own
        TaskLock only supports a single holder, with no built-in
        "N concurrent" primitive. Used to allow a bounded amount of
        concurrency (e.g. 2 simultaneous yt-dlp calls instead of a hard
        single-holder lock) for tasks that still need an upper bound to
        avoid an unbounded pile-up or tripping YouTube rate limits.

        Same acquire/raise-TaskLockedException-on-failure contract as a
        plain huey_lock_task, so it's a drop-in replacement as a
        decorator or context manager.

        The held lock is tracked per-thread (`threading.local`), not as
        a plain instance attribute -- a single LockPool instance (e.g.
        as a decorator, which is only instantiated once at import time)
        is shared by every huey worker *thread* that calls the decorated
        function, and queues like 'network' run with multiple threads.
        With a plain `self._held`, two threads racing through
        __enter__/__exit__ concurrently would stomp on the same
        attribute: thread A acquires slot 0 and sets self._held, then
        thread B acquires slot 1 and overwrites self._held before A's
        __exit__ runs -- so A's __exit__ ends up clearing B's lock
        instead of its own, and slot 0 is never cleared. Seen live in
        production: both slots of the 'sync.tasks.yt_dlp_aux_call.slot'
        pool ended up permanently held with no task actually running,
        silently deadlocking every format-refresh/metadata task
        forever (they'd raise TaskLockedException, retry with backoff,
        and pile up) until the entire task queue was manually reset.
    '''
    def __init__(self, base_name, n, /, *, queue):
        from django_huey import lock_task
        self._locks = [lock_task(f'{base_name}.{i}', queue=queue) for i in range(n)]
        self._local = threading.local()

    def __call__(self, fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return inner

    def __enter__(self):
        for lock in self._locks:
            try:
                lock.acquire()
            except TaskLockedException:
                continue
            self._local.held = lock
            return self
        raise TaskLockedException(
            f'unable to acquire any of {len(self._locks)} locks in pool'
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        held = getattr(self._local, 'held', None)
        if held is not None:
            held.clear()
            self._local.held = None

    def is_locked(self):
        return all(lock.is_locked() for lock in self._locks)


class SqliteStorage(huey_SqliteStorage):
    begin_sql = 'BEGIN IMMEDIATE'
    auto_vacuum = 'INCREMENTAL'
    journal_size_limit = 1024 * 1024 * 64
    wal_autocheckpoint = 100
    vacuum_pages = 10

    def _create_connection(self):
        conn = super()._create_connection()
        conn.execute(f'PRAGMA auto_vacuum = {self.auto_vacuum}')
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute(f'PRAGMA journal_size_limit = {self.journal_size_limit}')
        conn.execute('PRAGMA user_version = 1')
        conn.execute(f'PRAGMA wal_autocheckpoint={self.wal_autocheckpoint}')
        # copied from huey_SqliteHuey to use EXTRA or NORMAL
        # instead of FULL or OFF
        conn.execute('pragma synchronous=%s' % (3 if self._fsync else 1))
        # take at most vacuum_pages off the free list
        conn.execute(f'PRAGMA incremental_vacuum({self.vacuum_pages})')
        return conn

    def initialize_schema(self):
        super().initialize_schema()
        auto_vacuum_dict = {'NONE': 0, 'FULL': 1, 'INCREMENTAL': 2}
        valid = set(auto_vacuum_dict.values()) | set(auto_vacuum_dict.keys())
        assert self.auto_vacuum in valid, 'auto_vacuum was invalid'
        current = self.sql('PRAGMA auto_vacuum', results=True)
        expected = auto_vacuum_dict.get(self.auto_vacuum) or self.auto_vacuum
        vacuum_db = True
        try:
            current = int(current[0][0])
            expected = int(expected)
        except (TypeError, ValueError,):
            pass
        else:
            vacuum_db = (current != expected)
        with self.db(close=True) as curs:
            curs.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            if vacuum_db:
                curs.execute(f'PRAGMA auto_vacuum = {self.auto_vacuum}')
                curs.execute('VACUUM')


class Huey(huey_Huey):

    # do not use __len__ (pending_count) for bool
    def __bool__(self):
        if not (self._registry._registry or self._registry._periodic_tasks):
            return False
        return True

    def _emit(self, signal, task, *args, **kwargs):
        kwargs['huey'] = self
        super()._emit(signal, task, *args, **kwargs)

    def create_consumer(self, **options):
        consumer = super().create_consumer(**options)
        suffix = f'worker.{consumer.worker_type}'
        consumer._logger = consumer._logger.getChild(suffix)
        return consumer

    def reschedule(self, task_id, eta):
        pc = self.pending_count()
        found = [
            t
            for t in self.pending(100)
            if t.id == task_id
        ]
        if not found and pc > 100:
            found = [
                t
                for t in self.pending()
                if t.id == task_id
            ]
        if not found:
            found = [
                t
                for t in self.scheduled()
                if t.id == task_id
            ]
        if not found:
            return False
        for t in found:
            self.revoke(t, revoke_once=True)
        task_eta = utils.normalize_time(eta=eta, utc=self.utc)
        for t in found:
            previous_id = t.id
            t.eta = task_eta
            t.id = t.create_id()
            t.revoke_id = f'r:{t.id}'
            try:
                from common.models import TaskHistory
                th = TaskHistory.objects.get(task_id=previous_id)
            except:
                pass
            else:
                # clone the old task history
                th.pk = None
                th._state.adding = True
                th.task_id = t.id
                th.save()
            self.enqueue(t)
        return True

    def scheduled_at_from_task(self, task_obj, /):
        scheduled_at = None
        if not (hasattr(task_obj, 'eta') and task_obj.eta):
            return None
        if self.utc:
            scheduled_at = task_obj.eta.replace(tzinfo=datetime.timezone.utc)
        else: # this path is unlikely
            scheduled_at = timestamp_to_datetime(
                datetime_to_timestamp(task_obj.eta, integer=False),
            ).astimezone(tz=datetime.timezone.utc)
        return scheduled_at


class SqliteHuey(Huey):
    storage_class = SqliteStorage


def CancelExecution_init(self, *args, retry=None, **kwargs):
    self.retry = retry
    super(CancelExecution, self).__init__(*args, **kwargs)
CancelExecution.__init__ = CancelExecution_init


def delay_to_eta(delay, /):
    return utils.normalize_time(delay=delay)


def h_q_dict(q, /):
    return dict(
        scheduled=(q.scheduled_count(), q.scheduled(),),
        pending=(q.pending_count(), q.pending(),),
        results=(q.result_count(), list(q.all_results().keys()),),
    )


def h_q_tuple(q, /):
    if isinstance(q, str):
        from django_huey import get_queue
        q = get_queue(q)
    return (
        q.name,
        list(q._registry._registry.keys()),
        h_q_dict(q),
    )

def start_consumer(queue_name):
    assert isinstance(queue_name, str), type(queue_name)
    svc_name = queue_name.replace('_', '-')
    svc_dir = Path('/run') / 'service' / svc_name

    try:
        if not all((
            svc_dir.is_dir(),
            (svc_dir / 'supervise').is_dir(),
            (svc_dir / 'supervise' / 'control').is_fifo(),
        )):
            return
    except PermissionError:
        pass
    else:
        subprocess.Popen(
            [ '/command/s6-svc', '-U', str(svc_dir) ],
            stdout=-1, stderr=-1, start_new_session=True,
        )

def h_q_reset_maint_func(queue, /, exception=None, status=None):
        if status is None:
            return
        elif 'started' == status:
            start_consumer(queue.name)
        elif 'exception' == status and exception is not None:
            # log, but do not raise an exception
            from huey.api import logger
            logger.error(
                f'{queue.name}: maintenance function exception: {exception}'
            )
            return
        return True

# Configuration convenience helpers

def h_q_reset_tasks(q, /, *, maint_func=None):
    if isinstance(q, str):
        from django_huey import get_queue
        q = get_queue(q)
    # revoke to prevent pending tasks from executing
    for t in q._registry._registry.values():
        q.revoke_all(t, revoke_until=delay_to_eta(600))
    # clear scheduled tasks
    q.storage.flush_schedule()
    # clear pending tasks
    q.storage.flush_queue()
    # run the maintenance function
    def default_maint_func(queue, /, exception=None, status=None):
        if status is None:
            return
        if 'exception' == status and exception is not None:
            # log, but do not raise an exception
            from huey.api import logger
            logger.error(
                f'{queue.name}: maintenance function exception: {exception}'
            )
            return
        return True
    maint_result = None
    if maint_func is None:
        maint_func = default_maint_func
    if maint_func and callable(maint_func):
        try:
            maint_result = maint_func(q, status='started')
        except Exception as exc:
            maint_result = maint_func(q, exception=exc, status='exception')
        finally:
            maint_func(q, status='finished')
    # clear everything now that we are done
    q.storage.flush_all()
    q.flush()
    # return the results from the maintenance function
    return maint_result


def sqlite_tasks(key, /, prefix=None, thread=None, workers=None, *, tasks_dir=None):
    name_fmt = 'huey_{}'
    if prefix:
        name_fmt = f'huey_{prefix}_' + '{}'
    name = name_fmt.format(key)
    thread = thread is True
    try:
        workers = int(workers)
    except TypeError:
        workers = 2
    finally:
        if 0 >= workers:
            workers = max(2, get_usable_cpu_count() // 2)
        elif 1 == workers:
            thread = False
    return dict(
        huey_class='common.huey.SqliteHuey',
        name=name,
        immediate=False,
        results=True,
        store_none=False,
        utc=True,
        compression=True,
        connection=dict(
            filename=str(Path(tasks_dir or '/config/tasks') / f'{name}.db'),
            fsync=True,
            isolation_level='IMMEDIATE', # _create_connection sets this to None
            strict_fifo=True,
            timeout=60,
        ),
        consumer=dict(
            workers=workers if thread else 1,
            worker_type='thread' if thread else 'process',
            max_delay=20.0,
            max_tasks=10_000,
            check_worker_health=True,
            health_check_interval=30.0,
            flush_locks=True,
            scheduler_interval=10,
            simple_log=False,
            # verbose has three positions:
            # DEBUG: True
            # INFO: None
            # WARNING: False
            verbose=False,
        ),
    )

# Decorators

def dynamic_retry(task_func=None, /, *args, **kwargs):
    if task_func is None:
        from django_huey import task as huey_task
        task_func = huey_task
    backoff_func = kwargs.pop('backoff_func', None)
    def default_backoff(attempt, /):
        return (5+(attempt**4))
    if backoff_func is None or not callable(backoff_func):
        backoff_func = default_backoff
    def deco(fn):
        @wraps(fn)
        def inner(*a, **kwa):
            backoff = backoff_func
            # the scoping becomes complicated when reusing functions
            try:
                _task = kwa.pop('task')
            except KeyError:
                pass
            else:
                task = _task
            try:
                return fn(*a, **kwa)
            except Exception as exc:
                try:
                    task is not None
                except NameError:
                    raise exc
                # A TaskLockedException here means this call lost the race
                # for a LockPool slot (e.g. sync.tasks.yt_dlp_aux_call) --
                # that's contention between our own worker threads, not a
                # real failure of the underlying work, and it fails near-
                # instantly (no actual yt-dlp/network call ever happened).
                # Running it through the caller's real-failure backoff_func
                # (tuned for genuine repeated errors, e.g. hours-scale for
                # refresh_formats) let a handful of lock-contention misses
                # push a task hours into the future within minutes, making
                # a perfectly healthy backlog look permanently stuck. Use
                # a short, capped backoff for this case regardless of what
                # the decorated function configured.
                if isinstance(exc, TaskLockedException):
                    lock_backoff = lambda attempt: min(30 * attempt, 300)
                else:
                    lock_backoff = backoff
                for attempt in range(1, 240):
                    if lock_backoff(attempt) > task.retry_delay:
                        task.retry_delay = lock_backoff(attempt)
                        break
                    # insanity, but handle it anyway
                    if 239 == attempt:
                        task.retry_delay = lock_backoff(attempt)
                raise exc
        kwargs.update(dict(
            context=True,
            retry_delay=backoff_func(1),
        ))
        return task_func(*args, **kwargs)(inner)
    return deco

# Signal handlers shared between queues

def on_executing_remove_duplicates(signal_name, task_obj, exception_obj=None, /, *, huey=None):
    if signals.SIGNAL_EXECUTING != signal_name:
        return
    assert exception_obj is None
    assert huey is not None
    assert hasattr(huey, 'pending') and callable(huey.pending)
    assert hasattr(huey, 'revoke_by_id') and callable(huey.revoke_by_id)
    assert hasattr(huey, 'scheduled') and callable(huey.scheduled)
    assert signals.SIGNAL_EXECUTING == signal_name

    from common.models import TaskHistory
    try:
        th = TaskHistory.objects.get(task_id=str(task_obj.id))
    except TaskHistory.DoesNotExist:
        return
    else:
        if not th.remove_duplicates:
            return

    def task_generator(queue):
        for t in queue.pending():
            yield t
        for t in queue.scheduled():
            yield t

    def waiting_id_generator(queue, task_obj):
        for t in task_generator(queue):
            matches = all((
                task_obj.id != t.id,
                task_obj.name == t.name,
                task_obj.data == t.data,
                task_obj.priority >= t.priority,
                task_obj.retries <= t.retries,
            ))
            if matches:
                yield t.id

    for task_id in waiting_id_generator(queue=huey, task_obj=task_obj):
        huey.revoke_by_id(task_id, revoke_once=True)

def on_interrupted(signal_name, task_obj, exception_obj=None, /, *, huey=None):
    if signals.SIGNAL_INTERRUPTED != signal_name:
        return
    assert exception_obj is None
    assert huey is not None
    assert hasattr(huey, 'enqueue') and callable(huey.enqueue)
    huey.enqueue(task_obj)

storage_key_prefix = 'task_history:'

def historical_task(signal_name, task_obj, exception_obj=None, /, *, huey=None):
    signal_time = time.monotonic()
    signal_dt = datetime.datetime.now(datetime.timezone.utc)
    assert huey is not None
    assert hasattr(huey, 'get') and callable(huey.get)
    assert hasattr(huey, 'put') and callable(huey.put)

    from common.models import TaskHistory
    add_to_elapsed_signals = frozenset((
        signals.SIGNAL_TIMEOUT,
        signals.SIGNAL_INTERRUPTED,
        signals.SIGNAL_ERROR,
        signals.SIGNAL_CANCELED,
        signals.SIGNAL_COMPLETE,
    ))
    recorded_signals = frozenset((
        signals.SIGNAL_REVOKED,
        signals.SIGNAL_EXPIRED,
        signals.SIGNAL_LOCKED,
        signals.SIGNAL_EXECUTING,
        signals.SIGNAL_RETRYING,
        signals.SIGNAL_RATE_LIMITED,
    )) | add_to_elapsed_signals
    storage_key = f'{storage_key_prefix}{task_obj.id}'
    task_obj_attr = '_signals_history'

    history = getattr(task_obj, task_obj_attr, None)
    if history is None:
        # pull it from storage, or initialize it
        history = huey.get(
            key=storage_key,
            peek=True,
        ) or dict(
            created=signal_dt,
            data=(task_obj.args, repr(task_obj.kwargs),),
            elapsed=0,
            module=task_obj.__module__,
            name=task_obj.name,
        )
        setattr(task_obj, task_obj_attr, history)
    assert history is not None
    history['modified'] = signal_dt

    if signal_name in recorded_signals:
        history[signal_name] = signal_time
    if signal_name in add_to_elapsed_signals and signals.SIGNAL_EXECUTING in history:
        history['elapsed'] += signal_time - history[signals.SIGNAL_EXECUTING]
    if signals.SIGNAL_COMPLETE in history:
        huey.get(key=storage_key)
    else:
        huey.put(key=storage_key, data=history)
    th, created = TaskHistory.objects.get_or_create(
        task_id=str(task_obj.id),
        name=f"{task_obj.__module__}.{task_obj.name}",
        queue=huey.name,
    )
    th.priority = task_obj.priority
    th.task_params = list((
        list(task_obj.args),
        repr(task_obj.kwargs),
    ))
    if signal_name == signals.SIGNAL_EXECUTING:
        th.attempts += 1
        th.start_at = signal_dt
    elif exception_obj is not None:
        th.failed_at = signal_dt
        th.last_error = str(exception_obj)
    elif signal_name == signals.SIGNAL_REVOKED:
        # A revoked task (e.g. remove_duplicates canceling a stale
        # duplicate -- see on_executing_remove_duplicates) never gets a
        # SIGNAL_EXECUTING and has no exception, so without this branch
        # start_at/failed_at both stay NULL forever: identical to a task
        # that's still genuinely pending. That inflated
        # get_task_breakdown()'s "pending by task type" (and the Tasks
        # page's total) with hundreds of thousands of dead rows that will
        # never run again, and bloated this table to 600k+ rows. Mark it
        # failed (not pending) so it stops being counted as backlog.
        th.failed_at = signal_dt
        th.last_error = 'revoked (duplicate or stale task, will not run)'
    elif signal_name == signals.SIGNAL_ENQUEUED:
        scheduled_at = huey.scheduled_at_from_task(task_obj)
        if scheduled_at:
            th.scheduled_at = scheduled_at
        if not th.verbose_name:
            try:
                from django.core.exceptions import ValidationError
                from sync.models import Media, Source
            except:
                pass
            else:
                try:
                    key = task_obj.args[0]
                    instance_uuid_str = str(key)
                    instance_uuid = uuid.UUID(instance_uuid_str)
                except IndexError:
                    key = None
                except RuntimeError:
                    instance_uuid_str = None
                except ValueError:
                    instance_uuid = None
                else:
                    for model in (Source, Media,):
                        try:
                            model_instance = model.objects.get(pk=instance_uuid)
                        except (model.DoesNotExist, ValidationError,):
                            pass
                        else:
                            if hasattr(model_instance, 'key'):
                                th.verbose_name = f'{th.name} with: {model_instance.key}'
                                if hasattr(model_instance, 'name'):
                                    th.verbose_name += f' / {model_instance.name}'
                            break
    elif signal_name == signals.SIGNAL_SCHEDULED:
        scheduled_at = huey.scheduled_at_from_task(task_obj)
        if scheduled_at:
            th.scheduled_at = scheduled_at
    th.end_at = signal_dt
    th.elapsed = history['elapsed']
    th.save()

# Registration of shared signal handlers

def prune_task_history(q, /, *, max_age=datetime.timedelta(days=7)):
    '''
        Removes `task_history:*` KV entries (and their associated result,
        if any) older than `max_age` from a queue's storage.

        This used to run only once, from register_huey_signals() at
        worker startup. Tasks that are interrupted/killed before firing a
        SIGNAL_COMPLETE never get pruned by that signal handler either, so
        between worker restarts this storage can grow unbounded. Called
        here so it can also be scheduled periodically (see
        prune_all_task_history() in sync/tasks.py).
    '''
    now_time = time.monotonic()
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    pruned = 0
    for key in q.all_results().keys():
        if not key.startswith(storage_key_prefix):
            continue
        history = q.get(peek=True, key=key)
        if not isinstance(history, dict):
            continue
        if 'created' not in history:
            # Set an incorrect created time to ensure eventual removal.
            history['created'] = now_dt
            q.put(key=key, data=history)
            # Attempt to calculate an accurate age.
            # This doesn't work all the time,
            # so the created datetime is the fail-safe case.
            seconds = now_time - history.get(signals.SIGNAL_EXECUTING, now_time)
            age = datetime.timedelta(
                seconds=(seconds if seconds > 0 else 0),
            )
        else:
            age = now_dt - history['created']
        if age > max_age:
            result_key = key[len(storage_key_prefix) :]
            q.get(peek=False, key=result_key)
            q.get(peek=False, key=key)
            pruned += 1
    return pruned


def register_huey_signals(on_worker_startup=None):
    '''
        `on_worker_startup`, if given, is called once per queue via huey's
        own on_startup() hook -- which only fires in actual consumer/worker
        processes (`manage.py djangohuey --queue ...`), not in gunicorn web
        workers that merely import this module. This is the right place
        for "recover any state left behind by an ungracefully-killed
        previous pod" work: it runs the moment a fresh worker process comes
        up, rather than waiting for that recovery's own periodic schedule
        to happen to fire (which could be up to its full interval late,
        e.g. up to 10 minutes for clear_stale_media_locks).
    '''
    from django import db
    from django_huey import DJANGO_HUEY, get_queue, pre_execute, post_execute, signal
    def close_db(task, task_value=None, exception=None, /, *, huey=None):
        assert huey is not None
        assert hasattr(huey, 'immediate')
        # Guard against immediate mode and transactions
        if (
            any((huey is None, not hasattr(huey, 'immediate'), huey.immediate,)) or
            any(dbw.in_atomic_block for dbw in db.connections.all())
        ):
            return
        db.close_old_connections()

    for qn in DJANGO_HUEY.get('queues', dict()):
        q = get_queue(qn)

        # hooks to clean up database connections at task boundaries
        pre_execute(queue=qn, name='close_db')(partial(close_db, huey=q))
        post_execute(queue=qn, name='close_db')(partial(close_db, huey=q))

        # signals for tracking / maintenance of tasks
        signal(signals.SIGNAL_INTERRUPTED, queue=qn)(on_interrupted)
        signal(queue=qn)(historical_task)
        signal(signals.SIGNAL_EXECUTING, queue=qn)(on_executing_remove_duplicates)

        # clean up old history and results from storage
        prune_task_history(q)

        if on_worker_startup is not None:
            # If a huey TaskWrapper (@db_task/@db_periodic_task-decorated)
            # is passed, calling it directly would enqueue() it as an
            # async task rather than actually running it here -- and it
            # has no __name__ for on_startup()'s default naming, since
            # it's a TaskWrapper, not a plain function. call_local() runs
            # the wrapped function synchronously instead, which is what a
            # startup hook needs.
            hook = getattr(on_worker_startup, 'call_local', on_worker_startup)
            hook_name = getattr(on_worker_startup, '__name__', None) or getattr(
                getattr(on_worker_startup, 'func', None), '__name__', 'on_worker_startup',
            )
            q.on_startup(name=hook_name)(hook)

