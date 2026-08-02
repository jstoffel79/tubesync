import os
from pathlib import Path
from random import random
from django.conf import settings
from django.shortcuts import render
from django.views.generic import View
from django.http import HttpResponse, HttpResponseServerError, JsonResponse
from django.core.exceptions import PermissionDenied
from django.db import connection
from .utils import get_client_ip


def error403(request, *args, **kwargs):
    return render(request, 'error403.html', status=403) 


def error404(request, *args, **kwargs):
    return render(request, 'error404.html', status=404) 


def error500(request, *args, **kwargs):
    return render(request, 'error500.html', status=500) 


class HealthCheckView(View):
    '''
        A basic healthcheck view. SELECTs a random int via the database connection
        and verifies it matches. This checks that the application server, django and
        the database connection are all working correctly.
    '''

    ALLOWED_IPS = settings.HEALTHCHECK_ALLOWED_IPS

    def get(self, request, *args, **kwargs):
        if settings.HEALTHCHECK_FIREWALL:
            client_ip = get_client_ip(request)
            if client_ip not in self.ALLOWED_IPS:
                raise PermissionDenied
        randomint = int(random() * (10 ** 10))
        with connection.cursor() as cursor:
            cursor.execute('select {}'.format(randomint))
            row = cursor.fetchone()
        try:
            pong = row[0]
        except IndexError:
            pong = False
        if str(pong) != str(randomint):
            err = 'Failed healtcheck, expected "{}" got "{}"'
            return HttpResponseServerError(err.format(randomint, pong))
        else:
            return HttpResponse('ok')


class LivenessView(View):
    '''
        Cheap process-alive check for a kubernetes livenessProbe: no DB, no
        filesystem I/O. Deliberately does not check dependencies (DB, NFS
        mount) that can be slow/degraded without the process itself being
        dead -- a livenessProbe failing on those causes a restart loop that
        makes a transient NFS/DB blip *worse*, not better. Use ReadinessView
        (readinessProbe) for dependency checks instead.
    '''

    def get(self, request, *args, **kwargs):
        return HttpResponse('ok')


class ReadinessView(View):
    '''
        Dependency check for a kubernetes readinessProbe: DB connectivity and
        that /config and /downloads are actually writable (catches a stuck/
        read-only NFS mount before it routes traffic to this pod). No IP
        firewall here -- kubelet probes this via the pod IP from the node,
        not from 127.0.0.1, so applying HEALTHCHECK_FIREWALL would make every
        pod permanently unready.
    '''

    def _check_db(self):
        randomint = int(random() * (10 ** 10))
        with connection.cursor() as cursor:
            cursor.execute('select {}'.format(randomint))
            row = cursor.fetchone()
        pong = row[0] if row else None
        if str(pong) != str(randomint):
            raise RuntimeError(f'db check failed: expected "{randomint}" got "{pong}"')

    def _check_writable(self, dirpath):
        path = Path(dirpath)
        probe = path / f'.healthz-{os.getpid()}'
        try:
            probe.write_text('ok')
        finally:
            probe.unlink(missing_ok=True)

    def get(self, request, *args, **kwargs):
        checks = {}
        healthy = True
        for name, fn in (
            ('database', self._check_db),
            ('config_writable', lambda: self._check_writable('/config')),
            ('downloads_writable', lambda: self._check_writable('/downloads')),
        ):
            try:
                fn()
                checks[name] = 'ok'
            except Exception as e:
                checks[name] = f'error: {e}'
                healthy = False
        status = 200 if healthy else 503
        return JsonResponse({'status': 'ok' if healthy else 'error', 'checks': checks}, status=status)


class MetricsView(View):
    '''
        Plain-text Prometheus exposition of the same figures already
        computed for the Tasks page System Status panel (queue depths,
        cgroup accounting, lock/throttle state, live ffmpeg processes) --
        reusing those functions rather than a separate metrics-specific
        code path, so this can never drift from what the UI shows. No auth/
        IP firewall: scraped in-cluster only by the Prometheus Operator via
        a ServiceMonitor, same trust boundary as the rest of the pod.
    '''

    def get(self, request, *args, **kwargs):
        from sync.tasks import (
            get_queue_status, get_cgroup_status, get_ffmpeg_status,
            get_lock_status, get_throttle_status,
        )
        # name -> (help, type, [(labels_dict, value), ...])
        metrics = {}

        def sample(name, help_text, value, mtype='gauge', labels=None):
            metrics.setdefault(name, (help_text, mtype, []))[2].append((labels or {}, value))

        for q in get_queue_status():
            sample('tubesync_queue_pending',
                   'Tasks ready to run right now in this Huey queue.',
                   q['pending'], labels={'queue': q['name']})
            sample('tubesync_queue_scheduled',
                   'Tasks scheduled for a future eta in this Huey queue.',
                   q['scheduled'], labels={'queue': q['name']})

        cgroup_status = get_cgroup_status()
        if 'cpu_throttled_pct' in cgroup_status:
            sample('tubesync_cpu_throttled_ratio',
                   'Fraction of cgroup CPU accounting periods this container was throttled in.',
                   cgroup_status['cpu_throttled_pct'] / 100)
        if 'memory_current_mb' in cgroup_status:
            sample('tubesync_memory_bytes',
                   'Current cgroup memory usage in bytes.',
                   int(cgroup_status['memory_current_mb'] * 1024 * 1024))
        if 'memory_max_mb' in cgroup_status:
            sample('tubesync_memory_limit_bytes',
                   'Cgroup memory limit in bytes.',
                   int(cgroup_status['memory_max_mb'] * 1024 * 1024))

        sample('tubesync_ffmpeg_processes',
               'Number of ffmpeg/ffprobe processes currently running in this container.',
               len(get_ffmpeg_status()))

        lock_status = get_lock_status()
        sample('tubesync_locks_held', 'Media/index locks currently held.',
               lock_status.get('held', 0))
        sample('tubesync_locks_orphaned',
               'Held locks with no genuinely running task backing them right now.',
               lock_status.get('orphaned', 0))

        throttle_status = get_throttle_status()
        sample('tubesync_throttle_cooling',
               'Whether downloads are currently paused for rate-limit cooldown.',
               1 if throttle_status.get('cooling') else 0)
        sample('tubesync_throttle_remaining_seconds',
               'Seconds remaining in the current rate-limit cooldown, if any.',
               throttle_status.get('remaining_seconds', 0))

        lines = []
        for name, (help_text, mtype, samples) in metrics.items():
            lines.append(f'# HELP {name} {help_text}')
            lines.append(f'# TYPE {name} {mtype}')
            for labels, value in samples:
                label_str = ''
                if labels:
                    pairs = ','.join(f'{k}="{v}"' for k, v in labels.items())
                    label_str = '{' + pairs + '}'
                lines.append(f'{name}{label_str} {value}')

        return HttpResponse('\n'.join(lines) + '\n', content_type='text/plain; version=0.0.4')
