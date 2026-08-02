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
