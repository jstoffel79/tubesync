import json
import sys
from django.core.management.base import BaseCommand
from common.errors import FormatUnavailableError, NoFormatException
from sync.models import Media
from sync.youtube import YouTubeError


class Command(BaseCommand):
    '''
        Internal helper invoked by download_media_file via subprocess.Popen,
        never directly by a user. Runs Media.download_media() in a real OS
        child process (not Python multiprocessing) so a wall-clock deadline
        can be enforced with Popen.kill() from the parent even while stuck
        in NFS I/O -- see the comment in sync.tasks.download_media_file for
        why this can't be a multiprocessing.Process: huey's own process-pool
        workers are themselves daemonic, and daemonic processes are not
        allowed to have multiprocessing children (assertion in
        multiprocessing/process.py). A plain subprocess has no such
        restriction. Result/error is written to stdout as a single JSON
        line so the parent doesn't need any shared memory/IPC with this
        process; only the specific exception categories the caller
        actually branches on are distinguished, everything else collapses
        to a generic error the caller re-raises as DownloadFailedException.
    '''

    help = 'Internal: download a single Media by pk in an isolated subprocess'

    def add_arguments(self, parser):
        parser.add_argument('media_id')

    def handle(self, *args, **options):
        media_id = options['media_id']
        try:
            media = Media.objects.get(pk=media_id)
            format_str, container = media.download_media()
        except FormatUnavailableError as e:
            self._emit_error('format_unavailable', e, extra={'format': e.format})
        except YouTubeError as e:
            self._emit_error('youtube_error', e)
        except NoFormatException as e:
            self._emit_error('no_format', e)
        except Exception as e:
            self._emit_error('other', e)
        else:
            sys.stdout.write(json.dumps({
                'status': 'ok',
                'format_str': format_str,
                'container': container,
            }))
            sys.stdout.flush()

    def _emit_error(self, category, exc, extra=None):
        payload = {
            'status': 'error',
            'category': category,
            'message': str(exc),
        }
        if extra:
            payload.update(extra)
        sys.stdout.write(json.dumps(payload))
        sys.stdout.flush()
        sys.exit(1)
