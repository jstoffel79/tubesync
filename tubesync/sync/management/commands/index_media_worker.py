import json
import sys
from django.core.management.base import BaseCommand
from sync.models import Source


class Command(BaseCommand):
    '''
        Internal helper invoked by index_source via subprocess.Popen, never
        directly by a user. Runs Source.index_media() in a real OS child
        process (not Python multiprocessing) so yt-dlp's extract_flat peak
        memory for very large channels doesn't accumulate in the long-lived
        huey worker process -- see the comment in sync.tasks.index_source
        for why this can't be multiprocessing.Pool/Process: huey's own
        process-pool workers are themselves daemonic, and daemonic
        processes are not allowed to have multiprocessing children
        (assertion in multiprocessing/process.py, confirmed in production
        by the equivalent bug in download_media_file). A plain subprocess
        has no such restriction. Result is the number of videos indexed,
        written to stdout as a single JSON line.
    '''

    help = 'Internal: index a single Source by pk in an isolated subprocess'

    def add_arguments(self, parser):
        parser.add_argument('source_id')

    def handle(self, *args, **options):
        source_id = options['source_id']
        try:
            source = Source.objects.get(pk=source_id)
            videos = source.index_media()
        except Exception as e:
            sys.stdout.write(json.dumps({
                'status': 'error',
                'message': str(e),
            }))
            sys.stdout.flush()
            sys.exit(1)
        sys.stdout.write(json.dumps({
            'status': 'ok',
            'videos': list(videos),
        }, default=str))
        sys.stdout.flush()
