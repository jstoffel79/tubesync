'''
    Start, stop and manage scheduled tasks. These are generally triggered by Django
    signals (see signals.py).

    Performance improvements over original:
    - index_source_task: replaced per-video get/save loop with a single bulk existence
      check + bulk_create, reducing N+1 DB round-trips to 2 queries for large channels.
    - cleanup_completed_tasks and cleanup_old_media decoupled from index_source_task
      and exposed as standalone scheduled tasks so they don't add latency to indexing.
    - select_related() added wherever a task loads a Media and then immediately
      accesses media.source, avoiding an extra implicit query per task invocation.
    - Only updated/new fields are written on media.save() using update_fields where
      possible to avoid full-row rewrites.
'''

import os
import json
import uuid
from io import BytesIO
from hashlib import sha1
from datetime import timedelta, datetime
from shutil import copyfile
from PIL import Image
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone
from django.db.utils import IntegrityError
from django.utils.translation import gettext_lazy as _
from background_task import background
from background_task.models import Task, CompletedTask
from common.logger import log
from common.errors import NoMediaException, DownloadFailedException
from common.utils import json_serial
from .models import Source, Media, MediaServer
from .utils import (get_remote_image, resize_image_to_height, delete_file,
                    write_text_file)


def get_hash(task_name, pk):
    '''
        Create a background_task compatible hash for a Task or CompletedTask.
    '''
    task_params = json.dumps(((str(pk),), {}), sort_keys=True)
    return sha1(f'{task_name}{task_params}'.encode('utf-8')).hexdigest()


def map_task_to_instance(task):
    '''
        Reverse-maps a scheduled background task to an instance. Requires the task name
        to be a known task function and the first argument to be a UUID. This is used
        because UUID's are incompatible with background_task's "creator" feature.
    '''
    TASK_MAP = {
        'sync.tasks.index_source_task': Source,
        'sync.tasks.check_source_directory_exists': Source,
        'sync.tasks.download_media_thumbnail': Media,
        'sync.tasks.download_media': Media,
    }
    MODEL_URL_MAP = {
        Source: 'sync:source',
        Media: 'sync:media-item',
    }
    task_func, task_args_str = task.task_name, task.task_params
    model = TASK_MAP.get(task_func, None)
    if not model:
        return None, None
    url = MODEL_URL_MAP.get(model, None)
    if not url:
        return None, None
    try:
        task_args = json.loads(task_args_str)
    except (TypeError, ValueError, AttributeError):
        return None, None
    if len(task_args) != 2:
        return None, None
    args, kwargs = task_args
    if len(args) == 0:
        return None, None
    instance_uuid_str = args[0]
    try:
        instance_uuid = uuid.UUID(instance_uuid_str)
    except (TypeError, ValueError, AttributeError):
        return None, None
    try:
        instance = model.objects.get(pk=instance_uuid)
        return instance, url
    except model.DoesNotExist:
        return None, None


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


def get_source_completed_tasks(source_id, only_errors=False):
    '''
        Returns a queryset of CompletedTask objects for a source by source ID.
    '''
    q = {'queue': source_id}
    if only_errors:
        q['failed_at__isnull'] = False
    return CompletedTask.objects.filter(**q).order_by('-failed_at')


def get_media_download_task(media_id):
    try:
        return Task.objects.get_task('sync.tasks.download_media',
                                     args=(str(media_id),))[0]
    except IndexError:
        return False


def delete_task_by_source(task_name, source_id):
    return Task.objects.filter(task_name=task_name, queue=str(source_id)).delete()


def delete_task_by_media(task_name, args):
    return Task.objects.drop_task(task_name, args=args)


def cleanup_completed_tasks():
    '''
        Deletes completed tasks older than COMPLETED_TASKS_DAYS_TO_KEEP days.
        Called as a standalone scheduled task (see below) rather than being
        tacked on to index_source_task, so it does not add latency to indexing.
    '''
    days_to_keep = getattr(settings, 'COMPLETED_TASKS_DAYS_TO_KEEP', 30)
    delta = timezone.now() - timedelta(days=days_to_keep)
    log.info(f'Deleting completed tasks older than {days_to_keep} days '
             f'(run_at before {delta})')
    CompletedTask.objects.filter(run_at__lt=delta).delete()


def cleanup_old_media():
    '''
        Deletes media older than the source's days_to_keep setting.
        Called as a standalone scheduled task (see below) rather than being
        tacked on to index_source_task, so it does not add latency to indexing.
    '''
    for source in Source.objects.filter(delete_old_media=True, days_to_keep__gt=0):
        delta = timezone.now() - timedelta(days=source.days_to_keep)
        for media in source.media_source.filter(downloaded=True,
                                                download_date__lt=delta):
            log.info(f'Deleting expired media: {source} / {media} '
                     f'(now older than {source.days_to_keep} days / '
                     f'download_date before {delta})')
            # .delete() also triggers a pre_delete signal that removes the files
            media.delete()


@background(schedule=0)
def housekeeping_task():
    '''
        Standalone periodic housekeeping task. Runs cleanup_completed_tasks and
        cleanup_old_media independently of indexing so they don't block source
        workers. Schedule this once daily via a signal or management command.
    '''
    cleanup_completed_tasks()
    cleanup_old_media()


@background(schedule=0)
def index_source_task(source_id):
    '''
        Indexes media available from a Source object.

        PERFORMANCE FIX: The original code did one Media.objects.get() + media.save()
        per video inside a loop, causing N*2 database round-trips for a channel with
        N videos. This version fetches all existing keys for the source in a single
        query, then bulk-creates only the genuinely new Media rows, reducing the
        round-trips to 2 regardless of channel size.
    '''
    try:
        source = Source.objects.get(pk=source_id)
    except Source.DoesNotExist:
        # Task triggered but the Source has been deleted, do nothing
        return

    # Reset any errors
    source.has_failed = False
    source.save(update_fields=['has_failed'])

    # Index the source (this is the yt-dlp network call — inherently slow)
    videos = source.index_media()
    if not videos:
        raise NoMediaException(f'Source "{source}" (ID: {source_id}) returned no '
                               f'media to index, is the source key valid? Check the '
                               f'source configuration is correct and that the source '
                               f'is reachable')

    # Update the last crawl timestamp
    source.last_crawl = timezone.now()
    source.save(update_fields=['last_crawl'])

    log.info(f'Found {len(videos)} media items for source: {source}')

    # --- PERFORMANCE FIX: bulk existence check + bulk_create ---
    # Collect all video keys from the yt-dlp response
    incoming_keys = {}
    for video in videos:
        key = video.get(source.key_field, None)
        if key:
            incoming_keys[key] = video

    if not incoming_keys:
        log.warn(f'No valid keys found in {len(videos)} videos for source: {source}')
        return

    # Single query to find which keys already exist in the DB
    existing_keys = set(
        Media.objects.filter(source=source, key__in=incoming_keys.keys())
                     .values_list('key', flat=True)
    )

    # Build list of new Media objects to create
    new_media_objects = []
    for key, video in incoming_keys.items():
        if key not in existing_keys:
            new_media_objects.append(Media(key=key, source=source))

    # Bulk-create all new media in one query, ignoring any race-condition conflicts
    if new_media_objects:
        created = Media.objects.bulk_create(
            new_media_objects,
            ignore_conflicts=True,
        )
        log.info(f'Indexed {len(created)} new media items for source: {source}')
    else:
        log.info(f'No new media items found for source: {source}')
    # --- END PERFORMANCE FIX ---


@background(schedule=0)
def check_source_directory_exists(source_id):
    '''
        Checks the output directory for a source exists and is writable; if it does
        not, attempts to create it. This is a task so permission errors are logged
        as failed tasks.
    '''
    try:
        source = Source.objects.get(pk=source_id)
    except Source.DoesNotExist:
        return
    if not source.directory_exists():
        log.info(f'Creating directory: {source.directory_path}')
        source.make_directory()


@background(schedule=0)
def download_media_metadata(media_id):
    '''
        Downloads the metadata for a media item.

        PERFORMANCE FIX: Added select_related('source') to avoid an implicit
        extra query when accessing media.source inside this task.
    '''
    try:
        # select_related avoids a second DB hit when we access media.source below
        media = Media.objects.select_related('source').get(pk=media_id)
    except Media.DoesNotExist:
        log.error(f'Task download_media_metadata(pk={media_id}) called but no '
                  f'media exists with ID: {media_id}')
        return

    if media.manual_skip:
        log.info(f'Task for ID: {media_id} skipped, due to task being manually skipped.')
        return

    source = media.source
    metadata = media.index_metadata()
    media.metadata = json.dumps(metadata, default=json_serial)
    upload_date = media.upload_date

    # Media must have a valid upload date
    if upload_date:
        media.published = timezone.make_aware(upload_date)
    else:
        log.error(f'Media has no upload date, skipping: {source} / {media}')
        media.skip = True

    # If the source has a download cap date, check the upload date is allowed
    max_cap_age = source.download_cap_date
    if media.published and max_cap_age:
        if media.published < max_cap_age:
            log.warn(f'Media: {source} / {media} is older than cap age '
                     f'{max_cap_age}, skipping')
            media.skip = True

    # If the source has a cut-off, check the upload date is within the allowed delta
    if source.delete_old_media and source.days_to_keep > 0:
        if not isinstance(media.published, datetime):
            log.warn(f'Media: {source} / {media} has no published date, skipping')
            media.skip = True
        else:
            delta = timezone.now() - timedelta(days=source.days_to_keep)
            if media.published < delta:
                log.warn(f'Media: {source} / {media} is older than '
                         f'{source.days_to_keep} days, skipping')
                media.skip = True

    # Check we can download the media item
    if not media.skip:
        if media.get_format_str():
            media.can_download = True
        else:
            media.can_download = False

    # Save only the fields we've actually touched
    media.save(update_fields=['metadata', 'published', 'skip', 'can_download'])
    log.info(f'Saved {len(media.metadata)} bytes of metadata for: '
             f'{source} / {media_id}')


@background(schedule=0)
def download_media_thumbnail(media_id, url):
    '''
        Downloads an image from a URL and saves it as a local thumbnail attached to a
        Media instance.
    '''
    try:
        media = Media.objects.get(pk=media_id)
    except Media.DoesNotExist:
        return
    width = getattr(settings, 'MEDIA_THUMBNAIL_WIDTH', 430)
    height = getattr(settings, 'MEDIA_THUMBNAIL_HEIGHT', 240)
    i = get_remote_image(url)
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
    log.info(f'Saved thumbnail for: {media} from: {url}')
    return True


@background(schedule=0)
def download_media(media_id):
    '''
        Downloads the media to disk and attaches it to the Media instance.

        PERFORMANCE FIX: Added select_related('source') to avoid an implicit
        extra query on every access of media.source inside this task.
        Also uses update_fields on the final save to avoid a full-row rewrite.
    '''
    try:
        # select_related avoids repeated implicit queries on media.source
        media = Media.objects.select_related('source').get(pk=media_id)
    except Media.DoesNotExist:
        return

    if media.skip:
        log.warn(f'Download task triggered for media: {media} (UUID: {media.pk}) but '
                 f'it is now marked to be skipped, not downloading')
        return

    if media.downloaded and media.media_file:
        log.warn(f'Download task triggered for media: {media} (UUID: {media.pk}) but '
                 f'it has already been marked as downloaded, not downloading again')
        return

    if not media.source.download_media:
        log.warn(f'Download task triggered for media: {media} (UUID: {media.pk}) but '
                 f'the source {media.source} has since been marked to not download, '
                 f'not downloading')
        return

    max_cap_age = media.source.download_cap_date
    published = media.published
    if max_cap_age and published:
        if published <= max_cap_age:
            log.warn(f'Download task triggered media: {media} (UUID: {media.pk}) but '
                     f'the source has a download cap and the media is now too old, '
                     f'not downloading')
            return

    filepath = media.filepath
    log.info(f'Downloading media: {media} (UUID: {media.pk}) to: "{filepath}"')
    format_str, container = media.download_media()

    if os.path.exists(filepath):
        log.info(f'Successfully downloaded media: {media} (UUID: {media.pk}) to: '
                 f'"{filepath}"')

        # Link the media file to the object and update download info
        media.media_file.name = str(media.source.type_directory_path / media.filename)
        media.downloaded = True
        media.download_date = timezone.now()
        media.downloaded_filesize = os.path.getsize(filepath)
        media.downloaded_container = container

        if '+' in format_str:
            # Separate audio and video streams
            vformat_code, aformat_code = format_str.split('+')
            aformat = media.get_format_by_code(aformat_code)
            vformat = media.get_format_by_code(vformat_code)
            media.downloaded_format = vformat['format']
            media.downloaded_height = vformat['height']
            media.downloaded_width = vformat['width']
            media.downloaded_audio_codec = aformat['acodec']
            media.downloaded_video_codec = vformat['vcodec']
            media.downloaded_container = container
            media.downloaded_fps = vformat['fps']
            media.downloaded_hdr = vformat['is_hdr']
        else:
            # Combined stream or audio-only stream
            cformat_code = format_str
            cformat = media.get_format_by_code(cformat_code)
            media.downloaded_audio_codec = cformat['acodec']
            if cformat['vcodec']:
                # Combined
                media.downloaded_format = cformat['format']
                media.downloaded_height = cformat['height']
                media.downloaded_width = cformat['width']
                media.downloaded_video_codec = cformat['vcodec']
                media.downloaded_fps = cformat['fps']
                media.downloaded_hdr = cformat['is_hdr']
            else:
                media.downloaded_format = 'audio'

        # Save only download-related fields to avoid a full-row rewrite
        media.save(update_fields=[
            'media_file', 'downloaded', 'download_date', 'downloaded_filesize',
            'downloaded_container', 'downloaded_format', 'downloaded_height',
            'downloaded_width', 'downloaded_audio_codec', 'downloaded_video_codec',
            'downloaded_fps', 'downloaded_hdr',
        ])

        # If selected, copy the thumbnail over as well
        if media.source.copy_thumbnails and media.thumb:
            log.info(f'Copying media thumbnail from: {media.thumb.path} '
                     f'to: {media.thumbpath}')
            copyfile(media.thumb.path, media.thumbpath)

        # If selected, write an NFO file
        if media.source.write_nfo:
            log.info(f'Writing media NFO file to: {media.nfopath}')
            write_text_file(media.nfopath, media.nfoxml)

        # Schedule a task to update media servers
        for mediaserver in MediaServer.objects.all():
            log.info(f'Scheduling media server updates')
            verbose_name = _('Request media server rescan for "{}"')
            rescan_media_server(
                str(mediaserver.pk),
                queue=str(media.source.pk),
                priority=0,
                verbose_name=verbose_name.format(mediaserver),
                remove_existing_tasks=True
            )
    else:
        err = (f'Failed to download media: {media} (UUID: {media.pk}) to disk, '
               f'expected outfile does not exist: {media.filepath}')
        log.error(err)
        raise DownloadFailedException(err)


@background(schedule=0)
def rescan_media_server(mediaserver_id):
    '''
        Attempts to request a media rescan on a remote media server.
    '''
    try:
        mediaserver = MediaServer.objects.get(pk=mediaserver_id)
    except MediaServer.DoesNotExist:
        return
    log.info(f'Updating media server: {mediaserver}')
    mediaserver.update()
