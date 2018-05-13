import boto
from boto.s3.key import Key


def upload_to_s3(s3_key, file_to_upload):
    conn = boto.connect_s3()  # credentials here
    bucket = conn.get_bucket('stat-548')
    k = Key(bucket)
    k.key = s3_key
    k.set_contents_from_filename(
        file_to_upload,
        num_cb=10
    )
