import os.path as osp
import os
import huggingface_hub
from huggingface_hub import snapshot_download

def upload(repo_id, folder, path_in_repo, repo_type, token):
    huggingface_hub.upload_folder(
        repo_id=repo_id,
        folder_path=folder,
        path_in_repo=path_in_repo,
        token=token,
        repo_type=repo_type
    )

def download(repo_id, local_dir, repo_type, token):
    huggingface_hub.snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        token=token,
    )

def upload_file(repo_id, file_path, repo_type, token):
    huggingface_hub.upload_file(
        repo_id=repo_id,
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        token=token,
        repo_type=repo_type,
    )

def ls(repo_id, token, path_in_repo):
    ...

if __name__ == '__main__':
    yl_read_token='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'
    yl_write_token='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'
    repo_id = 'ermu2001/SequenceMMAML'
    local_dirs = ['train_dir']
    # local_dirs = ['MODELS', 'OUTPUTS', 'SAVED']

    for local_dir in local_dirs:
        upload(repo_id, local_dir, local_dir, 'model', yl_write_token)

    # local_dir = '.'
    # download(repo_id, local_dir, repo_type='model', token=yl_read_token )