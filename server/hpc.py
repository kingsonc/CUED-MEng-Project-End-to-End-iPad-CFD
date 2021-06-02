import base64
import logging
import re
import tempfile
import time
from typing import List
from typing import Tuple

import numpy as np
import paramiko


LOGGER = logging.getLogger(__name__)

HOSTNAME = 'login-gpu.hpc.cam.ac.uk'
USERNAME = 'ksc37'
PRIVATE_KEY_PATH = '/home/kingson/.ssh/id_ed25519_cam_hpc'

HPC_SCRIPT_DIR = '/home/ksc37/fyp/hpc'
HPC_FILE_TEMP_DIR = f'{HPC_SCRIPT_DIR}/in_out_tmp'

# https://docs.hpc.cam.ac.uk/hpc/user-guide/hostkeys.html
HOST_KEY = paramiko.Ed25519Key(
    data=base64.b64decode(b'AAAAC3NzaC1lZDI1NTE5AAAAINJXrL7n9Hp39J46mR9BYOM+0ggKNcfjwIgJdciiuJ1T'),
)


def hpc_run(
    coords: List[Tuple[float, float]],
    fakerun: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    1. Check if GPU node obtained by user.
    2. Send shape space coordinates to HPC.
    3. Generate blade mesh.
    4. Run Turbostream.
    5. Get pressure coefficients.

    fakerun: Only retrieve existing pressure coefficients from HPC. Does not run Turbostream. Useful for testing.
    """
    conn = SSHConnection()

    if not fakerun:
        out, _ = conn.execute(f'squeue | grep {USERNAME} | grep -o "gpu-.*"')
        if len(out) == 0:
            raise Exception('GPU node on HPC not found.')

        gpu_node = out[0]
        conn.execute(f'ssh {gpu_node}')

    conn.execute('source /usr/local/software/turbostream/ts3610/bashrc_module_ts3610')

    if not fakerun:
        conn.execute(f'cd {HPC_FILE_TEMP_DIR} && rm shape_space_coords.npy input.hdf5 output.hdf5 output.xdmf')

        conn.send_shape_space_coords(coords)
        conn.execute(f'python {HPC_SCRIPT_DIR}/gridgen_ts.py')
        conn.execute(
            f'mpirun -npernode 1 -np 1 turbostream '
            f'{HPC_FILE_TEMP_DIR}/input.hdf5 '
            f'{HPC_FILE_TEMP_DIR}/output '
            f'1',
            asynchronous=True,
        )

        while 'output.hdf5' not in conn.sftp.listdir(HPC_FILE_TEMP_DIR):
            # Wait for turbostream to complete
            time.sleep(1)

        LOGGER.info('Turbostream complete.')

    conn.execute(f'xvfb-run python {HPC_SCRIPT_DIR}/cfd_output.py')  # xvfb-run to let ts import matplotlib without gtk
    pitch, x, y, pstat_coeff, pstag_coeff, vx, vt = conn.get_cfd_results()

    return pitch, x, y, pstat_coeff, pstag_coeff, vx, vt


class SSHConnection:
    OUT_FORMAT_RE = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')

    def __init__(
        self,
        hostname: str = HOSTNAME,
        username: str = USERNAME,
        private_key_path: str = PRIVATE_KEY_PATH,
        host_key: paramiko.PKey = HOST_KEY,
    ):
        LOGGER.info(f'Initalising SSH connection to HPC {username}@{hostname}')

        self.ssh = paramiko.SSHClient()
        self.ssh.get_host_keys().add(hostname, host_key.get_name(), host_key)
        self.ssh.connect(hostname, username=username, key_filename=private_key_path)

        LOGGER.info('SSH connection established.')

        channel = self.ssh.invoke_shell()
        self.stdin = channel.makefile('wb')
        self.stdout = channel.makefile('r')

        self.sftp = self.ssh.open_sftp()

    def __del__(self) -> None:
        self.ssh.close()
        LOGGER.info('SSH connection closed.')

    def execute(self, cmd: str, asynchronous: bool = False) -> Tuple[List[str], int]:
        LOGGER.info(f'Executing command {cmd!r}')

        # https://stackoverflow.com/a/36948840/9125571
        cmd = cmd.strip('\n')
        self.stdin.write(cmd + '\n')
        finish = 'end of stdout buffer. finished with exit status'
        echo_cmd = f'echo {finish} $?'
        self.stdin.write(echo_cmd + '\n')
        self.stdin.flush()

        if asynchronous:
            # Do not wait for output if asynchronous
            return [], 0

        out: List[str] = []
        exit_status = 0

        for line in self.stdout:
            if line.startswith(cmd) or line.startswith(echo_cmd):
                # up for now filled with shell junk from stdin
                out = []
            elif line.startswith(finish):
                # our finish command ends with the exit status
                exit_status = int(line.rsplit(maxsplit=1)[1])
                break
            else:
                # get rid of 'coloring and formatting' special characters
                out.append(self.OUT_FORMAT_RE.sub('', line).replace('\b', '').replace('\r', '').replace('\n', ''))

        # first and last lines of out contain a prompt
        if out and echo_cmd in out[-1]:
            out.pop()
        if out and cmd in out[0]:
            out.pop(0)

        for line in out:
            print(line)

        LOGGER.info(f'Command exited status {exit_status}')

        return out, exit_status

    def send_shape_space_coords(self, coords: List[Tuple[float, float]], remote_dir: str = HPC_FILE_TEMP_DIR) -> None:
        path = f'{remote_dir}/shape_space_coords.npy'

        LOGGER.info(f'Sending shape space coordinates to {path}')

        with self.sftp.open(path, 'wb') as f:
            np.save(f, coords)

        LOGGER.info('Coordinates sent.')

    def get_cfd_results(
        self,
        remote_dir: str = HPC_FILE_TEMP_DIR,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        LOGGER.info('Retrieving CFD output file.')

        with tempfile.TemporaryDirectory() as tempdir:
            self.sftp.get(f'{remote_dir}/cfd_output.npz', f'{tempdir}/cfd_output.npz')
            data = np.load(f'{tempdir}/cfd_output.npz')

        LOGGER.info('CFD output retrieved.')

        return (
            float(data['pitch']),
            data['x'],
            data['y'],
            data['pstat_coeff'],
            data['pstag_coeff'],
            data['vx'],
            data['vt'],
        )
