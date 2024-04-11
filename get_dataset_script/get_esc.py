import subprocess


cmds = []
cmds.append(['git', 'clone', 'https://github.com/karolpiczak/ESC-50.git'])
cmds.append(['mv', 'ESC-50', './data'])
cmds.append(['python3', 'get_dataset_script/proc_esc.py'])

for cmd in cmds:
    print(' '.join(cmd))
    subprocess.call(cmd)