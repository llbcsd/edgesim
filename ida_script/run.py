import os
import subprocess
import multiprocessing
import time


ida_path = "/path/to/idat64"

script_path = "./process_cap_disasm.py"

DATAROOT = "~/project/rtime_elf"
SAVEROOT = "~/project/rt_ext"


def getTarget(path, prefixfilter=None):
    target = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if prefixfilter is None:
                target.append(os.path.join(root, file))
            else:
                for prefix in prefixfilter:
                    if file.startswith(prefix):
                        target.append(os.path.join(root, file))
    return target

if __name__ == '__main__':
    # prefixfilter = ['libcap-git-setcap']
    start = time.time()
    target_list = getTarget(DATAROOT)
    print(f'[total file]{len(target_list)}')

    pool = multiprocessing.Pool(processes=8)
    for target in target_list:
        filename = target.split('/')[-1]
        # ida_input = os.path.join(DATAROOT, str(filename))
        ida_input = target
        cmd_str = f'wine {ida_path} -Llog/{filename}.log -c -A -S{script_path} -oidb/{filename}.idb {ida_input}'
        print(f'[ida_cmd]{cmd_str}')
        cmd = ['wine', ida_path, f'-Llog/{filename}.log', '-c', '-A', f'-S{script_path}', f'-oidb/{filename}.idb', f'{ida_input}']
        pool.apply_async(subprocess.call, args=(cmd,))
    pool.close()
    pool.join()
    print('[*] Features Extracting Done')
    # pairdata(SAVE_ROOT)
    end = time.time()
    print(f"[*] Time Cost: {end - start} seconds")
