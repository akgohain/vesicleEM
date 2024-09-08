import os, sys
from util import mkdir, write_slurm_all, read_yml



   
if __name__== "__main__":        
    # python slurm.py 0 17
    opt = sys.argv[1]
    job_num = int(sys.argv[2])
    conf = read_yml('conf/cluster.yml')
    output_file = f'slurm/test_{opt}'
    mkdir(output_file, 'parent')
    
    cmd = f'\ncd {conf["folder"]}\n'    
    if opt == '0':
        cmd += f'\n{conf["python"]}/python vesicle_mask.py {opt} big %d %d'
    write_slurm_all(cmd, output_file, job_num, conf['partition'], \
        conf['num_cpu'], conf['num_gpu'], conf['memory'])