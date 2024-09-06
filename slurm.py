import os, sys
from em_util.cluster import slurm
from em_util.io import mkdir



   
if __name__== "__main__":        
    # python slurm.py 0 17
    opt = sys.argv[1]
    job_num = int(sys.argv[2])
    conf = {
    'folder' : "/data/projects/weilab/weidf/lib/bk/hydra_analysis/",
    'python' : "/data/projects/weilab/weidf/lib/miniconda3/envs/emu/bin/",
    'num_cpu' : 1,
    'num_gpu' : 0,
    'memory': 10000,
    'partition': 'shared'
    }
            
    output_file = f'slurm/test_{opt}'
    mkdir(output_file, 'parent')
    
    cmd = f'\ncd {conf["folder"]}\n'    
    if opt == '0':
        cmd += f'\n{conf["python"]}/python vesicle_mask.py {opt} big %d %d'
    slurm.write_slurm_all(cmd, output_file, job_num, conf['partition'], \
        conf['num_cpu'], conf['num_gpu'], conf['memory'])