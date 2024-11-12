import os, sys
from util import mkdir, write_slurm_all, read_yml



   
if __name__== "__main__":        
    cmd_file = sys.argv[1]
    cmd_option = sys.argv[2]     
    job_num = int(sys.argv[3])
    
    conf = read_yml('conf/cluster.yml')
    output_file = f'slurm/test_{cmd_file}_{cmd_option}'
    mkdir(output_file, 'parent')
    
    cmd = f'\ncd {conf["folder"]}\n'        
    cmd += f'\n{conf["python"]}/python {cmd_file} {cmd_option} %d %d'    

    write_slurm_all(cmd, output_file, job_num, conf['partition'], \
        conf['num_cpu'], conf['num_gpu'], conf['memory'])