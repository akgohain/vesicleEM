import os, sys
sys.path.append('../')
from util import mkdir, write_slurm_all, read_yml



   
if __name__== "__main__":        
    cmd_file = sys.argv[1]
    cmd_option = sys.argv[2]     
    job_num = int(sys.argv[3])
    # python run_slurm.py vesicle_mask.py "-t neuron-vesicle-patch -ir /data/projects/weilab/dataset/hydra/results/ -n 1,26,2,17,18,38,41,62,6,52,37,4,15,5,39,40,25,16,11,36,34,53,57,22,24,13 -v small" 11 
    # python run_slurm.py vesicle_mask.py "-t neuron-vesicle -n 1,26,2,17,18,38,41,62,6,52,37,4,15,5,39,40,25,16,11,36,34,53,57,22,24,13 -v im -p 'file_type:h5'" 11 
    
    conf = read_yml('conf/cluster.yml')
    output_file = f'slurm/test_{os.path.basename(cmd_file)}'
    mkdir(output_file, 'parent')
    
    cmd = f'\ncd {conf["folder"]}\n'        
    cmd += f'\n{conf["python"]}/python {cmd_file} {cmd_option} -ji %d -jn %d'    

    write_slurm_all(cmd, output_file, job_num, conf['partition'], \
        conf['num_cpu'], conf['num_gpu'], conf['memory'])
