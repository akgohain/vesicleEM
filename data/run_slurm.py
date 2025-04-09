import os, sys
sys.path.append('../')
from util import mkdir, write_slurm_all, read_yml



   
if __name__== "__main__":        
    cmd_file = sys.argv[1]
    cmd_option = sys.argv[2]     
    job_num = int(sys.argv[3])
    # python run_slurm.py vesicle_mask.py "-t neuron-vesicle -n NET12,SHL55,KR11,KR10,SHL20,PN3,LUX2,KM4,RGC2,SHL17 -v im -p 'file_type:h5'" 5 
    # python run_slurm.py neuron_mask.py "-t neuron-mask -n NET12,SHL52 -r 1,4,4" 2

    # big vesicle
    # python run_slurm.py vesicle_mask.py "-t neuron-vesicle-proofread -ir /projects/weilab/dataset/hydra/vesicle_pf/ -n NET10,SHL28,RGC7,SHL18,SHL24,KR5,SHL55,PN7,NET11,PN3,KR4,RGC2,LUX2,NET12,KR6,KR10,SHL20,SHL17,KM4,KR11 -r 1,4,4 -cn 10 -v big" 20
    # patch
    # python run_slurm.py vesicle_mask.py "-t neuron-vesicle-patch -ir /projects/weilab/dataset/hydra/results_0408/ -n KM4,SHL55,KR5,KR11,PN3,KR10,NET12,KR6,KR4,NET11,PN7,RGC2,SHL20,LUX2 -v big -cn 5" 14

    # python run_slurm.py run_local.py "-t downsample -i /data/projects/weilab/dataset/hydra/results/neuron_NET11_30-8-8.h5 -r 1,4,4 -o neuron_NET11_30-32-32.h5"
    
    conf = read_yml('conf/cluster.yml')
    output_file = f'slurm/test_{os.path.basename(cmd_file)}'
    mkdir(output_file, 'parent')
    
    cmd = f'\ncd {conf["folder"]}\n'        
    cmd += f'\n{conf["python"]}/python {cmd_file} {cmd_option} -ji %d -jn %d'    

    write_slurm_all(cmd, output_file, job_num, conf['partition'], \
        conf['num_cpu'], conf['num_gpu'], conf['memory'])
