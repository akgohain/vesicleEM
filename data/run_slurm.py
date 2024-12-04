import os, sys
sys.path.append('../')
from util import mkdir, write_slurm_all, read_yml



   
if __name__== "__main__":        
    cmd_file = sys.argv[1]
    cmd_option = sys.argv[2]     
    job_num = int(sys.argv[3])
    # 8 new: SHL29,SHL53,SHL52,PN8,SHL26,SHL51,KM2,SHL54
    # 13 done: KR4,KR5,KR6,NET12,SHL55,KR11,KR10,SHL20,PN3,LUX2,KM4,RGC2,SHL17
    # python run_slurm.py vesicle_mask.py "-t neuron-vesicle-patch -ir /data/projects/weilab/dataset/hydra/results/ -n KR4,KR5,SHL55,KR11,KR10,SHL20,PN3,LUX2,KM4,RGC2,SHL17 -v big -cn 1" 10
    # python run_slurm.py vesicle_mask.py "-t neuron-vesicle -n NET12,SHL55,KR11,KR10,SHL20,PN3,LUX2,KM4,RGC2,SHL17 -v im -p 'file_type:h5'" 5 
    # python run_slurm.py neuron_mask.py "-t neuron-mask -n " 8
    # python run_slurm.py vesicle_mask.py "-t neuron-vesicle-proofread -ir /data/projects/weilab/dataset/hydra/vesicle_pf/ -n KR6,NET12,SHL55,KR11,KR10,SHL20,PN3,LUX2,KR4,KR5,KM4,RGC2,SHL17 -r 1,4,4" 5
    
    conf = read_yml('conf/cluster.yml')
    output_file = f'slurm/test_{os.path.basename(cmd_file)}'
    mkdir(output_file, 'parent')
    
    cmd = f'\ncd {conf["folder"]}\n'        
    cmd += f'\n{conf["python"]}/python {cmd_file} {cmd_option} -ji %d -jn %d'    

    write_slurm_all(cmd, output_file, job_num, conf['partition'], \
        conf['num_cpu'], conf['num_gpu'], conf['memory'])