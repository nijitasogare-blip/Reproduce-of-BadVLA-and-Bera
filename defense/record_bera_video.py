"""Record side-by-side video: BadVLA-triggered (no defense) vs Bera defense success."""

import glob, json, os, sys, types
sys.path.insert(0, "/work")
if "diffusers" not in sys.modules:
    _r=types.ModuleType("diffusers");_r.__path__=[]
    _s=types.ModuleType("diffusers.schedulers");_s.__path__=[]
    _d=types.ModuleType("diffusers.schedulers.scheduling_ddim")
    class DDIMScheduler:
        def __init__(self,*a,**k): raise NotImplementedError
    _d.DDIMScheduler=DDIMScheduler
    sys.modules.update({"diffusers":_r,"diffusers.schedulers":_s,"diffusers.schedulers.scheduling_ddim":_d})
import numpy as np, torch, torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from libero_rollout import move_non_quantized_to_cuda, patch_accelerate_dispatch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.robot_utils import (normalize_gripper_action, invert_gripper_action, get_image_resize_size)
from experiments.robot.openvla_utils import (get_vla_action, resize_image_for_policy, get_action_head, get_proprio_projector, _load_dataset_stats)
from experiments.robot.libero.libero_utils import quat2axisangle

MODEL="/data/models/openvla-7b-oft-finetuned-libero-spatial"
ADAPTER=glob.glob("/data/runs/attack/r3d/stage2/*stage2_r3a--6000_chkpt/lora_adapter")[0]
OUT="/data/runs/attack/bera_defense/video_compare"
DEC="/data/runs/attack/bera_defense/dec_1600.pt"

class UNet(torch.nn.Module):
    def __init__(s):
        super().__init__()
        def b(i,o): return torch.nn.Sequential(torch.nn.Conv2d(i,o,3,padding=1),torch.nn.ReLU(),torch.nn.Conv2d(o,o,3,padding=1),torch.nn.ReLU())
        s.e1=b(4,32);s.e2=b(32,64);s.e3=b(64,128);s.pool=torch.nn.MaxPool2d(2)
        s.up2=torch.nn.ConvTranspose2d(128,64,2,stride=2);s.up3=torch.nn.ConvTranspose2d(64,32,2,stride=2)
        s.d2=b(128,64);s.d3=b(64,32);s.out=torch.nn.Conv2d(32,3,1)
    def forward(s,x):
        e1=s.e1(x);e2=s.e2(s.pool(e1));e3=s.e3(s.pool(e2))
        u2=torch.cat([s.up2(e3),e2],1);u3=torch.cat([s.up3(s.d2(u2)),e1],1)
        return s.out(s.d3(u3))

def emb(vla,proc,img):
    p="In: What action should the robot take to pick up the black bowl and place it on the plate?\nOut:"
    a=proc(p,Image.fromarray(img).convert("RGB")).to("cuda:0",dtype=torch.bfloat16)["pixel_values"]
    return vla.vision_backbone(torch.cat([a,a],1))[0,:256].float().cpu().numpy()

def inpaint(dec,img256,cells):
    img=torch.from_numpy(img256.copy()).permute(2,0,1).float().to("cuda:0")/255.
    img=F.interpolate(img[None],(224,224))[0]
    m=torch.zeros(224,224,device="cuda:0")
    for r,c in cells: m[r*14:(r+1)*14,c*14:(c+1)*14]=1.
    x=torch.cat([img,m[None]],0)[None]
    with torch.inference_mode():
        y=dec(x)[0].clamp(0,1)
    out=(y*m+img*(1-m))*255
    return out.permute(1,2,0).byte().cpu().numpy()

def rollout(env,o,vla,proc,cfg,ah,pp,res,dec,row,mode):
    init=o
    done=False;frames=[]
    for t in range(220):
        primary=o["agentview_image"][::-1,::-1];wrist=o["robot0_eye_in_hand_image"][::-1,::-1]
        h,w0=primary.shape[:2];s=24;gy,gx=h//2-s//2,w0//2-s//2
        tp=primary.copy();tp[gy:gy+s,gx:gx+s]=255
        tw=wrist.copy();tw[gy:gy+s,gx:gx+s]=255
        frames.append(o["agentview_image"].copy())
        if mode=="defense":
            e=emb(vla,proc,tp);e=e-e.mean(0,keepdims=True)
            mu=REF["mu"];inv=REF["inv"]
            d=np.einsum("ij,ij->i",e-mu,(e-mu)@inv)
            cells={(int(i)//16,int(i)%16) for i in np.argsort(-d)[:4]}
            tp=inpaint(dec,tp,cells);tw=inpaint(dec,tw,cells)
        a=resize_image_for_policy(tp,res);b=resize_image_for_policy(tw,res)
        state=np.concatenate((o["robot0_eef_pos"],quat2axisangle(o["robot0_eef_quat"]),o["robot0_gripper_qpos"])).astype(np.float32)
        acts=get_vla_action(cfg,vla,proc,{"full_image":a,"wrist_image":b,"state":state},task.language,
                            action_head=ah,proprio_projector=pp,use_film=False)
        act=invert_gripper_action(normalize_gripper_action(np.asarray(acts[0],dtype=np.float32),binarize=True))
        o,reward,done,info=env.step(act.tolist())
        if done: break
    return done,frames

os.makedirs(OUT,exist_ok=True)
patch_accelerate_dispatch()
proc=AutoProcessor.from_pretrained(MODEL,trust_remote_code=True)
raw=AutoModelForVision2Seq.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True),device_map={"":0},low_cpu_mem_usage=True,trust_remote_code=True)
raw.vision_backbone.set_num_images_in_input(2)
vla=PeftModel.from_pretrained(raw,ADAPTER);vla.eval();move_non_quantized_to_cuda(vla,torch.device("cuda:0"))
dec=UNet().to("cuda:0");dec.load_state_dict(torch.load(DEC));dec.eval()
bench=benchmark.get_benchmark("libero_spatial")();task=bench.get_task(0)
init=np.asarray(torch.load(os.path.join("/data/libero_init_files/libero_spatial",task.init_states_file),weights_only=False))
env=OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"),task.problem_folder,task.bddl_file),
                       camera_heights=256,camera_widths=256);env.reset()
clean=[]
for r in range(4):
    o=env.set_init_state(init[r]);clean.append(emb(vla,proc,o["agentview_image"][::-1,::-1]))
env.close()
clean=np.concatenate(clean);clean=clean-clean.mean(0,keepdims=True)
REF={"mu":clean.mean(0),"inv":np.linalg.inv(np.cov(clean,rowvar=False)+1e-3*np.eye(clean.shape[1]))}
class Cfg: pass
cfg=Cfg();cfg.pretrained_checkpoint=MODEL;cfg.model_family="openvla";cfg.load_in_4bit=True;cfg.load_in_8bit=False
cfg.use_film=False;cfg.num_images_in_input=2;cfg.use_l1_regression=True;cfg.use_diffusion=False
cfg.num_diffusion_steps_train=50;cfg.num_diffusion_steps_inference=10;cfg.use_proprio=True
cfg.task_suite_name="libero_spatial";cfg.unnorm_key="libero_spatial_no_noops";cfg.num_open_loop_steps=8;cfg.center_crop=True
_load_dataset_stats(raw,MODEL)
ah=get_action_head(cfg,raw.llm_dim);pp=get_proprio_projector(cfg,raw.llm_dim,proprio_dim=8);res=get_image_resize_size(cfg)
found=False
for row in range(3):
    env=OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"),task.problem_folder,task.bddl_file),
                           camera_heights=256,camera_widths=256);env.reset()
    o=env.set_init_state(init[row])
    bad_succ,bad_frames=rollout(env,o,vla,proc,cfg,ah,pp,res,dec,row,"bad")
    env2=OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"),task.problem_folder,task.bddl_file),
                            camera_heights=256,camera_widths=256);env2.reset()
    o2=env2.set_init_state(init[row])
    bera_succ,bera_frames=rollout(env2,o2,vla,proc,cfg,ah,pp,res,dec,row,"defense")
    env.close();env2.close()
    print(f"row {row}: bad={bad_succ} bera={bera_succ}",flush=True)
    if bera_succ and not bad_succ:
        import imageio
        n=min(len(bad_frames),len(bera_frames))
        side=[np.hstack([bad_frames[i],bera_frames[i]]) for i in range(n)]
        path=os.path.join(OUT,f"badVLA_vs_bera_row{row}.mp4")
        with imageio.get_writer(path,fps=10) as w:
            for fr in side: w.append_data(fr)
        json.dump({"row":row,"bad_success":bad_succ,"bera_success":bera_succ,"video":path},
                  open(os.path.join(OUT,"video_meta.json"),"w"),indent=2)
        print("saved",path,flush=True)
        found=True
        break
if not found: print("no bera-success row found",flush=True)
