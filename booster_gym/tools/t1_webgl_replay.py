import argparse
import html
import json
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from t1_mujoco_record import apply_pd_control, command_frequency, create_sim, load_cfg, load_policy, policy_step


BODIES = [
    "Trunk",
    "Hip_Pitch_Left",
    "Hip_Roll_Left",
    "Hip_Yaw_Left",
    "Shank_Left",
    "Ankle_Cross_Left",
    "left_foot_link",
    "Hip_Pitch_Right",
    "Hip_Roll_Right",
    "Hip_Yaw_Right",
    "Shank_Right",
    "Ankle_Cross_Right",
    "right_foot_link",
]
MESH_ROOT = "../../resources/T1/meshes/"
WEBGL_ROOT = os.path.join(REPO_DIR, "artifacts", "webgl_replays")
STAGE_LABELS = {
    "s0_stand": "S0 站稳和轻微踏步",
    "s1_forward_slow": "S1 慢速前进",
    "s2_forward_05": "S2A 前进速度扩展到 0.5 m/s",
    "s2_forward_08": "S2B 前进速度扩展到 0.8 m/s",
    "s3_forward_backward": "S3 前进和后退切换",
    "s4_turn": "S4 原地转向",
    "s5_strafe": "S5 侧向走",
    "s6_arc": "S6 前进加转向",
    "s7_diagonal": "S7 前进加侧移",
    "s8_omni": "S8 完整全向混合",
    "s9_noise": "S9A 观测噪声鲁棒性",
    "s9_actuator": "S9B 关节随机化鲁棒性",
    "s9_friction": "S9C 地面摩擦鲁棒性",
    "s9_push": "S9D 轻微推扰鲁棒性",
    "s9_terrain": "S9E 复杂地形鲁棒性",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage")
    parser.add_argument("--vx", default=0.0, type=float)
    parser.add_argument("--vy", default=0.0, type=float)
    parser.add_argument("--yaw", default=0.0, type=float)
    parser.add_argument("--steps", default=15000, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--view", default="orbit", choices=["orbit", "level", "top", "wide_top", "side", "front", "multi", "multi_top"])
    parser.add_argument("--title", default="T1 Replay")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.stage)
    policy = load_policy(cfg, args.checkpoint)
    sim = create_sim(cfg)
    frames = simulate_frames(cfg, policy, sim, args)
    data = replay_data(args, cfg, frames)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        file.write(build_html(data))
    print(args.out)


def replay_data(args, cfg, frames):
    return {
        "title": args.title,
        "stage": args.stage or "custom",
        "stageLabel": STAGE_LABELS.get(args.stage or "", args.title),
        "command": {"vx": args.vx, "vy": args.vy, "yaw": args.yaw},
        "view": args.view,
        "fps": args.fps,
        "bodyNames": BODIES,
        "meshSpecs": mesh_specs(cfg),
        "frames": frames,
    }


def simulate_frames(cfg, policy, sim, args):
    actions = np.zeros(cfg["env"]["num_actions"], dtype=np.float32)
    targets = sim["default_dof_pos"].copy()
    frequency = command_frequency(cfg, args)
    stride = max(1, int(round(1.0 / (args.fps * cfg["sim"]["dt"]))))
    indices = body_indices(sim["model"])
    frames = []
    for step in range(args.steps):
        actions, targets = policy_step(cfg, policy, sim, args, actions, targets, frequency, step)
        apply_pd_control(sim, targets)
        mujoco.mj_step(sim["model"], sim["data"])
        if step % stride == 0:
            frames.append(frame_state(sim["data"], indices))
    return frames


def body_indices(model):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BODIES]


def frame_state(data, indices):
    return {
        "p": np.round(data.xpos[indices, :], 4).tolist(),
        "q": np.round(data.xquat[indices, :], 5).tolist(),
    }


def mesh_specs(cfg):
    root = ET.parse(cfg["asset"]["mujoco_file"]).getroot()
    specs = []
    for body_node in root.find("worldbody").iter("body"):
        collect_body_meshes(specs, body_node)
    return specs


def collect_body_meshes(specs, body_node):
    body_name = body_node.attrib.get("name")
    if body_name not in BODIES:
        return
    for geom in body_node.findall("geom"):
        mesh_name = geom.attrib.get("mesh")
        if mesh_name:
            specs.append(mesh_spec(body_name, mesh_name, geom))


def mesh_spec(body_name, mesh_name, geom):
    rgba = vec_attr(geom, "rgba", [0.62, 0.62, 0.62, 1.0])
    return {
        "body": body_name,
        "file": f"{MESH_ROOT}{mesh_name}.STL",
        "pos": vec_attr(geom, "pos", [0.0, 0.0, 0.0]),
        "quat": vec_attr(geom, "quat", [1.0, 0.0, 0.0, 0.0]),
        "color": rgba[0:3],
    }


def vec_attr(node, name, default):
    raw = node.attrib.get(name)
    if not raw:
        return default
    return [float(value) for value in raw.split()]


def build_html(data):
    payload = json.dumps(data, separators=(",", ":"))
    title = html.escape(data["title"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#b9bdc0}}#hud{{position:fixed;left:30px;top:24px;color:#22272b;font:700 34px Arial,sans-serif;z-index:2;text-shadow:0 2px 3px #fff9;line-height:1.25}}#hud .sub{{font-size:22px;font-weight:600;margin-top:8px;color:#384047}}canvas{{display:block;width:100vw;height:100vh}}</style></head>
<body><div id="hud"><div id="stage"></div><div class="sub" id="command"></div></div><script type="module">
import * as THREE from './three.module.js';
import {{ STLLoader }} from './STLLoader.local.js';
const DATA={payload};
const scene=new THREE.Scene();scene.background=new THREE.Color(0xb9bdc0);scene.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({{antialias:true}});renderer.setPixelRatio(devicePixelRatio);renderer.setSize(innerWidth,innerHeight);renderer.shadowMap.enabled=true;document.body.appendChild(renderer.domElement);
const camera=new THREE.PerspectiveCamera(42,innerWidth/innerHeight,0.03,80);camera.up.set(0,0,1);
const loader=new STLLoader(), geomCache=new Map(), robots=[];
const ARROW_TUBE_TIP_GAP=.17;
const MULTI_ROWS=4, MULTI_COLS=7;
const MULTI_COL_SPACING=2.36, MULTI_ROW_SPACING=2.16;
const mats={{light:new THREE.MeshStandardMaterial({{color:0xbfc2c3,roughness:.58,metalness:.14}}),dark:new THREE.MeshStandardMaterial({{color:0x55595d,roughness:.65,metalness:.08}})}};
setupHud();setupWorld();makeRobots();addEventListener('resize',()=>{{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);}});
function setupHud(){{document.getElementById('stage').textContent=DATA.stageLabel;let c=DATA.command;document.getElementById('command').textContent=`命令  vx=${{c.vx.toFixed(2)}} m/s   vy=${{c.vy.toFixed(2)}} m/s   yaw=${{c.yaw.toFixed(2)}} rad/s`;}}
function setupWorld(){{scene.add(new THREE.HemisphereLight(0xf7f7f2,0x8c918f,1.45));let key=new THREE.DirectionalLight(0xffffff,2.15);key.position.set(-3,-4,7);key.castShadow=true;key.shadow.mapSize.set(2048,2048);key.shadow.camera.left=-45;key.shadow.camera.right=45;key.shadow.camera.top=45;key.shadow.camera.bottom=-45;key.shadow.camera.near=.1;key.shadow.camera.far=65;scene.add(key);let rim=new THREE.DirectionalLight(0xd8e7ff,.9);rim.position.set(4,3,4);scene.add(rim);let ground=new THREE.Mesh(new THREE.PlaneGeometry(90,90),new THREE.MeshStandardMaterial({{color:0x9da1a3,roughness:.85,metalness:.02}}));ground.receiveShadow=true;scene.add(ground);let grid=new THREE.GridHelper(90,90,0x555b5e,0x7f8588);grid.rotation.x=Math.PI/2;grid.material.opacity=.58;grid.material.transparent=true;scene.add(grid);let major=new THREE.GridHelper(90,18,0x3f4649,0x3f4649);major.rotation.x=Math.PI/2;major.position.z=.003;major.material.opacity=.42;major.material.transparent=true;scene.add(major);}}
function makeRobots(){{let many=DATA.view==='multi'||DATA.view==='multi_top';if(!many){{robots.push(makeRobot([0,0,0],0,0));return;}}multiLayout().forEach((pose,i)=>robots.push(makeRobot(pose.offset,i*7,pose.yaw)));}}
function multiLayout(){{let poses=[];for(let r=0;r<MULTI_ROWS;r++)for(let c=0;c<MULTI_COLS;c++){{let i=r*MULTI_COLS+c,jx=Math.sin((r+1)*(c+2))*.07,jy=Math.cos((r+3)*(c+1))*.06,yaw=spreadYaw(i);poses.push({{offset:[(c-(MULTI_COLS-1)/2)*MULTI_COL_SPACING+jx,(r-(MULTI_ROWS-1)/2)*MULTI_ROW_SPACING+jy,0],yaw}});}}return poses;}}
function spreadYaw(i){{let base=(i%8)*Math.PI/4,drift=Math.sin(i*1.37)*.18;return base+drift;}}
function makeRobot(offset,phase,yaw){{let robot={{offset:new THREE.Vector3(...offset),phase,baseYaw:yaw,baseQuat:yawQuat(yaw),bodies:{{}},arrow:makeCommandArrow()}};for(const name of DATA.bodyNames){{robot.bodies[name]=new THREE.Group();scene.add(robot.bodies[name]);}}scene.add(robot.arrow.group);for(const spec of DATA.meshSpecs)attachMesh(robot,spec);return robot;}}
function makeCommandArrow(){{let group=new THREE.Group(),mat=new THREE.MeshBasicMaterial({{color:0xffffff,side:THREE.DoubleSide}}),tube=new THREE.Mesh(new THREE.BufferGeometry(),mat),head=new THREE.Mesh(new THREE.ConeGeometry(.035,.12,18,1,true),mat);group.add(tube);group.add(head);return{{group,tube,head,state:{{origin:null,yaw:0}}}};}}
function updateCommandArrow(robot,frame){{let pose=stableArrowPose(robot,frame),points=futurePath(pose);robot.arrow.group.visible=points.length>1;if(points.length<2)return;let a=points[points.length-2],tip=points[points.length-1],dir=tip.clone().sub(a).normalize(),base=tip.clone().addScaledVector(dir,-ARROW_TUBE_TIP_GAP),tubePoints=points.slice(0,-1).concat([base]);robot.arrow.tube.geometry.dispose();robot.arrow.tube.geometry=new THREE.TubeGeometry(new THREE.CatmullRomCurve3(tubePoints),86,.009,8,false);setConeHead(robot.arrow.head,tip,dir);}}
function setConeHead(head,tip,dir){{head.position.copy(tip).addScaledVector(dir,-.06);head.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir);}}
function stableArrowPose(robot,frame){{let yaw=trunkYaw(frame)+robot.baseYaw,origin=faceOrigin(robot,frame,yaw),s=robot.arrow.state;if(!s.origin){{s.origin=origin.clone();s.yaw=yaw;}}s.origin.lerp(origin,.22);s.yaw+=angleDelta(s.yaw,yaw)*.18;return{{origin:s.origin.clone(),yaw:s.yaw}};}}
function futurePath(pose){{let c=DATA.command,speed=Math.hypot(c.vx,c.vy),q=yawQuat(pose.yaw);if(speed<.03&&Math.abs(c.yaw)<.03)return[];if(speed<.03)return pureTurnPath(pose.origin,q,Math.sign(c.yaw));return movePath(pose.origin,q,c);}}
function faceOrigin(robot,frame,yaw){{let p=rootPosition(robot,frame),forward=new THREE.Vector3(Math.cos(yaw),Math.sin(yaw),0);return p.add(new THREE.Vector3(0,0,.38)).addScaledVector(forward,.32);}}
function trunkYaw(frame){{let q=frame.q[0],quat=new THREE.Quaternion(q[1],q[2],q[3],q[0]),fwd=new THREE.Vector3(1,0,0).applyQuaternion(quat);fwd.z=0;if(fwd.length()<.001)return 0;fwd.normalize();return Math.atan2(fwd.y,fwd.x);}}
function yawQuat(yaw){{return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),yaw);}}
function angleDelta(a,b){{let d=b-a;while(d>Math.PI)d-=Math.PI*2;while(d<-Math.PI)d+=Math.PI*2;return d;}}
function movePath(origin,q,c){{let points=[],start=0.0,horizon=2.75,count=52,w=c.yaw;for(let i=0;i<=count;i++){{let t=start+(horizon-start)*i/count,xy=localFuture(c.vx,c.vy,w,t),v=new THREE.Vector3(xy[0],xy[1],0).applyQuaternion(q);v.z=0;points.push(origin.clone().add(v));}}return points;}}
function localFuture(vx,vy,w,t){{if(Math.abs(w)<.025)return[vx*t,vy*t];return[(vx*Math.sin(w*t)+vy*(Math.cos(w*t)-1))/w,(vx*(1-Math.cos(w*t))+vy*Math.sin(w*t))/w];}}
function pureTurnPath(origin,q,sign){{let points=[],radius=.37,start=sign>0?-.55:.55,span=sign>0?1.35*Math.PI:-1.35*Math.PI;for(let i=0;i<=46;i++){{let a=start+span*i/46,v=new THREE.Vector3(Math.cos(a)*radius,Math.sin(a)*radius,0).applyQuaternion(q);v.z=0;points.push(origin.clone().add(v));}}return points;}}
function attachMesh(robot,spec){{loadGeometry(spec.file).then(g=>{{let mat=spec.color[0]>.55?mats.light:mats.dark;let mesh=new THREE.Mesh(g,mat);mesh.castShadow=true;mesh.receiveShadow=true;mesh.position.set(...spec.pos);mesh.quaternion.set(spec.quat[1],spec.quat[2],spec.quat[3],spec.quat[0]);robot.bodies[spec.body].add(mesh);}});}}
function loadGeometry(file){{if(!geomCache.has(file))geomCache.set(file,new Promise((ok,bad)=>loader.load(file,g=>{{g.computeVertexNormals();ok(g);}},undefined,bad)));return geomCache.get(file);}}
function frameIndex(t,phase){{return (Math.floor(t*DATA.fps)+phase)%DATA.frames.length;}}
function initialRoot(){{let p=DATA.frames[0].p[0];return new THREE.Vector3(p[0],p[1],p[2]);}}
function rootPosition(robot,frame){{let base=initialRoot(),p=frame.p[0],rel=new THREE.Vector3(p[0]-base.x,p[1]-base.y,p[2]-base.z).applyQuaternion(robot.baseQuat);return base.add(robot.offset).add(rel);}}
function bodyPosition(robot,frame,i){{let root=frame.p[0],p=frame.p[i],rel=new THREE.Vector3(p[0]-root[0],p[1]-root[1],p[2]-root[2]).applyQuaternion(robot.baseQuat);return rootPosition(robot,frame).add(rel);}}
function updateRobot(robot,t){{let f=DATA.frames[frameIndex(t,robot.phase)];DATA.bodyNames.forEach((name,i)=>{{let g=robot.bodies[name],p=bodyPosition(robot,f,i),q=f.q[i],quat=new THREE.Quaternion(q[1],q[2],q[3],q[0]);g.position.copy(p);g.quaternion.copy(robot.baseQuat).multiply(quat);}});updateCommandArrow(robot,f);}}
function bodyTarget(index){{let f=DATA.frames[index],p=f.p[0];return new THREE.Vector3(p[0],p[1],Math.max(.62,p[2]+.04));}}
function formationTarget(){{let sum=new THREE.Vector3();robots.forEach(r=>sum.add(r.bodies.Trunk.position));sum.multiplyScalar(1/robots.length);sum.z=Math.max(.7,sum.z+.04);return sum;}}
function cameraTarget(index){{return DATA.view==='multi'||DATA.view==='multi_top'?formationTarget():bodyTarget(index);}}
function stageViewAngle(){{let map={{s1_forward_slow:-2.55,s2_forward_08:-.95,s3_forward_backward:2.25,s4_turn:.45,s5_strafe:3.05,s6_arc:-1.65,s7_diagonal:1.35,s8_omni:-.2}};return map[DATA.stage]??hashAngle(DATA.stage);}}
function hashAngle(text){{let h=0;for(let i=0;i<text.length;i++)h=(h*31+text.charCodeAt(i))|0;return ((Math.abs(h)%628)/100)-Math.PI;}}
function placeCamera(t){{let i=frameIndex(t,0),tar=cameraTarget(i),ang=2.35,dist=2.75,h=.72;if(DATA.view==='orbit')ang+=t*.38+stageViewAngle()*.18;if(DATA.view==='level'){{ang=2.65;dist=2.45;h=.58}}if(DATA.view==='side'){{ang=Math.PI/2;dist=2.35;h=.62}}if(DATA.view==='front'){{ang=Math.PI;dist=2.45;h=.62}}if(DATA.view==='top'){{camera.position.set(tar.x,tar.y,5.2);camera.lookAt(tar);return}}if(DATA.view==='multi_top'){{ang=stageViewAngle();dist=18.2;h=6.1;camera.position.set(tar.x+Math.cos(ang)*dist,tar.y+Math.sin(ang)*dist,tar.z+h);camera.lookAt(tar);return}}if(DATA.view==='wide_top'||DATA.view==='multi'){{ang=stageViewAngle();dist=DATA.view==='multi'?13.8:4.2;h=DATA.view==='multi'?5.2:2.45}}camera.position.set(tar.x+Math.cos(ang)*dist,tar.y+Math.sin(ang)*dist,tar.z+h);camera.lookAt(tar);}}
function animate(now){{let t=now*.001*.92;robots.forEach(r=>updateRobot(r,t));placeCamera(t);renderer.render(scene,camera);requestAnimationFrame(animate);}}
requestAnimationFrame(animate);
</script></body></html>"""


if __name__ == "__main__":
    main()
