"""A small local 3D view of measured quadratic DG surface snapshots."""

import json

import numpy as np


def write_view(output, comparison):
    models = []
    for level in comparison.get("levels", ("coarse", "fine")):
        with np.load(output / level / "surface.npz", allow_pickle=False) as data:
            vz = data["surface_state"][..., 2].reshape(len(data["times_s"]), -1)
            models.append({
                "name": f'{level}: {comparison["runs"][level]["elements"]:,} tetrahedra',
                "level": level,
                "points": data["face_positions"].reshape(-1, 3).tolist(),
                "triangles": data["triangles"].tolist(), "vz": vz.tolist(),
            })
    maximum = max(float(np.max(np.abs(model["vz"]))) for model in models)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("nonzero finite measured surface wavefield required for colour scale")
    frame_max = np.max(np.abs(models[-1]["vz"]), axis=1)
    data = {
        "models": models, "times": comparison["spatial"]["times_s"], "limit": maximum,
        "initialFrame": int(np.argmax(frame_max)), "comparison": comparison,
    }
    html = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>DGWave - surface wavefield refinement</title>
<style>
body{font:15px system-ui;color:#1f2328;background:white;max-width:1600px;margin:24px auto;padding:0 16px}
h1{font-size:25px;margin-bottom:8px}p{line-height:1.55}.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
canvas{width:100%;height:auto;border:1px solid #d0d7de;border-radius:8px;touch-action:none;cursor:grab}
h2{font-size:18px}.bar{height:20px;background:linear-gradient(to right,#2166ac,#f7f7f7,#b2182b);border:1px solid #8c959f}
.labels{display:flex;justify-content:space-between;font-variant-numeric:tabular-nums}.scale{max-width:620px;margin:15px auto}
.control{display:flex;gap:15px;align-items:center;padding:12px;background:#f6f8fa;border-radius:6px}input{flex:1}
table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:7px;border:1px solid #d0d7de;text-align:right}th:first-child,td:first-child{text-align:left}
small{color:#57606a}.notice{border-left:4px solid #bf8700;padding-left:12px}
@media(max-width:720px){.views{grid-template-columns:1fr}}
</style>
<h1>Actual free-surface wavefield - preliminary mesh comparison</h1>
<p>Vertical velocity <b>v<sub>z</sub> in m/s</b>: red = upward, blue = downward.
One common symmetric colour scale across all meshes and every saved time.
Drag any terrain to rotate all views; <b>no vertical exaggeration</b>.</p>
<div class="control"><label for="time">Snapshot time</label><input id="time" type="range" min="0" step="1"><b id="timeLabel"></b><button id="reset">Reset view</button></div>
<div class="scale"><div class="bar"></div><div class="labels"><span id="negative"></span><span>0 m/s</span><span id="positive"></span></div></div>
<div class="views" id="views"></div>
<p><b>Source:</b> vertical Gaussian body force (not an explosion), (800,800,700) m,
600 m beneath the summit. The dashed marker is an <b>x-ray location/depth marker</b>, not a visible cavity.
Velocity-RHS acceleration amplitude 0.2 m/s<sup>2</sup>; spatial width 180 m; delay 0.5 s;
C2 quintic turn-on over 0-0.3 s.
Fourier weighting band 0.5-3 Hz, multiplied by the ramp: <b>not strictly bandlimited</b>.</p>
<p><small>These are real DG free-face snapshots, not element-mean or conceptual images.
Each displayed microtriangle is coloured by the mean of three actual degree-2 face nodal values.
This display tessellation adds no numerical resolution; discontinuous traces are retained.
Refined meshes subdivide the same original planar terrain/material facets exactly.</small></p>
<p id="difference" class="notice"></p>
<p>Synthetic basement: rho 2400 kg/m3, vp 3600 m/s, vs 2000 m/s;
upper unit: rho 2100 kg/m3, vp 2800 m/s, vs 1500 m/s.
Both Qp=60, Qs=40; reference 1.5 Hz. Input speeds specify reference storage moduli.</p>
<h2>Measured four-core runtime and memory</h2>
<table><thead><tr><th>Run</th><th>Compile (s)</th><th>Solve (s)</th><th>Output (s)</th><th>Job wall (s)</th><th>Max rank peak (MiB)</th><th>Sum rank peaks (MiB)</th></tr></thead><tbody id="metrics"></tbody></table>
<p><small>Linux ru_maxrss rank peaks; their sum is NOT a simultaneous node peak and excludes
launcher/orchestrator/previews. Solve/output/compilation report maximum-rank phase times; job wall
includes launch through exit. Actual rank-to-logical-CPU/socket/core and thread settings are in each metrics.json.</small></p>
<p class="notice">Mesh levels show preliminary waveform differences, not an established convergence order or proof
of resolution. Geometry is held fixed. No arrival-time/reflection acceptance sweep or PML study.
GPU work remains Pending; all runs use four bound physical CPU cores.</p>
<p id="dataLinks"><a href="comparison.json">Comparison and run metrics (JSON)</a> |
<a href="comparison.npz">Matched-sample velocities (NPZ)</a> |
<a href="coarse/surface.npz">Coarse full surface DG state</a> |
<a href="fine/surface.npz">Fine full surface DG state</a></p>
<script>
const study=__DATA__,slider=document.getElementById('time');
let az=-.65,el=.62,frame=study.initialFrame,drag=null;
slider.max=study.times.length-1;slider.value=frame;
document.getElementById('negative').textContent=(-study.limit).toExponential(3)+' m/s';
document.getElementById('positive').textContent=study.limit.toExponential(3)+' m/s';
study.models.forEach((m,i)=>{
 const section=document.createElement('section'),name=document.createElement('h2'),canvas=document.createElement('canvas');
 name.textContent=m.name+'; degree 2';canvas.id='view'+i;canvas.width=600;canvas.height=470;
 section.append(name,canvas);document.getElementById('views').appendChild(section);
 if(i>=2){const link=document.createElement('a');link.href=m.level+'/surface.npz';link.textContent=m.name+' full surface DG state';
 document.getElementById('dataLinks').append(' | ',link);}
});
const spatial=study.comparison.spatial.spacetime_relative_l2,temporal=study.comparison.temporal_coarse.spacetime_relative_l2;
document.getElementById('difference').textContent=`Surface velocity space-time relative L2 difference: ${(100*spatial).toFixed(2)}% (coarse vs fine reference); coarse half-timestep difference: ${temporal.toExponential(3)}. Matched area quadrature points/times, all three velocity components. This is a difference, not exact error.`;
if(study.comparison.spatial_fine_veryfine){
 const next=study.comparison.spatial_fine_veryfine.spacetime_relative_l2,control=study.comparison.next_time_control;
 document.getElementById('difference').textContent+=` Fine vs veryfine: ${(100*next).toFixed(2)}%, using the IDENTICAL 1536 original quadrature points and saved times.`;
 if(control)document.getElementById('difference').textContent+=` Fine timestep control relative L2: ${control.fine_relative_temporal.spacetime_relative_l2.toExponential(3)}. Original fine dt=${study.comparison.runs.fine.dt_s}s; veryfine dt=${study.comparison.runs.veryfine.dt_s}s.`;
}
for(const key of Object.keys(study.comparison.runs)){
 const m=study.comparison.runs[key],row=document.createElement('tr');
 for(const value of [key,m.compile_seconds_max_rank,m.solve_seconds_max_rank,m.output_seconds_max_rank,m.launcher_wall_seconds,m.peak_rss_max_rank_mib,m.sum_rank_peak_rss_mib]){
  const c=document.createElement('td');c.textContent=typeof value==='number'?value.toFixed(2):value;row.appendChild(c);
 }document.getElementById('metrics').appendChild(row);
}
function rotated(v){const x=v[0]-800,y=v[1]-800,z=v[2]-1000,h=x*Math.sin(az)+y*Math.cos(az);
return [x*Math.cos(az)-y*Math.sin(az),z*Math.cos(el)-h*Math.sin(el),z*Math.sin(el)+h*Math.cos(el)]}
function colour(value){
 const t=Math.min(1,Math.abs(value)/study.limit),end=value<0?[33,102,172]:[178,24,43];
 return `rgb(${end.map(c=>Math.round(247*(1-t)+c*t)).join(',')})`;
}
function draw(){
 document.getElementById('timeLabel').textContent=study.times[frame].toFixed(2)+' s';
 const domain=study.models.flatMap(m=>m.points).concat([[800,800,700]]).map(rotated);
 const minX=Math.min(...domain.map(p=>p[0])),maxX=Math.max(...domain.map(p=>p[0])),minY=Math.min(...domain.map(p=>p[1])),maxY=Math.max(...domain.map(p=>p[1]));
 const scale=Math.min(530/(maxX-minX),340/(maxY-minY)),ox=300-scale*(minX+maxX)/2,oy=225+scale*(minY+maxY)/2;
 const project=v=>{const p=rotated(v);return [ox+scale*p[0],oy-scale*p[1],p[2]]};
 study.models.forEach((m,index)=>{
  const ctx=document.getElementById('view'+index).getContext('2d');ctx.clearRect(0,0,600,470);ctx.fillStyle='#f6f8fa';ctx.fillRect(0,0,600,470);
  const p=m.points.map(project),order=m.triangles.map((t,i)=>[i,t.reduce((s,j)=>s+p[j][2],0)/3]).sort((a,b)=>a[1]-b[1]);
  for(const [i] of order){const t=m.triangles[i];ctx.beginPath();t.forEach((j,k)=>k?ctx.lineTo(p[j][0],p[j][1]):ctx.moveTo(p[j][0],p[j][1]));ctx.closePath();
   ctx.fillStyle=colour(t.reduce((sum,j)=>sum+m.vz[frame][j],0)/3);ctx.fill();ctx.strokeStyle='#58677a66';ctx.lineWidth=.35;ctx.stroke();
  }
  const source=project([800,800,700]),summit=project([800,800,1300]);ctx.setLineDash([5,4]);ctx.strokeStyle='#24292f';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(...source.slice(0,2));ctx.lineTo(...summit.slice(0,2));ctx.stroke();ctx.setLineDash([]);
  ctx.beginPath();ctx.arc(source[0],source[1],5,0,2*Math.PI);ctx.fillStyle='#fff';ctx.fill();ctx.stroke();ctx.font='12px system-ui';ctx.fillStyle='#24292f';ctx.fillText('source -600 m',source[0]+8,source[1]+13);
  ctx.beginPath();ctx.moveTo(24,438);ctx.lineTo(24+400*scale,438);ctx.stroke();ctx.fillText('400 m (view plane)',24,458);
  ctx.fillText('z upward / actual faceted terrain',20,24);
 });
}
slider.oninput=()=>{frame=Number(slider.value);draw()};
document.getElementById('reset').onclick=()=>{az=-.65;el=.62;draw()};
for(const c of document.querySelectorAll('canvas')){c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};
c.onpointermove=e=>{if(!drag)return;az+=(e.clientX-drag[0])*.008;el=Math.max(.15,Math.min(1.25,el+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];draw()};c.onpointerup=()=>drag=null;}
draw();
</script></html>"""
    (output / "view.html").write_text(html.replace("__DATA__", json.dumps(data)), encoding="utf-8")
