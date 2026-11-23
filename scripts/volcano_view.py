"""Render the generated terrain mesh without external plotting dependencies."""

import json
from pathlib import Path

import numpy as np

from dg_wave.mesh import read_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.reference import FACE_VERTICES

mesh = read_mesh("volcano.msh")
prepared = prepare_mesh(mesh, 2)
elements, faces = np.nonzero(mesh.boundary_tags)
triangles = mesh.cells[elements[:, None], FACE_VERTICES[faces]]
data = {
    "vertices": mesh.vertices.tolist(), "triangles": triangles.tolist(),
    "regions": mesh.physical_tags[elements].tolist(),
    "boundary": mesh.boundary_tags[elements, faces].tolist(),
}
assert np.all(prepared.geometry.volumes > 0)
assert np.count_nonzero(mesh.boundary_tags == 10) > 0
lower, upper = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
html = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Synthetic volcano - actual DGWave mesh</title>
<style>
body{font:16px system-ui;margin:28px auto;max-width:1080px;padding:0 20px;background:#fff;color:#1f2328}
h1{font-size:26px;margin-bottom:8px}p{line-height:1.55}.legend{display:flex;gap:24px;flex-wrap:wrap}
.swatch{display:inline-block;width:17px;height:13px;border:1px solid #536474;vertical-align:middle;margin-right:6px}
canvas{width:100%;height:auto;border:1px solid #d0d7de;border-radius:8px;touch-action:none;cursor:grab}
button{padding:7px 12px;border:1px solid #8c959f;border-radius:6px;background:#f6f8fa}
small{color:#57606a}
</style>
<h1>Synthetic volcanic terrain - Stage 7.1</h1>
<p>Actual generated mesh: <b>__CELLS__ conforming linear tetrahedra, __VERTICES__ vertices</b>.
Footprint __LENGTH_X__ x __LENGTH_Y__ m; elevation __LOW_Z__ to __HIGH_Z__ m.
Piecewise planar terrain, not curved elements. <b>No vertical exaggeration.</b></p>
<div class="legend"><span><i class="swatch" style="background:#5b9bd5"></i>Region 1: synthetic basement (__REGION1__ tetrahedra)</span>
<span><i class="swatch" style="background:#e5ad58"></i>Region 2: synthetic upper unit (__REGION2__ tetrahedra)</span></div>
<p><small>Drag to rotate. Visible exterior triangles are actual tetrahedron faces.
The coloured side walls expose the conforming material interface.</small> <button id="reset">Reset view</button></p>
<canvas id="model" width="1080" height="660" aria-label="Rotatable actual volcanic tetrahedral mesh with two geological regions"></canvas>
<p><b>Terrain boundary:</b> __TERRAIN__ triangles, physical tag 10, traction-free using outward
geometric normals. <b>Lateral/base:</b> physical tag 20, characteristic absorbing.</p>
<p><b>Status:</b> geometry/tag/orientation and non-axis-aligned zero-traction tests passed;
focused serial/four-rank terrain and full attenuation-state restart comparisons passed.
This is a geometry view, not a resolved wavefield or geological accuracy claim.
GPU checks remain blocked; the demonstrator uses CPU FP64.</p>
<p><a href="volcano-model.msh">Download actual Gmsh mesh</a> |
<a href="volcano-model.toml">Synthetic solver input</a></p>
<script>
const mesh=DATA;
const centre=__CENTRE__,axes=__AXES__,scaleDistance=__SCALE__;
const canvas=document.getElementById('model'),ctx=canvas.getContext('2d');
let az=-0.70,el=0.56,drag=null,scale=0.28,offsetX=540,offsetY=320;
function rotate(v) {
 const [x,y,z]=v.map((q,i)=>q-centre[i]),h=x*Math.sin(az)+y*Math.cos(az);
 return [x*Math.cos(az)-y*Math.sin(az),z*Math.cos(el)-h*Math.sin(el),z*Math.sin(el)+h*Math.cos(el)];
}
function project(v) {
 const q=rotate(v);return [offsetX+scale*q[0],offsetY-scale*q[1],q[2]];
}
function draw(){
 ctx.clearRect(0,0,1080,660);ctx.fillStyle='#f6f8fa';ctx.fillRect(0,0,1080,660);
 const r=mesh.vertices.map(rotate),xmin=Math.min(...r.map(q=>q[0])),xmax=Math.max(...r.map(q=>q[0]));
 const ymin=Math.min(...r.map(q=>q[1])),ymax=Math.max(...r.map(q=>q[1]));
 scale=Math.min(940/(xmax-xmin),510/(ymax-ymin));
 offsetX=540-scale*(xmin+xmax)/2;offsetY=320+scale*(ymin+ymax)/2;
 const p=mesh.vertices.map(project);
 const order=mesh.triangles.map((t,i)=>[i,t.reduce((a,j)=>a+p[j][2],0)/3]).sort((a,b)=>a[1]-b[1]);
 for(const [i] of order){
  const t=mesh.triangles[i];ctx.beginPath();
  t.forEach((v,j)=>j?ctx.lineTo(p[v][0],p[v][1]):ctx.moveTo(p[v][0],p[v][1]));
  ctx.closePath();ctx.fillStyle=mesh.regions[i]===1?'#5b9bd5':(mesh.boundary[i]===10?'#f0bf74':'#e5ad58');
  ctx.fill();ctx.strokeStyle='#40566c';ctx.lineWidth=0.65;ctx.stroke();
 }
 ctx.font='15px system-ui';ctx.lineWidth=2;
 const origin=project(__LOWER__);
 for(const [end,label] of axes){
  const q=project(end);ctx.strokeStyle='#24292f';ctx.beginPath();ctx.moveTo(origin[0],origin[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();
  ctx.fillStyle='#24292f';ctx.fillText(label,Math.max(14,Math.min(950,q[0]+7)),Math.max(22,Math.min(630,q[1]-8)));
 }
 ctx.strokeStyle='#24292f';ctx.beginPath();ctx.moveTo(36,610);ctx.lineTo(36+scaleDistance*scale,610);ctx.stroke();
 ctx.fillText(`${scaleDistance.toLocaleString()} m (view-plane scale)`,36,640);
 ctx.fillText('Terrain: tag 10 / traction-free',36,32);
}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!drag)return;az+=(e.clientX-drag[0])*0.008;el=Math.max(-0.2,Math.min(1.35,el+(e.clientY-drag[1])*0.006));drag=[e.clientX,e.clientY];draw()};
canvas.onpointerup=()=>drag=null;
document.getElementById('reset').onclick=()=>{az=-0.70;el=0.56;draw()};
draw();
</script></html>"""
html = html.replace("DATA", json.dumps(data))
axes = []
for axis, name in enumerate("xyz"):
    end = lower.copy()
    end[axis] = upper[axis]
    axes.append([end.tolist(), f"{name}: {upper[axis]:,.0f} m"])
values = {
    "__CELLS__": f"{len(mesh.cells):,}", "__VERTICES__": f"{len(mesh.vertices):,}",
    "__REGION1__": str(np.count_nonzero(mesh.physical_tags == 1)),
    "__REGION2__": str(np.count_nonzero(mesh.physical_tags == 2)),
    "__TERRAIN__": str(np.count_nonzero(mesh.boundary_tags == 10)),
    "__LENGTH_X__": f"{upper[0]-lower[0]:,.0f}", "__LENGTH_Y__": f"{upper[1]-lower[1]:,.0f}",
    "__LOW_Z__": f"{lower[2]:,.0f}", "__HIGH_Z__": f"{upper[2]:,.0f}",
    "__CENTRE__": json.dumps(((lower+upper)/2).tolist()), "__AXES__": json.dumps(axes),
    "__SCALE__": str(max(upper[:2]-lower[:2])/4), "__LOWER__": json.dumps(lower.tolist()),
}
for key, value in values.items():
    html = html.replace(key, value)
Path("output/volcano-model.html").write_text(html, encoding="utf-8")
for suffix in (".msh", ".toml"):
    Path("output/volcano-model"+suffix).write_bytes(Path("volcano"+suffix).read_bytes())
print(json.dumps({
    "elements": len(mesh.cells), "vertices": len(mesh.vertices),
    "region_counts": dict(zip(*[a.tolist() for a in np.unique(mesh.physical_tags, return_counts=True)])),
    "terrain_triangles": int(np.count_nonzero(mesh.boundary_tags == 10)),
    "positive_volume_min_m3": float(prepared.geometry.volumes.min()),
    "file": "output/volcano-model.html",
}))
