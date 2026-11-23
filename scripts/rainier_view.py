"""Local orthographic view of actual Rainier mesh faces and surface stations."""

import json

import numpy as np

from dg_wave.reference import FACE_VERTICES


def write_view(output, mesh, metadata, *, base_href=None):
    rows, faces = np.nonzero(mesh.boundary_tags)
    full = mesh.cells[rows[:, None], FACE_VERTICES[faces]]
    tags = mesh.boundary_tags[rows, faces]
    full_materials = mesh.physical_tags[rows]
    keep = mesh.vertices[mesh.cells, 0].mean(axis=1) <= 0
    candidates = mesh.cells[keep][:, FACE_VERTICES].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(candidates, axis=1), axis=0, return_index=True, return_counts=True)
    cut = candidates[first[counts == 1]]
    cut_materials = mesh.physical_tags[np.flatnonzero(keep)[first[counts == 1]//4]]
    top_vertices = set(full[tags == 10].ravel().tolist())
    cut_tags = [10 if all(int(v) in top_vertices for v in triangle) else 20 for triangle in cut]
    data = {
        "vertices": mesh.vertices.tolist(), "full": full.tolist(), "cut": cut.tolist(),
        "fullTags": tags.tolist(), "cutTags": cut_tags, "metadata": metadata,
        "fullMaterials": full_materials.tolist(), "cutMaterials": cut_materials.tolist(),
    }
    html = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Mount Rainier - terrain and model geometry</title>
<style>
body{font:15px system-ui;max-width:1300px;margin:20px auto;padding:0 16px;color:#1f2328;background:#fff}
h1{font-size:26px;margin-bottom:8px}p{line-height:1.55}.notice{border-left:4px solid #bf8700;padding-left:12px}
.controls{display:flex;flex-wrap:wrap;gap:20px;background:#f6f8fa;padding:12px;border-radius:8px;align-items:center}
canvas{width:100%;height:auto;touch-action:none;border:1px solid #d0d7de;border-radius:8px;cursor:grab}
.scale{margin:12px auto;width:70%}.bar{height:16px;background:linear-gradient(to right,#32679a,#568a61,#c0b078,#b98153,#eee)}
.labels{display:flex;justify-content:space-between}small{color:#57606a}
</style>
<h1 id="title">Mount Rainier - actual terrain and tetrahedral model</h1>
<p id="facts"></p>
<div class="controls"><label><input id="cut" type="checkbox">Cut away eastern half (x &gt; 0)</label>
<label><input id="stations" type="checkbox" checked>Show 10 surface seismometers</label>
<label id="velocityControl" hidden><input id="vp" type="checkbox" checked>Colour actual regions by Vp</label>
<button id="reset">Reset view</button><span>Drag to rotate. No vertical exaggeration.</span></div>
<div class="scale" id="elevationScale"><div class="bar"></div><div class="labels"><span id="low"></span><span>Surface elevation, EGM2008</span><span id="high"></span></div></div>
<div id="velocityLegend" style="display:none;margin:14px 0"></div>
<canvas id="terrain" width="1280" height="760"></canvas>
<p class="notice" id="notice"><b>Geometry only - NOT simulation-ready.</b> One unassigned placeholder volume, no inferred
geological layers or material properties. Lines on side/cut faces are real tetrahedral mesh edges;
terrain-following mesh levels are <b>not geology</b>.</p>
<p id="heights"></p><p id="fidelity"></p>
<p>Terrain source: Microsoft Planetary Computer Copernicus GLO-30 <b>DSM</b>, not a bare-earth DTM.
The available 3DEP tiles do not cover the summit/domain. No products or vertical datums were mixed.
Glacier/ice, vegetation and infrastructure are represented as an <b>effective surface</b>, not separate media.
Main satellite acquisition 2011-2015; exact local acquisition dates are not supplied by this MPC tile.</p>
<p>Surface stations R01-R10 are deterministic random XY locations, seed 20261120, at least 500 m from lateral edges.
Their elevations are on the <b>actual model triangles</b>, not the unmeshed DSM.
Station markers/labels are overlaid for visibility; stations outside the retained cutaway half are hidden.</p>
<p id="attribution"><small></small></p>
<p><a href="model.json">Model metadata</a> | <a href="rainier.msh">Tagged Gmsh mesh</a> |
<a href="mesh.npz">Mesh arrays</a> | <a href="stations.csv">10 stations (CSV)</a> |
<a href="receivers.json">Config-ready receiver positions</a></p>
<p id="geology"></p><p id="layerLinks"></p>
<script>
const data=__DATA__,m=data.metadata,canvas=document.getElementById('terrain'),ctx=canvas.getContext('2d');
const hasLayers=Array.isArray(m.materials);
let az=-.65,el=.62,drag=null;
const lo=m.dimensions.retained_terrain_min_max_m[0],hi=m.dimensions.mesh_highest_elevation_m,base=m.dimensions.flat_base_elevation_m;
document.getElementById('facts').textContent=`25.000 km x 25.000 km; total vertical extent 6.000 km; ${m.mesh.elements.toLocaleString()} conforming affine tetrahedra. Summit-centred UTM10N metric square. Terrain vertices spaced 500 m (not a 30 m simulation mesh).`;
document.getElementById('low').textContent=lo.toFixed(1)+' m';document.getElementById('high').textContent=hi.toFixed(1)+' m';
document.getElementById('heights').textContent=`Retained mesh peak ${hi.toFixed(2)} m; flat bottom ${base.toFixed(2)} m EGM2008. Local depth ${m.dimensions.local_depth_min_max_m.map(x=>(x/1000).toFixed(3)).join(' to ')} km. The published nominal 4392 m summit height was NOT substituted for the sampled terrain.`;
document.getElementById('fidelity').textContent=`500 m facets versus the same DSM at ${m.terrain_sampling.fidelity_points.toLocaleString()} points (125 m spacing): RMS height difference ${m.terrain_sampling.model_minus_DSM_rms_m.toFixed(1)} m; maximum absolute ${m.terrain_sampling.model_minus_DSM_max_abs_m.toFixed(1)} m. This measures geometric approximation, not DEM accuracy or wave resolution.`;
document.querySelector('#attribution small').textContent=m.dataset.attribution;
if(hasLayers){
 document.title='Mount Rainier - six-layer effective Vp model';
 document.getElementById('title').textContent='Mount Rainier - six-layer effective P-wave velocity model';
 document.getElementById('velocityControl').hidden=false;
 document.getElementById('notice').textContent='Depths, Vp, density, Vs and Q are user-adopted MODELLING INPUTS, NOT a measured/calibrated Rainier profile. All five interfaces follow local terrain at fixed vertical depth, including the basement approximation; these are not mapped geological contacts. Layer-specific Q is implemented, but no source band/reference or run is selected. No propagation or Q fitting; NOT simulation-ready.';
 if(m.run_source)document.getElementById('notice').textContent='Run proposal awaiting explicit approval. True DC at3km depth; scalar moment1e12Nm and Gaussian sigma750m are PROPOSED. The slow causal pulse delivers only21.77% of its final moment by6s and is still active at stop. Band/reference and source normalisation were checked statically; no Rainier propagation/pilot/full-model JIT. All geological simplifications still apply.';
 if(m.run_source&&m.run_source.temporal_normalization_seconds!==undefined)document.getElementById('notice').textContent='EXPLORATORY UNDER-RESOLVED proposal awaiting explicit run approval. Requested nominal0.01-2Hz source is NOT filtered to the mesh estimate near0.23Hz; no2Hz accuracy claim.0.01Hz has100s period, not resolved in6s. True DC at3km depth. Proposed scalar moment1e12Nm, sigma750m and C2-tapered sign-changing moment rate; finite-time spectral leakage. No propagation/pilot/full-model JIT.';
 if(m.run_blocked)document.getElementById('notice').textContent='SUPERSEDED LOW-BAND DRAFT. User now requests0.01-2Hz; current conservative mesh estimate is0.228Hz. Source/band finalisation and run approval are paused pending mesh/scope decision. Marker and previous source proposal are retained for context only. No propagation or pilot.';
 if(m.run_approved)document.getElementById('notice').textContent='EXPLORATORY UNDER-RESOLVED run APPROVED. Nominal0.01-2Hz source is not filtered to the mesh estimate0.2298Hz; no2Hz accuracy claim.0.01Hz has100s period, not resolved in6s. True DC3km below summit, net scalar moment1e12Nm, sigma750m, C2-tapered sign-changing moment rate and finite-window leakage. This is the static model/source view; execution status is reported separately in the inbox.';
 const legend=document.getElementById('velocityLegend'),heading=document.createElement('b');
 heading.textContent='P-wave velocity Vp (km/s), piecewise constant in actual material regions. Colours shaded only for geometric relief.';legend.appendChild(heading);
 const table=document.createElement('table');table.style.cssText='border-collapse:collapse;font-size:14px;margin-top:8px;width:100%';
 const header=table.insertRow();for(const text of ['Region / effective unit','Depth below local terrain (km)','Vp (km/s)','Density (kg/m3)','Vs (km/s)','Qp / Qs targets']){const c=document.createElement('th');c.textContent=text;c.style.cssText='text-align:left;padding:6px;border-bottom:1px solid #bbb';header.appendChild(c)}
 for(const a of m.materials){const row=table.insertRow();const texts=[a.tag+' '+a.name,(a.depth_top_m/1000).toFixed(3)+' - '+(a.depth_bottom_m===null?'flat base':(a.depth_bottom_m/1000).toFixed(3)),(a.vp_m_s/1000).toFixed(1),a.rho_kg_m3,(a.vs_m_s/1000).toFixed(1),a.qp+' / '+a.qs];texts.forEach((text,i)=>{const c=row.insertCell();c.textContent=text;c.style.cssText='padding:5px;border-bottom:1px solid #ddd';if(i===0){c.style.borderLeft='12px solid '+a.color;c.style.paddingLeft='8px'}})}
 legend.appendChild(table);
 document.getElementById('geology').textContent=m.geological_interpretation.limitations.join(' ');
 const links=document.getElementById('layerLinks');for(const [href,title] of [['materials.json','Adopted material assumptions and elastic/Q feasibility'],['material_inputs.json','Partial material/Q configuration (band unselected)'],['interfaces.npz','Actual mesh-conforming interfaces'],['../rainier_model/view.html','Preserved neutral geometry view']]){const a=document.createElement('a');a.href=href;a.textContent=title;links.appendChild(a);links.appendChild(document.createTextNode(' | '))}
 if(m.run_source)for(const [file,title] of [['rainier.toml','Run configuration (approval pending)'],['approval.json','Source, frequency, Q and timing evidence']]){const a=document.createElement('a');a.href=new URL(file,window.location.href).href;a.textContent=title;links.appendChild(a);links.appendChild(document.createTextNode(' | '))}
}
function rotate(v){const x=v[0],y=v[1],z=v[2]-(base+3000),h=x*Math.sin(az)+y*Math.cos(az);
return [x*Math.cos(az)-y*Math.sin(az),z*Math.cos(el)-h*Math.sin(el),z*Math.sin(el)+h*Math.cos(el)]}
function colour(z){const t=Math.max(0,Math.min(.99999,(z-lo)/(hi-lo))),p=[[50,103,154],[86,138,97],[192,176,120],[185,129,83],[238,238,238]],i=Math.floor(t*4),u=t*4-i;
return `rgb(${p[i].map((c,j)=>Math.round(c*(1-u)+p[i+1][j]*u)).join(',')})`}
function shadedMaterial(tag,t){
 const hex=m.materials.find(a=>a.tag===tag).color,rgb=[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)),v=t.map(i=>data.vertices[i]),a=v[1].map((x,i)=>x-v[0][i]),b=v[2].map((x,i)=>x-v[0][i]);
 const n=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],light=.72+.28*Math.abs(.3*n[0]-.4*n[1]+.8660254*n[2])/Math.hypot(...n);
 return `rgb(${rgb.map(c=>Math.round(c*light)).join(',')})`;
}
function draw(){
 ctx.clearRect(0,0,1280,760);ctx.fillStyle='#f6f8fa';ctx.fillRect(0,0,1280,760);
 const rv=data.vertices.map(rotate),frame=rv.concat([rotate([-12500,-12500,hi])]),minX=Math.min(...frame.map(v=>v[0])),maxX=Math.max(...frame.map(v=>v[0])),minY=Math.min(...frame.map(v=>v[1])),maxY=Math.max(...frame.map(v=>v[1]));
 const s=Math.min(1080/(maxX-minX),590/(maxY-minY)),ox=640-s*(maxX+minX)/2,oy=365+s*(minY+maxY)/2;
 const project=v=>{const r=rotate(v);return [ox+s*r[0],oy-s*r[1],r[2]]};
 const p=rv.map(v=>[ox+s*v[0],oy-s*v[1],v[2]]),cut=document.getElementById('cut').checked,tri=cut?data.cut:data.full,tags=cut?data.cutTags:data.fullTags;
 const vpMode=hasLayers&&document.getElementById('vp').checked,materials=cut?data.cutMaterials:data.fullMaterials;
 document.getElementById('velocityLegend').style.display=vpMode?'block':'none';document.getElementById('elevationScale').hidden=vpMode;
 const order=tri.map((t,i)=>[i,t.reduce((a,j)=>a+p[j][2],0)/3]).sort((a,b)=>a[1]-b[1]);
 for(const [i] of order){const t=tri[i];ctx.beginPath();t.forEach((j,k)=>k?ctx.lineTo(p[j][0],p[j][1]):ctx.moveTo(p[j][0],p[j][1]));ctx.closePath();
 ctx.fillStyle=vpMode?shadedMaterial(materials[i],t):(tags[i]===10?colour(t.reduce((sum,j)=>sum+data.vertices[j][2],0)/3):'#cbd3dd');ctx.fill();
 ctx.strokeStyle=tags[i]===10?'#36465544':'#67778866';ctx.lineWidth=.35;ctx.stroke();}
 if(document.getElementById('stations').checked)for(const station of m.stations.items){
  if(cut&&station.xyz_m[0]>0)continue;const a=project(station.xyz_m);ctx.beginPath();ctx.arc(a[0],a[1],4,0,2*Math.PI);ctx.fillStyle='#00a6a6';ctx.fill();ctx.strokeStyle='#003d3d';ctx.lineWidth=1;ctx.stroke();
  ctx.font='bold 13px system-ui';ctx.lineWidth=3;ctx.strokeStyle='#fff';ctx.strokeText(station.id,a[0]+7,a[1]-6);ctx.fillStyle='#003d3d';ctx.fillText(station.id,a[0]+7,a[1]-6);}
 if(m.run_source){const a=project(m.run_source.position_local_enu_m);ctx.strokeStyle='#cf222e';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(a[0]-7,a[1]-7);ctx.lineTo(a[0]+7,a[1]+7);ctx.moveTo(a[0]-7,a[1]+7);ctx.lineTo(a[0]+7,a[1]-7);ctx.stroke();ctx.font='bold 14px system-ui';ctx.fillStyle='#cf222e';ctx.fillText('DC source: 3 km below summit (overlaid)',a[0]+12,a[1]);}
 const origin=[-12500,-12500,base];ctx.font='13px system-ui';ctx.fillStyle='#24292f';ctx.strokeStyle='#24292f';ctx.lineWidth=1.5;
 for(const [end,label] of [[[12500,-12500,base],'E: 25 km'],[[-12500,12500,base],'N: 25 km'],[[-12500,-12500,hi],'Z: 6 km']]){
 const a=project(origin),b=project(end);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.fillText(label,b[0]+5,b[1]-5);}
 ctx.beginPath();ctx.moveTo(40,710);ctx.lineTo(40+5000*s,710);ctx.stroke();ctx.fillText('5 km (view plane)',40,735);
 ctx.fillText(cut?'Actual mesh cutaway: west half retained':'Full volume and actual faceted terrain',30,30);
 if(vpMode){ctx.fillText(m.run_source?'Exploratory DC run proposal; source/mesh limitations apply; approval pending':'Adopted effective inputs; geology simplified; no source/band or wave run',30,52);
 ctx.fillText('Vp (km/s)',650,688);m.materials.forEach((a,i)=>{const x=650+85*i;ctx.fillStyle=a.color;ctx.fillRect(x,700,75,15);ctx.fillStyle='#24292f';ctx.fillText((a.vp_m_s/1000).toFixed(1),x+25,735)});}
}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!drag)return;az+=(e.clientX-drag[0])*.008;el=Math.max(.15,Math.min(1.3,el+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];draw()};
canvas.onpointerup=()=>drag=null;document.getElementById('reset').onclick=()=>{az=-.65;el=.62;draw()};
document.getElementById('cut').onchange=()=>{az=document.getElementById('cut').checked?.65:-.65;draw()};document.getElementById('stations').onchange=draw;document.getElementById('vp').onchange=draw;draw();
</script></html>"""
    if base_href is not None:
        from html import escape
        html = html.replace("<title>", f'<base href="{escape(base_href, quote=True)}"><title>', 1)
    (output / "view.html").write_text(html.replace("__DATA__", json.dumps(data).replace("</", "<\\/")),
                                      encoding="utf-8")
