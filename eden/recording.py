"""Export a compact spectator snapshot and build a self-contained offline viewer.

The full audit ledger remains authoritative. This extract omits raw model calls,
observation payloads, and scheduler internals; it preserves recorded display states.
"""
import argparse
import gzip
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

from .store import canonical, decode_state, digest


def extract(database):
    uri = "file:" + quote(str(Path(database).resolve())) + "?immutable=1"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    pool, pool_index, frames, events = [], {}, [], []
    previous, count = "0" * 64, 0
    try:
        for row in db.execute("SELECT * FROM events ORDER BY id"):
            state, payload = decode_state(row["state"]), json.loads(row["payload"])
            body = {key: row[key] for key in ("id", "time", "kind", "actor", "state_hash", "prev_hash")}
            body["payload"] = payload
            count += 1
            if row["id"] != count or row["prev_hash"] != previous or digest(state) != row["state_hash"] or digest(body) != row["hash"]:
                raise ValueError(f"Archive verification failed at event {row['id']}")
            previous = row["hash"]
            objects, agents = [], {}
            for ident, entity in state["entities"].items():
                if entity["kind"] == "agent":
                    agents[ident] = {k: v for k, v in entity.items() if k in {
                        "id", "name", "kind", "type", "pos", "alive", "born", "needs", "conditions",
                        "busy", "deliberation", "last_result", "files", "action_history"}}
                else:
                    signature = canonical(entity)
                    if signature not in pool_index:
                        pool_index[signature] = len(pool)
                        pool.append(entity)
                    objects.append(pool_index[signature])
            frames.append({"id": row["id"], "time": row["time"], "agents": agents, "objects": objects,
                           "seed": state["seed"], "voxels": state["voxels"], "weather": state["weather"],
                           "config": {k: state["config"][k] for k in ("model", "backend", "speed")}})
            events.append({**body, "hash": row["hash"],
                           "payload": {k: v for k, v in payload.items() if k not in {"observation", "scheduled_event"}}})
    finally:
        db.close()
    if not frames:
        raise ValueError("Archive has no events")
    return {"format": "eden-spectator-1", "objects": pool, "frames": frames, "events": events,
            "source_verification": {"valid": True, "events": count, "head_hash": previous,
                "scope": "Full source ledger verified during export. This compact viewer omits raw model calls and observation payloads."}}


BRIDGE = r'''
const recording = JSON.parse(document.getElementById('eden-recording').textContent);
const frameFor = n => recording.frames[Math.max(0, Math.min(recording.frames.length-1, n-1))];
function recordedState(frame) {
  return {time:frame.time, seed:frame.seed, voxels:frame.voxels, weather:frame.weather,
    config:frame.config, entities:Object.fromEntries([...frame.objects.map(i=>{
      const e=recording.objects[i];return [e.id,e];}), ...Object.entries(frame.agents)])};
}
function recordedSurface(state,x,z){
  const ix=Math.floor(x), iz=Math.floor(z);
  let y=Math.floor(7+3*Math.sin((x+state.seed%97)/19)+2*Math.cos(z/23)+Math.sin((x+z)/7));
  for(const [key,material] of Object.entries(state.voxels)){
    const [vx,vy,vz]=key.split(',').map(Number);
    if(vx===ix&&vz===iz&&!['air','water'].includes(material))y=Math.max(y,vy);
  }
  while(state.voxels[`${ix},${y},${iz}`]==='air')y--;
  return y;
}
window.fetch=async function(url, options){
  if(options?.method==='POST')throw new Error('This is an archived snapshot; simulation controls are disabled.');
  const address=new URL(url,'http://localhost'), q=address.searchParams;
  let answer;
  if(address.pathname==='/api/session')answer={readonly:true,admin_token:null};
  else if(address.pathname==='/api/verify')answer=recording.source_verification;
  else if(address.pathname==='/api/state'){
    const frame=frameFor(q.has('at')?Number(q.get('at')):recording.frames.length);
    const state=recordedState(frame), agents=Object.values(frame.agents);
    const selected=agents.find(a=>a.id===q.get('follow'))||agents[0], [x,,z]=selected.pos;
    const tiles=[];
    for(let px=Math.trunc(x)-24;px<=Math.trunc(x)+24;px+=2)
      for(let pz=Math.trunc(z)-24;pz<=Math.trunc(z)+24;pz+=2){
        const y=recordedSurface(state,px,pz);tiles.push({x:px,z:pz,y,biome:y<=3?'water':'land'});
      }
    answer={revision:frame.id,head:recording.frames.length,state,terrain:tiles,status:'archived snapshot',paused:true,error:null};
  }else if(address.pathname==='/api/events'){
    answer=recording.events.filter(e=>e.id>Number(q.get('after')||0)&&e.id<=Number(q.get('through')||recording.frames.length)).slice(0,Number(q.get('limit')||200));
  }else throw new Error('Unsupported archived endpoint');
  return new Response(JSON.stringify(answer),{headers:{'Content-Type':'application/json'}});
};
'''

TRAILS = r'''
const drawOriginal=draw;
draw=function(){
  drawOriginal();if(!data)return;
  const canvas=document.getElementById('map'), ctx=canvas.getContext('2d');
  const agents=Object.values(data.state.entities).filter(e=>e.kind==='agent');
  const center=(data.state.entities[follow]||agents[0]).pos;
  const project=p=>[(p[0]-center[0]-p[2]+center[2])*zoom*.87+canvas.clientWidth/2+panX,
    (p[0]-center[0]+p[2]-center[2])*zoom*.43-(p[1]-center[1])*zoom*.7+canvas.clientHeight*.48+panY];
  ctx.save();ctx.strokeStyle='#d8e2d8';ctx.globalAlpha=.5;ctx.lineWidth=1.5;
  agents.forEach((agent,index)=>{
    ctx.setLineDash(index?[4,4]:[]);ctx.beginPath();
    const frames=recording.frames.slice(Math.max(0,data.revision-120),data.revision);
    frames.forEach((f,i)=>{const a=f.agents[agent.id];if(!a)return;const p=project(a.pos);i?ctx.lineTo(...p):ctx.moveTo(...p);});ctx.stroke();
  });ctx.restore();
};
document.getElementById('live').textContent='Latest snapshot';
document.getElementById('verify').textContent='Source verification';
document.getElementById('notice').textContent='Recorded Bouchet snapshot · drag the history slider or press Play replay. No model calls. Trails: Adam solid, Eve dashed (last 120 events).';
zoom=25;
if(data)draw();
'''


def write_html(recording, destination):
    static = Path(__file__).parent / "static"
    html = (static / "index.html").read_text()
    encoded = canonical(recording).replace("<", "\\u003c")
    html = html.replace('<link rel="stylesheet" href="/style.css">', "<style>" + (static / "style.css").read_text() + "</style>")
    scripts = ('<script id="eden-recording" type="application/json">' + encoded + '</script><script>' + BRIDGE
               + '</script><script>' + (static / "app.js").read_text() + '</script><script>' + TRAILS + '</script>')
    html = html.replace('<script src="/app.js"></script>', scripts)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--database", help="Standalone checkpoint only; opens in immutable read-only mode")
    group.add_argument("--recording", help="Previously exported .json.gz")
    parser.add_argument("--output", required=True, help="Output .html or .json.gz")
    args = parser.parse_args()
    recording = extract(args.database) if args.database else json.loads(gzip.decompress(Path(args.recording).read_bytes()))
    if args.output.endswith(".html"):
        write_html(recording, args.output)
    elif args.output.endswith(".json.gz"):
        Path(args.output).write_bytes(gzip.compress(canonical(recording).encode()))
    else:
        parser.error("Output must end in .html or .json.gz")
    print(json.dumps({"output": args.output, "events": len(recording["frames"]),
                      "sim_seconds": recording["frames"][-1]["time"], "bytes": Path(args.output).stat().st_size}))


if __name__ == "__main__":
    main()
