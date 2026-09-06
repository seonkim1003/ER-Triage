'use strict';
const $ = id => document.getElementById(id);
let patient = null, position = 0, timer = null, requestVersion = 0, searchVersion = 0;
const units = {HR:'bpm',O2Sat:'%',Temp:'°C',SBP:'mmHg',MAP:'mmHg',DBP:'mmHg',Resp:'/min'};
const fmt = (n, digits=3) => n == null ? 'Not available' : Number(n).toFixed(digits);
async function get(url) {
  const response = await fetch(url);
  let data;
  try { data = await response.json(); } catch { throw new Error('The local server returned an unreadable response.'); }
  if (!response.ok) throw new Error(data.error || 'The local server could not load this record.');
  return data;
}
function stop() { if (timer) clearInterval(timer); timer=null; $('play').textContent='▶ Play'; }
function controls(disabled) {
  for (const id of ['play','step','restart','scrubber','retrospective','vital','speed']) $(id).disabled=disabled;
}
async function loadPatient(id) {
  stop(); controls(true); const version=++requestVersion;
  $('workspace').hidden=true; $('previous').disabled=true; $('next').disabled=true;
  $('notice').className=''; $('notice').textContent='Checking patient file and prediction alignment…';
  try {
    const next = await get('/api/patient?id='+encodeURIComponent(id));
    if (version !== requestVersion) return;
    patient=next; position=0; $('retrospective').checked=false;
    $('patient-title').textContent=id; $('patient-search').value=id;
    $('scrubber').max=patient.rows.length-1; $('scrubber').value=0;
    $('previous').disabled=!patient.previous; $('next').disabled=!patient.next;
    $('workspace').hidden=false; controls(false); $('notice').textContent='Verified historical record · Frozen model scores · No live patient input';
    render();
  } catch(error) {
    if (version !== requestVersion) return;
    patient=null; $('notice').className='error'; $('notice').textContent=error.message+' Choose a held-out identifier and try Open again.';
    $('review-status').textContent='Record unavailable'; $('review-reason').textContent='No history is being displayed.'; $('next-review').textContent='';
  }
}
function svgChart(rows, key, height, bounds, missing=false) {
  const width=Math.max(300,$('score-chart').clientWidth), left=48, right=16, top=18, bottom=27;
  const x=i=>left+i*(width-left-right)/Math.max(patient.rows.length-1,1);
  const y=v=>top+(bounds[1]-v)*(height-top-bottom)/(bounds[1]-bounds[0]);
  let s='';
  if ($('retrospective').checked && patient.timing.onset_proxy_hour != null) {
    const onsetIndex=patient.timing.onset_proxy_hour-patient.rows[0].hour;
    const lo=Math.max(0,onsetIndex-12), hi=Math.min(patient.rows.length-1,onsetIndex-6);
    if (hi>=lo) s+=`<rect class="window" x="${x(lo)}" y="${top}" width="${Math.max(1,x(hi)-x(lo))}" height="${height-top-bottom}"/>`;
    if(onsetIndex>=0 && onsetIndex<patient.rows.length) s+=`<line class="onset" x1="${x(onsetIndex)}" x2="${x(onsetIndex)}" y1="${top}" y2="${height-bottom}"/>`;
  }
  for(let tick=0;tick<=3;tick++) {
    const v=bounds[0]+(bounds[1]-bounds[0])*tick/3, yy=y(v);
    s+=`<line class="grid" x1="${left}" x2="${width-right}" y1="${yy}" y2="${yy}"/><text x="${left-9}" y="${yy+4}" text-anchor="end">${v.toFixed(key==='score'?2:0)}</text>`;
  }
  for(let tick=0;tick<=4;tick++) {
    const i=Math.round((patient.rows.length-1)*tick/4);
    s+=`<text x="${x(i)}" y="${height-7}" text-anchor="middle">${patient.rows[i].hour}</text>`;
  }
  if (key==='score' && patient.threshold<=1) s+=`<line class="threshold" x1="${left}" x2="${width-right}" y1="${y(patient.threshold)}" y2="${y(patient.threshold)}"/>`;
  let path='', pen=false;
  rows.forEach((row,i)=>{
    const value=row[key];
    if(value==null){pen=false;if(missing)s+=`<circle class="missing" cx="${x(i)}" cy="${height-bottom+4}" r="1.2"/>`;return;}
    path+=(pen?' L':' M')+x(i)+' '+y(value); pen=true;
    s+=`<circle class="observation" cx="${x(i)}" cy="${y(value)}" r="${rows.length>150?1.5:2.6}"><title>ICU hour ${row.hour}: ${value.toFixed(key==='score'?3:1)}</title></circle>`;
  });
  s+=`<path class="trace" d="${path}"/><line class="cursor" x1="${x(position)}" x2="${x(position)}" y1="${top}" y2="${height-bottom}"/>`;
  if (!rows.some(r=>r[key]!=null)) s+=`<text x="${width/2}" y="${height/2}" text-anchor="middle">No recorded measurements in the visible history</text>`;
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${key==='score'?'Model score':'Recorded '+key} by ICU hour. Exact values are available below.">${s}</svg>`;
}
function eventChart(rows) {
  const width=Math.max(300,$('events-chart').clientWidth), x=i=>48+i*(width-64)/Math.max(patient.rows.length-1,1);
  let s='<text x="0" y="23">Alert</text><text x="0" y="57">Review</text>';
  s+=`<line class="grid" x1="48" x2="${width-16}" y1="20" y2="20"/><line class="grid" x1="48" x2="${width-16}" y1="54" y2="54"/>`;
  rows.forEach((r,i)=>{
    const xx=x(i);
    if(r.alert)s+=`<path class="alert" d="M${xx} 14 l-4 8 h8 Z"><title>Threshold alert at hour ${r.hour}</title></path>`;
    if(r.review_now)s+=`<circle class="review" cx="${xx}" cy="54" r="3"><title>Review at hour ${r.hour}</title></circle>`;
  });
  s+=`<line class="cursor" x1="${x(position)}" x2="${x(position)}" y1="5" y2="68"/>`;
  return `<svg viewBox="0 0 ${width} 75" role="img" aria-label="Threshold alerts and scheduled reviews by ICU hour">${s}</svg>`;
}
function renderTable(rows, retrospective) {
  const fragment=document.createDocumentFragment();
  const vital=$('vital').value;
  $('vital-heading').textContent=vital+' ('+units[vital]+')';
  for(const row of rows) {
    const tr=document.createElement('tr');
    const age=row[vital+'_age'];
    const values=[row.hour,fmt(row.score),row.alert?'Yes':'No',row.review_now?'Yes':'No',
      row[vital]==null?'Missing':fmt(row[vital],1),age===999?'Never observed':age+'h',row.reason];
    if(retrospective) values.push(row.retrospective_label);
    for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}
    fragment.append(tr);
  }
  $('hourly-body').replaceChildren(fragment); $('label-heading').hidden=!retrospective;
}
function render() {
  if(!patient)return;
  const retro=$('retrospective').checked, current=patient.rows[position];
  const rows=retro?patient.rows:patient.rows.slice(0,position+1);
  const prefix=patient.rows.slice(0,position+1);
  $('scrubber').value=position; $('clock').textContent='ICU hour '+current.hour;
  $('position').textContent=(position+1)+' / '+patient.rows.length+' recorded hours';
  $('mode-label').textContent=retro?'Retrospective view':'History prefix';
  $('outcome-note').textContent=retro?outcomeText():'Future scores, observations and outcomes are hidden.';
  $('review-status').textContent=current.review_now?'Review at this hour':'Review not due yet';
  $('review-reason').textContent=current.reason;
  $('next-review').textContent='Next scheduled review in '+current.next_review_in_hours+'h, unless new information changes the plan.';
  const maxScore=Math.min(1,Math.max(.05,Math.min(patient.threshold,1)*1.2,...rows.map(r=>r.score*1.15)));
  $('score-chart').innerHTML=svgChart(rows,'score',190,[0,maxScore]);
  $('score-value').textContent='Current '+fmt(current.score)+' · Threshold '+(patient.threshold>1?'no alerts':fmt(patient.threshold));
  $('events-chart').innerHTML=eventChart(rows);
  $('event-count').textContent=`Through current hour: ${prefix.filter(r=>r.alert).length} threshold-positive hours · ${prefix.filter(r=>r.review_now).length} review events`;
  $('timing-summary').hidden=!retro;
  const timing=patient.timing;
  $('timing-summary').textContent=retro?`Full record: ${timing.alert_episodes} alert episodes · ${timing.repeated_episodes} repeated episodes. `+
    (timing.first_alert_hour==null?'No threshold alert.':`First threshold alert: ICU hour ${timing.first_alert_hour}. `)+
    (timing.timing_eligible?(timing.early_warning?'Alert present in the early window.':'No alert in the early window.'):''):'';
  const vital=$('vital').value, values=rows.map(r=>r[vital]).filter(v=>v!=null);
  let lo=values.length?Math.min(...values):0, hi=values.length?Math.max(...values):100;
  const pad=Math.max((hi-lo)*.2,1);lo-=pad;hi+=pad;
  $('vital-chart').innerHTML=svgChart(rows,vital,180,[lo,hi],true);
  const age=current[vital+'_age'];
  $('vital-value').textContent=(current[vital]==null?'No measurement this hour':fmt(current[vital],1)+' '+units[vital])+' · '+(age===999?'Never observed so far':age+'h since last observation');
  renderTable(rows,retro);
  $('step').disabled=position===patient.rows.length-1;
  if(position===patient.rows.length-1)stop();
}
function outcomeText() {
  const t=patient.timing;
  if(!t.positive_patient)return 'Retrospective: no positive label in this record.';
  if(t.onset_proxy_hour==null)return 'Retrospective: onset timing unavailable ('+t.timing_status.replaceAll('_',' ')+').';
  return 'Label-derived onset proxy: ICU hour '+t.onset_proxy_hour+'. Shading marks 12–6h before it. '+(t.timing_eligible?'Timing eligible.':t.timing_status.replaceAll('_',' ')+'.');
}
function play() {
  if(!patient)return;
  if(timer){stop();return;}
  if(position===patient.rows.length-1)position=0;
  $('play').textContent='Ⅱ Pause';render();
  timer=setInterval(()=>{position=Math.min(position+1,patient.rows.length-1);render();},Number($('speed').value));
}
function renderEvaluation(e) {
  const dest=$('evaluation');dest.replaceChildren();
  if(!e){dest.textContent='Start the viewer with --evaluation to show the matching cohort report.';return;}
  const dl=document.createElement('dl');
  const frac=e.measures.early_warning_fraction;
  const ci=e.bootstrap.intervals.early_warning_fraction;
  for(const [label,value] of [
    ['Warned in the 12–6h window',frac==null?'Undefined':(100*frac).toFixed(1)+'%'+(ci?' ['+(ci.low*100).toFixed(1)+', '+(ci.high*100).toFixed(1)+']':'' )],
    ['Patients in timing denominator',e.timing_eligible_patients.toLocaleString()+' / '+e.positive_patients.toLocaleString()+' positive'],
    ['Missed that early window',e.missed_early_window_patients.toLocaleString()],
    ['Nonsepsis patients ever alerted',e.measures.nonsepsis_patients_with_any_alert==null?'Undefined':(100*e.measures.nonsepsis_patients_with_any_alert).toFixed(1)+'%']]) {
    const group=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;group.append(dt,dd);dl.append(group);
  }
  const note=document.createElement('p');note.className='eval-note muted';note.textContent='95% patient-bootstrap interval. “Missed” refers to this analysis window, not a clinical diagnosis.';dest.append(dl,note);
}
$('patient-form').addEventListener('submit',e=>{e.preventDefault();loadPatient($('patient-search').value.trim());});
$('patient-search').addEventListener('input',async()=>{
  const version=++searchVersion;
  try{const data=await get('/api/patients?q='+encodeURIComponent($('patient-search').value));if(version!==searchVersion)return;
    $('patient-options').replaceChildren(...data.patients.map(p=>{const o=document.createElement('option');o.value=p;return o;}));
    $('search-count').textContent=data.matches+' matching records'+(data.matches>50?' · showing the first 50':'');
  }catch{$('search-count').textContent='Search unavailable. Check that the local server is running.';}
});
$('previous').addEventListener('click',()=>patient?.previous&&loadPatient(patient.previous));
$('next').addEventListener('click',()=>patient?.next&&loadPatient(patient.next));
$('play').addEventListener('click',play);
$('step').addEventListener('click',()=>{stop();position=Math.min(position+1,patient.rows.length-1);render();});
$('restart').addEventListener('click',()=>{stop();position=0;render();});
$('scrubber').addEventListener('input',()=>{stop();position=Number($('scrubber').value);render();});
$('speed').addEventListener('change',()=>{if(timer){stop();play();}});
$('vital').addEventListener('change',render);
$('retrospective').addEventListener('change',()=>{stop();render();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
window.addEventListener('resize',render);
async function init(){
  controls(true);
  try{const meta=await get('/api/meta');$('run-name').textContent=meta.run;$('run-detail').textContent=meta.patients.toLocaleString()+' held-out records · '+meta.model;renderEvaluation(meta.evaluation);await loadPatient(meta.first_patient);}
  catch(error){$('notice').className='error';$('notice').textContent=error.message+' Reload after restarting the local server.';}
}
init();
