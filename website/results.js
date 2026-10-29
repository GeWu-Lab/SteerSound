'use strict';

// Capability: compare the final editor with other methods (paper Table 1).
const capabilityRows = [
  ['Audio-Omni','7.907B','0.231','0.196','0.154','0.001','—','—'],
  ['MMEdit','9.057B','—','0.159','0.085','0.012','—','14.043'],
  ['AUDIT','1.067B','—','0.151','0.082','0.038','—','—'],
  ['PPAE','1.318B','—','0.182','0.094','0.044','—','—'],
  ['SDEdit','1.118B','—','0.111','0.129','0.129','0.923','5.047'],
  ['ZETA','1.118B','—','0.135','0.140','0.135','0.570','4.199'],
  ['SAM Audio (Small)','0.714B','—','—','—','0.059','—','—'],
  ['SAM Audio (Large)','3.154B','—','—','—','0.056','—','—'],
  ['UNISON','8.086B','0.247','0.250','0.222','0.112','—','—'],
  ['Our base editor','1.348B','0.223','0.249','0.230','0.137','0.055','0.702'],
  ['SteerSound','1.348B','0.220','0.251','0.235','0.144','0.047','0.634'],
];

// Effectiveness: evaluate each backbone before and after the same two STEER stages.
// Selective: [overall success, selective error] (%), paper Table 3.
// Selective-setting task columns: Generate success, Add success, then success/error pairs for
// Replace, Remove, Pitch, Volume. Null means an unsupported task.
// Complex: [TBG_all, TBG_sel], relative to that backbone's baseline, Table 2.
const steerResults = [
  {
    name:'Our editor', tasks:'6 tasks',
    selective:[[70.42,28.48],[84.02,9.99],[85.78,8.45]],
    selectiveTasks:[
      [92.80,93.20,67.90,24.10,72.50,22.80,51.20,48.40,39.10,24.70],
      [94.00,94.60,78.90,14.00,88.10,7.90,87.90,11.70,61.60,6.80],
      [94.80,93.50,79.10,12.30,88.20,7.90,92.80,6.80,68.00,6.40],
    ],
    complex:[[0,0],[1.67,2.97],[5.36,8.25]],
    primary:[['0.22','0.25','0.23','0.14','0.06','0.70'],['0.22','0.25','0.23','0.14','0.05','0.71'],['0.22','0.25','0.23','0.14','0.05','0.63']],
  },
  {
    name:'UNISON', tasks:'4 tasks',
    selective:[[61.30,38.90],[74.28,11.05],[75.25,10.55]],
    selectiveTasks:[
      [96.40,90.10,33.40,50.90,25.30,26.90,null,null,null,null],
      [94.80,85.60,78.40,13.60,38.30,8.50,null,null,null,null],
      [94.90,85.40,79.60,13.10,41.10,8.00,null,null,null,null],
    ],
    complex:[[0,0],[10.27,17.85],[10.81,18.42]],
    primary:[['0.25','0.25','0.22','0.11','—','—'],['0.25','0.26','0.24','0.15','—','—'],['0.25','0.27','0.23','0.15','—','—']],
  },
  {
    name:'SAM Audio (Small)', tasks:'Removal',
    selective:[[37.60,55.90],[80.60,19.10],[84.10,15.70]],
    selectiveTasks:[
      [null,null,null,null,37.60,55.90,null,null,null,null],
      [null,null,null,null,80.60,19.10,null,null,null,null],
      [null,null,null,null,84.10,15.70,null,null,null,null],
    ],
    complex:[[0,0],[108.62,108.62],[124.53,124.53]],
    primary:[['—','—','—','0.06','—','—'],['—','—','—','0.12','—','—'],['—','—','—','0.13','—','—']],
  },
];
const steerStageNames = ['Original model','+ Emphasis','+ Recovery (Full STEER)'];
const steerStages = [
  {name:'Original model', note:'Before STEER', className:'original'},
  {name:'+ Emphasis', note:'Stage 1', className:'emphasis'},
  {name:'+ Recovery', note:'Stage 2 · Full STEER', className:'full'},
];

function resultTable(headers, rows, caption) {
  return `<table><caption class="sr-only">${caption}</caption><thead><tr>${headers.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
}

function renderCapability() {
  const rows = capabilityRows.map((row, i) => `<tr${i===capabilityRows.length-1?' class="highlight-row"':''}>${row.map((v,j)=>j===0?`<th scope="row">${v}</th>`:`<td>${v}</td>`).join('')}</tr>`);
  $('#capability-results').innerHTML = `<div class="table-wrap" tabindex="0" role="region" aria-label="Overall unified audio editing capability; scroll horizontally on small screens">${resultTable(
    ['Model','Total params.','Generate<br><small>CLAP ↑</small>','Add<br><small>CLAP ↑</small>','Replace<br><small>CLAP ↑</small>','Remove<br><small>CLAP ↑</small>','Pitch<br><small>MAE (st) ↓</small>','Volume<br><small>MAE (dB) ↓</small>'], rows,
    'Complex setting test set, primary metrics from paper Table 1. Generate / Add / Replace / Remove: task-aware CLAP ↑. Pitch: MAE in semitones ↓. Volume: MAE in dB ↓. “—” indicates an unsupported task.'
  )}</div>`;
}

function stageComparison(model, setting) {
  const selective = setting==='selective';
  const headings = selective
    ? ['Success<span>(%) ↑</span>','Selective error<span>(%) ↓</span>']
    : ['Overall<span>TBG<sub>all</sub> ↑</span>','Selective<span>TBG<sub>sel</sub> ↑</span>'];
  const rows = model[setting].map((values,i)=>`<tr class="stage-${steerStages[i].className}"><th scope="row"><span class="stage-checkpoint" aria-hidden="true">${i}</span><span><strong>${steerStages[i].name}</strong><small>${steerStages[i].note}</small></span></th><td>${values[0].toFixed(2)}</td><td>${values[1].toFixed(2)}</td></tr>`).join('');
  return `<table class="stage-comparison"><caption class="sr-only">${model.name}: original model, Emphasis, and Emphasis plus Recovery on ${selective?'the Selective setting':'the Complex setting'}</caption><thead><tr><th scope="col">Checkpoint</th>${headings.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>`;
}

function selectiveCard(model) {
  const data = model.selective;
  const reduction = (data[0][1]-data[2][1])/data[0][1]*100;
  return `<article class="backbone-card"><div class="backbone-heading"><h5>${model.name}</h5><span>${model.tasks}</span></div>${stageComparison(model,'selective')}<div class="stage-outcome"><span>Full STEER vs original model</span><strong>−${reduction.toFixed(1)}%</strong><p>relative selective error</p></div></article>`;
}

function complexCard(model) {
  const stages = model.complex.slice(1).map((values,index) => {
    const full = index===1;
    const metrics = values.map((value,i)=>`<div><span>${i===0?'Overall':'Selective'} gain</span><strong>+${value.toFixed(2)}<small>%</small></strong></div>`).join('');
    return `<section class="gain-stage ${full?'gain-full':'gain-emphasis'}" aria-label="${model.name}: ${full?'Emphasis plus Recovery':'Emphasis alone'}, gains over the original model"><div class="gain-stage-heading"><h6>${full?'+ Recovery':'+ Emphasis'}</h6><span>${full?'Full STEER':'Stage 1'}</span></div><div class="gain-stage-metrics">${metrics}</div></section>`;
  }).join('');
  return `<article class="backbone-card gain-card"><div class="backbone-heading"><h5>${model.name}</h5><span>${model.tasks}</span></div>${stages}</article>`;
}

function effectDetails(setting) {
  const selective = setting==='selective';
  const rows = steerResults.flatMap((model) => model[setting].map((values,i) => {
    const scores = values.map(value => `<td>${selective?value.toFixed(2):i===0?'Reference':`+${value.toFixed(2)}%`}</td>`).join('');
    const taskValues = selective
      ? model.selectiveTasks[i].map(value=>value===null?'—':value.toFixed(2))
      : model.primary[i];
    return `<tr class="${i===0?'model-group ':''}${i===2?'highlight-row':''}"><th scope="row">${model.name}</th><td>${steerStageNames[i]}</td>${taskValues.map(value=>`<td>${value}</td>`).join('')}${scores}</tr>`;
  }));
  const taskNames = ['Generate','Add','Replace','Remove','Pitch','Volume'];
  const taskHeaders = taskNames.map((task,i) => `<th scope="colgroup" colspan="${selective&&i>1?2:1}">${task}</th>`).join('');
  const metricHeaders = taskNames.map((task,i) => selective
    ? '<th scope="col">Succ. (%) ↑</th>'+(i>1?'<th scope="col">SErr. (%) ↓</th>':'')
    : `<th scope="col">${i<4?'CLAP ↑':i===4?'MAE (st) ↓':'MAE (dB) ↓'}</th>`).join('');
  const overallHeaders = selective
    ? '<th scope="col">Succ. (%) ↑</th><th scope="col">SErr. (%) ↓</th>'
    : '<th scope="col">Overall ↑</th><th scope="col">Selective ↑</th>';
  const header = `<thead><tr><th scope="col" rowspan="2">Backbone</th><th scope="col" rowspan="2">Training stage</th>${taskHeaders}<th scope="colgroup" colspan="2">${selective?'Overall':'Task-balanced gain'}</th></tr><tr>${metricHeaders}${overallHeaders}</tr></thead>`;
  const caption = selective
    ? 'Task-wise results from paper Table 3. Succ. = success rate; SErr. = selective error rate. All rates are percentages. Generate and Add report success only; the four selective tasks also report SErr. “—” indicates an unsupported task. Overall rates follow the paper’s valid-sample aggregation; unmeasurable pitch examples are excluded.'
    : 'Paper Table 2, at its reported precision. Generate / Add / Replace / Remove: task-aware CLAP ↑. Pitch: MAE in semitones ↓. Volume: MAE in dB ↓. Overall / selective gains are TBG_all / TBG_sel, expressed as percentages relative to the original model (Reference). “—” indicates an unsupported task.';
  const table = `<table class="task-stage-table"><caption class="sr-only">${caption}</caption>${header}<tbody>${rows.join('')}</tbody></table>`;
  return `<details class="effect-details"><summary>Task-wise results · all training stages <span aria-hidden="true">+</span></summary><div class="table-wrap" tabindex="0" role="region" aria-label="Task-wise STEER training-stage comparison; scroll horizontally to view all tasks">${table}</div></details>`;
}

function renderEffectiveness(setting) {
  const panel = $('#effectiveness-panel');
  panel.setAttribute('aria-labelledby',`effect-tab-${setting}`);
  if (setting==='selective') {
    panel.innerHTML = `<div class="effect-finding"><div><h4>From the original model to full STEER.</h4><p>Original model → Emphasis → Emphasis + Recovery.</p></div><div class="effect-badge">&gt;70%<span>relative selective error reduction<br>full STEER vs each original model</span></div></div><div class="backbone-grid">${steerResults.map(selectiveCard).join('')}</div>${effectDetails(setting)}`;
  } else {
    panel.innerHTML = `<div class="effect-finding"><div><h4>STEER improves each original model.</h4><p>Cumulative gains over each original model.</p></div><div class="effect-badge">+8.25%<span>SteerSound’s selective gain<br>over our original editor</span></div></div><div class="backbone-grid">${steerResults.map(complexCard).join('')}</div>${effectDetails(setting)}`;
  }
}

function initializeResults() {
  renderCapability();
  bindTabs('.effectiveness-switch [role=tab]','setting',renderEffectiveness);
  renderEffectiveness('selective');
}
